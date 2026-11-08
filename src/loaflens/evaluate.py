"""CLI evaluasi: ``python -m loaflens.evaluate --config configs/{baseline,anomaly}.yaml``.

Model (hasil ``train``) dinilai pada val dan test secara terpisah dengan threshold dari config
dan threshold terkalibrasi (dari CV train+val, bukan test). Ditambah cross-validation per grup pada
train+val (test tidak ikut), domain check foto konveyor (``eval.domain_check``), cek shortcut per
fitur (baseline) dan overlay heatmap (anomaly). Output di ``output.dir``: report.md, metrics.json,
scores.csv, calibration.yaml, roc_test.png, domain_check.jpg (+ salinan ``eval.docs_copy``).

``--domain-check FOTO``: hanya domain check satu foto (tanpa evaluasi penuh). ``--domain-scale S``:
foto diperkecil dulu ×S (INTER_AREA) untuk menguji ketahanan resolusi; bila ``labels.json`` di
folder foto punya entri foto itu, jumlah kempes tertangkap / roti OK ter-reject ikut dilaporkan.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from loaflens.audit import name_prefix
from loaflens.calibrate import calibrate, write_calibration
from loaflens.config import Config, load_config
from loaflens.data import dhash, group_ids, load_image, near_duplicate_pairs, read_manifest
from loaflens.score import Scorer, load_scorer, model_path, new_scorer
from loaflens.segment import segment_all
from loaflens.train import LABEL_TO_Y

log = logging.getLogger(__name__)

HEADLINE = "Hasil pada 48 foto studio; indikasi kelayakan, bukan klaim akurasi produksi."
GROUP_NAMES = {
    ("normal", "bagus"): "bagus",
    ("normal", "medium"): "medium-good",
    ("defect", "bad"): "bad",
    ("defect", "medium"): "medium-bad",
}
GROUP_ORDER = ("bagus", "medium-good", "medium-bad", "bad")
SUSPICIOUS_AUROC = 0.98
TITLES = {
    "baseline": "scorer A (baseline: fitur tekstur/warna/bentuk + logistic regression)",
    "anomaly": "scorer B (anomaly: PatchCore-lite, ResNet18 beku)",
}


# --- metrik (murni, mudah di-test) ---------------------------------------------------------


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC; NaN bila hanya satu kelas."""
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def recall_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> tuple[float, float, float]:
    """(recall, FPR tercapai, threshold skor) terbaik dengan FPR <= target, dari kurva ROC."""
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.flatnonzero(fpr <= target + 1e-12)
    i = ok[np.argmax(tpr[ok])]
    return float(tpr[i]), float(fpr[i]), float(thr[i])


def binary_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion matrix + accuracy/precision/recall untuk prediksi reject = skor >= threshold."""
    pred = s >= threshold
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }


def prefix_group(path: str, label: str) -> str:
    """Kelompok nama file: bagus / medium-good / bad / medium-bad (prefix sebelum ' (n)')."""
    prefix = name_prefix(path).strip().lower()
    return GROUP_NAMES.get((label, prefix), f"{label}:{prefix}")


# --- data skor -----------------------------------------------------------------------------


@dataclass
class Scored:
    """Skor satu himpunan gambar."""

    paths: list[str]
    labels: list[str]
    scores: np.ndarray
    latency_ms: list[float] = field(default_factory=list)

    @property
    def y(self) -> np.ndarray:
        return np.array([LABEL_TO_Y[lab] for lab in self.labels])

    def prefix_means(self) -> dict[str, float]:
        """Skor rata-rata per kelompok prefix."""
        groups: dict[str, list[float]] = {}
        for p, lab, s in zip(self.paths, self.labels, self.scores, strict=True):
            groups.setdefault(prefix_group(p, lab), []).append(float(s))
        return {g: float(np.mean(v)) for g, v in groups.items()}


def score_split(scorer: Scorer, rows: list[dict[str, str]], root: Path) -> Scored:
    """Skor tiap gambar + latency (segmentasi + representasi + skor, tanpa baca file)."""
    paths, labels, scores, lat = [], [], [], []
    if rows:  # pemanasan (alokasi pertama OpenCV/sklearn/torch) agar latency tidak bias
        for _ in range(2):
            scorer.score(load_image(rows[0]["path"], root))
    for r in rows:
        img = load_image(r["path"], root)
        t0 = time.perf_counter()
        scores.append(scorer.score(img))
        lat.append((time.perf_counter() - t0) * 1000)
        paths.append(r["path"])
        labels.append(r["label"])
    return Scored(paths, labels, np.array(scores), lat)


def cv_groups(rows: list[dict[str, str]], root: Path, max_dist: int) -> list[int]:
    """Grup CV = batch_id digabung near-duplicate (aturan yang sama dengan split manifest)."""
    hashes = [dhash(load_image(r["path"], root)) for r in rows]
    pairs = near_duplicate_pairs(hashes, max_dist)
    return group_ids([r["batch_id"] for r in rows], pairs)


@dataclass
class CVResult:
    aurocs: list[float]
    accs: list[float]
    oof: np.ndarray  # skor out-of-fold per gambar


def cv_splits(y: np.ndarray, groups: list[int], cfg: Config) -> list[tuple[np.ndarray, ...]]:
    """Fold StratifiedGroupKFold deterministik (seed config) — sama untuk semua scorer."""
    folds = StratifiedGroupKFold(n_splits=cfg.eval.cv_folds, shuffle=True, random_state=cfg.seed)
    return list(folds.split(np.zeros(len(y)), y, groups))


def cross_validate(
    scorer: Scorer, reprs: list[Any], y: np.ndarray, groups: list[int], cfg: Config
) -> CVResult:
    """CV pada train+val: scorer di-fit ulang per fold dari representasi (anomaly: normal saja)."""
    aurocs, accs, oof = [], [], np.zeros(len(y))
    for tr, te in cv_splits(y, groups, cfg):
        s = scorer.fit_repr([reprs[i] for i in tr], y[tr]).score_repr([reprs[i] for i in te])
        oof[te] = s
        aurocs.append(auroc(y[te], s))
        accs.append(binary_metrics(y[te], s, cfg.threshold)["accuracy"])
    return CVResult(aurocs, accs, oof)


def feature_shortcuts(
    x: np.ndarray, y: np.ndarray, groups: list[int], cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    """AUROC satu fitur: rata-rata di fold uji (arah dari fold latih) dan gabungan (arah bebas)."""
    splits = cv_splits(y, groups, cfg)
    feat = np.zeros((len(splits), x.shape[1]))
    for k, (tr, te) in enumerate(splits):
        for j in range(x.shape[1]):
            sign = 1.0 if auroc(y[tr], x[tr, j]) >= 0.5 else -1.0
            feat[k, j] = auroc(y[te], sign * x[te, j])
    pooled = np.array([auroc(y, x[:, j]) for j in range(x.shape[1])])
    return np.nanmean(feat, axis=0), np.maximum(pooled, 1 - pooled)


@dataclass
class CVRun:
    result: CVResult
    scored: Scored
    reprs: list[Any]
    y: np.ndarray
    groups: list[int]


def run_cv(cfg: Config, scorer: Scorer, rows: list[dict[str, str]], root: Path) -> CVRun:
    """Representasi train+val dihitung sekali (scorer terlatih), lalu CV dengan scorer baru."""
    reprs = [scorer.represent(load_image(r["path"], root)) for r in rows]
    y = np.array([LABEL_TO_Y[r["label"]] for r in rows])
    groups = cv_groups(rows, root, cfg.data.near_dup_hamming)
    cv = cross_validate(new_scorer(cfg), reprs, y, groups, cfg)  # model train tidak disentuh
    scored = Scored([r["path"] for r in rows], [r["label"] for r in rows], cv.oof)
    log.info("CV AUROC %.3f ± %.3f", np.nanmean(cv.aurocs), np.nanstd(cv.aurocs))
    return CVRun(cv, scored, reprs, y, groups)


def split_rows(cfg: Config, root: Path) -> dict[str, list[dict[str, str]]]:
    """Baris manifest per split."""
    rows = read_manifest(root / cfg.data.manifest)
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}


def calibration_inputs(
    cfg: Config, root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(y_cv, skor_cv_oof, y_val, skor_val) — data NON-test untuk kalibrasi threshold."""
    scorer = load_scorer(cfg, root)
    by_split = split_rows(cfg, root)
    val = score_split(scorer, by_split["val"], root)
    cv = run_cv(cfg, scorer, by_split["train"] + by_split["val"], root)
    return cv.y, cv.result.oof, val.y, val.scores


# --- domain check (foto konveyor; semua roti OK) -------------------------------------------


@dataclass
class DomainCheck:
    path: str
    scores: list[float]
    bboxes: list[tuple[int, int, int, int]]
    thresholds: dict[str, float]  # nama → threshold

    def n_reject(self, name: str) -> int:
        return int(sum(s >= self.thresholds[name] for s in self.scores))


def domain_check(
    scorer: Scorer,
    cfg: Config,
    path: str,
    thresholds: dict[str, float],
    root: Path,
    out: Path,
    scale: float = 1.0,
) -> DomainCheck:
    """Skor tiap roti terdeteksi di foto konveyor; tulis gambar beranotasi ke ``out``.

    Tool tidak tahu label tiap roti: bila semua roti di foto OK, setiap REJECT = false positive.
    Tidak ada retraining/tuning dari foto ini. ``scale`` < 1: foto diperkecil dulu (INTER_AREA);
    bbox di hasil tetap dalam piksel foto yang sudah diperkecil.
    """
    img = load_image(path, root)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    segs = segment_all(img, cfg.segment, cfg.camera.roi)
    scores = [scorer.score(img, sg) for sg in segs]
    res = DomainCheck(path, scores, [sg.bbox for sg in segs], thresholds)
    vis = img.copy()
    t_cfg = thresholds["config"]
    t_cal = thresholds.get("kalibrasi")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (sg, s) in enumerate(zip(segs, scores, strict=True)):
        rej = s >= t_cfg
        color = (0, 0, 255) if rej else (0, 200, 0)
        cv2.drawContours(vis, [sg.contour], -1, color, 2)
        x, y, w, h = sg.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        text = f"#{i + 1} {s:.0f} {'REJECT' if rej else 'OK'}"
        if t_cal is not None:
            text += f" | cal {'REJ' if s >= t_cal else 'OK'}"
        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
        ty = max(y - 6, th + 4)
        cv2.rectangle(vis, (x, ty - th - 4), (x + tw + 4, ty + 3), (255, 255, 255), cv2.FILLED)
        cv2.putText(vis, text, (x + 2, ty), font, 0.55, color, 2)
    legend = f"{cfg.model.type}: thr config {t_cfg:g}" + (
        f", kalibrasi {t_cal:g}" if t_cal is not None else ""
    )
    hh = vis.shape[0]
    cv2.rectangle(vis, (0, hh - 28), (vis.shape[1], hh), (255, 255, 255), cv2.FILLED)
    cv2.putText(vis, legend, (6, hh - 8), font, 0.6, (0, 0, 0), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    if hasattr(scorer, "heatmap_crop"):  # scorer B: strip heatmap per roti
        tiles = []
        for sg, s in zip(segs, scores, strict=True):
            c_img, c_mask = scorer.prepare(img, sg)
            tiles.append(heat_tile(scorer, c_img, c_mask, f"{s:.0f}"))
        if tiles:
            cv2.imwrite(str(out.with_name(out.stem + "_heatmaps.jpg")), grid_sheet(tiles, 8))
    log.info(
        "Domain check %s: %d roti, REJECT %s → %s",
        path,
        len(scores),
        {k: res.n_reject(k) for k in thresholds},
        out,
    )
    return res


def domain_table(dc: DomainCheck, image_note: str) -> list[str]:
    """Bagian laporan domain check (markdown)."""
    names = list(dc.thresholds)
    n = len(dc.scores)
    out = [
        f"Foto: `{dc.path}` — roti terdeteksi: **{n}** (roti terpotong di tepi frame dibuang). "
        "Label per roti tidak diketahui tool; bila semua roti di foto OK, **setiap REJECT = "
        "false positive** di domain konveyor. Tidak ada retraining/tuning dari foto ini. "
        + image_note,
        "",
        "| threshold | nilai | REJECT |",
        "|---|---|---|",
    ]
    for k in names:
        out.append(f"| {k} | {dc.thresholds[k]:g} | {dc.n_reject(k)}/{n} |")
    if n:
        s = np.array(dc.scores)
        out += [
            "",
            f"Skor: min {s.min():.1f}, median {np.median(s):.1f}, maks {s.max():.1f}. "
            "Per roti (urut atas→bawah): "
            + ", ".join(f"#{i + 1}={v:.0f}" for i, v in enumerate(s)),
        ]
    return out


# --- gambar --------------------------------------------------------------------------------


def grid_sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    """Susun tile berukuran sama jadi grid."""
    h, w = tiles[0].shape[:2]
    tiles = tiles + [np.zeros((h, w, 3), np.uint8)] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def heat_tile(scorer: Any, c_img: np.ndarray, c_mask: np.ndarray, caption: str) -> np.ndarray:
    """Tile [crop | overlay heatmap] + keterangan (scorer B)."""
    from loaflens.anomaly import heatmap_overlay

    hm = scorer.heatmap_crop(c_img, c_mask)
    tile = np.hstack([c_img, heatmap_overlay(c_img, c_mask, hm)])
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 18), (255, 255, 255), cv2.FILLED)
    cv2.putText(tile, caption, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return tile


HEAT_NOTE = (
    "Overlay heatmap: JET biru = skor patch ≤ 25, merah = ≥ 75 (skala 0–100 yang sama dengan skor "
    "gambar); hanya patch di dalam mask roti yang dierosi."
)


def save_heatmaps(scorer: Any, sets: dict[str, Scored], root: Path, out_dir: Path) -> None:
    """Overlay heatmap (hanya di dalam mask roti) untuk val/test → out_dir + sheet.jpg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for split, sc in sets.items():
        for p, lab, s in zip(sc.paths, sc.labels, sc.scores, strict=True):
            c_img, c_mask = scorer.prepare(load_image(p, root))
            caption = f"{split} {prefix_group(p, lab)} {Path(p).stem} skor {s:.0f}"
            tile = heat_tile(scorer, c_img, c_mask, caption)
            cv2.imwrite(str(out_dir / f"{split}_{lab}_{Path(p).stem}.jpg"), tile)
            tiles.append(tile)
    if tiles:
        cv2.imwrite(str(out_dir / "sheet.jpg"), grid_sheet(tiles, 2))
    log.info("Heatmap: %d gambar → %s", len(tiles), out_dir)


def draw_roc(y: np.ndarray, s: np.ndarray, path: Path, title: str, size: int = 400) -> None:
    """Kurva ROC sederhana digambar dengan OpenCV (tanpa matplotlib)."""
    fpr, tpr, _ = roc_curve(y, s)
    pad = 50
    img = np.full((size + 2 * pad, size + 2 * pad, 3), 255, np.uint8)

    def pt(fx: float, ty: float) -> tuple[int, int]:
        return int(pad + fx * size), int(pad + (1 - ty) * size)

    cv2.rectangle(img, pt(0, 0), pt(1, 1), (0, 0, 0), 1)
    cv2.line(img, pt(0, 0), pt(1, 1), (180, 180, 180), 1)
    pts = np.array([pt(f, t) for f, t in zip(fpr, tpr, strict=True)], np.int32)
    cv2.polylines(img, [pts], False, (200, 80, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title, (pad, 30), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "FPR", (pad + size // 2 - 15, size + pad + 35), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "TPR", (pad - 40, pad - 8), font, 0.5, (0, 0, 0), 1)
    for v in (0.0, 0.5, 1.0):
        cv2.putText(img, f"{v:.1f}", (pt(v, 0)[0] - 10, size + pad + 15), font, 0.4, (0, 0, 0), 1)
        cv2.putText(img, f"{v:.1f}", (pad - 30, pt(0, v)[1] + 4), font, 0.4, (0, 0, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def write_scores_csv(path: Path, sets: dict[str, Scored], threshold: float) -> None:
    """scores.csv: split, path, label, prefix, score, pred (cv_oof = skor out-of-fold CV)."""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["split", "path", "label", "prefix", "score", "pred"])
        for split, sc in sets.items():
            for p, lab, s in zip(sc.paths, sc.labels, sc.scores, strict=True):
                pred = "defect" if s >= threshold else "normal"
                w.writerow([split, p, lab, prefix_group(p, lab), f"{s:.2f}", pred])


# --- laporan -------------------------------------------------------------------------------


def fmt(v: float, nd: int = 3) -> str:
    return "n/a" if v != v else f"{v:.{nd}f}"  # NaN → n/a


def prefix_table(sc: Scored, threshold: float) -> list[str]:
    """Rata-rata skor + jumlah salah per kelompok prefix."""
    out = ["| kelompok | n | skor rata-rata | min–max | salah |", "|---|---|---|---|---|"]
    groups: dict[str, list[int]] = {}
    for i, (p, lab) in enumerate(zip(sc.paths, sc.labels, strict=True)):
        groups.setdefault(prefix_group(p, lab), []).append(i)
    for g in (*GROUP_ORDER, *sorted(groups)):
        if g not in groups:
            continue
        idx = groups.pop(g)
        s, y = sc.scores[idx], sc.y[idx]
        wrong = int(np.sum((s >= threshold) != (y == 1)))
        out.append(f"| {g} | {len(idx)} | {s.mean():.1f} | {s.min():.1f}–{s.max():.1f} | {wrong} |")
    return out


def top_errors(sc: Scored, threshold: float, n: int) -> list[str]:
    """Kesalahan dengan margin salah-sisi terbesar."""
    wrong = [i for i in range(len(sc.paths)) if (sc.scores[i] >= threshold) != (sc.y[i] == 1)]
    wrong.sort(key=lambda i: -abs(sc.scores[i] - threshold))
    if not wrong:
        return [f"(tidak ada kesalahan pada threshold {threshold:g})"]
    return [
        f"- `{sc.paths[i]}` — label {sc.labels[i]}, skor {sc.scores[i]:.1f} "
        f"(margin {abs(sc.scores[i] - threshold):.1f})"
        for i in wrong[:n]
    ]


def holdout_rows(sets: dict[str, Scored], threshold: float, fpr_target: float) -> list[str]:
    """Baris tabel hold-out untuk satu threshold."""
    out = []
    for name, sc in sets.items():
        y = sc.y
        rec, fpr_at, _ = recall_at_fpr(y, sc.scores, fpr_target)
        m = binary_metrics(y, sc.scores, threshold)
        out.append(
            f"| {name} | {threshold:g} | {len(y)} ({int((y == 0).sum())}/{int(y.sum())}) | "
            f"{fmt(auroc(y, sc.scores))} | {fmt(rec, 2)} ({fmt(fpr_at, 2)}) | "
            f"{fmt(m['accuracy'], 2)} | {fmt(m['precision'], 2)} | {fmt(m['recall'], 2)} | "
            f"{m['tn']} {m['fp']} {m['fn']} {m['tp']} |"
        )
    return out


@dataclass
class EvalResult:
    """Semua hasil evaluasi satu config (bahan laporan + metrics.json)."""

    cfg: Config
    scorer: Scorer
    sets: dict[str, Scored]
    cv: CVRun
    cal: dict[str, Any]
    n_train: int
    seg_failures: int
    model_bytes: int
    domain: list[DomainCheck]
    shortcuts: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def t_cal(self) -> float:
        return float(self.cal["suggested_threshold"])

    @property
    def latencies(self) -> list[float]:
        return [v for sc in self.sets.values() for v in sc.latency_ms]


def model_lines(r: EvalResult) -> list[str]:
    """Deskripsi model di kepala laporan."""
    p = r.cfg.model.params
    if r.cfg.model.type == "baseline":
        return [
            f"- Classifier: `{p['classifier']}` + StandardScaler; "
            f"{len(r.scorer.feature_names)} fitur (tekstur, warna/kilap, bentuk bebas skala) "
            f"dari crop roti {p['crop_size']} px. Dilatih dari normal + kempes.",
        ]
    return [
        f"- PatchCore-lite: crop roti {p['image_size']} px (latar diisi abu) → ResNet18 "
        "pretrained ImageNet **beku** (feature extractor; layer2 + layer3, avg-pool "
        f"{p.get('pool', 3)}x3) → patch di dalam mask roti (erosi {p.get('mask_erode_px', 0)} px) "
        f"→ memory bank {len(r.scorer.bank)} patch dari roti **normal saja** → jarak kNN "
        f"(k={p.get('knn', 1)}), skor gambar = `{p.get('image_score', 'max')}` jarak patch.",
        f"- Skor 0–100 = clip(50 · mentah / ref); ref = p{p['ref_percentile']:g} skor "
        f"leave-one-image-out roti normal train = {r.scorer.ref:.3f}. Skor 50 ≈ seaneh roti "
        "normal train paling aneh; > 50 = di luar variasi normal yang pernah dilihat.",
    ]


def render_report(r: EvalResult, with_images: bool) -> str:
    """Susun laporan markdown (bahasa Indonesia)."""
    cfg, e, t, tc = r.cfg, r.cfg.eval, r.cfg.threshold, r.t_cal
    cv, cv_set = r.cv.result, r.cv.scored
    out = [
        f"# Hasil evaluasi {TITLES[cfg.model.type]} — LoafLens",
        "",
        f"> **{HEADLINE}**",
        "",
        f"- Tanggal: {date.today().isoformat()}",
        *model_lines(r),
        f"- Train: {r.n_train} gambar (split `train` manifest beku; label "
        f"{'/'.join(cfg.model.train_labels)}). Threshold config: **{t:g}**; threshold "
        f"terkalibrasi (CV train+val, FPR ≤ {e.fpr_target:g}): **{tc:g}** (reject jika skor ≥ "
        "threshold).",
        "- **Test dievaluasi dengan dua threshold yang keduanya TIDAK dipilih dari test** "
        "(config & kalibrasi CV train+val). Fitur dan hyperparameter tidak disetel dari test.",
        f"- Gagal segmentasi pada val+test (fallback seluruh gambar): {r.seg_failures}",
        "",
        "## Hold-out (model dilatih dari train saja)",
        "",
        f"Recall @ FPR ≤ {e.fpr_target:g} dihitung dari kurva ROC (threshold optimal pasca-hoc, "
        "hanya informasi — bukan threshold operasi). Dengan 3–4 roti normal per split, FPR hanya "
        "bisa 0, 0.25/0.33, … — angka ini praktis = recall pada FPR 0 (sangat kasar).",
        "",
        "| split | threshold | n (N/D) | AUROC | recall@FPR≤target (FPR) | akurasi | presisi | "
        "recall | TN FP FN TP |",
        "|---|---|---|---|---|---|---|---|---|",
        *holdout_rows(r.sets, t, e.fpr_target),
        *holdout_rows(r.sets, tc, e.fpr_target),
    ]
    oof_y = cv_set.y
    rec, fpr_at, _ = recall_at_fpr(oof_y, cv.oof, e.fpr_target)
    au_m, au_s = np.nanmean(cv.aurocs), np.nanstd(cv.aurocs)
    acc_m, acc_s = np.mean(cv.accs), np.std(cv.accs)
    fit_note = (
        "memory bank di-fit dari roti normal fold latih saja; semua gambar fold uji diskor"
        if cfg.model.type == "anomaly"
        else "classifier di-fit ulang dari fold latih"
    )
    out += [
        "",
        f"## Cross-validation per grup ({e.cv_folds}-fold StratifiedGroupKFold, train+val, "
        "test tidak ikut)",
        "",
        "Lebih bisa dipercaya daripada hold-out kecil di atas. Grup = `batch_id` digabung "
        f"near-duplicate (aturan yang sama dengan split manifest); {fit_note}. Fold sama untuk "
        "scorer A dan B (seed config).",
        "",
        "| metrik | nilai |",
        "|---|---|",
        f"| AUROC per fold (mean ± std) | {au_m:.3f} ± {au_s:.3f} |",
        f"| akurasi @ threshold {t:g} (mean ± std) | {acc_m:.3f} ± {acc_s:.3f} |",
        f"| AUROC gabungan out-of-fold (n={len(oof_y)}) | {fmt(auroc(oof_y, cv.oof))} |",
        f"| recall @ FPR ≤ {e.fpr_target:g} out-of-fold (FPR) | {fmt(rec, 2)} ({fmt(fpr_at, 2)}) |",
        "",
        "AUROC per fold: " + ", ".join(fmt(a) for a in cv.aurocs),
        "",
        *calibration_section(r.cal),
        "",
        "## Rincian per kelompok nama file",
        "",
        "`medium-good` = masih layak walau permukaan tidak sempurna; `medium-bad` = tampak "
        "lumayan tapi permukaan terlalu kempes → reject. Kolom `salah` memakai threshold config.",
    ]
    for name, sc in {**r.sets, "CV out-of-fold (train+val)": cv_set}.items():
        out += ["", f"**{name}**", "", *prefix_table(sc, t)]
    out += ["", "## Kesalahan terbesar (margin salah-sisi dari threshold config)"]
    for name, sc in {**r.sets, "CV out-of-fold": cv_set}.items():
        out += ["", f"**{name}**", "", *top_errors(sc, t, e.top_errors)]

    lat = r.latencies
    out += [
        "",
        "## Latency & ukuran model",
        "",
        f"Rata-rata {np.mean(lat):.1f} ms/gambar (median {np.median(lat):.1f}, maks "
        f"{np.max(lat):.1f}; n={len(lat)}, setelah pemanasan) untuk segmentasi + representasi + "
        "skor, CPU, tanpa waktu baca file. Target PRD: ≤ 100 ms. File model: "
        f"{r.model_bytes / 1e6:.2f} MB"
        + (
            " (memory bank float32; bobot backbone ResNet18 ~47 MB terpisah di cache torch)."
            if cfg.model.type == "anomaly"
            else "."
        ),
    ]
    if r.domain:
        img_note = (
            "Gambar beranotasi: `domain_check.jpg`"
            + (" dan `domain_check_heatmaps.jpg`." if cfg.model.type == "anomaly" else ".")
            if with_images
            else f"Gambar beranotasi ada di `{cfg.output_dir}/` (tidak di-versi)."
        )
        out += ["", "## Domain check: foto konveyor asli (semua roti OK)", ""]
        for dc in r.domain:
            out += domain_table(dc, img_note)
    if r.shortcuts is not None:
        out += shortcut_section(r)
    out += notes(r, with_images)
    return "\n".join(out) + "\n"


def calibration_section(cal: dict[str, Any]) -> list[str]:
    """Bagian kalibrasi threshold (F9)."""
    cv, val = cal["cv"], cal["val"]
    return [
        f"## Kalibrasi threshold (F9, target FPR ≤ {cal['fpr_target']:g}, tanpa test)",
        "",
        "| sumber | n normal / kempes | langkah FPR | threshold | FPR | recall | FPR / recall @ "
        f"config {cal['config_threshold']:g} |",
        "|---|---|---|---|---|---|---|",
        f"| CV out-of-fold train+val (**saran**) | {cv['n_normal']} / {cv['n_defect']} | "
        f"{cv['fpr_step']:.3f} | **{cv['threshold']:g}** | {cv['fpr']:.2f} | {cv['recall']:.2f} "
        f"| {cv['fpr_at_config']:.2f} / {cv['recall_at_config']:.2f} |",
        f"| val saja (pembanding) | {val['n_normal']} / {val['n_defect']} | "
        f"{val['fpr_step']:.3f} | {val['threshold']:g} | {val['fpr']:.2f} | {val['recall']:.2f} "
        f"| {val['fpr_at_config']:.2f} / {val['recall_at_config']:.2f} |",
        "",
        "Aturan: boleh k = floor(target · n normal) false positive → batas bawah = sedikit di atas "
        "skor normal tertinggi ke-(k+1); threshold lalu digeser ke tengah celah antara batas itu "
        "dan skor kempes terendah di atasnya (margin maksimum, FPR/recall data kalibrasi tidak "
        "berubah). **Kasar:** dengan 20 normal di CV satu roti = langkah FPR 0.05, jadi target "
        "5% = paling banyak 1 normal di atas batas; di val (3 normal) target 5% = FPR 0. Saran "
        "ditulis ke `calibration.yaml`; config TIDAK diubah otomatis.",
    ]


def shortcut_section(r: EvalResult) -> list[str]:
    """Cek shortcut AUROC satu fitur + koefisien (baseline)."""
    feat_cv, feat_pooled = r.shortcuts
    names = r.scorer.feature_names
    out = [
        "",
        "## Cek shortcut: AUROC satu fitur",
        "",
        f"CV = rata-rata AUROC di fold uji (arah fitur dipilih dari fold latih); gabungan = "
        f"seluruh train+val, arah bebas. Fitur dengan AUROC CV ≥ {SUSPICIOUS_AUROC} ditandai.",
        "",
        "| fitur | AUROC CV | AUROC gabungan | catatan |",
        "|---|---|---|---|",
    ]
    for j in np.argsort(-np.nan_to_num(feat_cv)):
        flag = (
            "**mencurigakan — pastikan bukan artefak pengambilan foto**"
            if feat_cv[j] >= SUSPICIOUS_AUROC
            else ""
        )
        out.append(f"| `{names[j]}` | {fmt(feat_cv[j])} | {fmt(feat_pooled[j])} | {flag} |")
    clf = r.scorer.pipeline[-1] if r.scorer.pipeline is not None else None
    if hasattr(clf, "coef_"):
        coef = clf.coef_[0]
        top = np.argsort(-np.abs(coef))[:8]
        out += [
            "",
            "Koefisien logistic regression terbesar (fitur terstandardisasi; + = ke arah kempes): "
            + ", ".join(f"`{names[j]}` {coef[j]:+.2f}" for j in top),
        ]
    return out


COMMON_NOTES = [
    "- Data sangat kecil (24 + 24 foto; val 3+3, test 4+4): satu gambar menggeser metrik "
    "hold-out 12–17 poin. Pakai angka CV sebagai acuan utama, itu pun dengan rentang lebar.",
    "- Foto studio: tiap folder diambil dengan kondisi berbeda (ukuran gambar, latar, "
    "pencahayaan/warna). Ukuran gambar sudah dinetralkan (crop + resize tetap), tetapi angka yang "
    "(hampir) sempurna bisa sebagian mencerminkan **perbedaan pengambilan foto**, bukan kempes. "
    "Sinyal paling realistis: domain check foto konveyor (di atas) dan rekaman kamera (M4).",
    "- Tidak ada info batch/sesi: grup = satu gambar (+ near-duplicate). Kebocoran antar foto dari "
    "sesi yang sama tidak bisa disingkirkan.",
    "- Hasil CV bergantung pada pembagian fold: dengan 40 gambar (±8 per fold), satu gambar salah "
    "urut menggeser AUROC fold ±0.06.",
]

BASELINE_NOTES = [
    "- Segmentasi saturasi HSV kadang memotong sebagian alas roti yang pucat (lihat contact "
    "sheet); dampaknya terutama pada fitur bentuk, bukan tekstur permukaan atas.",
    "- Sudut kamera bervariasi (samping/miring/atas) sehingga fitur bentuk (aspek, circularity) "
    "ikut dipengaruhi sudut pandang.",
    "",
    "## Riwayat iterasi M2 (sebelum test dibuka; dinilai hanya dengan CV train+val)",
    "",
    "- Versi awal (crop 256, tanpa blur): AUROC CV ~1.0, tetapi fitur tekstur halus (LBP, "
    "Laplacian, gradien, Canny) bergeser ~1 std bila foto asli di-resize 0.7x — sebanding dengan "
    "selisih antar kelas, dan arahnya terbalik dari dugaan (roti bagus punya lebih banyak detail "
    "halus). Ini risiko shortcut ketajaman/resolusi foto.",
    "- Perbaikan: Gaussian blur σ=1.5 sebelum tekstur + crop 192 px (dipilih karena paling tahan "
    "resize, pergeseran ≤ 0.4 std) dan fitur lipatan `tex_crease` (black-hat). Classifier tetap "
    "logistic regression (default); RF/SVM dicoba di CV, tidak lebih baik.",
    "- Akibatnya model kini bersandar terutama pada kontras luminans/kilap (`col_L_std`), warna "
    "(`col_a_mean`, `col_b_std`), pola LBP, dan aspek roti; fitur kerutan eksplisit "
    "(`tex_crease`, `tex_edge_density`) lemah sendirian. Permukaan kusam & pucat pada foto kempes "
    "bisa nyata (kempes) atau sebagian artefak pencahayaan per folder.",
]

ANOMALY_NOTES = [
    "- Backbone ResNet18 pretrained ImageNet dipakai **hanya sebagai feature extractor beku** "
    "(tidak di-fine-tune); bobot diunduh sekali saat training ke cache torch. Saat inferensi "
    "tidak ada panggilan jaringan — bila bobot tidak ada, scorer gagal dengan pesan jelas.",
    "- Memory bank hanya dari 17 roti normal train (20 per fold CV) dengan sudut kamera "
    'bervariasi: "normal" yang dikenal model sangat sempit, sehingga roti normal dengan sudut/'
    "pencahayaan baru mudah dianggap anomali (false positive). Skor 50 adalah batas variasi normal "
    "train, bukan probabilitas.",
    "- Kalibrasi ref (p95 leave-one-out) dari 17 nilai: praktis ditentukan 1–2 roti normal "
    "paling aneh; ref per fold CV berbeda-beda.",
    "- Heatmap menunjukkan patch dengan jarak terbesar ke bank; tidak selalu bertepatan dengan "
    "area kempes (bisa kilap, tepi crop, atau alas roti yang terpotong segmentasi).",
]


def notes(r: EvalResult, with_images: bool) -> list[str]:
    """Catatan & keterbatasan per tipe model."""
    extra = BASELINE_NOTES if r.cfg.model.type == "baseline" else [*ANOMALY_NOTES, f"- {HEAT_NOTE}"]
    out = ["", "## Catatan & keterbatasan", "", *COMMON_NOTES, *extra, ""]
    files = "`roc_test.png`, `segment_debug.jpg`, `scores.csv`, `calibration.yaml`, `metrics.json`"
    if r.cfg.model.type == "anomaly":
        files += ", `heatmaps/`"
    if with_images:
        out.append(f"Artefak di folder ini: {files}.")
    else:
        out.append(f"Artefak ({files}) ada di `{r.cfg.output_dir}/` (tidak di-versi).")
    return out


def metrics_json(r: EvalResult) -> dict[str, Any]:
    """Ringkasan angka untuk perbandingan A vs B (compare.py)."""
    e = r.cfg.eval

    def split_summary(sc: Scored) -> dict[str, Any]:
        rec, fpr_at, _ = recall_at_fpr(sc.y, sc.scores, e.fpr_target)
        return {
            "auroc": auroc(sc.y, sc.scores),
            "recall_at_fpr": rec,
            "fpr_at": fpr_at,
            "acc_config": binary_metrics(sc.y, sc.scores, r.cfg.threshold)["accuracy"],
            "acc_cal": binary_metrics(sc.y, sc.scores, r.t_cal)["accuracy"],
            "cm_config": {
                k: binary_metrics(sc.y, sc.scores, r.cfg.threshold)[k]
                for k in ("tn", "fp", "fn", "tp")
            },
            "cm_cal": {
                k: binary_metrics(sc.y, sc.scores, r.t_cal)[k] for k in ("tn", "fp", "fn", "tp")
            },
            "prefix_means": sc.prefix_means(),
        }

    cv, cv_set = r.cv.result, r.cv.scored
    rec, fpr_at, _ = recall_at_fpr(cv_set.y, cv.oof, e.fpr_target)
    return {
        "model_type": r.cfg.model.type,
        "output_dir": str(r.cfg.output_dir),
        "threshold_config": r.cfg.threshold,
        "threshold_cal": r.t_cal,
        "fpr_target": e.fpr_target,
        "val": split_summary(r.sets["val"]),
        "test": split_summary(r.sets["test"]),
        "cv": {
            "auroc_mean": float(np.nanmean(cv.aurocs)),
            "auroc_std": float(np.nanstd(cv.aurocs)),
            "auroc_pooled": auroc(cv_set.y, cv.oof),
            "recall_at_fpr": rec,
            "fpr_at": fpr_at,
            "acc_config": binary_metrics(cv_set.y, cv.oof, r.cfg.threshold)["accuracy"],
            "acc_cal": binary_metrics(cv_set.y, cv.oof, r.t_cal)["accuracy"],
            "prefix_means": cv_set.prefix_means(),
        },
        "latency_ms": {
            "mean": float(np.mean(r.latencies)),
            "median": float(np.median(r.latencies)),
        },
        "model_bytes": r.model_bytes,
        "domain_check": [
            {
                "path": dc.path,
                "n": len(dc.scores),
                "scores": [round(s, 2) for s in dc.scores],
                "thresholds": dc.thresholds,
                "n_reject": {k: dc.n_reject(k) for k in dc.thresholds},
            }
            for dc in r.domain
        ],
    }


# --- orkestrasi ----------------------------------------------------------------------------


def evaluate(cfg: Config, root: Path) -> str:
    """Jalankan seluruh evaluasi; tulis artefak; kembalikan teks laporan."""
    out_dir = root / cfg.output_dir
    scorer = load_scorer(cfg, root)
    by_split = split_rows(cfg, root)

    sets = {s: score_split(scorer, by_split[s], root) for s in ("val", "test")}
    seg_failures = scorer.seg_failures
    for name, sc in sets.items():
        log.info("%s: AUROC=%s", name, fmt(auroc(sc.y, sc.scores)))

    cv = run_cv(cfg, scorer, by_split["train"] + by_split["val"], root)
    cal = calibrate(cv.y, cv.result.oof, sets["val"].y, sets["val"].scores, cfg)
    write_calibration(cal, out_dir / "calibration.yaml")
    shortcuts = None
    if cfg.model.type == "baseline":
        x_clean = np.stack([np.atleast_2d(r)[0] for r in cv.reprs])  # tanpa salinan augmentasi
        shortcuts = feature_shortcuts(x_clean, cv.y, cv.groups, cfg)
    if cfg.model.type == "anomaly" and cfg.eval.heatmaps:
        save_heatmaps(scorer, sets, root, out_dir / "heatmaps")

    thresholds = {"config": cfg.threshold, "kalibrasi": float(cal["suggested_threshold"])}
    domain = [
        domain_check(
            scorer, cfg, p, thresholds, root, out_dir / f"domain_check{'' if i == 0 else i}.jpg"
        )
        for i, p in enumerate(cfg.eval.domain_check)
    ]

    test = sets["test"]
    title = f"ROC test (n={len(test.y)}), AUROC={fmt(auroc(test.y, test.scores))}"
    draw_roc(test.y, test.scores, out_dir / "roc_test.png", title)
    write_scores_csv(out_dir / "scores.csv", {**sets, "cv_oof": cv.scored}, cfg.threshold)
    result = EvalResult(
        cfg=cfg,
        scorer=scorer,
        sets=sets,
        cv=cv,
        cal=cal,
        n_train=sum(r["label"] in cfg.model.train_labels for r in by_split["train"]),
        seg_failures=seg_failures,
        model_bytes=model_path(cfg, root).stat().st_size,
        domain=domain,
        shortcuts=shortcuts,
    )
    report = render_report(result, with_images=True)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json(result), indent=2), "utf-8")
    if cfg.eval.docs_copy:
        (root / cfg.eval.docs_copy).write_text(render_report(result, with_images=False), "utf-8")
    log.info("Laporan: %s (salinan %s)", out_dir / "report.md", cfg.eval.docs_copy)
    return report


def label_counts(dc: DomainCheck, defects: list[list[float]], scale: float) -> dict[str, int]:
    """Banding dengan label acuan (titik tengah roti kempes, piksel foto asli).

    Objek terdeteksi = kempes bila bbox-nya memuat salah satu titik kempes (blob gabungan dua roti
    yang memuat roti kempes ikut dihitung kempes). Threshold = config.
    """
    out = {"kempes": 0, "kempes_reject": 0, "ok": 0, "ok_reject": 0}
    t = dc.thresholds["config"]
    for (x, y, w, h), s in zip(dc.bboxes, dc.scores, strict=True):
        hit = any(x <= px * scale <= x + w and y <= py * scale <= y + h for px, py in defects)
        key = "kempes" if hit else "ok"
        out[key] += 1
        out[f"{key}_reject"] += int(s >= t)
    return out


def run_domain_check(cfg: Config, root: Path, path: str, scale: float = 1.0) -> DomainCheck:
    """Domain check satu foto (CLI ``--domain-check``); threshold kalibrasi dari calibration.yaml
    bila sudah ada (hasil ``calibrate``/``evaluate``)."""
    import yaml

    scorer = load_scorer(cfg, root)
    thresholds = {"config": cfg.threshold}
    cal_path = root / cfg.output_dir / "calibration.yaml"
    if cal_path.exists():
        thresholds["kalibrasi"] = float(yaml.safe_load(cal_path.read_text())["suggested_threshold"])
    suffix = "" if scale == 1.0 else f"_x{scale:g}"
    out = root / cfg.output_dir / f"domain_check{suffix}.jpg"
    dc = domain_check(scorer, cfg, path, thresholds, root, out, scale)
    lines = ["# Domain check", "", *domain_table(dc, f"Gambar: `{out.name}` (skala ×{scale:g}).")]
    labels_path = root / Path(path).parent / "labels.json"
    entry = (
        json.loads(labels_path.read_text()).get(Path(path).name) if labels_path.exists() else None
    )
    if entry is not None:
        c = label_counts(dc, entry["defect"], scale)
        log.info(
            "Label acuan (thr config): kempes tertangkap %d/%d, roti OK ter-reject %d/%d",
            c["kempes_reject"],
            c["kempes"],
            c["ok_reject"],
            c["ok"],
        )
        lines += [
            "",
            f"Banding `{labels_path.name}` (threshold config): kempes tertangkap "
            f"**{c['kempes_reject']}/{c['kempes']}**, roti OK ter-reject "
            f"**{c['ok_reject']}/{c['ok']}**.",
        ]
    md = "\n".join(lines) + "\n"
    (out.with_suffix(".md")).write_text(md, encoding="utf-8")
    return dc


def main(argv: list[str] | None = None) -> int:
    """CLI evaluasi."""
    parser = argparse.ArgumentParser(description="Evaluasi scorer pada val/test + CV train+val.")
    parser.add_argument("--config", required=True, help="path configs/*.yaml")
    parser.add_argument("--domain-check", metavar="FOTO", help="hanya domain check foto konveyor")
    parser.add_argument(
        "--domain-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="perkecil foto domain check ×S dulu (uji ketahanan resolusi), mis. 0.383",
    )
    args = parser.parse_args(argv)
    if not 0 < args.domain_scale <= 1:
        parser.error("--domain-scale harus di (0, 1]")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.domain_check:
        run_domain_check(cfg, Path.cwd(), args.domain_check, args.domain_scale)
    else:
        evaluate(cfg, Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
