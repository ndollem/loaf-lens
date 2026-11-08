"""CLI evaluasi: ``python -m loaflens.evaluate --config configs/baseline.yaml``.

Model (hasil ``train``) dinilai pada val dan test secara terpisah dengan threshold dari config
(tanpa tuning). Ditambah cross-validation per grup pada train+val (test tidak ikut) dan cek
shortcut per fitur. Output: report.md, scores.csv, roc_test.png (+ salinan docs/results_m2.md).
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from loaflens.audit import name_prefix
from loaflens.config import Config, load_config
from loaflens.data import dhash, group_ids, load_image, near_duplicate_pairs, read_manifest
from loaflens.score import BaselineScorer
from loaflens.train import LABEL_TO_Y

log = logging.getLogger(__name__)

DOCS_COPY = Path("docs/results_m2.md")
HEADLINE = "Hasil pada 48 foto; indikasi kelayakan, bukan klaim akurasi produksi."
GROUP_NAMES = {
    ("normal", "bagus"): "bagus",
    ("normal", "medium"): "medium-good",
    ("defect", "bad"): "bad",
    ("defect", "medium"): "medium-bad",
}
SUSPICIOUS_AUROC = 0.98


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


def score_split(scorer: BaselineScorer, rows: list[dict[str, str]], root: Path) -> Scored:
    """Skor tiap gambar + latency (segmentasi + fitur + prediksi, tanpa baca file)."""
    paths, labels, scores, lat = [], [], [], []
    if rows:  # pemanasan (alokasi pertama OpenCV/sklearn) agar latency tidak bias
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
    feat_auroc: np.ndarray  # (n_fitur,) rata-rata AUROC 1 fitur di fold uji (arah dari fold latih)
    feat_auroc_pooled: np.ndarray  # (n_fitur,) AUROC 1 fitur pada train+val (arah bebas)


def cross_validate(
    scorer: BaselineScorer, x: np.ndarray, y: np.ndarray, groups: list[int], cfg: Config
) -> CVResult:
    """StratifiedGroupKFold pada train+val: scorer di-fit ulang per fold dari fitur."""
    folds = StratifiedGroupKFold(n_splits=cfg.eval.cv_folds, shuffle=True, random_state=cfg.seed)
    aurocs, accs, oof = [], [], np.zeros(len(y))
    feat = np.zeros((cfg.eval.cv_folds, x.shape[1]))
    for k, (tr, te) in enumerate(folds.split(x, y, groups)):
        s = scorer.fit_features(x[tr], y[tr]).score_features(x[te])
        oof[te] = s
        aurocs.append(auroc(y[te], s))
        accs.append(binary_metrics(y[te], s, cfg.threshold)["accuracy"])
        for j in range(x.shape[1]):
            sign = 1.0 if auroc(y[tr], x[tr, j]) >= 0.5 else -1.0
            feat[k, j] = auroc(y[te], sign * x[te, j])
    pooled = np.array([auroc(y, x[:, j]) for j in range(x.shape[1])])
    return CVResult(aurocs, accs, oof, np.nanmean(feat, axis=0), np.maximum(pooled, 1 - pooled))


# --- output --------------------------------------------------------------------------------


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


def fmt(v: float, nd: int = 3) -> str:
    return "n/a" if v != v else f"{v:.{nd}f}"  # NaN → n/a


def prefix_table(sc: Scored, threshold: float) -> list[str]:
    """Rata-rata skor + jumlah salah per kelompok prefix."""
    out = ["| kelompok | n | skor rata-rata | min–max | salah |", "|---|---|---|---|---|"]
    groups: dict[str, list[int]] = {}
    for i, (p, lab) in enumerate(zip(sc.paths, sc.labels, strict=True)):
        groups.setdefault(prefix_group(p, lab), []).append(i)
    for g in ("bagus", "medium-good", "medium-bad", "bad", *sorted(groups)):
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
        return ["(tidak ada kesalahan pada threshold config)"]
    return [
        f"- `{sc.paths[i]}` — label {sc.labels[i]}, skor {sc.scores[i]:.1f} "
        f"(margin {abs(sc.scores[i] - threshold):.1f})"
        for i in wrong[:n]
    ]


def render_report(
    cfg: Config,
    scorer: BaselineScorer,
    sets: dict[str, Scored],
    cv: CVResult,
    cv_set: Scored,
    n_train: int,
    seg_failures: int,
    with_images: bool,
) -> str:
    """Susun laporan markdown (bahasa Indonesia)."""
    t, e = cfg.threshold, cfg.eval
    p = cfg.model.params
    out = [
        "# Hasil evaluasi scorer A (baseline) — LoafLens M2",
        "",
        f"> **{HEADLINE}**",
        "",
        f"- Tanggal: {date.today().isoformat()}",
        f"- Classifier: `{p['classifier']}` + StandardScaler; {len(scorer.feature_names)} fitur "
        f"(tekstur, warna/kilap, bentuk bebas skala) dari crop roti {p['crop_size']} px",
        f"- Train: {n_train} gambar (split `train` manifest beku). Threshold config: **{t:g}** "
        "(reject jika skor ≥ threshold).",
        "- **Test dievaluasi sekali dengan threshold dari config.** Threshold, fitur, dan "
        "hyperparameter TIDAK disetel dari hasil test; val juga tidak dipakai untuk kalibrasi.",
        f"- Gagal segmentasi pada val+test (fallback seluruh gambar): {seg_failures}",
        "",
        "## Hold-out (model dilatih dari train saja)",
        "",
        f"Recall @ FPR ≤ {e.fpr_target:g} dihitung dari kurva ROC. Dengan 3–4 roti normal per "
        "split, FPR hanya bisa bernilai 0, 0.25/0.33, … — jadi angka ini praktis = recall pada "
        "FPR 0 (sangat kasar).",
        "",
        "| split | n (N/D) | AUROC | recall@FPR≤target (FPR) | akurasi | presisi | recall | "
        "TN FP FN TP |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, sc in sets.items():
        y = sc.y
        rec, fpr_at, _ = recall_at_fpr(y, sc.scores, e.fpr_target)
        m = binary_metrics(y, sc.scores, t)
        out.append(
            f"| {name} | {len(y)} ({int((y == 0).sum())}/{int(y.sum())}) | "
            f"{fmt(auroc(y, sc.scores))} | {fmt(rec, 2)} ({fmt(fpr_at, 2)}) | "
            f"{fmt(m['accuracy'], 2)} | {fmt(m['precision'], 2)} | {fmt(m['recall'], 2)} | "
            f"{m['tn']} {m['fp']} {m['fn']} {m['tp']} |"
        )
    oof_y = cv_set.y
    rec, fpr_at, _ = recall_at_fpr(oof_y, cv.oof, e.fpr_target)
    au_m, au_s = np.nanmean(cv.aurocs), np.nanstd(cv.aurocs)
    acc_m, acc_s = np.mean(cv.accs), np.std(cv.accs)
    out += [
        "",
        f"## Cross-validation per grup ({e.cv_folds}-fold StratifiedGroupKFold, train+val, "
        "test tidak ikut)",
        "",
        "Lebih bisa dipercaya daripada hold-out kecil di atas. Grup = `batch_id` digabung "
        "near-duplicate (aturan yang sama dengan split manifest).",
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
        "## Rincian per kelompok nama file",
        "",
        "`medium-good` = masih layak walau permukaan tidak sempurna; `medium-bad` = tampak "
        "lumayan tapi permukaan terlalu kempes → reject. Kelompok borderline ini yang paling "
        "penting.",
    ]
    for name, sc in {**sets, "CV out-of-fold (train+val)": cv_set}.items():
        out += ["", f"**{name}**", "", *prefix_table(sc, t)]
    out += ["", "## Kesalahan terbesar (margin salah-sisi dari threshold)"]
    for name, sc in {**sets, "CV out-of-fold": cv_set}.items():
        out += ["", f"**{name}**", "", *top_errors(sc, t, e.top_errors)]

    lat = [v for sc in sets.values() for v in sc.latency_ms]
    out += [
        "",
        "## Latency",
        "",
        f"Rata-rata {np.mean(lat):.1f} ms/gambar (median {np.median(lat):.1f}, maks "
        f"{np.max(lat):.1f}; n={len(lat)}) untuk segmentasi + fitur + prediksi, CPU, tanpa "
        "waktu baca file. Target PRD: ≤ 100 ms.",
        "",
        "## Cek shortcut: AUROC satu fitur",
        "",
        f"CV = rata-rata AUROC di fold uji (arah fitur dipilih dari fold latih); gabungan = "
        f"seluruh train+val, arah bebas. Fitur dengan AUROC CV ≥ {SUSPICIOUS_AUROC} ditandai.",
        "",
        "| fitur | AUROC CV | AUROC gabungan | catatan |",
        "|---|---|---|---|",
    ]
    order = np.argsort(-np.nan_to_num(cv.feat_auroc))
    for j in order:
        flag = (
            "**mencurigakan — pastikan bukan artefak pengambilan foto**"
            if cv.feat_auroc[j] >= SUSPICIOUS_AUROC
            else ""
        )
        out.append(
            f"| `{scorer.feature_names[j]}` | {fmt(cv.feat_auroc[j])} | "
            f"{fmt(cv.feat_auroc_pooled[j])} | {flag} |"
        )
    clf = scorer.pipeline[-1] if scorer.pipeline is not None else None
    if hasattr(clf, "coef_"):
        coef = clf.coef_[0]
        top = np.argsort(-np.abs(coef))[:8]
        out += [
            "",
            "Koefisien logistic regression terbesar (fitur terstandardisasi; + = ke arah kempes): "
            + ", ".join(f"`{scorer.feature_names[j]}` {coef[j]:+.2f}" for j in top),
        ]
    out += [
        "",
        "## Catatan & keterbatasan",
        "",
        "- Data sangat kecil (24 + 24 foto; val 3+3, test 4+4): satu gambar menggeser metrik "
        "hold-out 12–17 poin. Pakai angka CV sebagai acuan utama, itu pun dengan rentang lebar.",
        "- Foto studio: tiap folder diambil dengan kondisi berbeda (ukuran gambar, latar, "
        "pencahayaan/warna). Fitur ukuran gambar sudah dibuang (crop + resize tetap), tetapi "
        "angka yang (hampir) sempurna bisa sebagian mencerminkan **perbedaan pengambilan foto**, "
        "bukan kempes itu sendiri. Uji sebenarnya adalah rekaman kamera/konveyor (M4).",
        "- Tidak ada info batch/sesi: grup = satu gambar (+ near-duplicate). Kebocoran antar "
        "foto dari sesi yang sama tidak bisa disingkirkan.",
        "- Segmentasi saturasi HSV kadang memotong sebagian alas roti yang pucat (lihat "
        "contact sheet); dampaknya terutama pada fitur bentuk, bukan tekstur permukaan atas.",
        "- Sudut kamera bervariasi (samping/miring/atas) sehingga fitur bentuk (aspek, "
        "circularity) ikut dipengaruhi sudut pandang.",
        f"- Threshold {t:g} belum dikalibrasi (kalibrasi di val = M3, F9).",
        "- Hasil CV bergantung pada pembagian fold: dengan 40 gambar (±8 per fold), satu gambar "
        "salah urut menggeser AUROC fold ±0.06.",
        "",
        "## Riwayat iterasi (sebelum test dibuka; dinilai hanya dengan CV train+val)",
        "",
        "- Versi awal (crop 256, tanpa blur): AUROC CV ~1.0, tetapi fitur tekstur halus "
        "(LBP, Laplacian, gradien, Canny) bergeser ~1 std bila foto asli di-resize 0.7x — "
        "sebanding dengan selisih antar kelas, dan arahnya terbalik dari dugaan (roti bagus "
        "punya lebih banyak detail halus). Ini risiko shortcut ketajaman/resolusi foto.",
        "- Perbaikan: Gaussian blur σ=1.5 sebelum tekstur + crop 192 px (dipilih karena paling "
        "tahan resize, pergeseran ≤ 0.4 std) dan fitur lipatan `tex_crease` (black-hat). "
        "Classifier tetap logistic regression (default); RF/SVM dicoba di CV, tidak lebih baik.",
        "- Akibatnya model kini bersandar terutama pada kontras luminans/kilap (`col_L_std`), "
        "warna (`col_a_mean`, `col_b_std`), pola LBP, dan aspek roti; fitur kerutan eksplisit "
        "(`tex_crease`, `tex_edge_density`) lemah sendirian. Permukaan kusam & pucat pada foto "
        "kempes bisa nyata (kempes) atau sebagian artefak pencahayaan per folder.",
    ]
    if with_images:
        out += [
            "",
            "Kurva ROC test: `roc_test.png`; contact sheet segmentasi: "
            "`segment_debug.jpg`; skor per gambar: `scores.csv`.",
        ]
    else:
        out += [
            "",
            "Gambar (ROC, contact sheet) dan `scores.csv` ada di `outputs/baseline/` "
            "(tidak di-versi).",
        ]
    return "\n".join(out) + "\n"


def evaluate(cfg: Config, root: Path) -> str:
    """Jalankan seluruh evaluasi; tulis artefak; kembalikan teks laporan."""
    if cfg.model.type != "baseline":
        raise SystemExit(f"model.type={cfg.model.type!r} belum didukung (anomaly = M3)")
    out_dir = root / cfg.output_dir
    scorer = BaselineScorer.load(out_dir / "model.joblib")
    rows = read_manifest(root / cfg.data.manifest)
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}

    sets = {s: score_split(scorer, by_split[s], root) for s in ("val", "test")}
    seg_failures = scorer.seg_failures
    for name, sc in sets.items():
        log.info("%s: AUROC=%s", name, fmt(auroc(sc.y, sc.scores)))

    cv_rows = by_split["train"] + by_split["val"]
    x = np.stack([scorer.features(load_image(r["path"], root)) for r in cv_rows])
    y = np.array([LABEL_TO_Y[r["label"]] for r in cv_rows])
    groups = cv_groups(cv_rows, root, cfg.data.near_dup_hamming)
    cv_scorer = BaselineScorer.from_config(cfg)  # model terpisah; model train tidak disentuh
    cv = cross_validate(cv_scorer, x, y, groups, cfg)
    cv_set = Scored([r["path"] for r in cv_rows], [r["label"] for r in cv_rows], cv.oof)
    log.info("CV AUROC %.3f ± %.3f", np.nanmean(cv.aurocs), np.nanstd(cv.aurocs))

    test = sets["test"]
    title = f"ROC test (n={len(test.y)}), AUROC={fmt(auroc(test.y, test.scores))}"
    draw_roc(test.y, test.scores, out_dir / "roc_test.png", title)
    write_scores_csv(out_dir / "scores.csv", {**sets, "cv_oof": cv_set}, cfg.threshold)
    n_train = len(by_split["train"])
    report = render_report(cfg, scorer, sets, cv, cv_set, n_train, seg_failures, with_images=True)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    docs = render_report(cfg, scorer, sets, cv, cv_set, n_train, seg_failures, with_images=False)
    (root / DOCS_COPY).write_text(docs, encoding="utf-8")
    log.info("Laporan: %s (salinan %s)", out_dir / "report.md", DOCS_COPY)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI evaluasi."""
    parser = argparse.ArgumentParser(description="Evaluasi scorer pada val/test + CV train+val.")
    parser.add_argument("--config", required=True, help="path configs/*.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    evaluate(load_config(args.config), Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
