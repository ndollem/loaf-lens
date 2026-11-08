"""Demo real-time: capture → segment → track → score → overlay/log (M4).

``python -m loaflens.app --config configs/baseline.yaml --source video:samples/demo.mp4``
``[--headless] [--out outputs/demo_out.mp4] [--max-frames N] [--gt samples/demo.gt.json]``

Tiap roti diputuskan sekali saat centroid-nya melewati garis keputusan (``track.*``); skor dari
segmentasi frame lintas itu; REJECT jika skor >= ``threshold``. REJECT disimpan ke
``app.snapshot_dir`` + ``app.rejects_csv`` (F8); semua keputusan ke ``app.decisions_csv``;
satu baris log ``DECISION ...`` per roti (F11-lite, belum ada serial/PLC). ``--gt``: bandingkan
dengan ground truth video SYNTHETIC (``loaflens.synth``) → ``app.gt_report``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from loaflens.capture import FrameSource, SourceError, parse_source
from loaflens.config import AppConfig, Config, load_config
from loaflens.segment import Segmentation, roi_rect, segment_all
from loaflens.synth import predict
from loaflens.track import CentroidTracker, Point, Track

log = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX
NEUTRAL, GREEN, RED, WHITE, BLACK = (
    (220, 220, 220),
    (0, 200, 0),
    (0, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
)
QUIT_KEYS = {ord("q"), 27}  # q / ESC


class FrameScorer(Protocol):
    """Bagian antarmuka Scorer yang dipakai app."""

    def score(self, image: np.ndarray, seg: Segmentation | None = None) -> float: ...


@dataclass
class Decision:
    frame: int
    track_id: int
    center: Point
    score: float
    reject: bool
    time: str = ""

    @property
    def label(self) -> str:
        return "REJECT" if self.reject else "OK"


@dataclass
class RunResult:
    frames: int = 0
    seconds: float = 0.0  # waktu loop total (baca + proses + gambar + tulis/tampil)
    decisions: list[Decision] = field(default_factory=list)
    seg_track_ms: list[float] = field(default_factory=list)  # per frame
    score_ms: list[float] = field(default_factory=list)  # per roti

    @property
    def n_reject(self) -> int:
        return sum(d.reject for d in self.decisions)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0

    def timing(self) -> str:
        """FPS loop penuh, segment+track per frame dan skor per roti (rata-rata, p95)."""

        def ms(v: list[float]) -> str:
            return f"{np.mean(v):.1f} ms (p95 {np.percentile(v, 95):.1f})" if v else "n/a"

        return (
            f"{self.frames} frame, {self.seconds:.1f} s, FPS rata-rata {self.fps:.1f} (loop "
            f"penuh) | segment+track {ms(self.seg_track_ms)}/frame | skor {ms(self.score_ms)}/roti"
        )


# --- overlay -------------------------------------------------------------------------------


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple, scale: float) -> None:
    """Teks berwarna di atas kotak putih (terbaca di latar apa pun)."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    x, y = org
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 4), WHITE, cv2.FILLED)
    cv2.putText(img, text, (x + 2, y), FONT, scale, color, 2, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    tracker: CentroidTracker,
    result: RunResult,
    fps: float,
    banner: str | None,
    roi: list[float] | None,
) -> np.ndarray:
    """Frame beranotasi: kotak + id (+ skor/keputusan), garis keputusan, ROI, FPS, counter."""
    out = frame.copy()
    if roi is not None:
        x0, y0, x1, y1 = roi_rect(out.shape, roi)
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 200, 0), 1)
    p0, p1 = tracker.line_points()
    cv2.line(out, p0, p1, (0, 255, 255), 2)
    for t in tracker.visible():
        color = NEUTRAL if not t.decided else (RED if t.reject else GREEN)
        x, y, w, h = t.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3 if t.decided else 2)
        text = f"#{t.id}"
        if t.decided and t.score is not None:
            text += f" {t.score:.0f} {'REJECT' if t.reject else 'OK'}"
        _label(out, text, (x, y - 6), BLACK if not t.decided else color, 0.55)
    n = len(result.decisions)
    hud = f"FPS {fps:4.1f} | total {n} | OK {n - result.n_reject} | REJECT {result.n_reject}"
    top = 0
    if banner:
        (_, bh), _ = cv2.getTextSize(banner, FONT, 0.7, 2)
        cv2.rectangle(out, (0, 0), (out.shape[1], bh + 16), (0, 0, 160), cv2.FILLED)
        cv2.putText(out, banner, (8, bh + 7), FONT, 0.7, WHITE, 2, cv2.LINE_AA)
        top = bh + 16
    _label(out, hud, (6, top + 28), BLACK, 0.7)
    return out


# --- output --------------------------------------------------------------------------------


def snapshot_crop(frame: np.ndarray, bbox: tuple[int, int, int, int], margin: float) -> np.ndarray:
    """Crop bbox + margin (fraksi sisi bbox), dibatasi tepi frame."""
    x, y, w, h = bbox
    mx, my = round(w * margin), round(h * margin)
    ih, iw = frame.shape[:2]
    return frame[max(y - my, 0) : min(y + h + my, ih), max(x - mx, 0) : min(x + w + mx, iw)]


def append_reject(
    app: AppConfig, stamp: str, raw: np.ndarray, annotated: np.ndarray, t: Track, d: Decision
) -> None:
    """F8: simpan crop roti + frame beranotasi, tambah baris ke rejects.csv."""
    app.snapshot_dir.mkdir(parents=True, exist_ok=True)
    base = f"{stamp}_f{d.frame:06d}_id{d.track_id}"
    crop_path = app.snapshot_dir / f"{base}_crop.jpg"
    frame_path = app.snapshot_dir / f"{base}_frame.jpg"
    cv2.imwrite(str(crop_path), snapshot_crop(raw, t.bbox, app.snapshot_margin))
    cv2.imwrite(str(frame_path), annotated)
    app.rejects_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not app.rejects_csv.exists()
    with app.rejects_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(["time", "frame", "track_id", "score", "crop", "frame_image"])
        w.writerow([d.time, d.frame, d.track_id, f"{d.score:.2f}", crop_path, frame_path])


# --- loop ----------------------------------------------------------------------------------


def run(
    cfg: Config,
    source: FrameSource,
    scorer: FrameScorer,
    *,
    headless: bool = True,
    out: Path | None = None,
    max_frames: int | None = None,
    banner: str | None = None,
) -> RunResult:
    """Loop utama; kembalikan keputusan + statistik waktu. Tidak memuat model/config sendiri."""
    app = cfg.app
    res = RunResult()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tracker: CentroidTracker | None = None
    writer: cv2.VideoWriter | None = None
    fps_ema = 0.0
    app.decisions_csv.parent.mkdir(parents=True, exist_ok=True)
    dec_file = app.decisions_csv.open("w", encoding="utf-8", newline="")
    dec_csv = csv.writer(dec_file, lineterminator="\n")
    dec_csv.writerow(["frame", "track_id", "cx", "cy", "score", "decision"])
    if not headless:
        cv2.namedWindow(app.window_name, cv2.WINDOW_NORMAL)
    t_start = time.perf_counter()
    try:
        for i, frame in enumerate(source):
            if max_frames is not None and i >= max_frames:
                break
            t_loop = time.perf_counter()
            if tracker is None:
                tracker = CentroidTracker(cfg.track, (frame.shape[1], frame.shape[0]))
            segs = segment_all(frame, cfg.segment, cfg.camera.roi)
            crossed = tracker.update([s.bbox for s in segs])
            res.seg_track_ms.append((time.perf_counter() - t_loop) * 1000)

            new = []
            for t in crossed:
                t0 = time.perf_counter()
                t.score = float(scorer.score(frame, segs[t.det]))
                res.score_ms.append((time.perf_counter() - t0) * 1000)
                t.reject = t.score >= cfg.threshold
                d = Decision(i, t.id, t.center, t.score, t.reject, datetime.now().isoformat())
                log.info("DECISION id=%d frame=%d score=%.1f %s", d.track_id, i, d.score, d.label)
                dec_csv.writerow(
                    [i, t.id, f"{t.center[0]:.1f}", f"{t.center[1]:.1f}", f"{t.score:.2f}", d.label]
                )
                res.decisions.append(d)
                new.append((t, d))

            annotated = draw_overlay(frame, tracker, res, fps_ema, banner, cfg.camera.roi)
            for t, d in new:
                if d.reject:
                    append_reject(app, stamp, frame, annotated, t, d)
            if out is not None:
                if writer is None:
                    writer = open_writer(out, app.out_fps or source.fps, annotated.shape)
                writer.write(annotated)
            res.frames += 1
            if not headless:
                cv2.imshow(app.window_name, annotated)
                delay = 1
                if not source.spec.is_camera:  # video: tempo seperti fps sumber
                    spent = (time.perf_counter() - t_loop) * 1000
                    delay = max(1, round(1000 / source.fps - spent))
                if cv2.waitKey(delay) & 0xFF in QUIT_KEYS:
                    log.info("Dihentikan pengguna (q/ESC)")
                    break
            dt = time.perf_counter() - t_loop
            fps_ema = 1 / dt if fps_ema == 0 else 0.9 * fps_ema + 0.1 / dt
    finally:
        res.seconds = time.perf_counter() - t_start
        dec_file.close()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
    return res


def open_writer(path: Path, fps: float, shape: tuple[int, ...]) -> cv2.VideoWriter:
    """VideoWriter mp4v (.mp4) / MJPG (lainnya)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = "mp4v" if path.suffix.lower() == ".mp4" else "MJPG"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (shape[1], shape[0]))
    if not w.isOpened():
        raise RuntimeError(f"VideoWriter gagal dibuka: {path}")
    return w


def log_summary(res: RunResult) -> None:
    """Ringkasan akhir run."""
    n = len(res.decisions)
    log.info(
        "Ringkasan: %s | keputusan %d (OK %d, REJECT %d)",
        res.timing(),
        n,
        n - res.n_reject,
        res.n_reject,
    )


# --- ground truth (video SYNTHETIC) ---------------------------------------------------------


def compare_gt(decisions: list[Decision], gt: dict[str, Any], max_dist: float) -> dict[str, Any]:
    """Cocokkan tiap keputusan ke roti GT terdekat (posisi prediksi pada frame keputusan).

    Duplikat = roti GT diputuskan > 1 kali; miss = roti GT yang diharapkan tanpa keputusan;
    tak cocok = keputusan tanpa roti GT dalam ``max_dist`` px atau ke roti yang tidak diharapkan.
    """
    buns = gt["buns"]
    by_bun: dict[int, list[Decision]] = {}
    unmatched: list[Decision] = []
    for d in decisions:
        dist, b = min(
            ((math.dist(predict(b, d.frame), d.center), b) for b in buns), key=lambda x: x[0]
        )
        if dist <= max_dist and b["expected"]:
            by_bun.setdefault(b["id"], []).append(d)
        else:
            unmatched.append(d)
    expected = [b for b in buns if b["expected"]]
    n_exp = len(expected)
    rows, per_pass = [], {}
    for b in expected:
        ds = by_bun.get(b["id"], [])
        first = ds[0] if ds else None
        rows.append({**b, "n_decisions": len(ds), "decision": first})
        p = per_pass.setdefault(
            b["pass"], {"defect": 0, "defect_rej": 0, "ok": 0, "ok_rej": 0, "missed": 0}
        )
        if first is None:
            p["missed"] += 1
            continue
        if b["label"] in ("defect", "ok"):
            p[b["label"]] += 1
            p[b["label"] + "_rej"] += int(first.reject)
    return {
        "expected": n_exp,
        "decided": len(decisions),
        "duplicates": sum(len(v) - 1 for v in by_bun.values()),
        "missed": sum(r["n_decisions"] == 0 for r in rows),
        "unmatched": len(unmatched),
        "miscount_pct": 100 * abs(len(decisions) - n_exp) / n_exp if n_exp else float("nan"),
        "per_pass": per_pass,
        "rows": rows,
        "unmatched_rows": unmatched,
    }


def gt_report(cmp: dict[str, Any], res: RunResult, gt: dict[str, Any], threshold: float) -> str:
    """Laporan markdown perbandingan app vs ground truth."""
    lines = [
        "# Demo M4 vs ground truth (SYNTHETIC — bukan untuk evaluasi akurasi)",
        "",
        f"Video: `{gt['video']}` {gt['frame_size'][0]}x{gt['frame_size'][1]}, {gt['n_frames']} "
        f"frame @ {gt['fps']:g} fps, {gt['direction']}, {gt['speed_px_per_frame']:g} px/frame, "
        f"garis {gt['line_pos']:g}; threshold {threshold:g}.",
        "",
        "| diharapkan | diputuskan | duplikat | miss | tak cocok | miscount |",
        "|---|---|---|---|---|---|",
        f"| {cmp['expected']} | {cmp['decided']} | {cmp['duplicates']} | {cmp['missed']} | "
        f"{cmp['unmatched']} | {cmp['miscount_pct']:.1f}% |",
        "",
        "| pass | kempes REJECT | OK ikut REJECT | miss |",
        "|---|---|---|---|",
    ]
    for p, v in sorted(cmp["per_pass"].items()):
        lines.append(
            f"| {p} | {v['defect_rej']}/{v['defect']} | {v['ok_rej']}/{v['ok']} | {v['missed']} |"
        )
    lines += [
        "",
        f"Waktu: {res.timing()}.",
        "",
        "| pass | foto | # | label | frame lintas (GT) | frame keputusan | track | skor | "
        "keputusan | n |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cmp["rows"]:
        d = r["decision"]
        label = r["label"] + (" (borderline)" if r["borderline"] else "")
        dec = f"{d.frame} | {d.track_id} | {d.score:.0f} | {d.label}" if d else "– | – | – | MISS"
        lines.append(
            f"| {r['pass']} | {r['photo']} | {r['photo_num']} | {label} | "
            f"{r['cross_frame']} | {dec} | {r['n_decisions']} |"
        )
    for d in cmp["unmatched_rows"]:
        lines.append(
            f"| – | tak cocok | – | – | – | {d.frame} | {d.track_id} | {d.score:.0f} | "
            f"{d.label} | – |"
        )
    return "\n".join(lines) + "\n"


# --- CLI -----------------------------------------------------------------------------------


def synthetic_sidecar(spec: str) -> dict[str, Any] | None:
    """Sidecar ``<video>.gt.json`` bila ada (penanda video SYNTHETIC)."""
    src = parse_source(spec)
    if src.is_camera:
        return None
    p = Path(str(src.target)).with_suffix(".gt.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main(argv: list[str] | None = None) -> int:
    """CLI demo real-time."""
    ap = argparse.ArgumentParser(description="Demo real-time LoafLens (video/kamera).")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--source", required=True, help="video:<path> | camera:<id>")
    ap.add_argument("--headless", action="store_true", help="tanpa jendela (cv2.imshow)")
    ap.add_argument("--out", type=Path, help="tulis video beranotasi")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--gt", type=Path, help="sidecar ground truth video SYNTHETIC")
    ap.add_argument("--synthetic", action="store_true", help="paksa banner SYNTHETIC")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    no_display = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not args.headless and sys.platform.startswith("linux") and no_display:
        log.error("Tidak ada display (DISPLAY kosong) — pakai --headless (mis. dengan --out)")
        return 2
    try:
        source = FrameSource(args.source)
    except SourceError as e:
        log.error("%s", e)
        return 2
    gt = json.loads(args.gt.read_text(encoding="utf-8")) if args.gt else None
    side = gt or synthetic_sidecar(args.source)
    banner = (
        cfg.app.synthetic_banner if args.synthetic or (side and side.get("synthetic")) else None
    )
    if gt and (gt["direction"] != cfg.track.direction or gt["line_pos"] != cfg.track.line_pos):
        log.warning(
            "GT dibuat untuk %s/garis %s, config %s/%s — perbandingan tidak valid",
            gt["direction"],
            gt["line_pos"],
            cfg.track.direction,
            cfg.track.line_pos,
        )

    from loaflens.score import load_scorer, model_path  # impor berat setelah validasi sumber

    if not model_path(cfg, Path.cwd()).is_file():
        log.error("Model %s tidak ada — jalankan `make train` dulu", model_path(cfg, Path.cwd()))
        source.release()
        return 2
    scorer = load_scorer(cfg, Path.cwd())
    with source:
        res = run(
            cfg,
            source,
            scorer,
            headless=args.headless,
            out=args.out,
            max_frames=args.max_frames,
            banner=banner,
        )
    log_summary(res)
    if gt:
        w, h = gt["frame_size"]
        cmp = compare_gt(res.decisions, gt, cfg.app.gt_match_dist * max(w, h))
        cfg.app.gt_report.parent.mkdir(parents=True, exist_ok=True)
        cfg.app.gt_report.write_text(gt_report(cmp, res, gt, cfg.threshold), encoding="utf-8")
        log.info(
            "GT: diharapkan %d, diputuskan %d, duplikat %d, miss %d, tak cocok %d, miscount "
            "%.1f%%; per pass %s → %s",
            cmp["expected"],
            cmp["decided"],
            cmp["duplicates"],
            cmp["missed"],
            cmp["unmatched"],
            cmp["miscount_pct"],
            cmp["per_pass"],
            cfg.app.gt_report,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
