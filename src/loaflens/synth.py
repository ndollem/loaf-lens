"""Video demo SYNTHETIC dari foto konveyor diam (SYNTHETIC — bukan untuk evaluasi akurasi).

``python -m loaflens.synth --config configs/baseline.yaml --out samples/demo.mp4``

Foto ``train-data/domain-check/`` ditumpuk jadi satu strip vertikal (foto pertama tampil dulu),
diberi bantalan polos warna konveyor di kedua ujung, lalu jendela selebar foto digeser dengan
kecepatan tetap sehingga roti bergerak searah ``--direction``. Roti yang terpotong tepi atas/bawah
foto (sambungan antar foto) ditutup warna konveyor agar tidak muncul sebagai roti setengah.
Ground truth (``<out>.gt.json``): roti utuh per foto dari ``segment_all`` (parameter = app,
batas luas disesuaikan ke luas frame), posisi tiap roti per frame, dan jumlah roti yang melewati
garis keputusan. Ini scroll foto diam: tanpa motion blur, perubahan cahaya, atau efek kamera.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from loaflens.config import DIRECTIONS, SegmentConfig, load_config
from loaflens.segment import binary_mask, segment_all
from loaflens.track import center

log = logging.getLogger(__name__)

SYNTHETIC_NOTE = "SYNTHETIC — bukan untuk evaluasi akurasi"
DEFAULT_IMAGES = [
    "train-data/domain-check/conveyor_mixed_01.webp",
    "train-data/domain-check/conveyor_good_01.webp",
]
LABELS_FILE = "labels.json"  # di folder foto: posisi roti kempes per foto
LABEL_MATCH_PX = 40.0  # jarak maks titik tengah roti ↔ posisi label (piksel foto)
BURN_TEXT = "SYNTHETIC - bukan untuk evaluasi akurasi"  # ditulis di sudut keluar tiap frame


# --- strip & frame (murni) -----------------------------------------------------------------


def fill_color(photo: np.ndarray, cfg: SegmentConfig) -> tuple[int, int, int]:
    """Warna median konveyor: pita tengah foto, di luar mask roti."""
    w = photo.shape[1]
    band = photo[:, w // 4 : 3 * w // 4]
    bg = binary_mask(band, cfg, cfg.obj_min_value) == 0
    return tuple(int(v) for v in np.median(band[bg], axis=0))


def cover_edge_cut(
    photo: np.ndarray, cfg: SegmentConfig, min_area_px: float, color: tuple[int, int, int]
) -> np.ndarray:
    """Tutup objek yang menyentuh tepi atas/bawah foto (roti terpotong sambungan) dengan ``color``.

    Objek yang juga menyentuh tepi kiri/kanan (rel, lantai) dibiarkan: app membuangnya karena
    menyentuh tepi frame.
    """
    out = photo.copy()
    h, w = photo.shape[:2]
    binary = binary_mask(photo, cfg, cfg.obj_min_value)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        vertical_edge = y <= 0 or y + bh >= h
        side_edge = x <= 0 or x + bw >= w
        if vertical_edge and not side_edge and cv2.contourArea(c) >= min_area_px:
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
            out[cv2.dilate(mask, kernel) > 0] = color
    return out


def build_strip(
    photos: list[np.ndarray], pad: int, color: tuple[int, int, int]
) -> tuple[np.ndarray, list[int]]:
    """Strip arah kanonik (roti bergerak ke bawah): foto pertama paling bawah (tampil lebih dulu).

    Kembalikan strip dan offset y tiap foto di strip (urutan sama dengan ``photos``).
    """
    w = photos[0].shape[1]
    blank = np.full((pad, w, 3), color, np.uint8)
    parts = [blank, *reversed(photos), blank]
    offsets, y = [], pad
    for p in reversed(photos):
        offsets.append(y)
        y += p.shape[0]
    return np.vstack(parts), list(reversed(offsets))


def n_frames(strip_h: int, win_h: int, speed: float) -> int:
    """Jumlah frame sampai jendela menyapu seluruh strip."""
    return int((strip_h - win_h) // speed) + 1


def window(strip: np.ndarray, f: int, win_h: int, speed: float) -> np.ndarray:
    """Frame kanonik ke-f: jendela mulai di dasar strip lalu naik ``speed`` px/frame."""
    top = round(strip.shape[0] - win_h - speed * f)
    return strip[top : top + win_h]


def orient(frame: np.ndarray, direction: str) -> np.ndarray:
    """Putar frame kanonik (gerak ke bawah) agar roti bergerak searah ``direction``."""
    if direction == "top_to_bottom":
        return frame
    if direction == "bottom_to_top":
        return cv2.rotate(frame, cv2.ROTATE_180)
    if direction == "left_to_right":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # right_to_left


def orient_point(p: tuple[float, float], direction: str, w: int, h: int) -> tuple[float, float]:
    """Titik di frame kanonik (lebar w, tinggi h) → koordinat setelah ``orient``."""
    x, y = p
    if direction == "top_to_bottom":
        return x, y
    if direction == "bottom_to_top":
        return w - 1 - x, h - 1 - y
    if direction == "left_to_right":
        return y, w - 1 - x
    return h - 1 - y, x  # right_to_left


def burn_label(frame: np.ndarray, direction: str) -> None:
    """Tulis label SYNTHETIC kecil di sudut sisi keluar (roti sudah diputuskan di sana)."""
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, tth), _ = cv2.getTextSize(BURN_TEXT, font, scale, th)
    h, w = frame.shape[:2]
    pos = {
        "top_to_bottom": (8, h - 10),
        "bottom_to_top": (8, tth + 10),
        "left_to_right": (w - tw - 8, h - 10),
        "right_to_left": (8, h - 10),
    }[direction]
    cv2.putText(frame, BURN_TEXT, pos, font, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(frame, BURN_TEXT, pos, font, scale, (255, 255, 255), th, cv2.LINE_AA)


# --- ground truth --------------------------------------------------------------------------


def scaled_segment_cfg(cfg: SegmentConfig, frame_area: float, photo_area: float) -> SegmentConfig:
    """Batas luas objek (fraksi luas ROI) disesuaikan agar luas pikselnya sama dengan di frame."""
    k = frame_area / photo_area
    return replace(
        cfg,
        obj_min_area_frac=cfg.obj_min_area_frac * k,
        obj_max_area_frac=min(cfg.obj_max_area_frac * k, 1.0),
    )


def bun_label(c: tuple[float, float], labels: dict[str, Any] | None) -> tuple[str, bool]:
    """(label, borderline) dari posisi: 'defect' | 'ok' | 'unknown' (foto tanpa label)."""
    if labels is None:
        return "unknown", False

    def near(points: list[list[float]]) -> bool:
        return any(math.dist(c, p) <= LABEL_MATCH_PX for p in points)

    if near(labels.get("defect", [])):
        return "defect", False
    return "ok", near(labels.get("borderline_ok", []))


def ground_truth(
    photo_buns: list[list[tuple[int, int, int, int]]],
    names: list[str],
    labels: list[dict[str, Any] | None],
    offsets: list[int],
    strip_h: int,
    win_size: tuple[int, int],
    speed: float,
    line_pos: float,
    direction: str,
    repeat_len: int,
) -> list[dict[str, Any]]:
    """Daftar roti GT: posisi di frame 0 + kecepatan (koordinat video akhir), frame lintas garis.

    ``expected`` = centroid melewati garis (frame sebelumnya < garis <= frame ini) selagi roti utuh
    di dalam frame pada kedua frame tersebut, dalam rentang video.
    """
    w, win_h = win_size
    nf = n_frames(strip_h, win_h, speed)
    line = line_pos * win_h
    o0 = strip_h - win_h
    v_canon = (0.0, speed)
    v = np.subtract(
        orient_point(v_canon, direction, w, win_h), orient_point((0.0, 0.0), direction, w, win_h)
    )
    buns = []
    for i, (bboxes, name, lab, off) in enumerate(
        zip(photo_buns, names, labels, offsets, strict=True)
    ):
        for k, bb in enumerate(bboxes):
            cx, cy = center(bb)
            y0 = off + cy - o0  # y kanonik pada frame 0
            half = bb[3] / 2
            f = max(1, math.ceil((line - y0) / speed))
            ok = (
                f < nf
                and y0 + speed * (f - 1) < line <= y0 + speed * f
                and y0 + speed * (f - 1) - half > 0
                and y0 + speed * f + half < win_h
            )
            label, borderline = bun_label((cx, cy), lab)
            buns.append(
                {
                    "id": len(buns),
                    "pass": i // repeat_len + 1,
                    "photo": name,
                    "photo_num": k + 1,
                    "photo_center": [round(cx, 1), round(cy, 1)],
                    "label": label,
                    "borderline": borderline,
                    "center_f0": [round(c, 2) for c in orient_point((cx, y0), direction, w, win_h)],
                    "velocity": [round(float(c), 4) for c in v],
                    "cross_frame": f if ok else None,
                    "expected": ok,
                }
            )
    return buns


def predict(bun: dict[str, Any], frame: int) -> tuple[float, float]:
    """Posisi centroid roti GT pada frame ke-``frame`` (koordinat video)."""
    (x0, y0), (vx, vy) = bun["center_f0"], bun["velocity"]
    return x0 + vx * frame, y0 + vy * frame


# --- CLI -----------------------------------------------------------------------------------


def load_labels(path: Path) -> dict[str, Any] | None:
    """Label posisi untuk foto ``path`` dari ``labels.json`` di foldernya (None bila tidak ada)."""
    f = path.parent / LABELS_FILE
    if not f.is_file():
        return None
    return json.loads(f.read_text(encoding="utf-8")).get(path.name)


def make_video(
    images: list[Path],
    out: Path,
    seg_cfg: SegmentConfig,
    direction: str,
    line_pos: float,
    speed: float,
    fps: float,
    win_h: int,
    repeat: int,
) -> dict[str, Any]:
    """Tulis video SYNTHETIC + sidecar ground truth; kembalikan isi sidecar."""
    photos = [cv2.imread(str(p)) for p in images]
    if any(p is None for p in photos):
        raise FileNotFoundError(f"gagal membaca salah satu foto: {images}")
    w = photos[0].shape[1]
    photos = [
        p if p.shape[1] == w else cv2.resize(p, (w, round(p.shape[0] * w / p.shape[1])))
        for p in photos
    ]
    if win_h >= min(p.shape[0] for p in photos):
        raise ValueError(f"--height {win_h} harus lebih kecil dari tinggi foto")
    color = fill_color(photos[0], seg_cfg)
    buns, covered = [], []
    for p in photos:
        cfg_p = scaled_segment_cfg(seg_cfg, w * win_h, p.shape[0] * p.shape[1])
        q = cover_edge_cut(p, cfg_p, cfg_p.obj_min_area_frac * p.shape[0] * p.shape[1], color)
        covered.append(q)
        buns.append([s.bbox for s in segment_all(q, cfg_p)])
    seq = covered * repeat
    strip, offsets = build_strip(seq, win_h, color)
    names = [p.name for p in images] * repeat
    labels = [load_labels(p) for p in images] * repeat
    gt = ground_truth(
        buns * repeat,
        names,
        labels,
        offsets,
        strip.shape[0],
        (w, win_h),
        speed,
        line_pos,
        direction,
        len(images),
    )
    nf = n_frames(strip.shape[0], win_h, speed)
    first = orient(window(strip, 0, win_h, speed), direction)
    size = (first.shape[1], first.shape[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter gagal dibuka: {out}")
    try:
        for f in range(nf):
            frame = orient(window(strip, f, win_h, speed), direction).copy()  # bukan view strip
            burn_label(frame, direction)
            writer.write(frame)
    finally:
        writer.release()
    expected = [b for b in gt if b["expected"]]
    side = {
        "synthetic": True,
        "note": SYNTHETIC_NOTE + ": scroll foto diam, label acuan visual; bukan rekaman konveyor.",
        "video": str(out),
        "frame_size": list(size),
        "fps": fps,
        "n_frames": nf,
        "direction": direction,
        "speed_px_per_frame": speed,
        "line_pos": line_pos,
        "images": [str(p) for p in images],
        "repeat": repeat,
        "expected_decisions": len(expected),
        "expected_by_label": {
            lab: sum(b["label"] == lab for b in expected) for lab in ("defect", "ok", "unknown")
        },
        "buns": gt,
    }
    gt_path = out.with_suffix(".gt.json")
    gt_path.write_text(json.dumps(side, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info(
        "Video SYNTHETIC %s: %dx%d, %d frame @ %g fps, %s %g px/frame; GT %d roti melintas "
        "(%s) → %s",
        out,
        size[0],
        size[1],
        nf,
        fps,
        direction,
        speed,
        len(expected),
        side["expected_by_label"],
        gt_path,
    )
    return side


def main(argv: list[str] | None = None) -> int:
    """CLI pembuat video demo SYNTHETIC."""
    parser = argparse.ArgumentParser(description="Video demo SYNTHETIC dari foto konveyor.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="segment + track")
    parser.add_argument("--out", default="samples/demo.mp4")
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES, help="urutan tampil")
    parser.add_argument("--direction", choices=sorted(DIRECTIONS), help="default: track config")
    parser.add_argument("--speed", type=float, default=8.0, help="px/frame")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--height", type=int, default=900, help="tinggi jendela (px) < foto")
    parser.add_argument("--repeat", type=int, default=2, help="ulang urutan foto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if cfg.camera.roi is not None:
        log.warning("camera.roi diabaikan untuk ground truth (diasumsikan seluruh frame)")
    if args.speed <= 0 or args.repeat < 1:
        parser.error("--speed harus > 0 dan --repeat >= 1")
    make_video(
        [Path(p) for p in args.images],
        Path(args.out),
        cfg.segment,
        args.direction or cfg.track.direction,
        cfg.track.line_pos,
        args.speed,
        args.fps,
        args.height,
        args.repeat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
