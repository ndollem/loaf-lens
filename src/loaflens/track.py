"""Tracking centroid sederhana + garis keputusan: satu roti = satu keputusan (F6).

Logika murni (tanpa I/O). Tiap frame: deteksi (bbox) dicocokkan ke track dengan tetangga terdekat
dari posisi prediksi (kecepatan konstan), maksimal ``max_distance``. Track diputuskan
**tepat sekali** saat centroid-nya melewati garis keputusan searah gerak (posisi sebelumnya
< garis <= posisi kini).
Track yang lahir sudah di seberang garis tidak pernah diputuskan (hindari hitung ganda).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from loaflens.config import TrackConfig

BBox = tuple[int, int, int, int]  # x, y, w, h
Point = tuple[float, float]
VERTICAL = {"top_to_bottom", "bottom_to_top"}


def center(bbox: BBox) -> Point:
    """Titik tengah bbox."""
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


@dataclass
class Track:
    id: int
    center: Point
    bbox: BBox
    velocity: Point = (0.0, 0.0)  # px/frame
    missed: int = 0  # frame berturut-turut tanpa deteksi
    det: int | None = None  # indeks deteksi di frame terakhir; None = tidak terlihat
    decided: bool = False
    score: float | None = None  # diisi pemanggil saat keputusan
    reject: bool | None = None


class CentroidTracker:
    """Tracker centroid tetangga terdekat dengan garis keputusan dari ``TrackConfig``."""

    def __init__(self, cfg: TrackConfig, frame_size: tuple[int, int]) -> None:
        self.direction = cfg.direction
        self.w, self.h = frame_size
        extent = self.h if cfg.direction in VERTICAL else self.w
        self.line = cfg.line_pos * extent  # dalam koordinat "kemajuan" (0 = tepi masuk)
        self.max_dist = cfg.max_distance * max(self.w, self.h)
        self.max_missed = cfg.max_missed
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def progress(self, p: Point) -> float:
        """Jarak tempuh searah gerak dari tepi masuk frame."""
        x, y = p
        return {
            "left_to_right": x,
            "right_to_left": self.w - x,
            "top_to_bottom": y,
            "bottom_to_top": self.h - y,
        }[self.direction]

    def line_points(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Dua titik ujung garis keputusan (koordinat piksel frame) untuk digambar."""
        if self.direction in VERTICAL:
            y = round(self.line if self.direction == "top_to_bottom" else self.h - self.line)
            return (0, y), (self.w - 1, y)
        x = round(self.line if self.direction == "left_to_right" else self.w - self.line)
        return (x, 0), (x, self.h - 1)

    def visible(self) -> list[Track]:
        """Track yang terdeteksi di frame terakhir."""
        return [t for t in self.tracks.values() if t.det is not None]

    def update(self, boxes: list[BBox]) -> list[Track]:
        """Proses deteksi satu frame; kembalikan track yang baru melewati garis (putuskan kini)."""
        dets = [center(b) for b in boxes]
        pairs = []
        for tid, t in self.tracks.items():
            gap = t.missed + 1
            px, py = t.center[0] + t.velocity[0] * gap, t.center[1] + t.velocity[1] * gap
            for j, (dx, dy) in enumerate(dets):
                d = math.hypot(dx - px, dy - py)
                if d <= self.max_dist:
                    pairs.append((d, tid, j))
        used_t: set[int] = set()
        used_d: set[int] = set()
        crossed = []
        for _, tid, j in sorted(pairs):  # greedy: pasangan terdekat dulu
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            t = self.tracks[tid]
            gap = t.missed + 1
            new = dets[j]
            if not t.decided and self.progress(t.center) < self.line <= self.progress(new):
                t.decided = True
                crossed.append(t)
            t.velocity = ((new[0] - t.center[0]) / gap, (new[1] - t.center[1]) / gap)
            t.center, t.bbox, t.missed, t.det = new, boxes[j], 0, j
        for tid in list(self.tracks):
            if tid in used_t:
                continue
            t = self.tracks[tid]
            t.missed += 1
            t.det = None
            if t.missed > self.max_missed:
                del self.tracks[tid]
        for j, b in enumerate(boxes):
            if j not in used_d:
                self.tracks[self.next_id] = Track(self.next_id, dets[j], b, det=j)
                self.next_id += 1
        return crossed
