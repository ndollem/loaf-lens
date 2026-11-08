"""Test segmentasi + crop ukuran tetap (gambar sintetis)."""

import cv2
import numpy as np

from conftest import LOAF, synth_loaf
from loaflens.config import SegmentConfig
from loaflens.segment import crop, full_frame, segment, segment_all


def test_segment_finds_ellipse_bbox() -> None:
    img = synth_loaf(center=(160, 125), axes=(100, 70))
    seg = segment(img, SegmentConfig())
    assert seg is not None
    x, y, w, h = seg.bbox
    assert abs(x - 60) <= 4 and abs(y - 55) <= 4
    assert abs(w - 201) <= 6 and abs(h - 141) <= 6
    assert seg.mask[125, 160] == 255 and seg.mask[5, 5] == 0


def test_segment_returns_none_on_blank_background() -> None:
    img = np.full((100, 100, 3), 200, np.uint8)
    assert segment(img, SegmentConfig()) is None


def test_crop_is_fixed_size_and_keeps_aspect() -> None:
    img = synth_loaf()
    seg = segment(img, SegmentConfig())
    c_img, c_mask = crop(img, seg, size=128, margin=0.05)
    assert c_img.shape == (128, 128, 3) and c_mask.shape == (128, 128)
    rows = np.flatnonzero(c_mask.any(axis=1))
    cols = np.flatnonzero(c_mask.any(axis=0))
    aspect = (rows[-1] - rows[0] + 1) / (cols[-1] - cols[0] + 1)
    assert abs(aspect - 141 / 201) < 0.05


def test_full_frame_fallback_covers_image() -> None:
    img = synth_loaf()
    seg = full_frame(img)
    assert seg.mask.all() and seg.bbox == (0, 0, img.shape[1], img.shape[0])


def conveyor_frame() -> np.ndarray:
    """SYNTHETIC: 4 elips roti di latar gelap + satu terpotong tepi kanan + bintik kecil."""
    img = np.full((300, 400, 3), 40, np.uint8)
    for c in [(80, 70), (220, 70), (80, 200), (220, 210)]:
        cv2.ellipse(img, c, (40, 30), 0, 0, 360, LOAF, thickness=cv2.FILLED)
    cv2.ellipse(img, (385, 150), (40, 30), 0, 0, 360, LOAF, thickness=cv2.FILLED)  # tepi
    cv2.circle(img, (220, 70), 12, (230, 230, 230), thickness=cv2.FILLED)  # kilap (S rendah)
    cv2.circle(img, (300, 270), 3, (0, 220, 255), thickness=cv2.FILLED)  # bintik kecil
    return img


def test_segment_all_finds_ellipses_and_drops_border_object() -> None:
    img = conveyor_frame()
    segs = segment_all(img, SegmentConfig())
    centers = [(x + w // 2, y + h // 2) for x, y, w, h in (s.bbox for s in segs)]
    assert len(segs) == 4
    assert centers == sorted(centers, key=lambda c: (c[1] - 30, c[0]))  # urut atas→bawah
    assert all(abs(w - 81) <= 4 and abs(h - 61) <= 4 for _, _, w, h in (s.bbox for s in segs))
    assert segs[1].mask[70, 220] == 255  # kilap di tengah roti ikut terisi

    assert len(segment_all(img, SegmentConfig(drop_border=False))) == 5


def test_segment_all_respects_roi() -> None:
    img = conveyor_frame()
    segs = segment_all(img, SegmentConfig(), roi=[0.0, 0.0, 0.5, 1.0])  # kiri: x < 200
    assert [s.bbox[0] < 200 for s in segs] == [True, True]
