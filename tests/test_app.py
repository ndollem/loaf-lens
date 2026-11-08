"""Test end-to-end headless: video SYNTHETIC kecil → app → keputusan (tanpa model, tanpa layar).

SYNTHETIC — bukan untuk evaluasi akurasi.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from conftest import LOAF
from loaflens.app import compare_gt, run
from loaflens.capture import FrameSource
from loaflens.config import Config, from_dict
from loaflens.segment import Segmentation
from loaflens.synth import (
    build_strip,
    cover_edge_cut,
    make_video,
    n_frames,
    orient,
    orient_point,
    window,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BG = (30, 30, 30)  # konveyor gelap (V < obj_min_value → bukan roti)
CENTERS = [(60, 60), (140, 110), (60, 170), (140, 230)]


def photo() -> np.ndarray:
    """SYNTHETIC 200x300: 4 roti utuh + 1 terpotong tepi bawah (harus ditutup)."""
    img = np.full((300, 200, 3), BG, np.uint8)
    for c in [*CENTERS, (100, 295)]:
        cv2.ellipse(img, c, (25, 18), 0, 0, 360, LOAF, thickness=cv2.FILLED)
    return img


def config(tmp_path: Path, direction: str) -> Config:
    raw = yaml.safe_load((CONFIGS / "baseline.yaml").read_text(encoding="utf-8"))
    raw["track"]["direction"] = direction
    raw["app"] = {
        "snapshot_dir": str(tmp_path / "rejects"),
        "rejects_csv": str(tmp_path / "rejects" / "rejects.csv"),
        "decisions_csv": str(tmp_path / "app" / "decisions.csv"),
        "gt_report": str(tmp_path / "app" / "gt.md"),
    }
    return from_dict(raw)


class StubScorer:
    """Bukan model: skor bergantian 90 / 10 per panggilan."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, image: np.ndarray, seg: Segmentation | None = None) -> float:
        assert seg is not None
        self.calls += 1
        return 90.0 if self.calls % 2 else 10.0


def test_cover_edge_cut_hides_seam_bun() -> None:
    from loaflens.config import SegmentConfig
    from loaflens.segment import segment_all

    cfg = SegmentConfig(obj_min_value=40)
    covered = cover_edge_cut(photo(), cfg, 200, BG)
    assert covered[295, 100].tolist() == list(BG)  # roti terpotong ditutup
    assert len(segment_all(covered, cfg)) == len(CENTERS)


def test_strip_window_and_orientation() -> None:
    a, b = np.zeros((30, 10, 3), np.uint8), np.full((20, 10, 3), 255, np.uint8)
    strip, offsets = build_strip([a, b], 5, (7, 7, 7))
    assert strip.shape[0] == 60 and offsets == [25, 5]  # foto pertama paling bawah
    assert n_frames(60, 10, 4) == 13
    assert window(strip, 0, 10, 4)[-1, 0, 0] == 7  # mulai di bantalan bawah
    canon = np.zeros((6, 8, 3), np.uint8)
    canon[1, 2] = 255
    for d in ("top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"):
        ys, xs = np.nonzero(orient(canon, d)[..., 0])
        assert (xs[0], ys[0]) == orient_point((2, 1), d, 8, 6)


@pytest.mark.parametrize("direction", ["top_to_bottom", "right_to_left"])
def test_end_to_end_headless_decides_each_bun_once(tmp_path: Path, direction: str) -> None:
    cfg = config(tmp_path, direction)
    img_path = tmp_path / "photo.png"
    cv2.imwrite(str(img_path), photo())
    video = tmp_path / "demo.mp4"
    side = make_video([img_path], video, cfg.segment, direction, 0.5, 4.0, 15.0, 160, 2)
    gt = json.loads(video.with_suffix(".gt.json").read_text(encoding="utf-8"))
    assert (
        gt["synthetic"]
        and gt["expected_decisions"] == 2 * len(CENTERS) == side["expected_decisions"]
    )

    with FrameSource(f"video:{video}") as src:
        res = run(cfg, src, StubScorer(), headless=True, out=tmp_path / "out.avi", banner="SYN")
    assert res.frames == gt["n_frames"]
    assert len(res.decisions) == 2 * len(CENTERS)
    cmp = compare_gt(res.decisions, gt, 20)
    assert (cmp["duplicates"], cmp["missed"], cmp["unmatched"]) == (0, 0, 0)
    n_rej = res.n_reject
    assert 0 < n_rej < len(res.decisions)
    rows = (tmp_path / "rejects" / "rejects.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + n_rej
    assert len(list((tmp_path / "rejects").glob("*_crop.jpg"))) == n_rej
    dec = (tmp_path / "app" / "decisions.csv").read_text(encoding="utf-8").splitlines()
    assert len(dec) == 1 + len(res.decisions)
    assert (tmp_path / "out.avi").stat().st_size > 0
