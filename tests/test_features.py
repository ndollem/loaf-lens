"""Test fitur: panjang vektor, LBP, dan invarian terhadap ukuran gambar asli (anti shortcut)."""

from pathlib import Path

import cv2
import numpy as np

from conftest import synth_loaf
from loaflens.config import load_config
from loaflens.features import FEATURE_NAMES, LBP_BINS, lbp_hist
from loaflens.score import BaselineScorer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def scorer() -> BaselineScorer:
    return BaselineScorer.from_config(load_config(CONFIGS / "baseline.yaml"))


def test_feature_vector_matches_names() -> None:
    vec = scorer().features(synth_loaf(wrinkles=10))
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vec).all()


def test_lbp_hist_flat_image_is_single_code() -> None:
    gray = np.full((20, 20), 100, np.uint8)
    hist = lbp_hist(gray, np.full((20, 20), 255, np.uint8))
    assert hist.shape == (LBP_BINS,)
    assert hist[8] == 1.0  # semua tetangga >= pusat → 8 bit satu


def test_features_invariant_to_image_upscale() -> None:
    """Gambar yang sama diperbesar 1.3x → fitur nyaris sama (ukuran gambar tak bocor)."""
    s = scorer()
    img = synth_loaf(wrinkles=12, seed=3)
    big = cv2.resize(img, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_LINEAR)
    a, b = s.features(img), s.features(big)
    rel = np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-3)
    assert np.median(rel) < 0.05, dict(zip(FEATURE_NAMES, rel.round(3), strict=True))
    assert rel.max() < 0.35, dict(zip(FEATURE_NAMES, rel.round(3), strict=True))
