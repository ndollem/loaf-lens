"""Test BaselineScorer fit/score/save/load (SYNTHETIC: elips mulus vs berkerut)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import synth_loaf
from loaflens.config import load_config
from loaflens.score import BaselineScorer, degrade

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def dataset(n: int, offset: int = 0) -> tuple[list[np.ndarray], list[int]]:
    imgs, labels = [], []
    for i in range(n):
        imgs.append(synth_loaf(seed=offset + i, axes=(95 + i % 10, 70)))
        labels.append(0)
        imgs.append(synth_loaf(seed=offset + 100 + i, axes=(95 + i % 10, 70), wrinkles=25))
        labels.append(1)
    return imgs, labels


@pytest.mark.parametrize("classifier", ["logistic_regression", "random_forest", "svm"])
def test_fit_score_save_load_roundtrip(tmp_path: Path, classifier: str) -> None:
    cfg = load_config(CONFIGS / "baseline.yaml")
    cfg.model.params = {**cfg.model.params, "classifier": classifier}
    imgs, labels = dataset(8)
    scorer = BaselineScorer.from_config(cfg).fit(imgs, labels)

    test_imgs, test_labels = dataset(4, offset=500)
    scores = np.array([scorer.score(im) for im in test_imgs])
    assert ((scores >= 0) & (scores <= 100)).all()
    y = np.array(test_labels)
    assert scores[y == 1].mean() > scores[y == 0].mean()

    scorer.save(tmp_path / "model.joblib")
    loaded = BaselineScorer.load(tmp_path / "model.joblib")
    assert loaded.score(test_imgs[0]) == pytest.approx(scores[0])
    assert loaded.params["classifier"] == classifier


def test_fit_requires_labels() -> None:
    scorer = BaselineScorer.from_config(load_config(CONFIGS / "baseline.yaml"))
    with pytest.raises(ValueError, match="labels"):
        scorer.fit([synth_loaf()])


def test_segmentation_failure_falls_back_to_full_frame() -> None:
    scorer = BaselineScorer.from_config(load_config(CONFIGS / "baseline.yaml"))
    vec = scorer.features(np.full((80, 80, 3), 200, np.uint8))
    assert scorer.seg_failures == 1 and np.isfinite(vec).all()


def big_dataset(n: int, offset: int = 0) -> tuple[list[np.ndarray], list[int]]:
    """Roti lebar ~200 px di frame 640x480 (SYNTHETIC)."""
    imgs, labels = [], []
    for i in range(n):
        for lab, wr in ((0, 0), (1, 25)):
            axes = (100 + i % 5 * 2, 70)
            imgs.append(synth_loaf((640, 480), (320, 240), axes, wr, offset + 100 * lab + i))
            labels.append(lab)
    return imgs, labels


def test_score_close_for_200px_vs_60px_bun() -> None:
    cfg = load_config(CONFIGS / "baseline.yaml")
    imgs, labels = big_dataset(6)
    scorer = BaselineScorer.from_config(cfg).fit(imgs, labels)
    test_imgs, test_labels = big_dataset(3, offset=500)
    for im, lab in zip(test_imgs, test_labels, strict=True):
        small = cv2.resize(im, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        big_s, small_s = scorer.score(im), scorer.score(small)
        assert abs(big_s - small_s) < 15, (lab, big_s, small_s)
        assert (small_s >= cfg.threshold) == bool(lab)


def test_represent_adds_deterministic_augmented_copies() -> None:
    cfg = load_config(CONFIGS / "baseline.yaml")
    scorer = BaselineScorer.from_config(cfg)
    img = synth_loaf((640, 480), (320, 240), (100, 70), wrinkles=10)
    rep = scorer.represent(img)
    copies = cfg.model.params["augment"]["copies"]
    assert rep.shape == (1 + copies, len(scorer.feature_names))
    np.testing.assert_array_equal(rep, scorer.represent(img))  # RNG = seed + isi gambar
    np.testing.assert_array_equal(rep[0], scorer.features(img))  # baris 0 = gambar asli
    cfg.model.params = {**cfg.model.params, "augment": None}
    assert BaselineScorer.from_config(cfg).represent(img).shape == (1, len(scorer.feature_names))


def test_degrade_downscales_but_never_upscales() -> None:
    img = synth_loaf((640, 480), (320, 240), (100, 70))
    assert degrade(img, bun_side=200, bun_px=50, quality=60).shape == (120, 160, 3)
    assert degrade(img, bun_side=200, bun_px=400, quality=60).shape == img.shape
