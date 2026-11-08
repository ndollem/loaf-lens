"""Test BaselineScorer fit/score/save/load (SYNTHETIC: elips mulus vs berkerut)."""

from pathlib import Path

import numpy as np
import pytest

from conftest import synth_loaf
from loaflens.config import load_config
from loaflens.score import BaselineScorer

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
