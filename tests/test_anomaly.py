"""Test AnomalyScorer (SYNTHETIC; backbone acak weights=null — tanpa jaringan, tanpa train-data)."""

import socket
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import synth_loaf
from loaflens.anomaly import AnomalyScorer, heatmap_overlay
from loaflens.config import SegmentConfig

PARAMS = {
    "weights": None,
    "weights_dir": None,
    "image_size": 64,
    "fill_background": True,
    "mask_erode_px": 2,
    "pool": 3,
    "bank_ratio": 0.5,
    "bank_max": 20000,
    "knn": 1,
    "image_score": "max",
    "topk_frac": 0.01,
    "ref_percentile": 95,
    "heatmap_sigma": 2,
    "num_threads": 1,
}


def normals(n: int, offset: int = 0) -> list[np.ndarray]:
    return [synth_loaf(seed=offset + i, axes=(95 + i % 5, 70)) for i in range(n)]


@pytest.fixture(scope="module")
def fitted() -> AnomalyScorer:
    return AnomalyScorer(PARAMS, SegmentConfig(), seed=0).fit(normals(6))


def test_fit_score_save_load_roundtrip(fitted: AnomalyScorer, tmp_path: Path) -> None:
    assert fitted.ref is not None and fitted.ref > 0 and len(fitted.loo_raw) == 6
    imgs = normals(2, offset=50) + [synth_loaf(seed=60, wrinkles=40)]
    scores = [fitted.score(im) for im in imgs]
    assert all(0 <= s <= 100 for s in scores)

    fitted.save(tmp_path / "model.pt")
    loaded = AnomalyScorer.load(tmp_path / "model.pt")
    assert torch.equal(loaded.bank, fitted.bank) and loaded.ref == fitted.ref
    assert loaded.score(imgs[2]) == pytest.approx(scores[2], abs=1e-4)


def test_heatmap_shape_and_range(fitted: AnomalyScorer) -> None:
    img = synth_loaf(seed=70, wrinkles=20)
    hm = fitted.heatmap(img)
    assert hm.shape == (64, 64) and hm.dtype == np.float32
    assert hm.min() >= 0 and hm.max() <= 100 and hm.max() > 0
    c_img, c_mask = fitted.prepare(img)
    assert heatmap_overlay(c_img, c_mask, hm).shape == c_img.shape


def test_fit_rejects_defect_labels() -> None:
    scorer = AnomalyScorer(PARAMS, SegmentConfig(), seed=0)
    with pytest.raises(ValueError, match="normal"):
        scorer.fit(normals(2), labels=[0, 1])


def test_fit_repr_ignores_defects(fitted: AnomalyScorer) -> None:
    embs = [fitted.embed(im) for im in normals(3)]
    defect = fitted.embed(synth_loaf(seed=9, wrinkles=40))
    a = AnomalyScorer(PARAMS, SegmentConfig(), seed=0).fit_repr(
        [*embs, defect], np.array([0, 0, 0, 1])
    )
    b = AnomalyScorer(PARAMS, SegmentConfig(), seed=0).fit_repr(embs, np.array([0, 0, 0]))
    assert torch.equal(a.bank, b.bank)


def test_inference_makes_no_network_calls(
    fitted: AnomalyScorer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted.save(tmp_path / "model.pt")

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("panggilan jaringan saat inferensi")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(torch.hub, "download_url_to_file", boom)
    loaded = AnomalyScorer.load(tmp_path / "model.pt")
    assert 0 <= loaded.score(synth_loaf(seed=3)) <= 100

    # bobot ImageNet hilang → error jelas, bukan unduhan
    missing = {**PARAMS, "weights": "imagenet", "weights_dir": str(tmp_path / "kosong")}
    with pytest.raises(FileNotFoundError, match="tidak ditemukan"):
        AnomalyScorer(missing, SegmentConfig(), seed=0).score(synth_loaf())
