"""Test helper metrik evaluasi pada contoh buatan tangan."""

import numpy as np
import pytest

from loaflens.evaluate import (
    DomainCheck,
    auroc,
    binary_metrics,
    label_counts,
    prefix_group,
    recall_at_fpr,
)


def test_recall_at_fpr_hand_made() -> None:
    # 4 normal, 4 defect; satu normal skor tinggi (90) di atas dua defect (60, 40).
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([10, 20, 30, 90, 95, 80, 60, 40])
    rec, fpr, thr = recall_at_fpr(y, s, target=0.05)
    assert rec == pytest.approx(0.25) and fpr == 0.0 and thr == 95  # FPR 0 → hanya 95 lolos
    rec, fpr, _ = recall_at_fpr(y, s, target=0.25)
    assert rec == pytest.approx(1.0) and fpr == pytest.approx(0.25)
    assert auroc(y, s) == pytest.approx(13 / 16)


def test_binary_metrics_and_nan_auroc() -> None:
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(y, np.array([10, 60, 70, 40]), threshold=50)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
    assert np.isnan(auroc(np.array([1, 1]), np.array([1.0, 2.0])))


def test_prefix_group() -> None:
    assert prefix_group("a/bagus (3).png", "normal") == "bagus"
    assert prefix_group("a/medium (3).png", "normal") == "medium-good"
    assert prefix_group("a/medium.png", "defect") == "medium-bad"
    assert prefix_group("a/bad (0).png", "defect") == "bad"


def test_label_counts_scaled_and_merged_blob() -> None:
    # foto diperkecil x0.5: titik kempes (100, 100) & (400, 100) piksel asli → (50, 50) & (200, 50)
    dc = DomainCheck(
        "x.webp",
        scores=[90, 70, 10, 20],
        bboxes=[(30, 30, 40, 40), (150, 20, 100, 60), (300, 30, 40, 40), (400, 30, 40, 40)],
        thresholds={"config": 50},
    )
    c = label_counts(dc, [[100, 100], [400, 100]], scale=0.5)
    assert c == {"kempes": 2, "kempes_reject": 2, "ok": 2, "ok_reject": 0}
