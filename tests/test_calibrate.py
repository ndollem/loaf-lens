"""Test helper kalibrasi threshold pada contoh buatan tangan."""

from pathlib import Path

import numpy as np
import pytest

from loaflens.calibrate import calibrate, rates, threshold_for_fpr
from loaflens.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_threshold_for_fpr_hand_made() -> None:
    normals = np.array([10.0, 20.0, 30.0, 40.0, 84.1, 26.3, 5.0, 1.0, 2.0, 3.0])
    # FPR 0 → tepat di atas normal tertinggi
    assert threshold_for_fpr(normals, 0.0) == pytest.approx(84.2)
    # 10 normal, target 0.1 → boleh 1 FP → di atas normal tertinggi ke-2 (40)
    assert threshold_for_fpr(normals, 0.1) == pytest.approx(40.1)
    # target 0.05 dengan 10 normal → floor(0.5) = 0 FP
    assert threshold_for_fpr(normals, 0.05) == pytest.approx(84.2)
    assert threshold_for_fpr(np.array([100.0]), 0.0) == 100.0  # dibatasi ke rentang skor


def test_threshold_moves_to_middle_of_gap_with_defects() -> None:
    normals = np.array([10.0, 20.0, 30.0, 40.0, 84.1, 26.3, 5.0, 1.0, 2.0, 3.0])
    defects = np.array([35.0, 71.7, 90.0])  # 35 di bawah batas 40.1 → tetap terlewat
    t = threshold_for_fpr(normals, 0.1, defects)
    assert t == pytest.approx(55.8)  # tengah (40 + 71.7) / 2, dibulatkan ke bawah 0.1
    y = np.array([0] * 10 + [1] * 3)
    fpr, rec = rates(y, np.concatenate([normals, defects]), t)
    assert fpr == pytest.approx(0.1) and rec == pytest.approx(2 / 3)


def test_threshold_is_strictly_above_boundary_score() -> None:
    t = threshold_for_fpr(np.array([12.3, 1.0]), 0.0)
    assert t > 12.3 and rates(np.array([0, 0]), np.array([12.3, 1.0]), t)[0] == 0.0


def test_calibrate_uses_cv_as_suggestion() -> None:
    cfg = load_config(CONFIGS / "baseline.yaml")  # fpr_target 0.05
    y = np.array([0] * 20 + [1] * 4)
    s = np.concatenate([np.linspace(0, 19, 20), [30, 40, 50, 10]])
    cal = calibrate(y, s, np.array([0, 0, 1]), np.array([5.0, 7.0, 60.0]), cfg)
    # 20 normal 0..19, boleh 1 FP → batas di atas 18; kempes terendah di atasnya = 30 → tengah 24
    assert cal["suggested_threshold"] == cal["cv"]["threshold"] == pytest.approx(24.0)
    assert cal["cv"]["fpr"] == pytest.approx(0.0) and cal["cv"]["recall"] == pytest.approx(0.75)
    assert cal["val"]["threshold"] == pytest.approx(33.5) and cal["val"]["fpr_step"] == 0.5
