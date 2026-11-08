"""Kalibrasi threshold (F9): ``python -m loaflens.calibrate --config configs/<x>.yaml``.

Threshold dipilih dari data NON-test saja untuk target FPR (``eval.fpr_target``): skor CV
out-of-fold train+val (lebih banyak titik) sebagai saran utama, val saja sebagai pembanding.
Hasil ditulis ke ``<output.dir>/calibration.yaml`` dan di-log. Config TIDAK diubah otomatis.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from loaflens.config import Config, load_config

log = logging.getLogger(__name__)


def _above(v: float) -> float:
    """Kelipatan 0.1 terkecil yang > v."""
    t = math.floor(v * 10 + 1e-6) / 10 + 0.1
    return round(t + 0.1 if t <= v else t, 1)


def threshold_for_fpr(
    normal_scores: np.ndarray, target: float, defect_scores: np.ndarray | None = None
) -> float:
    """Threshold (kelipatan 0.1) dengan FPR ≤ target pada skor roti normal; reject = skor ≥ t.

    Dengan n normal boleh ada k = floor(target·n) false positive → batas bawah = sedikit di atas
    skor normal tertinggi ke-(k+1). Bila ``defect_scores`` diberikan, threshold digeser ke tengah
    celah antara batas itu dan skor kempes terendah di atasnya (margin maksimum; FPR & recall pada
    data kalibrasi tidak berubah). Hasil dibatasi ke [0, 100].
    """
    s = np.sort(np.asarray(normal_scores, float))[::-1]
    k = int(math.floor(target * len(s) + 1e-9))
    if len(s) == 0 or k >= len(s):
        return 0.0
    t = _above(s[k])
    if defect_scores is not None:
        above = np.asarray(defect_scores, float)
        above = above[above >= t]
        if above.size:
            t = max(t, math.floor((s[k] + above.min()) / 2 * 10) / 10)
    return float(min(t, 100.0))


def rates(y: np.ndarray, s: np.ndarray, threshold: float) -> tuple[float, float]:
    """(FPR, recall) untuk reject = skor ≥ threshold."""
    pred = s >= threshold
    neg, pos = y == 0, y == 1
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    rec = float(pred[pos].mean()) if pos.any() else float("nan")
    return fpr, rec


def calibrate(
    y_cv: np.ndarray, s_cv: np.ndarray, y_val: np.ndarray, s_val: np.ndarray, cfg: Config
) -> dict[str, Any]:
    """Saran threshold dari CV out-of-fold (utama) dan val (pembanding) untuk ``fpr_target``."""
    target = cfg.eval.fpr_target
    out: dict[str, Any] = {
        "fpr_target": target,
        "config_threshold": cfg.threshold,
        "model_type": cfg.model.type,
    }
    for name, y, s in (("cv", y_cv, s_cv), ("val", y_val, s_val)):
        n_neg = int((y == 0).sum())
        t = threshold_for_fpr(s[y == 0], target, s[y == 1])
        fpr, rec = rates(y, s, t)
        fpr_c, rec_c = rates(y, s, cfg.threshold)
        out[name] = {
            "threshold": t,
            "n_normal": n_neg,
            "n_defect": int((y == 1).sum()),
            "fpr_step": round(1 / n_neg, 4) if n_neg else None,  # resolusi FPR (kasar)
            "fpr": round(fpr, 4),
            "recall": round(rec, 4),
            "fpr_at_config": round(fpr_c, 4),
            "recall_at_config": round(rec_c, 4),
        }
    out["suggested_threshold"] = out["cv"]["threshold"]
    return out


def write_calibration(cal: dict[str, Any], path: Path) -> None:
    """Tulis calibration.yaml (saran saja) + log ringkas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Saran threshold dari data NON-test (CV out-of-fold train+val; val pembanding).\n"
        "# Config TIDAK diubah otomatis — salin manual ke `threshold:` bila disetujui.\n"
    )
    path.write_text(header + yaml.safe_dump(cal, sort_keys=False, allow_unicode=True), "utf-8")
    cv, val = cal["cv"], cal["val"]
    log.info(
        "Kalibrasi (FPR ≤ %g): CV → threshold %.1f (FPR %.2f, recall %.2f; n normal %d, langkah "
        "FPR %.3f) | val saja → %.1f (FPR %.2f, recall %.2f; n normal %d) | config %g → %s",
        cal["fpr_target"],
        cv["threshold"],
        cv["fpr"],
        cv["recall"],
        cv["n_normal"],
        cv["fpr_step"],
        val["threshold"],
        val["fpr"],
        val["recall"],
        val["n_normal"],
        cal["config_threshold"],
        path,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI kalibrasi: CV train+val + val (test tidak disentuh) → calibration.yaml."""
    from loaflens.evaluate import calibration_inputs  # impor lambat: evaluate memakai modul ini

    parser = argparse.ArgumentParser(description="Saran threshold dari CV train+val (bukan test).")
    parser.add_argument("--config", required=True, help="path configs/*.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg, root = load_config(args.config), Path.cwd()
    y_cv, s_cv, y_val, s_val = calibration_inputs(cfg, root)
    write_calibration(
        calibrate(y_cv, s_cv, y_val, s_val, cfg), root / cfg.output_dir / "calibration.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
