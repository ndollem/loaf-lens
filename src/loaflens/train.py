"""CLI training: ``python -m loaflens.train --config configs/{baseline,anomaly}.yaml``.

Fit hanya pada split ``train`` dari manifest beku, label sesuai ``model.train_labels`` (anomaly:
normal saja). Simpan model + salinan config + contact sheet segmentasi untuk dicek mata.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from loaflens.config import Config, load_config
from loaflens.data import load_image, read_manifest
from loaflens.score import Scorer, model_path, new_scorer
from loaflens.segment import overlay, segment

log = logging.getLogger(__name__)

LABEL_TO_Y = {"normal": 0, "defect": 1}
TILE = 160  # px per tile di contact sheet
SHEET_COLS = 8


def segment_debug_sheet(cfg: Config, rows: list[dict[str, str]], root: Path, out: Path) -> int:
    """Contact sheet overlay mask (hijau) + bbox (merah) + nama file; kembalikan jumlah gagal."""
    tiles, failures = [], 0
    for row in rows:
        img = load_image(row["path"], root)
        seg = segment(img, cfg.segment)
        failures += seg is None
        tile = cv2.resize(overlay(img, seg), (TILE, TILE), interpolation=cv2.INTER_AREA)
        name = f"{row['label'][0].upper()} {Path(row['path']).stem}"
        cv2.rectangle(tile, (0, 0), (TILE, 16), (255, 255, 255), cv2.FILLED)
        cv2.putText(tile, name, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        tiles.append(tile)
    while len(tiles) % SHEET_COLS:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    grid = [np.hstack(tiles[i : i + SHEET_COLS]) for i in range(0, len(tiles), SHEET_COLS)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(grid))
    return failures


def train(cfg: Config, config_path: Path, root: Path) -> Scorer:
    """Latih scorer (baseline/anomaly) dari split train; simpan artefak ke ``output.dir``."""
    rows = read_manifest(root / cfg.data.manifest)
    train_rows = [r for r in rows if r["split"] == "train" and r["label"] in cfg.model.train_labels]
    images = [load_image(r["path"], root) for r in train_rows]
    labels = [LABEL_TO_Y[r["label"]] for r in train_rows]
    log.info("Train: %d gambar (defect=%d)", len(labels), sum(labels))

    scorer = new_scorer(cfg).fit(images, labels)
    out_dir = root / cfg.output_dir
    path = model_path(cfg, root)
    scorer.save(path)
    shutil.copyfile(config_path, out_dir / "config.yaml")
    log.info(
        "Model: %s (%.1f kB; gagal segmentasi saat train: %d)",
        path,
        path.stat().st_size / 1e3,
        scorer.seg_failures,
    )

    failures = segment_debug_sheet(cfg, rows, root, out_dir / "segment_debug.jpg")
    log.info(
        "Contact sheet segmentasi: %s (%d/%d gagal)",
        out_dir / "segment_debug.jpg",
        failures,
        len(rows),
    )
    return scorer


def main(argv: list[str] | None = None) -> int:
    """CLI training."""
    parser = argparse.ArgumentParser(description="Latih scorer dari split train.")
    parser.add_argument("--config", required=True, help="path configs/*.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config_path = Path(args.config)
    train(load_config(config_path), config_path, Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
