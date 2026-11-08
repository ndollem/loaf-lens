"""Dataset: scan folder label → manifest CSV dengan split train/val/test per grup.

Grup = batch_id (subfolder) yang digabung dengan near-duplicate (dHash), supaya gambar mirip
tidak bocor antar split. Split dibekukan di manifest; buat ulang hanya dengan ``--force``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from loaflens.config import Config, DataConfig, load_config

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "path",
    "label",
    "batch_id",
    "date",
    "camera",
    "camera_view",
    "lighting",
    "bread_type",
    "split",
]
META_COLUMNS = ["date", "camera", "camera_view", "lighting", "bread_type"]
SPLITS = ("train", "val", "test")
FALLBACK_PREFIX = "img_"  # batch_id per gambar bila tidak ada subfolder batch

Row = dict[str, str]


# --- I/O gambar ---------------------------------------------------------------------------


def load_image(path: str | Path, root: Path | None = None) -> np.ndarray:
    """Baca gambar sebagai array BGR uint8; raise ValueError bila tak terbaca."""
    full = (root / path) if root is not None else Path(path)
    img = cv2.imread(str(full), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"gambar tidak terbaca: {full}")
    return img


# --- scan ---------------------------------------------------------------------------------


def label_dirs(data: DataConfig) -> dict[str, Path]:
    """Peta label → folder sumber."""
    return {"normal": data.normal_dir, "defect": data.defect_dir}


def batch_id_for(file: Path, label_dir: Path, label: str) -> str:
    """batch_id = nama subfolder langsung di bawah folder label; fallback satu grup per gambar."""
    rel = file.relative_to(label_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return f"{FALLBACK_PREFIX}{label}_{file.name}"


def scan_dataset(data: DataConfig, root: Path) -> list[Row]:
    """Kumpulkan semua gambar di folder label → baris manifest (split masih kosong)."""
    rows: list[Row] = []
    for label, rel_dir in label_dirs(data).items():
        label_dir = root / rel_dir
        if not label_dir.is_dir():
            log.warning("Folder label '%s' tidak ada: %s", label, label_dir)
            continue
        files = sorted(
            f
            for f in label_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in data.extensions and not f.name.startswith(".")
        )
        for f in files:
            row = dict.fromkeys(MANIFEST_COLUMNS, "")
            row.update(
                path=f.relative_to(root).as_posix(),
                label=label,
                batch_id=batch_id_for(f, label_dir, label),
            )
            rows.append(row)
    return rows


# --- near-duplicate & grup ----------------------------------------------------------------


def dhash(img: np.ndarray, size: int = 8) -> int:
    """Difference hash (size*size bit) dari gambar BGR/gray."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate_pairs(hashes: list[int], max_dist: int) -> list[tuple[int, int, int]]:
    """Semua pasangan (i, j, jarak) dengan jarak Hamming dHash <= max_dist."""
    return [
        (i, j, d)
        for i in range(len(hashes))
        for j in range(i + 1, len(hashes))
        if (d := (hashes[i] ^ hashes[j]).bit_count()) <= max_dist
    ]


def group_ids(batch_ids: list[str], pairs: list[tuple[int, int, int]]) -> list[int]:
    """Union-find: gambar sebatch dan pasangan near-duplicate → satu grup."""
    parent = list(range(len(batch_ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    first: dict[str, int] = {}
    for i, b in enumerate(batch_ids):
        union(i, first.setdefault(b, i))
    for i, j, _ in pairs:
        union(i, j)
    return [find(i) for i in range(len(batch_ids))]


# --- split --------------------------------------------------------------------------------


def assign_splits(
    labels: list[str], groups: list[int], ratios: dict[str, float], seed: int
) -> list[str]:
    """Split per grup, terstratifikasi per label, deterministik.

    Grup tidak pernah terbelah. Tiap label: grup diacak (seed), lalu tiap split dengan rasio > 0
    diberi minimal satu grup (urutan train, test, val), sisanya ke split yang paling kurang dari
    target jumlah gambarnya. Grup campuran label ikut label mayoritasnya.
    """
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    by_label: dict[str, list[int]] = defaultdict(list)
    for g in sorted(members):
        counts = Counter(labels[i] for i in members[g])
        primary = min(counts, key=lambda lab: (-counts[lab], lab))
        by_label[primary].append(g)

    rng = random.Random(seed)
    active = [s for s in ("train", "test", "val") if ratios[s] > 0]
    out = [""] * len(labels)
    for label in sorted(by_label):
        gs = by_label[label]
        rng.shuffle(gs)
        total = sum(len(members[g]) for g in gs)
        count = dict.fromkeys(SPLITS, 0)
        for g in gs:
            empty = [s for s in active if count[s] == 0]
            s = empty[0] if empty else max(active, key=lambda s: ratios[s] * total - count[s])
            count[s] += len(members[g])
            for i in members[g]:
                out[i] = s
    return out


# --- manifest -----------------------------------------------------------------------------


def read_manifest(path: str | Path) -> list[Row]:
    """Baca manifest CSV → list baris."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[Row], path: str | Path, force: bool = False) -> None:
    """Tulis manifest CSV; tolak menimpa file yang ada kecuali force (test set dibekukan)."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"manifest sudah ada: {path} (pakai --force untuk split ulang)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_split(manifest: str | Path, split: str) -> list[tuple[str, str]]:
    """Daftar (path, label) untuk satu split dari manifest."""
    if split not in SPLITS:
        raise ValueError(f"split={split!r} tidak valid; pilihan: {list(SPLITS)}")
    return [(r["path"], r["label"]) for r in read_manifest(manifest) if r["split"] == split]


def build_manifest(cfg: Config, root: Path) -> list[Row]:
    """Scan → hash → grup (batch + near-duplicate) → split. Gambar tak terbaca dilewati."""
    rows, hashes = [], []
    for row in scan_dataset(cfg.data, root):
        try:
            hashes.append(dhash(load_image(row["path"], root)))
            rows.append(row)
        except ValueError as e:
            log.warning("Dilewati: %s", e)
    pairs = near_duplicate_pairs(hashes, cfg.data.near_dup_hamming)
    groups = group_ids([r["batch_id"] for r in rows], pairs)
    splits = assign_splits([r["label"] for r in rows], groups, cfg.data.split, cfg.seed)
    for row, s in zip(rows, splits, strict=True):
        row["split"] = s
    log.info(
        "%d gambar, %d grup, %d pasangan near-duplicate", len(rows), len(set(groups)), len(pairs)
    )
    return rows


def keep_metadata(rows: list[Row], old_rows: list[Row]) -> None:
    """Salin metadata (date, camera, ...) dari manifest lama untuk path yang sama."""
    old = {r["path"]: r for r in old_rows}
    for row in rows:
        if prev := old.get(row["path"]):
            row.update({k: prev.get(k, "") for k in META_COLUMNS})


def main(argv: list[str] | None = None) -> int:
    """CLI: bangun manifest + split beku."""
    parser = argparse.ArgumentParser(description="Bangun manifest + split train/val/test.")
    parser.add_argument("--config", required=True, help="path configs/*.yaml")
    parser.add_argument("--force", action="store_true", help="timpa manifest (split ulang!)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    root = Path.cwd()
    manifest = root / cfg.data.manifest
    if manifest.exists() and not args.force:
        log.warning(
            "Manifest sudah ada, TIDAK ditimpa (test set dibekukan): %s. "
            "Pakai --force hanya jika memang ingin split ulang.",
            cfg.data.manifest,
        )
        return 0
    rows = build_manifest(cfg, root)
    if manifest.exists():
        keep_metadata(rows, read_manifest(manifest))
        log.warning("--force: manifest lama ditimpa, split dibuat ulang: %s", cfg.data.manifest)
    write_manifest(rows, manifest, force=True)
    counts = Counter((r["split"], r["label"]) for r in rows)
    for s in SPLITS:
        log.info("%-5s normal=%d defect=%d", s, counts[(s, "normal")], counts[(s, "defect")])
    log.info("Manifest ditulis: %s", cfg.data.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
