"""Test scan, grup near-duplicate, split, dan manifest (gambar sintetis kecil di tmp_path)."""

import copy
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from loaflens.config import from_dict
from loaflens.data import (
    assign_splits,
    build_manifest,
    dhash,
    group_ids,
    load_split,
    main,
    near_duplicate_pairs,
    read_manifest,
    scan_dataset,
    write_manifest,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def make_image(path: Path, seed: int, size: tuple[int, int] = (40, 32)) -> None:
    """Tulis gambar acak (pola berbeda per seed) ke path."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8), size, cv2.INTER_NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def make_cfg(root: Path, **data: object) -> object:
    raw = copy.deepcopy(yaml.safe_load((CONFIGS / "baseline.yaml").read_text(encoding="utf-8")))
    raw["data"].update(
        normal_dir=str(root / "good"), defect_dir=str(root / "bad"), manifest="m.csv", **data
    )
    raw["data"]["split"] = RATIOS
    return from_dict(raw)


def test_batch_id_from_subfolder_else_per_image(tmp_path: Path) -> None:
    make_image(tmp_path / "good" / "s1" / "a (1).png", 1)
    make_image(tmp_path / "good" / "s1" / "deep" / "b.jpg", 2)
    make_image(tmp_path / "good" / "loose (1).png", 3)
    make_image(tmp_path / "bad" / "loose (1).png", 4)  # nama sama di label lain
    (tmp_path / "good" / "notes.txt").write_text("x")
    rows = {r["path"]: r for r in scan_dataset(make_cfg(tmp_path).data, tmp_path)}
    assert set(rows) == {
        "good/s1/a (1).png",
        "good/s1/deep/b.jpg",
        "good/loose (1).png",
        "bad/loose (1).png",
    }
    assert rows["good/s1/a (1).png"]["batch_id"] == "s1"
    assert rows["good/s1/deep/b.jpg"]["batch_id"] == "s1"
    assert rows["good/loose (1).png"]["batch_id"] != rows["bad/loose (1).png"]["batch_id"]
    assert rows["bad/loose (1).png"]["label"] == "defect"


def test_near_duplicates_grouped(tmp_path: Path) -> None:
    make_image(tmp_path / "a.png", 1)
    make_image(tmp_path / "b.png", 2)
    a = cv2.imread(str(tmp_path / "a.png"))
    hashes = [dhash(a), dhash(cv2.resize(a, (80, 64))), dhash(cv2.imread(str(tmp_path / "b.png")))]
    pairs = near_duplicate_pairs(hashes, max_dist=5)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    groups = group_ids(["x", "y", "z"], pairs)
    assert groups[0] == groups[1] != groups[2]
    assert group_ids(["x", "y", "x"], [])[0] == group_ids(["x", "y", "x"], [])[2]


def _random_groups(n: int = 40) -> tuple[list[str], list[int]]:
    rng = np.random.default_rng(0)
    labels = ["normal" if i % 2 else "defect" for i in range(n)]
    groups = [int(g) * 2 + (i % 2) for i, g in enumerate(rng.integers(0, 8, n))]  # label-murni
    return labels, groups


def test_groups_never_cross_splits_and_stratified() -> None:
    labels, groups = _random_groups()
    splits = assign_splits(labels, groups, RATIOS, seed=1)
    by_group = defaultdict(set)
    for g, s in zip(groups, splits, strict=True):
        by_group[g].add(s)
    assert all(len(s) == 1 for s in by_group.values())
    for label in ("normal", "defect"):
        present = {s for s, lab in zip(splits, labels, strict=True) if lab == label}
        assert present == {"train", "val", "test"}


def test_split_deterministic() -> None:
    labels, groups = _random_groups()
    assert assign_splits(labels, groups, RATIOS, 7) == assign_splits(labels, groups, RATIOS, 7)
    per_image = list(range(len(labels)))
    seeds = {tuple(assign_splits(labels, per_image, RATIOS, s)) for s in range(5)}
    assert len(seeds) > 1  # seed memang berpengaruh


def test_few_groups_go_to_train_first() -> None:
    assert assign_splits(["normal"], [0], RATIOS, 0) == ["train"]
    assert sorted(assign_splits(["normal"] * 2, [0, 1], RATIOS, 0)) == ["test", "train"]


def test_manifest_roundtrip_and_no_overwrite(tmp_path: Path) -> None:
    for i in range(5):
        make_image(tmp_path / "good" / f"g ({i}).png", i)
        make_image(tmp_path / "bad" / f"b ({i}).png", 100 + i)
    make_image(tmp_path / "good" / "copy.png", 0)  # duplikat `g (0).png`
    (tmp_path / "bad" / "broken.png").write_bytes(b"not an image")
    cfg = make_cfg(tmp_path)
    rows = build_manifest(cfg, tmp_path)
    assert len(rows) == 11  # file rusak dilewati
    split = {r["path"].rsplit("/", 1)[-1]: r["split"] for r in rows}
    assert split["copy.png"] == split["g (0).png"]  # near-duplicate → split sama
    manifest = tmp_path / "m.csv"
    write_manifest(rows, manifest)
    assert read_manifest(manifest)[0].keys() >= {"path", "label", "batch_id", "split", "camera"}
    test = load_split(manifest, "test")
    assert test and all(lab in {"normal", "defect"} for _, lab in test)
    with pytest.raises(FileExistsError):
        write_manifest([], manifest)
    write_manifest(rows[:1], manifest, force=True)
    assert len(read_manifest(manifest)) == 1


def test_cli_does_not_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        make_image(tmp_path / "good" / f"g{i}.png", i)
        make_image(tmp_path / "bad" / f"b{i}.png", 10 + i)
    raw = yaml.safe_load((CONFIGS / "baseline.yaml").read_text(encoding="utf-8"))
    raw["data"].update(normal_dir="good", defect_dir="bad", manifest="m.csv")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "m.csv"
    manifest.write_text("frozen\n")
    assert main(["--config", "c.yaml"]) == 0
    assert manifest.read_text() == "frozen\n"
    assert main(["--config", "c.yaml", "--force"]) == 0
    assert len(read_manifest(manifest)) == 6
