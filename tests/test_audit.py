"""Test audit: deteksi shortcut ukuran, kebocoran batch, prefix nama."""

from pathlib import Path

import yaml

from loaflens.audit import batch_leaks, name_number, name_prefix, run_audit, stump_accuracy
from loaflens.config import from_dict
from test_data import CONFIGS, make_image


def test_name_helpers() -> None:
    assert name_prefix("x/bagus (3).png") == "bagus"
    assert name_prefix("medium.png") == "medium"
    assert name_number("x/bagus (3).png") == 3
    assert name_number("medium.png") is None


def test_stump_accuracy() -> None:
    assert stump_accuracy([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert stump_accuracy([4, 3, 2, 1], [False, False, True, True]) == 1.0  # arah terbalik
    assert stump_accuracy([1, 1, 1, 1], [False, True, False, True]) == 0.5


def test_batch_leaks() -> None:
    rows = [
        {"batch_id": "a", "split": "train"},
        {"batch_id": "a", "split": "test"},
        {"batch_id": "b", "split": "val"},
    ]
    assert batch_leaks(rows) == {"a": {"train", "test"}}


def test_audit_warns_on_non_overlapping_sizes(tmp_path: Path) -> None:
    for i in range(4):
        make_image(tmp_path / "good" / f"bagus ({i}).png", i, size=(30, 40))
        make_image(tmp_path / "bad" / f"bad ({i}).png", 50 + i, size=(36, 35))
    raw = yaml.safe_load((CONFIGS / "baseline.yaml").read_text(encoding="utf-8"))
    raw["data"].update(normal_dir="good", defect_dir="bad", manifest="m.csv")
    report, warnings = run_audit(from_dict(raw), tmp_path)
    text = " ".join(warnings)
    assert "Ukuran gambar TIDAK overlap" in text
    assert "tanpa subfolder batch" in text
    assert "Manifest belum ada" in text
    assert "## Ukuran & rasio aspek per kelas" in report
    assert report.count("**WARNING:**") == len(warnings)
