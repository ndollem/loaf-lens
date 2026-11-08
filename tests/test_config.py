"""Test loader & validasi config."""

import copy
from pathlib import Path

import pytest
import yaml

from loaflens.config import ConfigError, from_dict, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load((CONFIGS / "baseline.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["baseline.yaml", "anomaly.yaml"])
def test_repo_configs_load(name: str) -> None:
    cfg = load_config(CONFIGS / name)
    assert 0 <= cfg.threshold <= 100
    assert cfg.model.type == name.removesuffix(".yaml")


def test_baseline_uses_normal_and_defect() -> None:
    cfg = load_config(CONFIGS / "baseline.yaml")
    assert set(cfg.model.train_labels) == {"normal", "defect"}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("threshold",), 101),
        (("threshold",), -1),
        (("track", "direction"), "diagonal"),
        (("track", "line_pos"), 1.0),
        (("camera", "view"), "side"),
        (("camera", "roi"), [0.5, 0.0, 0.6, 1.0]),
        (("data", "split"), {"train": 0.8, "val": 0.15, "test": 0.15}),
        (("data", "split"), {"train": 0.85, "val": 0.15}),
        (("data", "split_by"), "image"),
        (("data", "extensions"), ["png"]),
        (("data", "extensions"), []),
        (("data", "near_dup_hamming"), 65),
        (("audit", "border_frac"), 0.5),
        (("audit", "shortcut_acc"), 0.5),
        (("audit", "min_eval_per_class"), 0),
        (("model", "type"), "llm"),
        (("model", "train_labels"), []),
        (("model", "params", "classifier"), "xgboost"),
        (("model", "params", "crop_size"), 10),
        (("segment", "blur"), 4),
        (("segment", "sat_threshold"), 300),
        (("eval", "cv_folds"), 1),
        (("eval", "fpr_target"), 0.0),
    ],
)
def test_invalid_values_raise(raw: dict, path: tuple[str, ...], value: object) -> None:
    bad = copy.deepcopy(raw)
    node = bad
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ConfigError):
        from_dict(bad)


def test_unknown_segment_key_raises(raw: dict) -> None:
    raw["segment"]["thresh"] = 10
    with pytest.raises(ConfigError, match="segment"):
        from_dict(raw)


def test_anomaly_must_train_on_normal_only(raw: dict) -> None:
    raw["model"]["type"] = "anomaly"
    with pytest.raises(ConfigError, match="normal"):
        from_dict(raw)


def test_unknown_and_missing_keys_raise(raw: dict) -> None:
    with pytest.raises(ConfigError, match="tidak dikenal"):
        from_dict({**raw, "treshold": 50})
    with pytest.raises(ConfigError, match="track"):
        from_dict({**raw, "track": {"direction": "left_to_right", "linepos": 0.5}})
    del raw["threshold"]
    with pytest.raises(ConfigError, match="wajib"):
        from_dict(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("weights", "imagenet21k"),
        ("image_size", 16),
        ("image_score", "mean"),
        ("knn", 0),
        ("ref_percentile", 0),
        ("bank_ratio", 1.5),
        ("pool", 2),
    ],
)
def test_invalid_anomaly_params_raise(key: str, value: object) -> None:
    raw = yaml.safe_load((CONFIGS / "anomaly.yaml").read_text(encoding="utf-8"))
    raw["model"]["params"][key] = value
    with pytest.raises(ConfigError, match="model.params"):
        from_dict(raw)
