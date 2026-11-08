"""Muat config YAML → dataclass sederhana dengan validasi dasar."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

CAMERA_VIEWS = {"top", "oblique"}  # oblique = top-down miring ~45°
DIRECTIONS = {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"}
MODEL_TYPES = {"baseline", "anomaly"}
LABELS = {"normal", "defect"}
SPLIT_KEYS = {"train", "val", "test"}
SPLIT_BY = {"batch_id"}


class ConfigError(ValueError):
    """Config tidak valid."""


@dataclass
class DataConfig:
    normal_dir: Path
    defect_dir: Path
    manifest: Path
    split: dict[str, float]
    split_by: str = "batch_id"
    extensions: list[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    near_dup_hamming: int = 5  # dHash 64-bit; jarak <= ini dianggap near-duplicate


@dataclass
class AuditConfig:
    report: Path = Path("outputs/audit.md")
    docs_copy: Path | None = None  # salinan laporan yang di-versi (mis. docs/audit_m1.md)
    border_frac: float = 0.05  # lebar tepi untuk statistik warna latar
    center_frac: float = 0.5  # kotak tengah untuk statistik warna roti
    shortcut_acc: float = 0.9  # ambang WARNING akurasi 1-threshold fitur trivial
    pair_warn_ratio: float = 0.6  # ambang WARNING kemiripan pasangan nama bernomor sama
    min_eval_per_class: int = 30  # ambang WARNING ukuran val/test per label


@dataclass
class CameraConfig:
    view: str = "top"
    roi: list[float] | None = None  # [x, y, w, h] fraksi frame 0–1; None = seluruh frame


@dataclass
class TrackConfig:
    direction: str = "left_to_right"
    line_pos: float = 0.5  # posisi garis keputusan, fraksi frame searah gerak


@dataclass
class ModelConfig:
    type: str
    train_labels: list[str]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    seed: int
    data: DataConfig
    camera: CameraConfig
    track: TrackConfig
    model: ModelConfig
    threshold: float  # skor 0–100; reject jika skor >= threshold
    output_dir: Path
    segment: dict[str, Any] = field(default_factory=dict)  # placeholder, dipakai di M4
    audit: AuditConfig = field(default_factory=AuditConfig)


def _section(cls: type, raw: Any, name: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"section '{name}' harus berupa mapping")
    try:
        return cls(**raw)
    except TypeError as e:
        raise ConfigError(f"section '{name}': {e}") from e


def _check_choice(value: Any, allowed: set[str], name: str) -> None:
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} tidak valid; pilihan: {sorted(allowed)}")


def validate(cfg: Config) -> None:
    """Validasi nilai config; raise ConfigError dengan pesan jelas."""
    if not 0 <= cfg.threshold <= 100:
        raise ConfigError(f"threshold={cfg.threshold} harus di rentang [0, 100]")

    split = cfg.data.split
    if set(split) != SPLIT_KEYS:
        raise ConfigError(f"data.split harus berkunci {sorted(SPLIT_KEYS)}, dapat {sorted(split)}")
    if any(not 0 <= v <= 1 for v in split.values()):
        raise ConfigError(f"data.split: tiap rasio harus di [0, 1], dapat {split}")
    if not math.isclose(sum(split.values()), 1.0, abs_tol=1e-6):
        raise ConfigError(f"data.split harus berjumlah 1.0, dapat {sum(split.values()):.4f}")
    _check_choice(cfg.data.split_by, SPLIT_BY, "data.split_by")
    exts = cfg.data.extensions
    if not exts or any(not str(e).startswith(".") for e in exts):
        raise ConfigError(f"data.extensions harus list tak kosong berawalan '.', dapat {exts}")
    if not 0 <= cfg.data.near_dup_hamming <= 64:
        raise ConfigError(f"data.near_dup_hamming={cfg.data.near_dup_hamming} harus di [0, 64]")

    a = cfg.audit
    if not 0 < a.border_frac < 0.5 or not 0 < a.center_frac <= 1:
        raise ConfigError("audit.border_frac harus di (0, 0.5) dan audit.center_frac di (0, 1]")
    if not 0.5 < a.shortcut_acc <= 1 or not 0 < a.pair_warn_ratio <= 1:
        raise ConfigError(
            "audit.shortcut_acc harus di (0.5, 1] dan audit.pair_warn_ratio di (0, 1]"
        )
    if a.min_eval_per_class < 1:
        raise ConfigError("audit.min_eval_per_class harus >= 1")

    _check_choice(cfg.camera.view, CAMERA_VIEWS, "camera.view")
    roi = cfg.camera.roi
    if roi is not None:
        if len(roi) != 4 or any(not 0 <= v <= 1 for v in roi):
            raise ConfigError(f"camera.roi harus [x, y, w, h] dengan nilai 0–1, dapat {roi}")
        x, y, w, h = roi
        if w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
            raise ConfigError(f"camera.roi keluar dari frame: {roi}")

    _check_choice(cfg.track.direction, DIRECTIONS, "track.direction")
    if not 0 < cfg.track.line_pos < 1:
        raise ConfigError(f"track.line_pos={cfg.track.line_pos} harus di (0, 1)")

    _check_choice(cfg.model.type, MODEL_TYPES, "model.type")
    labels = set(cfg.model.train_labels)
    if not labels or not labels <= LABELS:
        raise ConfigError(f"model.train_labels harus subset tak kosong dari {sorted(LABELS)}")
    if cfg.model.type == "anomaly" and labels != {"normal"}:
        raise ConfigError("model anomaly hanya boleh dilatih dari 'normal'")


def from_dict(raw: dict[str, Any]) -> Config:
    """Bangun Config dari dict hasil YAML lalu validasi."""
    if not isinstance(raw, dict):
        raise ConfigError("isi config harus berupa mapping")
    known = {"seed", "data", "camera", "segment", "track", "model", "threshold", "output", "audit"}
    if unknown := set(raw) - known:
        raise ConfigError(f"kunci tidak dikenal: {sorted(unknown)}")
    try:
        data = _section(DataConfig, raw["data"], "data")
        data.normal_dir, data.defect_dir = Path(data.normal_dir), Path(data.defect_dir)
        data.manifest = Path(data.manifest)
        data.extensions = [str(e).lower() for e in data.extensions]
        audit = _section(AuditConfig, raw.get("audit") or {}, "audit")
        audit.report = Path(audit.report)
        audit.docs_copy = Path(audit.docs_copy) if audit.docs_copy else None
        cfg = Config(
            seed=int(raw["seed"]),
            data=data,
            camera=_section(CameraConfig, raw.get("camera", {}), "camera"),
            track=_section(TrackConfig, raw.get("track", {}), "track"),
            model=_section(ModelConfig, raw["model"], "model"),
            threshold=float(raw["threshold"]),
            output_dir=Path(raw["output"]["dir"]),
            segment=raw.get("segment") or {},
            audit=audit,
        )
    except KeyError as e:
        raise ConfigError(f"kunci wajib hilang: {e}") from e
    validate(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    """Baca file YAML dan kembalikan Config tervalidasi."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = from_dict(raw)
    log.info("Config dimuat: %s (model=%s)", path, cfg.model.type)
    return cfg
