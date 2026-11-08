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
CLASSIFIERS = {"logistic_regression", "random_forest", "svm"}
IMAGE_SCORES = {"max", "topk_mean"}
BACKBONE_WEIGHTS = {"imagenet", None}


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
    line_pos: float = 0.5  # posisi garis keputusan, fraksi frame searah gerak (dari tepi masuk)
    max_distance: float = 0.08  # jarak maks prediksi→deteksi, fraksi sisi terpanjang frame
    max_missed: int = 5  # track dihapus setelah sekian frame berturut-turut tanpa deteksi


@dataclass
class AppConfig:
    """Demo real-time (``python -m loaflens.app``)."""

    window_name: str = "LoafLens"
    out_fps: float | None = None  # fps video --out; None = fps sumber
    snapshot_dir: Path = Path("outputs/rejects")
    rejects_csv: Path = Path("outputs/rejects/rejects.csv")
    decisions_csv: Path = Path("outputs/app/decisions.csv")
    snapshot_margin: float = 0.15  # margin crop snapshot, fraksi sisi bbox
    synthetic_banner: str = "SYNTHETIC - bukan untuk evaluasi akurasi"
    gt_match_dist: float = 0.05  # --gt: jarak maks keputusan→roti GT, fraksi sisi terpanjang
    gt_report: Path = Path("outputs/app/gt_compare.md")


@dataclass
class SegmentConfig:
    """Isolasi roti dari latar polos (kanal saturasi HSV)."""

    blur: int = 5  # kernel Gaussian (ganjil) sebelum threshold
    sat_threshold: int | None = None  # None = Otsu per gambar; angka 0–255 = tetap
    morph_kernel: int = 7  # kernel elips open/close (ganjil)
    min_area_frac: float = 0.02  # kontur terbesar < fraksi ini dari frame = gagal
    crop_margin: float = 0.05  # margin crop di sekitar bbox, fraksi sisi bbox
    # banyak roti per frame (segment_all; konveyor): luas objek sebagai fraksi luas ROI/frame
    obj_min_area_frac: float = 0.003
    obj_max_area_frac: float = 0.15
    drop_border: bool = True  # buang objek yang menyentuh tepi ROI/frame (roti terpotong)
    obj_min_value: int = 0  # piksel V < ini bukan roti (bayangan gelap); 0 = mati


@dataclass
class EvalConfig:
    fpr_target: float = 0.05  # recall dilaporkan pada FPR <= target
    cv_folds: int = 5  # StratifiedGroupKFold pada train+val (test tidak ikut)
    top_errors: int = 5  # jumlah kesalahan terbesar yang dilaporkan per split
    docs_copy: str | None = None  # salinan laporan ber-versi (mis. docs/results_m2.md)
    domain_check: list[str] = field(default_factory=list)  # foto konveyor (semua roti OK)
    heatmaps: bool = True  # anomaly: simpan overlay heatmap val/test ke output.dir/heatmaps/


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
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    app: AppConfig = field(default_factory=AppConfig)


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
    if not 0 < cfg.track.max_distance <= 1 or cfg.track.max_missed < 0:
        raise ConfigError("track.max_distance harus di (0, 1] dan track.max_missed >= 0")

    ap = cfg.app
    if ap.out_fps is not None and ap.out_fps <= 0:
        raise ConfigError("app.out_fps harus null atau > 0")
    if not 0 <= ap.snapshot_margin < 1 or not 0 < ap.gt_match_dist <= 1:
        raise ConfigError("app.snapshot_margin harus di [0, 1) dan app.gt_match_dist di (0, 1]")

    _check_choice(cfg.model.type, MODEL_TYPES, "model.type")
    labels = set(cfg.model.train_labels)
    if not labels or not labels <= LABELS:
        raise ConfigError(f"model.train_labels harus subset tak kosong dari {sorted(LABELS)}")
    if cfg.model.type == "anomaly" and labels != {"normal"}:
        raise ConfigError("model anomaly hanya boleh dilatih dari 'normal'")
    if cfg.model.type == "baseline":
        p = cfg.model.params
        _check_choice(p.get("classifier"), CLASSIFIERS, "model.params.classifier")
        if not isinstance(p.get("crop_size"), int) or p["crop_size"] < 32:
            raise ConfigError("model.params.crop_size harus int >= 32")
    if cfg.model.type == "anomaly":
        _validate_anomaly(cfg.model.params)

    sg = cfg.segment
    if sg.blur % 2 == 0 or sg.morph_kernel % 2 == 0 or sg.blur < 1 or sg.morph_kernel < 1:
        raise ConfigError("segment.blur dan segment.morph_kernel harus ganjil >= 1")
    if sg.sat_threshold is not None and not 0 <= sg.sat_threshold <= 255:
        raise ConfigError("segment.sat_threshold harus null atau 0–255")
    if not 0 < sg.min_area_frac < 1 or not 0 <= sg.crop_margin < 1:
        raise ConfigError("segment.min_area_frac harus di (0, 1) dan crop_margin di [0, 1)")
    if not 0 < sg.obj_min_area_frac < sg.obj_max_area_frac <= 1:
        raise ConfigError("segment: harus 0 < obj_min_area_frac < obj_max_area_frac <= 1")
    if not 0 <= sg.obj_min_value <= 255:
        raise ConfigError("segment.obj_min_value harus 0–255")

    if not 0 < cfg.eval.fpr_target < 1 or cfg.eval.cv_folds < 2 or cfg.eval.top_errors < 0:
        raise ConfigError("eval.fpr_target harus di (0, 1), eval.cv_folds >= 2, top_errors >= 0")


def _validate_anomaly(p: dict[str, Any]) -> None:
    """Validasi model.params untuk scorer B (PatchCore-lite)."""
    if p.get("weights") not in BACKBONE_WEIGHTS:
        raise ConfigError("model.params.weights harus 'imagenet' atau null")
    if not isinstance(p.get("image_size"), int) or p["image_size"] < 32:
        raise ConfigError("model.params.image_size harus int >= 32")
    _check_choice(p.get("image_score", "max"), IMAGE_SCORES, "model.params.image_score")
    if not 0 < float(p.get("bank_ratio", 1.0)) <= 1 or int(p.get("bank_max", 1)) < 1:
        raise ConfigError("model.params.bank_ratio harus di (0, 1] dan bank_max >= 1")
    if int(p.get("knn", 1)) < 1 or int(p.get("pool", 3)) % 2 == 0:
        raise ConfigError("model.params.knn harus >= 1 dan pool ganjil")
    if not 0 < float(p.get("ref_percentile", -1)) <= 100:
        raise ConfigError("model.params.ref_percentile harus di (0, 100]")
    if not 0 < float(p.get("topk_frac", 0.01)) <= 1:
        raise ConfigError("model.params.topk_frac harus di (0, 1]")


def _app_section(raw: Any) -> AppConfig:
    app = _section(AppConfig, raw, "app")
    for name in ("snapshot_dir", "rejects_csv", "decisions_csv", "gt_report"):
        setattr(app, name, Path(getattr(app, name)))
    return app


def from_dict(raw: dict[str, Any]) -> Config:
    """Bangun Config dari dict hasil YAML lalu validasi."""
    if not isinstance(raw, dict):
        raise ConfigError("isi config harus berupa mapping")
    known = {
        "seed",
        "data",
        "camera",
        "segment",
        "track",
        "model",
        "threshold",
        "output",
        "audit",
        "eval",
        "app",
    }
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
            segment=_section(SegmentConfig, raw.get("segment") or {}, "segment"),
            audit=audit,
            eval=_section(EvalConfig, raw.get("eval") or {}, "eval"),
            app=_app_section(raw.get("app") or {}),
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
