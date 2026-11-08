"""Antarmuka scorer + scorer A (baseline): segmentasi → fitur → StandardScaler + classifier.

Skor 0–100, makin tinggi = makin kempes. Baseline: skor = 100 * P(defect).
Scorer A tahan resolusi: fitur dari crop kanonik kecil (``crop_size``) + augmentasi resolusi saat
fit (``model.params.augment``: salinan gambar latih diperkecil ke roti 50–200 px + JPEG).
Scorer B (anomaly) ada di ``loaflens.anomaly``; ``new_scorer``/``load_scorer`` memilih per config.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, Self

import cv2
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from loaflens.config import Config, SegmentConfig
from loaflens.features import FEATURE_NAMES, extract
from loaflens.segment import Segmentation, crop, full_frame, segment

log = logging.getLogger(__name__)

MODEL_FILES = {"baseline": "model.joblib", "anomaly": "model.pt"}


class Scorer(Protocol):
    """Antarmuka bersama semua scorer (CLAUDE.md).

    ``seg`` opsional: segmentasi siap pakai (mis. satu roti dari ``segment_all`` di frame konveyor);
    None = segmentasi roti tunggal terbesar di ``image``. ``represent``/``fit_repr``/``score_repr``
    dipakai evaluasi agar representasi (fitur/embedding) dihitung sekali untuk cross-validation.
    """

    seg_failures: int

    def fit(self, images: Sequence[np.ndarray], labels: Sequence[int] | None = None) -> Self: ...

    def score(self, image: np.ndarray, seg: Segmentation | None = None) -> float: ...

    def represent(self, image: np.ndarray, seg: Segmentation | None = None) -> Any: ...

    def fit_repr(self, reprs: Sequence[Any], y: np.ndarray) -> Self: ...

    def score_repr(self, reprs: Sequence[Any]) -> np.ndarray: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> Self: ...


def new_scorer(cfg: Config) -> Scorer:
    """Scorer baru (belum di-fit) sesuai ``model.type``."""
    if cfg.model.type == "anomaly":
        from loaflens.anomaly import AnomalyScorer  # impor torch hanya bila perlu

        return AnomalyScorer.from_config(cfg)
    return BaselineScorer.from_config(cfg)


def model_path(cfg: Config, root: Path) -> Path:
    """Path file model di ``output.dir`` sesuai ``model.type``."""
    return root / cfg.output_dir / MODEL_FILES[cfg.model.type]


def load_scorer(cfg: Config, root: Path) -> Scorer:
    """Muat scorer terlatih dari ``output.dir``."""
    if cfg.model.type == "anomaly":
        from loaflens.anomaly import AnomalyScorer

        return AnomalyScorer.load(model_path(cfg, root))
    return BaselineScorer.load(model_path(cfg, root))


def degrade(image: np.ndarray, bun_side: int, bun_px: int, quality: int) -> np.ndarray:
    """Salinan beresolusi rendah: perkecil (INTER_AREA) agar sisi terpanjang roti ≈ ``bun_px``
    (tidak diperbesar bila roti sudah lebih kecil), lalu encode/decode JPEG ``quality``."""
    s = bun_px / max(bun_side, 1)
    if s < 1:
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else image


def make_classifier(params: dict[str, Any], seed: int) -> Pipeline:
    """StandardScaler + classifier sesuai ``model.params.classifier``."""
    name = params["classifier"]
    hp = dict((params.get("classifier_params") or {}).get(name) or {})
    if name == "logistic_regression":
        clf = LogisticRegression(random_state=seed, **hp)
    elif name == "random_forest":
        clf = RandomForestClassifier(random_state=seed, **hp)
    elif name == "svm":
        # probabilitas via kalibrasi sigmoid (SVC(probability=True) deprecated di sklearn 1.9)
        clf = CalibratedClassifierCV(SVC(random_state=seed, **hp), ensemble=False, cv=3)
    else:
        raise ValueError(f"classifier tidak dikenal: {name!r}")
    return make_pipeline(StandardScaler(), clf)


class BaselineScorer:
    """Scorer A: fitur tekstur/warna/bentuk dari crop roti → classifier supervised."""

    def __init__(self, params: dict[str, Any], segment_cfg: SegmentConfig, seed: int) -> None:
        self.params = params
        self.segment_cfg = segment_cfg
        self.seed = seed
        self.pipeline: Pipeline | None = None
        self.feature_names = list(FEATURE_NAMES)
        self.seg_failures = 0  # jumlah gambar yang jatuh ke fallback full-frame

    @classmethod
    def from_config(cls, cfg: Config) -> Self:
        """Bangun scorer dari Config (model.params, segment, seed)."""
        return cls(cfg.model.params, cfg.segment, cfg.seed)

    def features(self, image: np.ndarray, seg: Segmentation | None = None) -> np.ndarray:
        """Segmentasi → crop ukuran tetap → vektor fitur. Gagal segmentasi = full frame."""
        if seg is None:
            seg = segment(image, self.segment_cfg)
        if seg is None:
            self.seg_failures += 1
            log.warning("Segmentasi gagal → fallback seluruh gambar (total %d)", self.seg_failures)
            seg = full_frame(image)
        c_img, c_mask = crop(image, seg, self.params["crop_size"], self.segment_cfg.crop_margin)
        return extract(c_img, c_mask, self.params.get("features") or {})

    def augmented_features(self, image: np.ndarray, seg: Segmentation | None = None) -> np.ndarray:
        """Fitur salinan augmentasi resolusi (k, d); k = 0 bila ``augment`` mati.

        Tiap salinan: roti diperkecil ke sisi ``bun_px`` (log-uniform) + JPEG ``jpeg_quality``,
        lalu disegmentasi ulang seperti frame kamera beresolusi rendah. RNG = seed config + CRC
        isi gambar → deterministik dan tidak bergantung urutan gambar.
        """
        aug = self.params.get("augment") or {}
        n = int(aug.get("copies", 0))
        d = len(self.feature_names)
        if seg is None:
            seg = segment(image, self.segment_cfg)
        if n <= 0 or seg is None:
            return np.empty((0, d))
        lo, hi = aug["bun_px"]
        q_lo, q_hi = aug["jpeg_quality"]
        rng = np.random.default_rng([self.seed, zlib.crc32(np.ascontiguousarray(image).data)])
        bun_side = max(seg.bbox[2:])
        rows = []
        for _ in range(n):
            bun_px = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
            small = degrade(image, bun_side, bun_px, int(rng.integers(q_lo, q_hi + 1)))
            small_seg = segment(small, self.segment_cfg)
            if small_seg is None:  # roti hilang saat diperkecil → salinan dilewati
                log.debug("Augmentasi: segmentasi gagal pada roti %d px, dilewati", bun_px)
                continue
            rows.append(self.features(small, small_seg))
        return np.stack(rows) if rows else np.empty((0, d))

    def fit_features(self, x: np.ndarray, y: Sequence[int]) -> Self:
        """Latih classifier dari matriks fitur (dipakai juga oleh cross-validation)."""
        self.pipeline = make_classifier(self.params, self.seed).fit(x, np.asarray(y))
        return self

    def fit(self, images: Sequence[np.ndarray], labels: Sequence[int] | None = None) -> Self:
        """Latih dari gambar BGR (+ salinan augmentasi); labels wajib (1 = kempes, 0 = normal)."""
        if labels is None:
            raise ValueError("BaselineScorer butuh labels (supervised: normal + kempes)")
        return self.fit_repr([self.represent(im) for im in images], np.asarray(labels))

    def score_features(self, x: np.ndarray) -> np.ndarray:
        """Skor 0–100 untuk matriks fitur (n, d)."""
        if self.pipeline is None:
            raise RuntimeError("scorer belum di-fit / di-load")
        return 100.0 * self.pipeline.predict_proba(np.atleast_2d(x))[:, 1]

    def score(self, image: np.ndarray, seg: Segmentation | None = None) -> float:
        """Skor kempes 0–100 untuk satu gambar BGR (opsional segmentasi siap pakai)."""
        return float(self.score_features(self.features(image, seg))[0])

    def represent(self, image: np.ndarray, seg: Segmentation | None = None) -> np.ndarray:
        """Representasi untuk CV = matriks (1 + k, d): baris 0 fitur gambar asli, sisanya fitur
        salinan augmentasi (hanya dipakai saat fit; skor selalu dari baris 0)."""
        if seg is None:
            seg = segment(image, self.segment_cfg)
        clean = self.features(image, seg)
        return np.vstack([clean, self.augmented_features(image, seg)])

    def fit_repr(self, reprs: Sequence[np.ndarray], y: np.ndarray) -> Self:
        """Fit dari representasi (asli + salinan augmentasi berlabel sama; normal + kempes)."""
        mats = [np.atleast_2d(r) for r in reprs]
        labels = np.concatenate([np.full(len(m), lab) for m, lab in zip(mats, y, strict=True)])
        return self.fit_features(np.vstack(mats), labels)

    def score_repr(self, reprs: Sequence[np.ndarray]) -> np.ndarray:
        """Skor 0–100 untuk daftar representasi (baris 0 = gambar asli)."""
        return self.score_features(np.stack([np.atleast_2d(r)[0] for r in reprs]))

    def save(self, path: str | Path) -> None:
        """Simpan pipeline + nama fitur + snapshot config ke joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "type": "baseline",
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "params": self.params,
                "segment": asdict(self.segment_cfg),
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Muat scorer dari file joblib hasil ``save``."""
        blob = joblib.load(path)
        if blob.get("type") != "baseline":
            raise ValueError(f"{path}: bukan model baseline")
        obj = cls(blob["params"], SegmentConfig(**blob["segment"]), blob["seed"])
        obj.pipeline = blob["pipeline"]
        if blob["feature_names"] != obj.feature_names:
            raise ValueError(f"{path}: daftar fitur berbeda dari kode saat ini — latih ulang")
        return obj
