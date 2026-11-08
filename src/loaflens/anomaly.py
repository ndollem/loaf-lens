"""Scorer B: "PatchCore-lite" — anomaly detection dari roti normal saja.

Crop roti (``segment.crop``, sama dengan baseline) → ResNet18 beku (layer2 + layer3, pretrained
ImageNet sebagai feature extractor) → embedding per patch di dalam mask roti → memory bank patch
normal. Skor mentah = jarak kNN patch terjauh dari bank; skor 0–100 = clip(50 · raw / ref) dengan
ref = persentil skor leave-one-image-out roti normal train (50 ≈ seaneh roti normal train paling
aneh). Tidak ada panggilan jaringan saat inferensi: bobot backbone harus sudah ada di disk.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from loaflens.config import Config, SegmentConfig
from loaflens.segment import Segmentation, crop, full_frame, segment

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
WEIGHTS_URL = ResNet18_Weights.IMAGENET1K_V1.url


def weights_path(params: dict[str, Any]) -> Path:
    """Lokasi file bobot ResNet18: ``weights_dir`` dari config, atau cache torch hub default."""
    base = params.get("weights_dir") or Path(torch.hub.get_dir()) / "checkpoints"
    return Path(base) / Path(WEIGHTS_URL).name


class Backbone(torch.nn.Module):
    """ResNet18 terpotong: keluarkan feature map layer2 dan layer3 (beku, mode eval)."""

    def __init__(self, params: dict[str, Any], seed: int, allow_download: bool) -> None:
        super().__init__()
        torch.manual_seed(seed)  # bobot acak (weights=null, test) harus deterministik
        net = resnet18(weights=None)
        if params.get("weights") == "imagenet":
            path = weights_path(params)
            if not path.exists():
                if not allow_download:
                    raise FileNotFoundError(
                        f"Bobot backbone tidak ditemukan: {path}. Inferensi tidak mengunduh apa "
                        "pun — jalankan training sekali (mengunduh bobot ImageNet ResNet18) atau "
                        "salin file itu ke model.params.weights_dir."
                    )
                log.info("Mengunduh bobot ResNet18 ImageNet (sekali) → %s", path)
                path.parent.mkdir(parents=True, exist_ok=True)
                prefix = path.stem.split("-")[-1]  # resnet18-<sha256 prefix>.pth
                torch.hub.download_url_to_file(
                    WEIGHTS_URL, str(path), hash_prefix=prefix, progress=False
                )
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        elif params.get("weights") is not None:
            raise ValueError(f"model.params.weights={params.get('weights')!r}: imagenet | null")
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f2 = self.layer2(self.stem(x))
        return f2, self.layer3(f2)


@dataclass
class Embedding:
    """Embedding patch di dalam mask roti untuk satu gambar."""

    feats: torch.Tensor  # (n_patch, D) float32
    idx: np.ndarray  # (n_patch,) indeks datar di grid
    grid: tuple[int, int]  # (h, w) grid patch (resolusi layer2)


class AnomalyScorer:
    """Scorer B (PatchCore-lite): memory bank patch roti normal + jarak kNN."""

    def __init__(self, params: dict[str, Any], segment_cfg: SegmentConfig, seed: int) -> None:
        self.params = params
        self.segment_cfg = segment_cfg
        self.seed = seed
        self.bank: torch.Tensor | None = None  # (N, D)
        self.ref: float | None = None  # skor mentah yang dipetakan ke 50
        self.loo_raw: list[float] = []  # skor mentah leave-one-out roti normal train
        self.seg_failures = 0
        self._backbone: Backbone | None = None
        self._allow_download = False  # hanya fit() (training) boleh mengunduh bobot

    @classmethod
    def from_config(cls, cfg: Config) -> Self:
        """Bangun scorer dari Config (model.params, segment, seed)."""
        return cls(cfg.model.params, cfg.segment, cfg.seed)

    # --- representasi ---------------------------------------------------------------------

    @property
    def backbone(self) -> Backbone:
        if self._backbone is None:
            torch.set_num_threads(int(self.params.get("num_threads", 4)))
            self._backbone = Backbone(self.params, self.seed, self._allow_download)
        return self._backbone

    def prepare(
        self, image: np.ndarray, seg: Segmentation | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segmentasi (bila ``seg`` None) → crop persegi ``image_size`` + mask; latar diisi abu.

        Latar diisi warna rata-rata ImageNet agar latar foto (beda per folder/konveyor) tidak
        ikut masuk receptive field patch tepi roti.
        """
        if seg is None:
            seg = segment(image, self.segment_cfg)
        if seg is None:
            self.seg_failures += 1
            log.warning("Segmentasi gagal → fallback seluruh gambar (total %d)", self.seg_failures)
            seg = full_frame(image)
        img, mask = crop(image, seg, int(self.params["image_size"]), self.segment_cfg.crop_margin)
        if self.params.get("fill_background", True):
            fill = (IMAGENET_MEAN[::-1] * 255).astype(np.uint8)  # BGR
            img = np.where(mask[..., None] > 0, img, fill)
        return img, mask

    def embed_crop(self, img: np.ndarray, mask: np.ndarray) -> Embedding:
        """Crop BGR + mask → embedding patch (layer2 ⊕ upsample(layer3), avg-pool 3x3)."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy((rgb - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1)[None]
        k = int(self.params.get("pool", 3))
        with torch.inference_mode():
            f2, f3 = self.backbone(x)
            f2 = F.avg_pool2d(f2, k, stride=1, padding=k // 2)
            f3 = F.avg_pool2d(f3, k, stride=1, padding=k // 2)
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            feats = torch.cat([f2, f3], 1)[0].flatten(1).T.contiguous()  # (h*w, D)
        gh, gw = f2.shape[-2:]
        idx = self.patch_indices(mask, (gh, gw))
        return Embedding(feats[torch.from_numpy(idx)], idx, (gh, gw))

    def patch_indices(self, mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
        """Indeks patch grid yang ≥ 50% berada di dalam mask roti yang sudah dierosi."""
        e = int(self.params.get("mask_erode_px", 0))
        if e > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
            mask = cv2.erode(mask, kernel)
        cover = cv2.resize((mask > 0).astype(np.float32), grid[::-1], interpolation=cv2.INTER_AREA)
        idx = np.flatnonzero(cover.ravel() >= 0.5)
        if idx.size == 0:  # roti sangat kecil/gagal: pakai semua patch daripada kosong
            idx = np.arange(grid[0] * grid[1])
        return idx

    def embed(self, image: np.ndarray, seg: Segmentation | None = None) -> Embedding:
        """Gambar BGR penuh → embedding patch roti."""
        return self.embed_crop(*self.prepare(image, seg))

    # --- fit ------------------------------------------------------------------------------

    def _knn(self, feats: torch.Tensor, exclude: slice | None = None) -> torch.Tensor:
        """Jarak rata-rata ke k tetangga terdekat di bank (opsional: kecualikan rentang bank)."""
        assert self.bank is not None
        d = torch.cdist(feats, self.bank)
        if exclude is not None:
            d[:, exclude] = float("inf")
        k = min(int(self.params.get("knn", 1)), d.shape[1])
        return d.topk(k, dim=1, largest=False).values.mean(dim=1)

    def _image_raw(self, patch_d: torch.Tensor) -> float:
        """Agregasi jarak patch → skor mentah gambar: max atau rata-rata top-k%."""
        if self.params.get("image_score", "max") == "max":
            return float(patch_d.max())
        n = max(1, round(float(self.params.get("topk_frac", 0.01)) * patch_d.numel()))
        return float(patch_d.topk(n).values.mean())

    def fit_embeddings(self, embs: Sequence[Embedding]) -> Self:
        """Bangun memory bank (subsample acak) + kalibrasi ref dari leave-one-image-out."""
        if not embs:
            raise ValueError("fit butuh minimal satu gambar normal")
        rng = np.random.default_rng(self.seed)
        ratio = float(self.params.get("bank_ratio", 1.0))
        cap = int(self.params.get("bank_max", 20000))
        n_total = sum(len(e.idx) for e in embs)
        keep_frac = min(ratio, cap / n_total)
        parts, spans, start = [], [], 0
        for e in embs:
            n = len(e.idx)
            k = max(1, round(keep_frac * n))
            sel = np.sort(rng.choice(n, size=k, replace=False))
            parts.append(e.feats[torch.from_numpy(sel)])
            spans.append(slice(start, start + k))
            start += k
        self.bank = torch.cat(parts)
        if len(embs) > 1:  # LOO: patch gambar itu sendiri dikeluarkan dari bank
            self.loo_raw = [
                self._image_raw(self._knn(e.feats, sp)) for e, sp in zip(embs, spans, strict=True)
            ]
            self.ref = float(np.percentile(self.loo_raw, float(self.params["ref_percentile"])))
        else:
            self.loo_raw, self.ref = [], 1.0
            log.warning("Hanya 1 gambar normal: ref tidak bisa dikalibrasi (ref=1)")
        log.info(
            "Memory bank: %d patch (dari %d, %d gambar); ref (p%g LOO) = %.3f",
            len(self.bank),
            n_total,
            len(embs),
            float(self.params["ref_percentile"]),
            self.ref,
        )
        return self

    def fit(self, images: Sequence[np.ndarray], labels: Sequence[int] | None = None) -> Self:
        """Fit dari gambar normal saja; ValueError bila ada label defect (1)."""
        if labels is not None and any(int(v) != 0 for v in labels):
            raise ValueError("AnomalyScorer hanya di-fit dari roti normal (label 0)")
        self._allow_download = True
        return self.fit_embeddings([self.embed(im) for im in images])

    # --- skor -----------------------------------------------------------------------------

    def to_score(self, raw: float | np.ndarray) -> Any:
        """Skor mentah → 0–100: clip(50 · raw / ref)."""
        if self.ref is None:
            raise RuntimeError("scorer belum di-fit / di-load")
        return np.clip(50.0 * np.asarray(raw) / self.ref, 0.0, 100.0)

    def score_embedding(self, emb: Embedding) -> float:
        """Skor 0–100 untuk satu embedding."""
        return float(self.to_score(self._image_raw(self._knn(emb.feats))))

    def score(self, image: np.ndarray, seg: Segmentation | None = None) -> float:
        """Skor kempes/anomali 0–100 untuk satu gambar BGR (opsional segmentasi siap pakai)."""
        return self.score_embedding(self.embed(image, seg))

    def represent(self, image: np.ndarray, seg: Segmentation | None = None) -> Embedding:
        """Representasi untuk CV = embedding patch."""
        return self.embed(image, seg)

    def fit_repr(self, reprs: Sequence[Embedding], y: np.ndarray) -> Self:
        """Fit bank dari embedding yang berlabel normal saja (kempes diabaikan)."""
        return self.fit_embeddings([r for r, v in zip(reprs, y, strict=True) if v == 0])

    def score_repr(self, reprs: Sequence[Embedding]) -> np.ndarray:
        """Skor 0–100 untuk daftar embedding."""
        return np.array([self.score_embedding(r) for r in reprs])

    def heatmap_crop(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Heatmap skor patch (skala 0–100 yang sama) seukuran crop; 0 di luar patch roti.

        Blur ternormalisasi (hanya patch roti yang dirata-rata) agar tepi tidak tampak dingin.
        """
        emb = self.embed_crop(img, mask)
        grid = np.zeros(emb.grid[0] * emb.grid[1], np.float32)
        valid = np.zeros_like(grid)
        grid[emb.idx] = self.to_score(self._knn(emb.feats).numpy())
        valid[emb.idx] = 1.0
        size = img.shape[:2][::-1]
        hm = cv2.resize(grid.reshape(emb.grid), size, interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(valid.reshape(emb.grid), size, interpolation=cv2.INTER_LINEAR)
        sigma = float(self.params.get("heatmap_sigma", 4))
        if sigma > 0:
            hm, v_b = cv2.GaussianBlur(hm, (0, 0), sigma), cv2.GaussianBlur(v, (0, 0), sigma)
            hm = hm / np.maximum(v_b, 1e-6)
        return np.where(v >= 0.5, hm, 0.0).astype(np.float32)

    def heatmap(self, image: np.ndarray, seg: Segmentation | None = None) -> np.ndarray:
        """Heatmap anomali (H, W) float32 0–100 pada crop roti ``image_size`` (F10)."""
        return self.heatmap_crop(*self.prepare(image, seg))

    # --- simpan / muat --------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Simpan memory bank + ref + snapshot config (tanpa bobot backbone)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "type": "anomaly",
                "bank": self.bank,
                "ref": self.ref,
                "loo_raw": self.loo_raw,
                "params": self.params,
                "segment": asdict(self.segment_cfg),
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Muat scorer dari file hasil ``save``. Backbone dibangun dari disk saja (tanpa unduh)."""
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("type") != "anomaly":
            raise ValueError(f"{path}: bukan model anomaly")
        obj = cls(blob["params"], SegmentConfig(**blob["segment"]), blob["seed"])
        obj.bank, obj.ref, obj.loo_raw = blob["bank"], blob["ref"], list(blob["loo_raw"])
        _ = obj.backbone  # gagal cepat dengan pesan jelas bila bobot tidak ada
        return obj


def heatmap_overlay(
    img: np.ndarray, mask: np.ndarray, hm: np.ndarray, vmin: float = 25, vmax: float = 75
) -> np.ndarray:
    """Crop BGR + heatmap (colormap JET, vmin→biru, vmax→merah) hanya pada patch roti."""
    norm = np.clip((hm - vmin) / (vmax - vmin), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(img, 0.45, color, 0.55, 0)
    return np.where(((mask > 0) & (hm > 0))[..., None], blend, img)
