# LoafLens

POC visi komputer low-cost untuk mendeteksi roti **kempes** (afkir) di konveyor.
Kamera + model ML lokal (CPU-only, tanpa LLM/API cloud) memberi **skor kempes 0–100** per roti;
roti di-REJECT jika skor ≥ `threshold` di config.

Detail tujuan & requirement: [`docs/PRD.md`](docs/PRD.md). Aturan kerja: [`CLAUDE.md`](CLAUDE.md).

## Setup

Butuh [`uv`](https://docs.astral.sh/uv/) dan `make`. Versi Python ada di `.python-version` (3.14).

```bash
make setup   # buat .venv, pasang dependency terkunci (requirements-dev.txt) + paket editable
make test    # pytest
make lint    # ruff check + ruff format --check
make format  # perbaiki format otomatis
make lock    # perbarui requirements*.txt setelah mengubah pyproject.toml
```

### PyTorch (CPU-only) untuk scorer B

Scorer B memakai `torch` + `torchvision` (bukan anomalib). Di Linux, `torch` dari PyPI menarik paket
CUDA berukuran beberapa GB — karena itu `make setup` dan `make lock` memakai
`uv ... --torch-backend cpu`, yang mengambil **hanya** paket ekosistem PyTorch dari index CPU
`https://download.pytorch.org/whl/cpu` (paket lain tetap dari PyPI):

- **Linux x86_64/aarch64:** terpasang `torch==2.15.0+cpu`, `torchvision==0.29.2+cpu` (tanpa paket
  `nvidia-*`/`triton`). Ukuran `.venv` naik ±0.9 GB (torch ±750 MB).
- **macOS Apple Silicon:** wheel macOS memang CPU/MPS-only; index CPU menyediakan versi yang sama
  (`torch==2.15.0`, butuh macOS ≥ 14). `make setup` sama persis.
- `requirements*.txt` mem-pin `torch==2.15.0` / `torchvision==0.29.2` **tanpa** label `+cpu`
  (`make lock` membuangnya) sehingga satu file berlaku di Linux dan macOS; pin tanpa label lokal
  cocok dengan wheel `+cpu` (PEP 440).
- Tanpa uv: `pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu`
  (pip memilih `+cpu` karena dianggap versi lebih tinggi). Jangan `pip install torch` polos di Linux.
- Wheel dari index CPU di-host di `download-r2.pytorch.org`; host itu harus bisa diakses saat setup.

## Data

Foto dari user ada di `train-data/` (di-commit atas keputusan user): `good-loaf/` = normal,
`bad-loaf/` = kempes, split beku di `train-data/manifest.csv`. Data client lain **tidak** masuk git.
Lihat [`data/README.md`](data/README.md) untuk cara menambah foto, `batch_id`, manifest, dan aturan
split; hasil audit di [`docs/audit_m1.md`](docs/audit_m1.md).

## Config

Semua parameter ada di `configs/*.yaml`:

- `configs/baseline.yaml` — scorer A (fitur bentuk/warna + scikit-learn, latih dari normal + kempes)
- `configs/anomaly.yaml` — scorer B (anomaly detection "PatchCore-lite", latih dari normal saja)

## Backbone pretrained (scorer B)

Scorer B memakai **ResNet18 pretrained ImageNet (torchvision) hanya sebagai feature extractor beku**:
tidak di-fine-tune, tidak ada kepala klasifikasi ImageNet yang dipakai. Yang "dilatih dari data kita"
adalah memory bank patch roti normal dari `train-data/` + kalibrasi skor. Bobot (±47 MB,
`resnet18-f37072fd.pth`, hash diverifikasi) diunduh **sekali saat training** ke cache torch
(`~/.cache/torch/hub/checkpoints/`, atau `model.params.weights_dir`). Saat inferensi/evaluasi tidak
ada panggilan jaringan: bila file bobot tidak ada, scorer berhenti dengan pesan jelas. Bobot dan
`model.pt` (memory bank ±10 MB) tidak di-commit (`outputs/` di-gitignore).

Skor B: jarak kNN patch roti terjauh ke memory bank → `skor = clip(50 · mentah / ref, 0, 100)`, dengan
`ref` = persentil `ref_percentile` (p95) skor *leave-one-image-out* roti normal train. Jadi **50 ≈
seaneh roti normal train yang paling aneh**; di atas 50 = di luar variasi normal yang pernah dilihat.

## Perintah

| Perintah | Status |
|---|---|
| `make manifest` = `python -m loaflens.data --config configs/baseline.yaml` | M1; tidak menimpa manifest, `--force` = split ulang |
| `make audit` = `python -m loaflens.audit --config configs/baseline.yaml` | M1; → `outputs/audit.md` + `docs/audit_m1.md` |
| `make train` = `python -m loaflens.train --config configs/baseline.yaml` | M2; fit dari split `train` → `outputs/baseline/model.joblib` + `segment_debug.jpg` |
| `make evaluate` = `python -m loaflens.evaluate --config configs/baseline.yaml` | M2/M3; val/test + CV + kalibrasi + domain check → `outputs/baseline/` + `docs/results_m3_baseline.md` |
| `make train-anomaly` = `python -m loaflens.train --config configs/anomaly.yaml` | M3; normal saja → `outputs/anomaly/model.pt` (unduh bobot ResNet18 sekali) |
| `make evaluate-anomaly` = `python -m loaflens.evaluate --config configs/anomaly.yaml` | M3; + heatmap → `outputs/anomaly/heatmaps/` + `docs/results_m3_anomaly.md` |
| `make calibrate [CONFIG=...]` = `python -m loaflens.calibrate --config configs/<x>.yaml` | M3; saran threshold dari CV train+val (bukan test) → `outputs/<run>/calibration.yaml` (config tidak diubah) |
| `make compare` = `python -m loaflens.compare configs/baseline.yaml configs/anomaly.yaml` | M3; tabel A vs B dari `metrics.json` → `outputs/compare.md` + `docs/results_m3.md` |
| `make domain-check [CONFIG=...]` = `python -m loaflens.evaluate --config configs/<x>.yaml --domain-check <foto>` | M3; skor tiap roti di foto konveyor → `outputs/<run>/domain_check.jpg` |
| `python -m loaflens.app --source video:samples/demo.mp4` | belum tersedia (M4) |
| `python -m loaflens.app --source camera:0` | belum tersedia (M4) |

Urutan lengkap M3: `make train train-anomaly evaluate evaluate-anomaly compare` (`evaluate` sudah
menjalankan kalibrasi + domain check; `calibrate`/`domain-check` untuk dijalankan terpisah).

## Status

Lihat `docs/PRD.md` §13. Saat ini: **M3** selesai — scorer B (PatchCore-lite) + kalibrasi threshold +
perbandingan A vs B + domain check foto konveyor: [`docs/results_m3.md`](docs/results_m3.md)
(rinci: [`results_m3_baseline.md`](docs/results_m3_baseline.md),
[`results_m3_anomaly.md`](docs/results_m3_anomaly.md); snapshot M2: [`results_m2.md`](docs/results_m2.md)).
Ringkas: di 48 foto studio keduanya memisahkan kelas hampir sempurna (CV AUROC A 0.988, B 1.000),
tetapi pada foto konveyor asli (15 roti bagus) A menolak 0 dan B menolak 5–11 → untuk demo M4 A
dipakai sebagai keputusan utama, B butuh memory bank dari kamera konveyor.
