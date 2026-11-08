# LoafLens

POC visi komputer low-cost untuk mendeteksi roti **kempes** (afkir) di konveyor.
Kamera + model ML lokal (CPU-only, tanpa LLM/API cloud) memberi **skor kempes 0–100** per roti;
roti di-REJECT jika skor ≥ `threshold` di config.

> **Panduan lengkap: [`docs/USAGE.md`](docs/USAGE.md)** — instalasi, data, referensi config, alur
> kerja (latih, evaluasi, kalibrasi, demo, kamera), arti output, Makefile/CLI, troubleshooting.

Tujuan & requirement: [`docs/PRD.md`](docs/PRD.md). Aturan kerja & konvensi: [`CLAUDE.md`](CLAUDE.md).
Aturan data: [`data/README.md`](data/README.md).

## Status

M0–M4 selesai (M4 untuk uji file video; demo kamera-ke-layar belum dijalankan), M5 opsional
([`docs/PRD.md` §13](docs/PRD.md#13-status)). Catatan penting:

- Model dilatih dari **48 foto studio** (24 normal + 24 kempes) — angka = indikasi kelayakan, bukan
  akurasi produksi. CV AUROC: scorer A 0.988, scorer B 1.000 ([`docs/results_m3.md`](docs/results_m3.md)).
- **Scorer A (baseline) direkomendasikan** dan dipakai demo: di 2 foto konveyor asli 0/15 roti bagus
  ter-reject dan 5/5 kempes tertangkap ([`docs/domain_check_mixed.md`](docs/domain_check_mixed.md)).
- **Scorer B (anomaly) gagal di domain konveyor** tanpa refit memory bank dari roti bagus kamera
  konveyor (5–11/15 dan 20/20 roti di-REJECT).
- **Video demo adalah SYNTHETIC** (foto diam digeser): 70/70 roti diputuskan tepat sekali, kempes 5/5
  per putaran, OK ter-reject 0/30 — menguji pipeline, bukan akurasi ([`docs/results_m4.md`](docs/results_m4.md)).

## Quickstart

Butuh [`uv`](https://docs.astral.sh/uv/) dan `make` (Linux, atau macOS Apple Silicon ≥ 14). Python
3.14 (`.python-version`) diunduh `uv` bila belum ada.

```bash
make setup         # .venv + dependency terkunci (torch CPU-only) + paket editable
make test          # pytest
make train         # latih scorer A → outputs/baseline/model.joblib
make evaluate      # val/test + CV + kalibrasi + domain check → outputs/baseline/, docs/results_m3_baseline.md
make demo-video    # samples/demo.mp4 + demo.gt.json (SYNTHETIC, tidak di-commit)
make demo          # headless + banding ground truth → outputs/demo_out.mp4, outputs/app/gt_compare.md
make demo-live     # sama, dengan jendela OpenCV (butuh display; q/ESC keluar)
```

Scorer B dan perbandingan: `make train-anomaly evaluate-anomaly compare`. Kalibrasi threshold:
`make calibrate` (default scorer A; scorer B: `CONFIG=configs/anomaly.yaml`; config tidak diubah
otomatis). Kamera: `make camera CAMERA=0`. Menjalankan di MacBook dengan kamera (termasuk izin
kamera macOS): [`docs/USAGE.md` §6.8](docs/USAGE.md#68-kamera--demo-kamera-ke-layar). Repo ini tidak
cocok dideploy ke Vercel/serverless (butuh kamera lokal & proses yang terus berjalan).
Semua target: [`docs/USAGE.md` §8](docs/USAGE.md#8-referensi-makefile).

## Catatan instalasi

- **PyTorch CPU-only:** `make setup`/`make lock` memakai `uv ... --torch-backend cpu` sehingga hanya
  paket PyTorch diambil dari index `https://download.pytorch.org/whl/cpu` (Linux: `torch==2.15.0+cpu`
  tanpa paket CUDA; macOS: wheel biasa). `requirements*.txt` mem-pin tanpa label `+cpu` agar sama di
  Linux dan macOS. Tanpa uv:
  `pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu`
  (jangan `pip install torch` polos di Linux). Host `download-r2.pytorch.org` harus bisa diakses.
- **ResNet18 (hanya scorer B):** bobot ImageNet (±47 MB) diunduh sekali saat `make train-anomaly` ke
  cache torch, dipakai sebagai feature extractor beku (tidak di-fine-tune). Inferensi tidak pernah
  mengunduh apa pun.
- **`.venv` lama (sebelum M4):** OpenCV kini `opencv-python` (GUI), bentrok dengan
  `opencv-python-headless`. Jalankan
  `uv pip uninstall --python .venv/bin/python opencv-python-headless` lalu `make setup`.
- Linux tanpa display: pakai `--headless` (`make demo`).

## Data & config

- `train-data/` (di-commit atas keputusan user): `good-loaf/` = normal, `bad-loaf/` = kempes,
  `domain-check/` = foto konveyor asli (tidak dilatih), split beku `manifest.csv`. Data client lain,
  model, dan `outputs/` tidak masuk git.
- `configs/baseline.yaml` = scorer A + parameter demo (`camera`, `segment`, `track`, `app`);
  `configs/anomaly.yaml` = scorer B. Referensi tiap kunci: [`docs/USAGE.md` §5](docs/USAGE.md#5-referensi-konfigurasi).
