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

## Data

Foto dari user ada di `train-data/` (di-commit atas keputusan user): `good-loaf/` = normal,
`bad-loaf/` = kempes, split beku di `train-data/manifest.csv`. Data client lain **tidak** masuk git.
Lihat [`data/README.md`](data/README.md) untuk cara menambah foto, `batch_id`, manifest, dan aturan
split; hasil audit di [`docs/audit_m1.md`](docs/audit_m1.md).

## Config

Semua parameter ada di `configs/*.yaml`:

- `configs/baseline.yaml` — scorer A (fitur bentuk/warna + scikit-learn, latih dari normal + kempes)
- `configs/anomaly.yaml` — scorer B (anomaly detection, latih dari normal saja)

## Perintah

| Perintah | Status |
|---|---|
| `make manifest` = `python -m loaflens.data --config configs/baseline.yaml` | tersedia (M1); tidak menimpa manifest, `--force` = split ulang |
| `make audit` = `python -m loaflens.audit --config configs/baseline.yaml` | tersedia (M1); → `outputs/audit.md` + `docs/audit_m1.md` |
| `python -m loaflens.train --config configs/baseline.yaml` | belum tersedia (M2) |
| `python -m loaflens.evaluate --config configs/baseline.yaml` | belum tersedia (M2) |
| `python -m loaflens.train --config configs/anomaly.yaml` | belum tersedia (M3) |
| `python -m loaflens.app --source video:samples/demo.mp4` | belum tersedia (M4) |
| `python -m loaflens.app --source camera:0` | belum tersedia (M4) |

## Status

Lihat `docs/PRD.md` §13. Saat ini: **M1** selesai (manifest, split, audit dataset).
