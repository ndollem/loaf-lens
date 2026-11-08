# Data LoafLens

> Semua isi `data/` **di-gitignore** kecuali file ini dan `manifest.example.csv`.
> Jangan pernah commit foto/video asli client.

## Struktur

```
data/
├── raw/          # foto/video asli dari client, apa adanya (jangan diubah)
├── normal/       # roti OK (dikonfirmasi manusia)
├── defect/       # roti kempes
├── manifest.csv  # metadata tiap gambar (lihat di bawah)
└── README.md
```

- `normal/` dipakai untuk melatih **kedua** scorer.
- `defect/` dipakai untuk melatih **scorer A (baseline, supervised)**, dan untuk validasi/test
  kedua scorer. Scorer B (anomaly) **tidak** melihat kempes saat `fit`.
- Kedua folder di-split train/val/test **per batch** dengan cara yang sama.

## Konvensi nama file

Satu subfolder per batch/sesi pengambilan:

```
data/normal/<batch_id>/<batch_id>_<nnn>.jpg
data/defect/<batch_id>/<batch_id>_<nnn>.jpg
```

`batch_id` = `<YYYYMMDD>_<sesi>`, misal `20261110_s1`. Satu sesi = satu kali pengambilan dengan
kamera, posisi, dan pencahayaan yang sama. Contoh: `data/normal/20261110_s1/20261110_s1_001.jpg`.

## Manifest (`data/manifest.csv`)

Satu baris per gambar. Kolom:

| Kolom | Isi |
|---|---|
| `path` | path relatif dari root repo |
| `label` | `normal` \| `defect` |
| `batch_id` | id batch/sesi (sama dengan nama subfolder) |
| `date` | tanggal pengambilan, `YYYY-MM-DD` |
| `camera` | nama/model kamera atau HP |
| `camera_view` | `top` (tegak lurus dari atas) \| `oblique` (miring ~45°) |
| `lighting` | catatan pencahayaan, misal `lampu_pabrik`, `siang_jendela` |
| `bread_type` | jenis roti / SKU |
| `split` | `train` \| `val` \| `test` (diisi saat split, M1) |

Contoh format: [`manifest.example.csv`](manifest.example.csv) (contoh, bukan data asli).

## Aturan

1. **Split per batch, bukan acak per gambar.** Semua gambar dari satu `batch_id` masuk ke split
   yang sama, supaya frame yang mirip tidak bocor dari train ke test. Rasio split di
   `configs/*.yaml` (`data.split`).
2. **Test set dibekukan** sebelum tuning apa pun. Threshold dikalibrasi di val, bukan test.
3. **Kebersihan `normal/`:** hanya roti yang benar-benar normal (konfirmasi manusia). Satu roti
   kempes yang nyelip ke `normal/` merusak model anomaly. Kalau ragu, jangan masukkan ke `normal/`.
4. Foto asli tetap di `raw/`; `normal/` dan `defect/` berisi salinan yang sudah dipilah.

## Ukuran dataset (perkiraan awal, divalidasi di M1)

- ≥ 200 foto roti normal
- ≥ 30–50 foto roti kempes (lebih banyak lebih baik, karena kempes juga dipakai melatih scorer A)
- Variasi pencahayaan, posisi, dan **beberapa batch** (minimal 3 batch agar train/val/test bisa
  dipisah per batch)
