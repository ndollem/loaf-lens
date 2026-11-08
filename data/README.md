# Data LoafLens

Folder data aktif diatur di `configs/*.yaml` (`data.normal_dir`, `data.defect_dir`,
`data.manifest`). Saat ini:

| Config | Path | Label |
|---|---|---|
| `data.normal_dir` | `train-data/good-loaf/` | `normal` |
| `data.defect_dir` | `train-data/bad-loaf/` | `defect` (kempes) |
| `data.manifest` | `train-data/manifest.csv` | — |

Format yang diterima: `data.extensions` (default `.png`, `.jpg`, `.jpeg`). Nama file boleh berisi
spasi/kurung (mis. `bagus (3).png`).

> **`train-data/` di-commit ke git atas keputusan user** (foto kecil, ~6.6 MB).
> Isi `data/` lainnya tetap **di-gitignore** (kecuali file ini dan `manifest.example.csv`):
> jangan commit foto/video client lain, model, atau output eksperimen.

## Domain check: `train-data/domain-check/` (M3)

Foto konveyor asli dari user untuk **cek domain saja — tidak pernah dipakai untuk training, split,
atau tuning** (folder ini di luar `data.normal_dir`/`data.defect_dir`, tidak masuk manifest).
`conveyor_good_01.webp` (941x1672): belt mesh abu gelap tampak atas/miring, ±19 roti bulat mengilap,
**semua layak**. `segment_all` + scorer menilai tiap roti; karena semua OK, setiap REJECT = false
positive (`eval.domain_check` di config; `make domain-check`). `conveyor_mixed_01.webp`: konveyor yang sama, 20 roti terdeteksi = **5 kempes** (#1, #3, #6, #17,
#20) + 15 OK (termasuk borderline #12/#19, dikonfirmasi user OK); label posisi di `labels.json`,
hasil di [`docs/domain_check_mixed.md`](../docs/domain_check_mixed.md). `labels.json` juga dipakai
sebagai ground truth video demo SYNTHETIC (`make demo-video`). Tambahkan foto lain dengan nama yang
menjelaskan isinya dan catat labelnya di sini (dan di `labels.json`).

## Isi saat ini (M1, 2026-11-08)

- `train-data/good-loaf/`: 24 PNG normal (`bagus (0..11)`, `medium (0..11)`), semua 312x417.
- `train-data/bad-loaf/`: 24 PNG kempes (`bad (0..11)`, `medium`, `medium (1..11)`), semua 360x355.
- Satu roti per foto, foto samping/miring gaya studio dengan latar polos — bukan foto konveyor.
- **Tidak ada info batch/sesi** → tiap gambar jadi grupnya sendiri (lihat di bawah).
- Hasil audit lengkap: [`docs/audit_m1.md`](../docs/audit_m1.md) (ukuran gambar beda per kelas =
  risiko shortcut).

### Arti prefix nama file (dari user, 2026-11-08)

| Folder | Prefix | Arti | Keputusan |
|---|---|---|---|
| `good-loaf/` | `bagus (n)` | jelas layak | terima |
| `good-loaf/` | `medium (n)` | masih layak walau permukaan tidak sempurna (borderline OK) | terima |
| `bad-loaf/` | `bad (n)` | kempes parah | reject |
| `bad-loaf/` | `medium (n)`, `medium.png` | tampak lumayan, tapi permukaan terlalu kempes (borderline reject) | reject |

`bagus (n)`/`medium (n)` dengan nomor sama adalah tingkat kualitas berbeda, **bukan** roti yang sama
→ split beku tidak diubah. Fokus pembeda: kondisi/tekstur permukaan (kerutan, lipatan, permukaan
cekung/datar, kilap), bukan sesi/tanggal. Prefix hanya dipakai untuk rincian per kelompok di laporan
evaluasi (`bagus` / `medium-good` / `bad` / `medium-bad`), tidak sebagai fitur.

## Menambah foto

Taruh foto baru di folder label, **sebaiknya satu subfolder per batch/sesi pengambilan**:

```
train-data/good-loaf/<batch_id>/<nama>.png
train-data/bad-loaf/<batch_id>/<nama>.png
```

`batch_id` disarankan `<YYYYMMDD>_<sesi>`, misal `20261110_s1`. Satu sesi = satu kali pengambilan
dengan kamera, posisi, dan pencahayaan yang sama. Normal dan kempes dari sesi yang sama memakai
nama subfolder yang sama.

### Cara `batch_id` ditentukan (`python -m loaflens.data`)

1. Ada subfolder langsung di bawah folder label → `batch_id` = nama subfolder itu (subfolder yang
   lebih dalam ikut batch yang sama).
2. File langsung di folder label → **satu grup per gambar**: `batch_id = img_<label>_<nama file>`.
   Audit memberi WARNING karena split per batch tidak mungkin.
3. Prefix nama file (`bagus`, `medium`, `bad`) **tidak** dipakai sebagai batch.

Saat split, gambar sebatch dan pasangan **near-duplicate** (dHash 64-bit, jarak ≤
`data.near_dup_hamming`) digabung jadi satu grup, sehingga selalu masuk split yang sama.

## Manifest (`train-data/manifest.csv`)

Dibuat oleh `make manifest` (`python -m loaflens.data --config configs/baseline.yaml`).
Satu baris per gambar. Kolom:

| Kolom | Isi |
|---|---|
| `path` | path relatif dari root repo |
| `label` | `normal` \| `defect` |
| `batch_id` | id batch/sesi (nama subfolder, atau `img_...` bila tidak ada) |
| `date` | tanggal pengambilan, `YYYY-MM-DD` (kosong = belum diketahui) |
| `camera` | nama/model kamera atau HP |
| `camera_view` | `top` (tegak lurus dari atas) \| `oblique` (miring ~45°) |
| `lighting` | catatan pencahayaan, misal `lampu_pabrik`, `siang_jendela` |
| `bread_type` | jenis roti / SKU |
| `split` | `train` \| `val` \| `test` |

Metadata yang belum diketahui dibiarkan kosong; isi manual di CSV bila sudah ada info.
Contoh format: [`manifest.example.csv`](manifest.example.csv) (contoh, bukan data asli).

**Manifest = split beku.** CLI tidak menimpa manifest yang sudah ada. `--force` membuat split
ulang (metadata manual dipertahankan per `path`) — lakukan hanya sebelum tuning/evaluasi apa pun,
dan catat alasannya di `docs/PRD.md`. Setelah menambah foto, audit akan melaporkan file yang belum
masuk manifest.

## Aturan

1. **Split per batch/grup, bukan acak per gambar.** Semua gambar dari satu grup masuk ke split yang
   sama, terstratifikasi per label, deterministik (`seed`). Rasio di `data.split`.
2. **Test set dibekukan** sebelum tuning apa pun. Threshold dikalibrasi di val, bukan test.
3. **Kebersihan folder normal:** hanya roti yang benar-benar normal (konfirmasi manusia). Satu roti
   kempes yang nyelip merusak model anomaly. Kalau ragu, jangan masukkan.
4. Jalankan `make audit` setiap kali data berubah; baca bagian WARNING.

## Ukuran dataset (perkiraan PRD §8)

- ≥ 200 foto roti normal, ≥ 30–50 foto roti kempes (saat ini 24 + 24 — jauh di bawah).
- Variasi pencahayaan, posisi, dan **beberapa batch** (minimal 3 batch agar train/val/test bisa
  dipisah per batch).
