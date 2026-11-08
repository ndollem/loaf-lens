# Audit dataset — LoafLens (M1)

- Tanggal: 2026-11-08
- Folder: normal=`train-data/good-loaf`, defect=`train-data/bad-loaf`
- Manifest: `train-data/manifest.csv` (ada)
- Near-duplicate: dHash 64-bit, jarak ≤ 5

## Peringatan (9)

- **WARNING:** 48/48 gambar tanpa subfolder batch → satu grup per gambar. Split per batch/sesi TIDAK mungkin; gambar dari sesi yang sama bisa tersebar ke train/val/test. Minta info batch/sesi, lalu taruh di train-data/<label>/<batch_id>/.
- **WARNING:** Prefix nama `medium` muncul di kedua label (defect, normal) — arti prefix perlu dikonfirmasi (mis. tingkat kempes?). Prefix TIDAK dipakai sebagai batch.
- **WARNING:** [normal] File bernomor sama dengan prefix berbeda (mis. `x (3)`/`y (3)`) jauh lebih mirip (median dHash 9 vs 21) — kemungkinan varian roti/foto yang sama; 6/12 pasangan terpisah ke split berbeda. Jika memang varian, pasangan harus satu grup (risiko bocor); konfirmasi ke user. Saat ini TIDAK digabung otomatis.
- **WARNING:** Ukuran gambar TIDAK overlap antar kelas (normal 312x417; defect 360x355): model bisa belajar ukuran/rasio gambar, bukan bentuk roti. Perbaikan di M2: crop hasil segmentasi + resize ke ukuran tetap, tanpa fitur ukuran gambar.
- **WARNING:** Fitur trivial `width` memisahkan kelas dengan akurasi 100% (satu threshold) — risiko shortcut (latar/pencahayaan/ukuran berbeda per kelas).
- **WARNING:** Fitur trivial `height` memisahkan kelas dengan akurasi 100% (satu threshold) — risiko shortcut (latar/pencahayaan/ukuran berbeda per kelas).
- **WARNING:** Fitur trivial `aspect` memisahkan kelas dengan akurasi 100% (satu threshold) — risiko shortcut (latar/pencahayaan/ukuran berbeda per kelas).
- **WARNING:** Val/test sangat kecil (val/defect=3, val/normal=3, test/defect=4, test/normal=4; < 30 per label): metrik (AUROC, recall) akan sangat noisy — satu gambar bisa menggeser hasil beberapa persen. Rekomendasi M2: cross-validation per grup selain hold-out.
- **WARNING:** Dataset di bawah perkiraan PRD §8 (normal 24/≥200, defect 24/≥30–50).

## Jumlah per label

| label | file | terbaca | batch_id unik | fallback per-gambar |
|---|---|---|---|---|
| defect | 24 | 24 | 24 | 24 |
| normal | 24 | 24 | 24 | 24 |

## Prefix nama file

| label | prefix | jumlah |
|---|---|---|
| defect | `bad` | 12 |
| defect | `medium` | 12 |
| normal | `bagus` | 12 |
| normal | `medium` | 12 |

Kemiripan pasangan bernomor sama, prefix berbeda (median jarak dHash; kecil = mirip):

| label | pasangan | median nomor sama | median lainnya | pasangan beda split |
|---|---|---|---|---|
| defect | 11 | 16 | 19 | 5 |
| normal | 12 | 9 | 21 | 6 |

## Split (dari manifest)

| split | normal | defect | total |
|---|---|---|---|
| train | 17 | 17 | 34 |
| val | 3 | 3 | 6 |
| test | 4 | 4 | 8 |

Kebocoran batch (batch_id di >1 split): tidak ada.

## Ukuran & rasio aspek per kelas

| label | ukuran WxH (jumlah) | aspek W/H min–max |
|---|---|---|
| defect | 360x355 (24) | 1.014–1.014 |
| normal | 312x417 (24) | 0.748–0.748 |

## Warna per kelas (rata-rata RGB 0–255, ± std antar gambar)

Tepi = 5% pinggir gambar (≈ latar); tengah = kotak 50% sisi (≈ roti).

| label | tepi R,G,B | tepi gray | tengah R,G,B | tengah gray |
|---|---|---|---|---|
| defect | 190±9, 187±10, 188±14 | 188±10 | 183±12, 113±13, 49±16 | 127±12 |
| normal | 188±19, 186±19, 186±19 | 186±19 | 185±8, 107±13, 45±17 | 123±11 |

## Pemisahan kelas oleh fitur trivial

Akurasi terbaik dengan satu threshold (50% = tak berguna, 100% = pemisah sempurna).

| fitur | akurasi |
|---|---|
| `width` | 100% |
| `height` | 100% |
| `aspect` | 100% |
| `border_gray` | 65% |
| `border_r_minus_b` | 65% |
| `center_gray` | 69% |
| `center_r_minus_b` | 60% |

## Duplikat

Duplikat persis: 0 grup.

Near-duplicate (dHash ≤ 5): 1 pasangan.
- `train-data/good-loaf/bagus (5).png` (train) ↔ `train-data/good-loaf/medium (5).png` (train), jarak 5

## File tak terbaca

(tidak ada)

## Konsistensi manifest

- File di disk belum di manifest: 0
- Entri manifest tidak ada di disk: 0
