# Hasil evaluasi scorer A (baseline) — LoafLens M2

> **Hasil pada 48 foto; indikasi kelayakan, bukan klaim akurasi produksi.**

- Tanggal: 2026-11-08
- Classifier: `logistic_regression` + StandardScaler; 25 fitur (tekstur, warna/kilap, bentuk bebas skala) dari crop roti 192 px
- Train: 34 gambar (split `train` manifest beku). Threshold config: **50** (reject jika skor ≥ threshold).
- **Test dievaluasi sekali dengan threshold dari config.** Threshold, fitur, dan hyperparameter TIDAK disetel dari hasil test; val juga tidak dipakai untuk kalibrasi.
- Gagal segmentasi pada val+test (fallback seluruh gambar): 0

## Hold-out (model dilatih dari train saja)

Recall @ FPR ≤ 0.05 dihitung dari kurva ROC. Dengan 3–4 roti normal per split, FPR hanya bisa bernilai 0, 0.25/0.33, … — jadi angka ini praktis = recall pada FPR 0 (sangat kasar).

| split | n (N/D) | AUROC | recall@FPR≤target (FPR) | akurasi | presisi | recall | TN FP FN TP |
|---|---|---|---|---|---|---|---|
| val | 6 (3/3) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 3 0 0 3 |
| test | 8 (4/4) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 4 0 0 4 |

## Cross-validation per grup (5-fold StratifiedGroupKFold, train+val, test tidak ikut)

Lebih bisa dipercaya daripada hold-out kecil di atas. Grup = `batch_id` digabung near-duplicate (aturan yang sama dengan split manifest).

| metrik | nilai |
|---|---|
| AUROC per fold (mean ± std) | 0.988 ± 0.025 |
| akurasi @ threshold 50 (mean ± std) | 0.975 ± 0.050 |
| AUROC gabungan out-of-fold (n=40) | 0.992 |
| recall @ FPR ≤ 0.05 out-of-fold (FPR) | 1.00 (0.05) |

AUROC per fold: 0.938, 1.000, 1.000, 1.000, 1.000

## Rincian per kelompok nama file

`medium-good` = masih layak walau permukaan tidak sempurna; `medium-bad` = tampak lumayan tapi permukaan terlalu kempes → reject. Kelompok borderline ini yang paling penting.

**val**

| kelompok | n | skor rata-rata | min–max | salah |
|---|---|---|---|---|
| bagus | 2 | 2.8 | 2.1–3.4 | 0 |
| medium-good | 1 | 2.2 | 2.2–2.2 | 0 |
| medium-bad | 2 | 93.9 | 93.3–94.5 | 0 |
| bad | 1 | 98.0 | 98.0–98.0 | 0 |

**test**

| kelompok | n | skor rata-rata | min–max | salah |
|---|---|---|---|---|
| bagus | 1 | 2.5 | 2.5–2.5 | 0 |
| medium-good | 3 | 7.7 | 3.1–14.1 | 0 |
| medium-bad | 1 | 75.9 | 75.9–75.9 | 0 |
| bad | 3 | 97.5 | 93.2–99.8 | 0 |

**CV out-of-fold (train+val)**

| kelompok | n | skor rata-rata | min–max | salah |
|---|---|---|---|---|
| bagus | 11 | 11.6 | 0.1–84.1 | 1 |
| medium-good | 9 | 7.1 | 0.5–26.3 | 0 |
| medium-bad | 11 | 90.7 | 78.5–97.4 | 0 |
| bad | 9 | 92.6 | 71.7–99.2 | 0 |

## Kesalahan terbesar (margin salah-sisi dari threshold)

**val**

(tidak ada kesalahan pada threshold config)

**test**

(tidak ada kesalahan pada threshold config)

**CV out-of-fold**

- `train-data/good-loaf/bagus (10).png` — label normal, skor 84.1 (margin 34.1)

## Latency

Rata-rata 4.2 ms/gambar (median 4.1, maks 5.8; n=14) untuk segmentasi + fitur + prediksi, CPU, tanpa waktu baca file. Target PRD: ≤ 100 ms.

## Cek shortcut: AUROC satu fitur

CV = rata-rata AUROC di fold uji (arah fitur dipilih dari fold latih); gabungan = seluruh train+val, arah bebas. Fitur dengan AUROC CV ≥ 0.98 ditandai.

| fitur | AUROC CV | AUROC gabungan | catatan |
|---|---|---|---|
| `col_L_std` | 0.912 | 0.925 |  |
| `col_a_mean` | 0.875 | 0.845 |  |
| `tex_lbp_3` | 0.863 | 0.867 |  |
| `tex_lbp_0` | 0.838 | 0.815 |  |
| `shape_aspect` | 0.838 | 0.835 |  |
| `col_b_std` | 0.825 | 0.820 |  |
| `col_L_mean` | 0.787 | 0.762 |  |
| `tex_lbp_1` | 0.762 | 0.733 |  |
| `tex_lbp_5` | 0.738 | 0.708 |  |
| `shape_circularity` | 0.688 | 0.672 |  |
| `tex_lbp_2` | 0.688 | 0.698 |  |
| `tex_lbp_6` | 0.650 | 0.637 |  |
| `tex_lbp_8` | 0.625 | 0.603 |  |
| `shape_solidity` | 0.625 | 0.630 |  |
| `tex_lbp_7` | 0.613 | 0.585 |  |
| `tex_lbp_9` | 0.588 | 0.613 |  |
| `col_specular_frac` | 0.575 | 0.575 |  |
| `tex_edge_density` | 0.525 | 0.537 |  |
| `col_a_std` | 0.525 | 0.562 |  |
| `shape_extent` | 0.425 | 0.540 |  |
| `tex_lbp_4` | 0.412 | 0.537 |  |
| `tex_lap_var` | 0.412 | 0.527 |  |
| `tex_crease` | 0.362 | 0.547 |  |
| `tex_grad_mean` | 0.362 | 0.522 |  |
| `col_b_mean` | 0.287 | 0.513 |  |

Koefisien logistic regression terbesar (fitur terstandardisasi; + = ke arah kempes): `col_L_std` -1.25, `col_b_std` -0.76, `shape_aspect` -0.64, `col_a_mean` -0.59, `tex_lbp_3` -0.58, `col_L_mean` +0.57, `tex_crease` +0.47, `tex_lbp_0` -0.43

## Catatan & keterbatasan

- Data sangat kecil (24 + 24 foto; val 3+3, test 4+4): satu gambar menggeser metrik hold-out 12–17 poin. Pakai angka CV sebagai acuan utama, itu pun dengan rentang lebar.
- Foto studio: tiap folder diambil dengan kondisi berbeda (ukuran gambar, latar, pencahayaan/warna). Fitur ukuran gambar sudah dibuang (crop + resize tetap), tetapi angka yang (hampir) sempurna bisa sebagian mencerminkan **perbedaan pengambilan foto**, bukan kempes itu sendiri. Uji sebenarnya adalah rekaman kamera/konveyor (M4).
- Tidak ada info batch/sesi: grup = satu gambar (+ near-duplicate). Kebocoran antar foto dari sesi yang sama tidak bisa disingkirkan.
- Segmentasi saturasi HSV kadang memotong sebagian alas roti yang pucat (lihat contact sheet); dampaknya terutama pada fitur bentuk, bukan tekstur permukaan atas.
- Sudut kamera bervariasi (samping/miring/atas) sehingga fitur bentuk (aspek, circularity) ikut dipengaruhi sudut pandang.
- Threshold 50 belum dikalibrasi (kalibrasi di val = M3, F9).
- Hasil CV bergantung pada pembagian fold: dengan 40 gambar (±8 per fold), satu gambar salah urut menggeser AUROC fold ±0.06.

## Riwayat iterasi (sebelum test dibuka; dinilai hanya dengan CV train+val)

- Versi awal (crop 256, tanpa blur): AUROC CV ~1.0, tetapi fitur tekstur halus (LBP, Laplacian, gradien, Canny) bergeser ~1 std bila foto asli di-resize 0.7x — sebanding dengan selisih antar kelas, dan arahnya terbalik dari dugaan (roti bagus punya lebih banyak detail halus). Ini risiko shortcut ketajaman/resolusi foto.
- Perbaikan: Gaussian blur σ=1.5 sebelum tekstur + crop 192 px (dipilih karena paling tahan resize, pergeseran ≤ 0.4 std) dan fitur lipatan `tex_crease` (black-hat). Classifier tetap logistic regression (default); RF/SVM dicoba di CV, tidak lebih baik.
- Akibatnya model kini bersandar terutama pada kontras luminans/kilap (`col_L_std`), warna (`col_a_mean`, `col_b_std`), pola LBP, dan aspek roti; fitur kerutan eksplisit (`tex_crease`, `tex_edge_density`) lemah sendirian. Permukaan kusam & pucat pada foto kempes bisa nyata (kempes) atau sebagian artefak pencahayaan per folder.

Gambar (ROC, contact sheet) dan `scores.csv` ada di `outputs/baseline/` (tidak di-versi).
