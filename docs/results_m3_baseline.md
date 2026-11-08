# Hasil evaluasi scorer A (baseline: fitur tekstur/warna/bentuk + logistic regression) — LoafLens

> **Hasil pada 48 foto studio; indikasi kelayakan, bukan klaim akurasi produksi.**

- Tanggal: 2026-11-08
- Classifier: `logistic_regression` + StandardScaler; 25 fitur (tekstur, warna/kilap, bentuk bebas skala) dari crop roti 64 px. Dilatih dari normal + kempes.
- Train: 34 gambar (split `train` manifest beku; label normal/defect). Threshold config: **50**; threshold terkalibrasi (CV train+val, FPR ≤ 0.05): **40.4** (reject jika skor ≥ threshold).
- **Test dievaluasi dengan dua threshold yang keduanya TIDAK dipilih dari test** (config & kalibrasi CV train+val). Fitur dan hyperparameter tidak disetel dari test.
- Gagal segmentasi pada val+test (fallback seluruh gambar): 0

## Hold-out (model dilatih dari train saja)

Recall @ FPR ≤ 0.05 dihitung dari kurva ROC (threshold optimal pasca-hoc, hanya informasi — bukan threshold operasi). Dengan 3–4 roti normal per split, FPR hanya bisa 0, 0.25/0.33, … — angka ini praktis = recall pada FPR 0 (sangat kasar).

| split | threshold | n (N/D) | AUROC | recall@FPR≤target (FPR) | akurasi | presisi | recall | TN FP FN TP |
|---|---|---|---|---|---|---|---|---|
| val | 50 | 6 (3/3) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 3 0 0 3 |
| test | 50 | 8 (4/4) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 4 0 0 4 |
| val | 40.4 | 6 (3/3) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 3 0 0 3 |
| test | 40.4 | 8 (4/4) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 4 0 0 4 |

## Cross-validation per grup (5-fold StratifiedGroupKFold, train+val, test tidak ikut)

Lebih bisa dipercaya daripada hold-out kecil di atas. Grup = `batch_id` digabung near-duplicate (aturan yang sama dengan split manifest); classifier di-fit ulang dari fold latih. Fold sama untuk scorer A dan B (seed config).

| metrik | nilai |
|---|---|
| AUROC per fold (mean ± std) | 1.000 ± 0.000 |
| akurasi @ threshold 50 (mean ± std) | 1.000 ± 0.000 |
| AUROC gabungan out-of-fold (n=40) | 1.000 |
| recall @ FPR ≤ 0.05 out-of-fold (FPR) | 1.00 (0.00) |

AUROC per fold: 1.000, 1.000, 1.000, 1.000, 1.000

## Kalibrasi threshold (F9, target FPR ≤ 0.05, tanpa test)

| sumber | n normal / kempes | langkah FPR | threshold | FPR | recall | FPR / recall @ config 50 |
|---|---|---|---|---|---|---|
| CV out-of-fold train+val (**saran**) | 20 / 20 | 0.050 | **40.4** | 0.05 | 1.00 | 0.00 / 1.00 |
| val saja (pembanding) | 3 / 3 | 0.333 | 49.9 | 0.00 | 1.00 | 0.00 / 1.00 |

Aturan: boleh k = floor(target · n normal) false positive → batas bawah = sedikit di atas skor normal tertinggi ke-(k+1); threshold lalu digeser ke tengah celah antara batas itu dan skor kempes terendah di atasnya (margin maksimum, FPR/recall data kalibrasi tidak berubah). **Kasar:** dengan 20 normal di CV satu roti = langkah FPR 0.05, jadi target 5% = paling banyak 1 normal di atas batas; di val (3 normal) target 5% = FPR 0. Saran ditulis ke `calibration.yaml`; config TIDAK diubah otomatis.

## Rincian per kelompok nama file

`medium-good` = masih layak walau permukaan tidak sempurna; `medium-bad` = tampak lumayan tapi permukaan terlalu kempes → reject. Kolom `salah` memakai threshold config.

**val**

| kelompok | n | skor rata-rata | min–max | salah |
|---|---|---|---|---|
| bagus | 2 | 1.0 | 0.5–1.4 | 0 |
| medium-good | 1 | 0.7 | 0.7–0.7 | 0 |
| medium-bad | 2 | 98.9 | 98.5–99.2 | 0 |
| bad | 1 | 99.5 | 99.5–99.5 | 0 |

**test**

| kelompok | n | skor rata-rata | min–max | salah |
|---|---|---|---|---|
| bagus | 1 | 0.4 | 0.4–0.4 | 0 |
| medium-good | 3 | 3.4 | 2.0–4.2 | 0 |
| medium-bad | 1 | 69.4 | 69.4–69.4 | 0 |
| bad | 3 | 99.6 | 98.9–100.0 | 0 |

**CV out-of-fold (train+val)**

| kelompok | n | skor rata-rata | min–max | salah |
|---|---|---|---|---|
| bagus | 11 | 5.5 | 0.0–42.8 | 0 |
| medium-good | 9 | 3.5 | 0.0–13.8 | 0 |
| medium-bad | 11 | 92.9 | 67.1–99.6 | 0 |
| bad | 9 | 98.6 | 96.0–99.9 | 0 |

## Kesalahan terbesar (margin salah-sisi dari threshold config)

**val**

(tidak ada kesalahan pada threshold 50)

**test**

(tidak ada kesalahan pada threshold 50)

**CV out-of-fold**

(tidak ada kesalahan pada threshold 50)

## Latency & ukuran model

Rata-rata 1.9 ms/gambar (median 1.9, maks 2.1; n=14, setelah pemanasan) untuk segmentasi + representasi + skor, CPU, tanpa waktu baca file. Target PRD: ≤ 100 ms. File model: 0.00 MB.

## Domain check: foto konveyor asli (semua roti OK)

Foto: `train-data/domain-check/conveyor_good_01.webp` — roti terdeteksi: **15** (roti terpotong di tepi frame dibuang). Label per roti tidak diketahui tool; bila semua roti di foto OK, **setiap REJECT = false positive** di domain konveyor. Tidak ada retraining/tuning dari foto ini. Gambar beranotasi ada di `outputs/baseline/` (tidak di-versi).

| threshold | nilai | REJECT |
|---|---|---|
| config | 50 | 0/15 |
| kalibrasi | 40.4 | 0/15 |

Skor: min 0.5, median 2.6, maks 14.9. Per roti (urut atas→bawah): #1=1, #2=8, #3=2, #4=15, #5=1, #6=2, #7=6, #8=3, #9=12, #10=4, #11=8, #12=7, #13=1, #14=1, #15=1

## Cek shortcut: AUROC satu fitur

CV = rata-rata AUROC di fold uji (arah fitur dipilih dari fold latih); gabungan = seluruh train+val, arah bebas. Fitur dengan AUROC CV ≥ 0.98 ditandai.

| fitur | AUROC CV | AUROC gabungan | catatan |
|---|---|---|---|
| `col_L_std` | 0.912 | 0.920 |  |
| `col_a_mean` | 0.875 | 0.848 |  |
| `shape_aspect` | 0.850 | 0.843 |  |
| `col_L_mean` | 0.787 | 0.772 |  |
| `col_b_std` | 0.775 | 0.790 |  |
| `tex_lbp_4` | 0.713 | 0.685 |  |
| `shape_circularity` | 0.700 | 0.708 |  |
| `tex_lbp_0` | 0.637 | 0.640 |  |
| `tex_lbp_9` | 0.637 | 0.635 |  |
| `shape_solidity` | 0.637 | 0.637 |  |
| `tex_lbp_6` | 0.625 | 0.650 |  |
| `tex_lbp_2` | 0.613 | 0.570 |  |
| `tex_lbp_8` | 0.600 | 0.610 |  |
| `tex_lap_var` | 0.600 | 0.605 |  |
| `tex_lbp_1` | 0.575 | 0.552 |  |
| `tex_crease` | 0.550 | 0.615 |  |
| `col_a_std` | 0.550 | 0.583 |  |
| `col_specular_frac` | 0.550 | 0.550 |  |
| `tex_lbp_3` | 0.463 | 0.557 |  |
| `tex_lbp_7` | 0.450 | 0.507 |  |
| `tex_lbp_5` | 0.425 | 0.557 |  |
| `shape_extent` | 0.400 | 0.535 |  |
| `tex_edge_density` | 0.375 | 0.510 |  |
| `tex_grad_mean` | 0.375 | 0.535 |  |
| `col_b_mean` | 0.338 | 0.530 |  |

Koefisien logistic regression terbesar (fitur terstandardisasi; + = ke arah kempes): `col_L_std` -2.03, `col_b_std` -1.55, `col_a_mean` -1.15, `col_L_mean` +0.88, `shape_aspect` -0.85, `shape_extent` +0.73, `tex_crease` +0.58, `shape_circularity` -0.46

## Catatan & keterbatasan

- Data sangat kecil (24 + 24 foto; val 3+3, test 4+4): satu gambar menggeser metrik hold-out 12–17 poin. Pakai angka CV sebagai acuan utama, itu pun dengan rentang lebar.
- Foto studio: tiap folder diambil dengan kondisi berbeda (ukuran gambar, latar, pencahayaan/warna). Ukuran gambar sudah dinetralkan (crop + resize tetap), tetapi angka yang (hampir) sempurna bisa sebagian mencerminkan **perbedaan pengambilan foto**, bukan kempes. Sinyal paling realistis: domain check foto konveyor (di atas) dan rekaman kamera (M4).
- Tidak ada info batch/sesi: grup = satu gambar (+ near-duplicate). Kebocoran antar foto dari sesi yang sama tidak bisa disingkirkan.
- Hasil CV bergantung pada pembagian fold: dengan 40 gambar (±8 per fold), satu gambar salah urut menggeser AUROC fold ±0.06.
- Segmentasi saturasi HSV kadang memotong sebagian alas roti yang pucat (lihat contact sheet); dampaknya terutama pada fitur bentuk, bukan tekstur permukaan atas.
- Sudut kamera bervariasi (samping/miring/atas) sehingga fitur bentuk (aspek, circularity) ikut dipengaruhi sudut pandang.

## Riwayat iterasi M2 (sebelum test dibuka; dinilai hanya dengan CV train+val)

- Versi awal (crop 256, tanpa blur): AUROC CV ~1.0, tetapi fitur tekstur halus (LBP, Laplacian, gradien, Canny) bergeser ~1 std bila foto asli di-resize 0.7x — sebanding dengan selisih antar kelas, dan arahnya terbalik dari dugaan (roti bagus punya lebih banyak detail halus). Ini risiko shortcut ketajaman/resolusi foto.
- Perbaikan: Gaussian blur σ=1.5 sebelum tekstur + crop 192 px (dipilih karena paling tahan resize, pergeseran ≤ 0.4 std) dan fitur lipatan `tex_crease` (black-hat). Classifier tetap logistic regression (default); RF/SVM dicoba di CV, tidak lebih baik.
- Akibatnya model kini bersandar terutama pada kontras luminans/kilap (`col_L_std`), warna (`col_a_mean`, `col_b_std`), pola LBP, dan aspek roti; fitur kerutan eksplisit (`tex_crease`, `tex_edge_density`) lemah sendirian. Permukaan kusam & pucat pada foto kempes bisa nyata (kempes) atau sebagian artefak pencahayaan per folder.

Artefak (`roc_test.png`, `segment_debug.jpg`, `scores.csv`, `calibration.yaml`, `metrics.json`) ada di `outputs/baseline/` (tidak di-versi).
