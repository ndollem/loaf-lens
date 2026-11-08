# Hasil evaluasi scorer A (baseline: fitur tekstur/warna/bentuk + logistic regression) — LoafLens

> **Hasil pada 48 foto studio; indikasi kelayakan, bukan klaim akurasi produksi.**

- Tanggal: 2026-11-08
- Classifier: `logistic_regression` + StandardScaler; 25 fitur (tekstur, warna/kilap, bentuk bebas skala) dari crop roti 192 px. Dilatih dari normal + kempes.
- Train: 34 gambar (split `train` manifest beku; label normal/defect). Threshold config: **50**; threshold terkalibrasi (CV train+val, FPR ≤ 0.05): **48.9** (reject jika skor ≥ threshold).
- **Test dievaluasi dengan dua threshold yang keduanya TIDAK dipilih dari test** (config & kalibrasi CV train+val). Fitur dan hyperparameter tidak disetel dari test.
- Gagal segmentasi pada val+test (fallback seluruh gambar): 0

## Hold-out (model dilatih dari train saja)

Recall @ FPR ≤ 0.05 dihitung dari kurva ROC (threshold optimal pasca-hoc, hanya informasi — bukan threshold operasi). Dengan 3–4 roti normal per split, FPR hanya bisa 0, 0.25/0.33, … — angka ini praktis = recall pada FPR 0 (sangat kasar).

| split | threshold | n (N/D) | AUROC | recall@FPR≤target (FPR) | akurasi | presisi | recall | TN FP FN TP |
|---|---|---|---|---|---|---|---|---|
| val | 50 | 6 (3/3) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 3 0 0 3 |
| test | 50 | 8 (4/4) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 4 0 0 4 |
| val | 48.9 | 6 (3/3) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 3 0 0 3 |
| test | 48.9 | 8 (4/4) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 4 0 0 4 |

## Cross-validation per grup (5-fold StratifiedGroupKFold, train+val, test tidak ikut)

Lebih bisa dipercaya daripada hold-out kecil di atas. Grup = `batch_id` digabung near-duplicate (aturan yang sama dengan split manifest); classifier di-fit ulang dari fold latih. Fold sama untuk scorer A dan B (seed config).

| metrik | nilai |
|---|---|
| AUROC per fold (mean ± std) | 0.988 ± 0.025 |
| akurasi @ threshold 50 (mean ± std) | 0.975 ± 0.050 |
| AUROC gabungan out-of-fold (n=40) | 0.992 |
| recall @ FPR ≤ 0.05 out-of-fold (FPR) | 1.00 (0.05) |

AUROC per fold: 0.938, 1.000, 1.000, 1.000, 1.000

## Kalibrasi threshold (F9, target FPR ≤ 0.05, tanpa test)

| sumber | n normal / kempes | langkah FPR | threshold | FPR | recall | FPR / recall @ config 50 |
|---|---|---|---|---|---|---|
| CV out-of-fold train+val (**saran**) | 20 / 20 | 0.050 | **48.9** | 0.05 | 1.00 | 0.05 / 1.00 |
| val saja (pembanding) | 3 / 3 | 0.333 | 48.3 | 0.00 | 1.00 | 0.00 / 1.00 |

Aturan: boleh k = floor(target · n normal) false positive → batas bawah = sedikit di atas skor normal tertinggi ke-(k+1); threshold lalu digeser ke tengah celah antara batas itu dan skor kempes terendah di atasnya (margin maksimum, FPR/recall data kalibrasi tidak berubah). **Kasar:** dengan 20 normal di CV satu roti = langkah FPR 0.05, jadi target 5% = paling banyak 1 normal di atas batas; di val (3 normal) target 5% = FPR 0. Saran ditulis ke `calibration.yaml`; config TIDAK diubah otomatis.

## Rincian per kelompok nama file

`medium-good` = masih layak walau permukaan tidak sempurna; `medium-bad` = tampak lumayan tapi permukaan terlalu kempes → reject. Kolom `salah` memakai threshold config.

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

## Kesalahan terbesar (margin salah-sisi dari threshold config)

**val**

(tidak ada kesalahan pada threshold 50)

**test**

(tidak ada kesalahan pada threshold 50)

**CV out-of-fold**

- `train-data/good-loaf/bagus (10).png` — label normal, skor 84.1 (margin 34.1)

## Latency & ukuran model

Rata-rata 4.0 ms/gambar (median 3.9, maks 4.4; n=14, setelah pemanasan) untuk segmentasi + representasi + skor, CPU, tanpa waktu baca file. Target PRD: ≤ 100 ms. File model: 0.00 MB.

## Domain check: foto konveyor asli (semua roti OK)

Foto: `train-data/domain-check/conveyor_good_01.webp` — roti terdeteksi: **15** (roti terpotong di tepi frame dibuang). Semua roti di foto ini OK menurut user, jadi **setiap REJECT = false positive** di domain konveyor. Tidak ada retraining/tuning dari foto ini. Gambar beranotasi ada di `outputs/baseline/` (tidak di-versi).

| threshold | nilai | REJECT (false positive) |
|---|---|---|
| config | 50 | 0/15 |
| kalibrasi | 48.9 | 0/15 |

Skor: min 0.9, median 5.4, maks 31.7. Per roti (urut atas→bawah): #1=7, #2=11, #3=11, #4=32, #5=3, #6=5, #7=8, #8=7, #9=2, #10=5, #11=6, #12=2, #13=4, #14=2, #15=1

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
