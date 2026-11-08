# Hasil evaluasi scorer B (anomaly: PatchCore-lite, ResNet18 beku) — LoafLens

> **Hasil pada 48 foto studio; indikasi kelayakan, bukan klaim akurasi produksi.**

- Tanggal: 2026-11-08
- PatchCore-lite: crop roti 224 px (latar diisi abu) → ResNet18 pretrained ImageNet **beku** (feature extractor; layer2 + layer3, avg-pool 3x3) → patch di dalam mask roti (erosi 8 px) → memory bank 6513 patch dari roti **normal saja** → jarak kNN (k=1), skor gambar = `max` jarak patch.
- Skor 0–100 = clip(50 · mentah / ref); ref = p95 skor leave-one-image-out roti normal train = 2.329. Skor 50 ≈ seaneh roti normal train paling aneh; > 50 = di luar variasi normal yang pernah dilihat.
- Train: 17 gambar (split `train` manifest beku; label normal). Threshold config: **50**; threshold terkalibrasi (CV train+val, FPR ≤ 0.05): **52.1** (reject jika skor ≥ threshold).
- **Test dievaluasi dengan dua threshold yang keduanya TIDAK dipilih dari test** (config & kalibrasi CV train+val). Fitur dan hyperparameter tidak disetel dari test.
- Gagal segmentasi pada val+test (fallback seluruh gambar): 0

## Hold-out (model dilatih dari train saja)

Recall @ FPR ≤ 0.05 dihitung dari kurva ROC (threshold optimal pasca-hoc, hanya informasi — bukan threshold operasi). Dengan 3–4 roti normal per split, FPR hanya bisa 0, 0.25/0.33, … — angka ini praktis = recall pada FPR 0 (sangat kasar).

| split | threshold | n (N/D) | AUROC | recall@FPR≤target (FPR) | akurasi | presisi | recall | TN FP FN TP |
|---|---|---|---|---|---|---|---|---|
| val | 50 | 6 (3/3) | 1.000 | 1.00 (0.00) | 0.83 | 0.75 | 1.00 | 2 1 0 3 |
| test | 50 | 8 (4/4) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 4 0 0 4 |
| val | 52.1 | 6 (3/3) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 3 0 0 3 |
| test | 52.1 | 8 (4/4) | 1.000 | 1.00 (0.00) | 1.00 | 1.00 | 1.00 | 4 0 0 4 |

## Cross-validation per grup (5-fold StratifiedGroupKFold, train+val, test tidak ikut)

Lebih bisa dipercaya daripada hold-out kecil di atas. Grup = `batch_id` digabung near-duplicate (aturan yang sama dengan split manifest); memory bank di-fit dari roti normal fold latih saja; semua gambar fold uji diskor. Fold sama untuk scorer A dan B (seed config).

| metrik | nilai |
|---|---|
| AUROC per fold (mean ± std) | 1.000 ± 0.000 |
| akurasi @ threshold 50 (mean ± std) | 0.925 ± 0.061 |
| AUROC gabungan out-of-fold (n=40) | 0.995 |
| recall @ FPR ≤ 0.05 out-of-fold (FPR) | 0.95 (0.00) |

AUROC per fold: 1.000, 1.000, 1.000, 1.000, 1.000

## Kalibrasi threshold (F9, target FPR ≤ 0.05, tanpa test)

| sumber | n normal / kempes | langkah FPR | threshold | FPR | recall | FPR / recall @ config 50 |
|---|---|---|---|---|---|---|
| CV out-of-fold train+val (**saran**) | 20 / 20 | 0.050 | **52.1** | 0.00 | 0.95 | 0.10 / 0.95 |
| val saja (pembanding) | 3 / 3 | 0.333 | 52.7 | 0.00 | 1.00 | 0.33 / 1.00 |

Aturan: boleh k = floor(target · n normal) false positive → batas bawah = sedikit di atas skor normal tertinggi ke-(k+1); threshold lalu digeser ke tengah celah antara batas itu dan skor kempes terendah di atasnya (margin maksimum, FPR/recall data kalibrasi tidak berubah). **Kasar:** dengan 20 normal di CV satu roti = langkah FPR 0.05, jadi target 5% = paling banyak 1 normal di atas batas; di val (3 normal) target 5% = FPR 0. Saran ditulis ke `calibration.yaml`; config TIDAK diubah otomatis.

## Rincian per kelompok nama file

`medium-good` = masih layak walau permukaan tidak sempurna; `medium-bad` = tampak lumayan tapi permukaan terlalu kempes → reject. Kolom `salah` memakai threshold config.

**val**

| kelompok | n | skor rata-rata | min–max | salah |
|---|---|---|---|---|
| bagus | 2 | 43.9 | 42.8–44.9 | 0 |
| medium-good | 1 | 51.2 | 51.2–51.2 | 1 |
| medium-bad | 2 | 55.8 | 54.4–57.2 | 0 |
| bad | 1 | 62.5 | 62.5–62.5 | 0 |

**test**

| kelompok | n | skor rata-rata | min–max | salah |
|---|---|---|---|---|
| bagus | 1 | 36.8 | 36.8–36.8 | 0 |
| medium-good | 3 | 45.9 | 44.8–47.8 | 0 |
| medium-bad | 1 | 57.5 | 57.5–57.5 | 0 |
| bad | 3 | 59.6 | 55.3–63.3 | 0 |

**CV out-of-fold (train+val)**

| kelompok | n | skor rata-rata | min–max | salah |
|---|---|---|---|---|
| bagus | 11 | 42.9 | 36.0–48.0 | 0 |
| medium-good | 9 | 45.9 | 40.5–51.3 | 2 |
| medium-bad | 11 | 56.0 | 49.6–59.1 | 1 |
| bad | 9 | 59.3 | 54.3–64.2 | 0 |

## Kesalahan terbesar (margin salah-sisi dari threshold config)

**val**

- `train-data/good-loaf/medium (7).png` — label normal, skor 51.2 (margin 1.2)

**test**

(tidak ada kesalahan pada threshold 50)

**CV out-of-fold**

- `train-data/good-loaf/medium (0).png` — label normal, skor 51.3 (margin 1.3)
- `train-data/good-loaf/medium (7).png` — label normal, skor 50.9 (margin 0.9)
- `train-data/bad-loaf/medium (1).png` — label defect, skor 49.6 (margin 0.4)

## Latency & ukuran model

Rata-rata 24.8 ms/gambar (median 24.4, maks 28.7; n=14, setelah pemanasan) untuk segmentasi + representasi + skor, CPU, tanpa waktu baca file. Target PRD: ≤ 100 ms. File model: 10.01 MB (memory bank float32; bobot backbone ResNet18 ~47 MB terpisah di cache torch).

## Domain check: foto konveyor asli (semua roti OK)

Foto: `train-data/domain-check/conveyor_good_01.webp` — roti terdeteksi: **15** (roti terpotong di tepi frame dibuang). Semua roti di foto ini OK menurut user, jadi **setiap REJECT = false positive** di domain konveyor. Tidak ada retraining/tuning dari foto ini. Gambar beranotasi ada di `outputs/anomaly/` (tidak di-versi).

| threshold | nilai | REJECT (false positive) |
|---|---|---|
| config | 50 | 11/15 |
| kalibrasi | 52.1 | 5/15 |

Skor: min 47.8, median 51.0, maks 57.9. Per roti (urut atas→bawah): #1=58, #2=50, #3=53, #4=51, #5=54, #6=52, #7=58, #8=51, #9=49, #10=48, #11=49, #12=52, #13=50, #14=51, #15=49

## Catatan & keterbatasan

- Data sangat kecil (24 + 24 foto; val 3+3, test 4+4): satu gambar menggeser metrik hold-out 12–17 poin. Pakai angka CV sebagai acuan utama, itu pun dengan rentang lebar.
- Foto studio: tiap folder diambil dengan kondisi berbeda (ukuran gambar, latar, pencahayaan/warna). Ukuran gambar sudah dinetralkan (crop + resize tetap), tetapi angka yang (hampir) sempurna bisa sebagian mencerminkan **perbedaan pengambilan foto**, bukan kempes. Sinyal paling realistis: domain check foto konveyor (di atas) dan rekaman kamera (M4).
- Tidak ada info batch/sesi: grup = satu gambar (+ near-duplicate). Kebocoran antar foto dari sesi yang sama tidak bisa disingkirkan.
- Hasil CV bergantung pada pembagian fold: dengan 40 gambar (±8 per fold), satu gambar salah urut menggeser AUROC fold ±0.06.
- Backbone ResNet18 pretrained ImageNet dipakai **hanya sebagai feature extractor beku** (tidak di-fine-tune); bobot diunduh sekali saat training ke cache torch. Saat inferensi tidak ada panggilan jaringan — bila bobot tidak ada, scorer gagal dengan pesan jelas.
- Memory bank hanya dari 17 roti normal train (20 per fold CV) dengan sudut kamera bervariasi: "normal" yang dikenal model sangat sempit, sehingga roti normal dengan sudut/pencahayaan baru mudah dianggap anomali (false positive). Skor 50 adalah batas variasi normal train, bukan probabilitas.
- Kalibrasi ref (p95 leave-one-out) dari 17 nilai: praktis ditentukan 1–2 roti normal paling aneh; ref per fold CV berbeda-beda.
- Heatmap menunjukkan patch dengan jarak terbesar ke bank; tidak selalu bertepatan dengan area kempes (bisa kilap, tepi crop, atau alas roti yang terpotong segmentasi).
- Overlay heatmap: JET biru = skor patch ≤ 25, merah = ≥ 75 (skala 0–100 yang sama dengan skor gambar); hanya patch di dalam mask roti yang dierosi.

Artefak (`roc_test.png`, `segment_debug.jpg`, `scores.csv`, `calibration.yaml`, `metrics.json`, `heatmaps/`) ada di `outputs/anomaly/` (tidak di-versi).
