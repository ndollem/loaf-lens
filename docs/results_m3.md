# Perbandingan scorer A vs B — LoafLens M3

> **Hasil pada 48 foto studio; indikasi kelayakan, bukan klaim akurasi produksi.**

- Tanggal: 2026-11-08. Target FPR: 0.05. Sumber: `outputs/baseline/metrics.json`, `outputs/anomaly/metrics.json`. Laporan lengkap: [`results_m3_baseline.md`](results_m3_baseline.md), [`results_m3_anomaly.md`](results_m3_anomaly.md).
- Kedua scorer memakai split beku dan fold CV yang sama. Threshold config dan threshold terkalibrasi keduanya TIDAK dipilih dari test. Hold-out sangat kecil → acuan utama CV; sinyal paling realistis = domain check foto konveyor (semua roti OK, setiap REJECT = false positive).

| metrik | A (baseline) | B (anomaly) |
|---|---|---|
| AUROC val (3+3) | 1.000 | 1.000 |
| AUROC test (4+4) | 1.000 | 1.000 |
| AUROC CV per fold (mean ± std) | 0.988 ± 0.025 | 1.000 ± 0.000 |
| AUROC CV gabungan out-of-fold (n=40) | 0.992 | 0.995 |
| recall @ FPR ≤ target, CV out-of-fold (FPR) | 1.00 (0.05) | 0.95 (0.00) |
| recall @ FPR ≤ target, test (FPR) | 1.00 (0.00) | 1.00 (0.00) |
| threshold config | 50 | 50 |
| threshold terkalibrasi (CV, FPR ≤ target) | 48.9 | 52.1 |
| akurasi val / test / CV @ config | 1.00 / 1.00 / 0.97 | 0.83 / 1.00 / 0.93 |
| test TN FP FN TP @ config | 4 0 0 4 | 4 0 0 4 |
| akurasi val / test / CV @ kalibrasi | 1.00 / 1.00 / 0.97 | 1.00 / 1.00 / 0.97 |
| test TN FP FN TP @ kalibrasi | 4 0 0 4 | 4 0 0 4 |
| latency per roti (ms, CPU, setelah pemanasan) | 4.0 | 24.8 |
| ukuran file model | 3 kB | 10.0 MB |
| domain check konveyor: REJECT config / kalibrasi (dari 15 roti OK) | 0 / 0 | 11 / 5 |
| domain check: skor min / median / maks | 1 / 5 / 32 | 48 / 51 / 58 |
| skor rata-rata bagus (CV out-of-fold) | 11.6 | 42.9 |
| skor rata-rata medium-good (CV out-of-fold) | 7.1 | 45.9 |
| skor rata-rata medium-bad (CV out-of-fold) | 90.7 | 56.0 |
| skor rata-rata bad (CV out-of-fold) | 92.6 | 59.3 |
| skor rata-rata bagus (test) | 2.5 | 36.8 |
| skor rata-rata medium-good (test) | 7.7 | 45.9 |
| skor rata-rata medium-bad (test) | 75.9 | 57.5 |
| skor rata-rata bad (test) | 97.5 | 59.6 |

<!-- REKOMENDASI: teks di bawah baris ini ditulis manual, dipertahankan compare.py -->

## Domain check: foto konveyor asli (`train-data/domain-check/conveyor_good_01.webp`)

Satu foto konveyor dari user (tampak atas/miring, belt mesh abu gelap, ±19 roti bulat mengilap,
**semua layak**). `segment_all` menemukan **15 roti** (roti yang terpotong di tepi atas/bawah dan
garis kuning lantai dibuang karena menyentuh tepi frame). Karena semua roti OK, setiap REJECT adalah
false positive di domain konveyor. Foto ini tidak dipakai untuk training/tuning apa pun.

- **Scorer A:** 0/15 REJECT pada threshold config 50 maupun terkalibrasi 48.9 (skor 1–32, median 5).
- **Scorer B:** **11/15 REJECT** pada threshold config 50 dan **5/15** pada threshold terkalibrasi 52.1
  (skor 48–58, median 51). Heatmap (`outputs/anomaly/domain_check_heatmaps.jpg`) menunjukkan patch
  "aneh" terutama di **kilap/highlight bagian atas roti** — tampilan yang tidak ada di 17 roti normal
  train (foto studio samping/miring). Ini domain gap, bukan kempes.

## Rekomendasi

1. **Untuk demo M4, pakai scorer A sebagai keputusan utama** (threshold config 50; saran kalibrasi
   48.9 praktis sama). A ringan (≈ 4 ms/roti, model 3 kB) dan tidak menolak roti bagus di foto
   konveyor. **Tetapi** A bersandar pada kilap/kontras luminans dan warna (`col_L_std`, `col_a_mean`,
   `col_b_std`): roti konveyor yang mengilap keemasan "terlihat seperti kelas bagus", sehingga 0/15 bisa
   benar karena alasan yang tidak sepenuhnya tepat. Recall A di domain konveyor **belum terukur**
   (belum ada foto roti kempes di konveyor).
2. **Scorer B belum layak dipakai di konveyor dengan bank dari foto studio** (5–11 dari 15 roti bagus
   ditolak). Di data studio B memisahkan kelas sama baiknya (CV AUROC 1.000) tetapi margin absolutnya
   sempit (normal 36–51 vs kempes 50–64), jadi sensitif terhadap pergeseran domain. B tetap pendekatan
   yang menjanjikan untuk lini produksi karena **hanya butuh roti normal**: refit memory bank dari
   ≥ 50–100 roti bagus yang direkam kamera konveyor sebenarnya (tanpa label kempes) adalah langkah
   termurah berikutnya. Heatmap B berguna sebagai diagnostik (F10) — mis. ditampilkan untuk roti REJECT.
3. **Data yang paling dibutuhkan:** foto/video dari kamera & sudut konveyor yang sama — roti bagus
   (untuk B dan untuk mengecek FPR A) dan beberapa roti kempes (untuk mengukur recall kedua scorer).
   Tanpa itu, angka studio di atas tidak bisa diterjemahkan ke akurasi lini.
4. Threshold di config **tidak diubah otomatis**. Saran (dari CV train+val, bukan test): A 48.9, B 52.1
   (`outputs/<run>/calibration.yaml`). Keputusan di tangan user.

## Catatan & keterbatasan

- Hold-out sangat kecil (val 3+3, test 4+4): AUROC 1.000 di kedua scorer tidak membedakan apa pun.
  CV (40 gambar) lebih informatif, tetap dengan rentang lebar.
- Foto studio per folder diambil dengan kondisi berbeda (latar, pencahayaan, sudut); angka hampir
  sempurna sebagian bisa mencerminkan perbedaan pengambilan foto. Scorer B mengisi latar dengan abu
  dan hanya memakai patch di dalam mask roti untuk meredam efek latar, tetapi pencahayaan/kilap tetap
  ikut.
- Domain check hanya **satu foto** dengan roti bagus saja → hanya mengukur false positive, bukan recall.
- Langkah FPR kalibrasi kasar (CV: 20 normal → 0.05 per roti; val: 3 normal → 0.33).
- **Riwayat aturan kalibrasi (transparansi):** versi pertama memilih threshold tepat di atas skor
  normal ke-(k+1). Pada CV scorer A itu menghasilkan 26.4 — jauh di bawah semua skor kempes (min 71.7)
  tanpa menambah recall CV — sehingga aturan diganti ke tengah celah antara batas normal dan skor kempes
  terendah di atasnya (margin maksimum; FPR/recall CV tidak berubah). Alasannya dari data CV, tetapi
  perubahan ini dibuat **setelah** hasil domain check versi pertama terlihat. Hasil versi pertama:
  A 26.4 → 1/15 REJECT konveyor, B 51.0 → 7/15; versi sekarang: A 48.9 → 0/15, B 52.1 → 5/15. Test
  set tidak berubah hasilnya (8/8 benar di kedua versi untuk kedua scorer).
