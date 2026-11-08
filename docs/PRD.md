# PRD — LoafLens (POC)

**Status dokumen:** Draft v0.1 · 2026-11-08
**Owner:** AI Tribe
**Tujuan satu kalimat:** membuktikan bahwa kamera + model ML lokal yang dilatih dari foto sendiri dapat memisahkan roti kempes dari roti normal di konveyor dengan biaya rendah.

## 1. Latar belakang
Client memiliki lini produksi roti di konveyor. Roti yang kempes harus di-reject sebagai produk afkir. Saat ini seleksi bergantung pada pengamatan manual (diasumsikan; konfirmasi ke client). Client meminta POC bagaimana teknologi dapat membantu seleksi ini.

## 2. Tujuan & non-tujuan
**Tujuan POC**
1. Repo kecil yang bisa dilatih dari kumpulan gambar di folder `data/`.
2. Model memberi **skor kempes** (0–100) per roti, bukan sekadar label biner.
3. Demo real-time: kamera diarahkan ke layar device lain yang memutar video konveyor + roti; aplikasi menandai tiap roti (OK / REJECT) dan skornya.
4. Bukti biaya rendah: inferensi di CPU, tanpa layanan cloud saat runtime.

**Non-tujuan (di luar POC)**
- Integrasi PLC / aktuator reject fisik (cukup keluarkan sinyal/log; hook opsional).
- Multi-lini, dashboard produksi, autentikasi, deployment pabrik.
- Deteksi cacat lain (gosong, retak, jamur) — hanya kempes, kecuali client menambah.
- Klaim akurasi produksi. Hasil POC = indikasi kelayakan.

## 3. Definisi "kempes" (HARUS dikonfirmasi client)
Placeholder: roti yang tingginya turun / bagian tengah cekung / bentuk kolaps dibanding roti standar, sehingga di bawah spesifikasi tampilan. Perlu: foto contoh batas (borderline), jumlah jenis roti (SKU), dan siapa yang memberi label ground truth.

**Definisi dari user (2026-11-08), dipakai untuk label `train-data/`:**
- **Diterima (normal):** `bagus` = jelas layak; `medium` (good-loaf) = masih layak walau permukaan tidak sempurna (cacat permukaan ringan).
- **Reject (kempes):** `bad` = kempes parah; `medium` (bad-loaf) = tampak lumayan tetapi permukaan terlalu kempes.
- **Fokus pembeda: kondisi/tekstur permukaan roti** — kerutan, lipatan, permukaan cekung/datar, kilap — bukan sesi/tanggal foto. Nomor sama dengan prefix berbeda = tingkat kualitas berbeda, bukan roti yang sama (split beku dipertahankan).
- Masih terbuka: definisi client (SKU, siapa pemberi label resmi, contoh borderline dari lini produksi).

## 4. Pengguna & skenario
- **Operator QC (target akhir):** melihat roti lewat, melihat tanda OK/REJECT.
- **Tim POC (kita):** melatih ulang model dari foto baru, menyetel threshold, menjalankan demo.
- **Skenario demo:** video konveyor diputar di device lain → kamera aplikasi menangkap layar → tiap roti dinilai real-time.

## 5. Pendekatan teknis
Dua scorer dengan antarmuka yang sama, dibandingkan pada hold-out set yang sama:

| | A. Baseline klasik | B. Anomaly detection |
|---|---|---|
| Input latih | roti normal + kempes (berlabel) | roti normal saja |
| Metode | fitur bentuk/warna (area, solidity, rasio dimensi, rata-rata L*, tekstur) → SVM/RF/IsolationForest | PatchCore / EfficientAd via anomalib (feature extractor pretrained) |
| Keunggulan | sangat ringan, mudah dijelaskan | tidak butuh banyak contoh kempes, ada heatmap |
| Risiko | kempes halus dari kamera atas sulit terlihat | lebih berat, sensitif pada kebersihan data normal |

Pipeline runtime: `capture → segment (ROI + isolasi roti) → track (satu roti = satu keputusan) → score → threshold → tampil/log`.

## 6. Requirement fungsional
| ID | Requirement | Prioritas |
|---|---|---|
| F1 | Training dari folder `data/normal/` dan `data/defect/` (kempes ikut dilatih untuk baseline A; anomaly B hanya normal) via CLI + config YAML | Must |
| F2 | Evaluasi pada hold-out set: AUROC, recall pada FPR tetap, confusion matrix, kurva ROC, daftar kesalahan terbesar | Must |
| F3 | Skor kontinu 0–100 per roti, threshold dari config | Must |
| F4 | Sumber input: `camera:<id>` dan `video:<file>` | Must |
| F5 | Segmentasi roti per frame (ROI dapat diatur) | Must |
| F6 | Tracking sederhana: satu roti diputuskan sekali saat melewati garis keputusan | Must |
| F7 | Overlay real-time: kotak, skor, OK/REJECT, FPS, counter | Must |
| F8 | Simpan snapshot + skor setiap roti REJECT ke `outputs/rejects/` | Should |
| F9 | Kalibrasi threshold dari set validasi (bukan test) dengan target FPR/recall | Should |
| F10 | Heatmap anomali pada roti yang di-reject (scorer B) | Should |
| F11 | Hook keluaran sinyal (stdout/file/serial) untuk aktuator masa depan | Could |
| F12 | Mode generator data sintetis untuk uji pipeline (berlabel jelas) | Could |

## 7. Requirement non-fungsional
- **Biaya:** inferensi CPU-only; tanpa GPU wajib. Target hardware referensi: laptop biasa; uji kelayakan Raspberry Pi 5 sebagai bonus.
- **Latency:** target ≤ 100 ms per roti pada hardware referensi (dikonfirmasi setelah tahu kecepatan konveyor, lihat §10).
- **Reproducibility:** seed tetap, config tersimpan bersama model.
- **Privasi data:** data client tidak masuk git; tidak ada upload eksternal.
- **Maintainability:** modul terpisah, test untuk logika non-kamera.

## 8. Data
- Sumber: foto roti asli dari client + (opsional) foto yang kita ambil sendiri.
- Folder: `data/normal/`, `data/defect/`, `data/raw/` (lihat CLAUDE.md). Saat ini (M1): `train-data/good-loaf/` (normal) dan `train-data/bad-loaf/` (kempes), diatur di config `data.*`.
- Kebutuhan awal (perkiraan, divalidasi di M1): ≥ 200 foto normal; ≥ 30–50 foto kempes (train baseline A + validasi/test); variasi pencahayaan, posisi, dan batch.
- Kedua folder (normal & kempes) di-split train/val/test per batch dengan aturan yang sama.
- Split per sesi/batch. Test set dibekukan sebelum tuning apa pun.
- **Perbedaan domain:** latih dari foto roti asli, uji dari video di layar → catat sebagai risiko; sediakan juga uji dengan file video langsung (tanpa rekam layar) untuk memisahkan efek model vs efek kamera/layar.

## 9. Metrik & acceptance criteria POC
Angka target adalah usulan awal, setel bersama client setelah melihat data.

| Kriteria | Target usulan |
|---|---|
| AUROC pada hold-out set | ≥ 0.90 |
| Recall roti kempes pada FPR ≤ 5% | ≥ 0.80 |
| Latency inferensi per roti (CPU laptop) | ≤ 100 ms |
| Demo end-to-end (video file) | tiap roti diputuskan tepat sekali; miscount < 2% |
| Demo kamera-ke-layar | berjalan stabil ≥ 5 menit; akurasi dilaporkan terpisah dari uji file video |
| Perbandingan A vs B | tabel metrik + rekomendasi tertulis |

Jika target tidak tercapai, laporan harus menjelaskan penyebab (data, sudut kamera, definisi kempes) dan langkah perbaikan.

## 10. Pertanyaan terbuka (perlu jawaban client / kita)
1. Definisi kempes yang disepakati + contoh borderline? Siapa pemberi label?
2. Berapa jenis roti (SKU)? Satu model per SKU atau satu model umum?
3. Kecepatan konveyor, jumlah roti/menit, jarak antar roti, warna latar konveyor?
4. Posisi kamera: dari atas saja atau ada sudut samping? Kempes terlihat jelas dari atas?
5. Berapa foto normal & kempes yang tersedia sekarang? Kapan datang?
6. Hardware target di lapangan (PC industri, mini PC, Raspberry Pi)?
7. Toleransi: lebih buruk mana — roti bagus ikut ter-reject (false positive) atau kempes lolos (false negative)?
8. Apakah POC cukup memberi sinyal, atau client ingin demo aktuator?

## 11. Risiko
| Risiko | Dampak | Mitigasi |
|---|---|---|
| Data kempes sedikit | Klasifikasi biasa gagal | Utamakan anomaly detection (B) |
| Foto "normal" terkontaminasi kempes | Model anomali menganggap kempes normal | Audit manual `data/normal/` |
| Moiré / refresh rate saat merekam layar | Skor tidak stabil | Uji video file langsung; atur sudut, kecerahan, exposure |
| Domain gap (foto asli vs video layar) | Akurasi demo lebih rendah dari uji | Laporkan terpisah; kumpulkan foto dari kamera yang sama |
| Kempes halus tak terlihat dari atas | Recall rendah | Diskusikan sudut/sensor tinggi dengan client |
| Beberapa roti saling menempel/tumpang tindih | Segmentasi gagal | Batasi POC ke roti terpisah; catat asumsi |
| Scope creep ke aktuator | Waktu habis | Non-tujuan; hanya hook F11 |

## 12. Milestone
| M | Deliverable | Verifikasi |
|---|---|---|
| M0 | Skeleton repo, config, CLAUDE.md, `.gitignore`, CI lokal (pytest, ruff), data README | `make test` hijau |
| M1 | Audit dataset + loader + split per batch + data sintetis untuk uji pipeline | Laporan audit; test loader |
| M2 | Scorer A (baseline) + evaluate.py + laporan metrik | Tabel metrik di `outputs/` |
| M3 | Scorer B (anomalib) + perbandingan A vs B + kalibrasi threshold | Laporan perbandingan |
| M4 | Pipeline real-time (segment+track+overlay), mode video & kamera, snapshot reject | Demo dengan video file lalu kamera-ke-layar |
| M5 (opsional) | Uji Raspberry Pi / export ONNX, hook sinyal | Benchmark latency |

## 13. Status
- [x] M0  - [x] M1  - [x] M2  - [ ] M3  - [ ] M4  - [ ] M5

- 2026-11-08 · M0 selesai: skeleton repo, `configs/{baseline,anomaly}.yaml` + validasi (`src/loaflens/config.py`), Makefile (`setup`/`test`/`lint`), `data/README.md`. Python **3.14** (numpy, opencv-python-headless, scikit-learn, torch, anomalib punya dukungan 3.14). Keputusan: training memakai normal **dan** kempes (scorer A supervised); scorer B tetap normal-only. Kamera `top`/`oblique` dan arah konveyor jadi field config.
- 2026-11-08 · M1 selesai: `src/loaflens/data.py` (scan → `train-data/manifest.csv`, split beku per grup, `load_split`, `load_image`) dan `src/loaflens/audit.py` (laporan `outputs/audit.md`, salinan [`docs/audit_m1.md`](audit_m1.md)); `make manifest`, `make audit`. Data dari user di `train-data/` (di-commit atas keputusan user): **24 normal + 24 kempes**, satu roti per foto, gaya studio (bukan konveyor). Split 70/15/15 → train 17+17, val 3+3, test 4+4 (normal+defect).
  Temuan audit: (1) **tidak ada info batch** → satu grup per gambar (+ near-duplicate digabung), split per batch belum mungkin; (2) **risiko shortcut**: ukuran gambar memisahkan kelas 100% (normal 312x417, kempes 360x355) — M2 wajib crop segmentasi + resize tetap, tanpa fitur ukuran gambar; statistik warna tepi/tengah hanya memisahkan ~60–70%; (3) prefix `medium` ada di **kedua** label dan pasangan `bagus (n)`/`medium (n)` di normal sangat mirip (kemungkinan varian roti yang sama; 6/12 pasangan beda split) — arti `medium` perlu dikonfirmasi user; (4) val/test sangat kecil (3–4 per label) → metrik sangat noisy, M2 sebaiknya pakai cross-validation per grup. Tidak ada generator data sintetis di M1 (data asli sudah ada; generator video sintetis ditunda ke M4).
- 2026-11-08 · M2 selesai: scorer A (baseline) — `segment.py` (saturasi HSV + Otsu → kontur terbesar → crop bbox, resize sisi terpanjang 192 px + pad; 0/48 gagal, alas roti pucat kadang terpotong sebagian), `features.py` (25 fitur: tekstur LBP/Laplacian/gradien/Canny/lipatan black-hat setelah blur σ=1.5, warna L\*a\*b\* + fraksi kilap, bentuk bebas skala; tanpa ukuran gambar asli), `score.py` (`Scorer` Protocol, `BaselineScorer` = StandardScaler + logistic regression, skor = 100·P(kempes), joblib), `train.py`, `evaluate.py`; `make train`, `make evaluate`; laporan [`docs/results_m2.md`](results_m2.md).
  Hasil (48 foto studio — indikasi kelayakan, bukan akurasi produksi): **test AUROC 1.000** (4+4, akurasi 8/8 pada threshold config 50, dievaluasi sekali, tanpa tuning), val AUROC 1.000 (3+3); **CV 5-fold per grup pada train+val: AUROC 0.988 ± 0.025**, akurasi 0.975 ± 0.050 (satu salah: `bagus (10)` skor 84 — roti kusam/pucat dengan sedikit retakan); kelompok borderline `medium-good` rata-rata skor 7, `medium-bad` 91 (CV out-of-fold). **Latency ~4.5 ms/roti** (CPU, segmentasi+fitur+prediksi). Tidak ada fitur tunggal dengan AUROC CV ≥ 0.98 (tertinggi `col_L_std` 0.91).
  Catatan: (1) hold-out sangat kecil, angka CV lebih bisa dipercaya; (2) model bersandar terutama pada kontras luminans/kilap, warna, LBP, dan aspek roti — fitur kerutan eksplisit lemah sendirian; (3) versi awal tanpa blur memakai tekstur halus yang peka resolusi (arah terbalik dari dugaan) → diganti blur + crop 192 berdasarkan uji resize & CV saja, sebelum test dibuka; (4) foto per folder diambil dengan kondisi berbeda (ukuran, latar, pencahayaan) sehingga angka hampir sempurna bisa sebagian mencerminkan perbedaan pengambilan foto, bukan kempes — uji sebenarnya adalah rekaman kamera/konveyor (M4); (5) threshold belum dikalibrasi (M3).
