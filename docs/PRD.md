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
| Metode | fitur bentuk/warna (area, solidity, rasio dimensi, rata-rata L*, tekstur) → SVM/RF/IsolationForest | PatchCore-lite (ResNet18 pretrained beku sebagai feature extractor, torch CPU; anomalib tidak dipakai — keputusan M3) |
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
| M3 | Scorer B (anomaly; PatchCore-lite, torch CPU) + perbandingan A vs B + kalibrasi threshold | Laporan perbandingan |
| M4 | Pipeline real-time (segment+track+overlay), mode video & kamera, snapshot reject | Demo dengan video file lalu kamera-ke-layar |
| M5 (opsional) | Uji Raspberry Pi / export ONNX, hook sinyal | Benchmark latency |

## 13. Status
- [x] M0  - [x] M1  - [x] M2  - [x] M3  - [x] M4 (uji file video; kamera-ke-layar menunggu user)  - [ ] M5

- 2026-11-08 · M0 selesai: skeleton repo, `configs/{baseline,anomaly}.yaml` + validasi (`src/loaflens/config.py`), Makefile (`setup`/`test`/`lint`), `data/README.md`. Python **3.14** (numpy, opencv-python-headless, scikit-learn, torch, anomalib punya dukungan 3.14). Keputusan: training memakai normal **dan** kempes (scorer A supervised); scorer B tetap normal-only. Kamera `top`/`oblique` dan arah konveyor jadi field config.
- 2026-11-08 · M1 selesai: `src/loaflens/data.py` (scan → `train-data/manifest.csv`, split beku per grup, `load_split`, `load_image`) dan `src/loaflens/audit.py` (laporan `outputs/audit.md`, salinan [`docs/audit_m1.md`](audit_m1.md)); `make manifest`, `make audit`. Data dari user di `train-data/` (di-commit atas keputusan user): **24 normal + 24 kempes**, satu roti per foto, gaya studio (bukan konveyor). Split 70/15/15 → train 17+17, val 3+3, test 4+4 (normal+defect).
  Temuan audit: (1) **tidak ada info batch** → satu grup per gambar (+ near-duplicate digabung), split per batch belum mungkin; (2) **risiko shortcut**: ukuran gambar memisahkan kelas 100% (normal 312x417, kempes 360x355) — M2 wajib crop segmentasi + resize tetap, tanpa fitur ukuran gambar; statistik warna tepi/tengah hanya memisahkan ~60–70%; (3) prefix `medium` ada di **kedua** label dan pasangan `bagus (n)`/`medium (n)` di normal sangat mirip (kemungkinan varian roti yang sama; 6/12 pasangan beda split) — arti `medium` perlu dikonfirmasi user; (4) val/test sangat kecil (3–4 per label) → metrik sangat noisy, M2 sebaiknya pakai cross-validation per grup. Tidak ada generator data sintetis di M1 (data asli sudah ada; generator video sintetis ditunda ke M4).
- 2026-11-08 · M2 selesai: scorer A (baseline) — `segment.py` (saturasi HSV + Otsu → kontur terbesar → crop bbox, resize sisi terpanjang 192 px + pad; 0/48 gagal, alas roti pucat kadang terpotong sebagian), `features.py` (25 fitur: tekstur LBP/Laplacian/gradien/Canny/lipatan black-hat setelah blur σ=1.5, warna L\*a\*b\* + fraksi kilap, bentuk bebas skala; tanpa ukuran gambar asli), `score.py` (`Scorer` Protocol, `BaselineScorer` = StandardScaler + logistic regression, skor = 100·P(kempes), joblib), `train.py`, `evaluate.py`; `make train`, `make evaluate`; laporan [`docs/results_m2.md`](results_m2.md).
  Hasil (48 foto studio — indikasi kelayakan, bukan akurasi produksi): **test AUROC 1.000** (4+4, akurasi 8/8 pada threshold config 50, dievaluasi sekali, tanpa tuning), val AUROC 1.000 (3+3); **CV 5-fold per grup pada train+val: AUROC 0.988 ± 0.025**, akurasi 0.975 ± 0.050 (satu salah: `bagus (10)` skor 84 — roti kusam/pucat dengan sedikit retakan); kelompok borderline `medium-good` rata-rata skor 7, `medium-bad` 91 (CV out-of-fold). **Latency ~4.5 ms/roti** (CPU, segmentasi+fitur+prediksi). Tidak ada fitur tunggal dengan AUROC CV ≥ 0.98 (tertinggi `col_L_std` 0.91).
  Catatan: (1) hold-out sangat kecil, angka CV lebih bisa dipercaya; (2) model bersandar terutama pada kontras luminans/kilap, warna, LBP, dan aspek roti — fitur kerutan eksplisit lemah sendirian; (3) versi awal tanpa blur memakai tekstur halus yang peka resolusi (arah terbalik dari dugaan) → diganti blur + crop 192 berdasarkan uji resize & CV saja, sebelum test dibuka; (4) foto per folder diambil dengan kondisi berbeda (ukuran, latar, pencahayaan) sehingga angka hampir sempurna bisa sebagian mencerminkan perbedaan pengambilan foto, bukan kempes — uji sebenarnya adalah rekaman kamera/konveyor (M4); (5) threshold belum dikalibrasi (M3).
- 2026-11-08 · M3 selesai: scorer B "PatchCore-lite" (`src/loaflens/anomaly.py`: crop roti `segment.crop` 224 px, latar diisi abu → ResNet18 ImageNet **beku** layer2+layer3, avg-pool 3x3 → patch di dalam mask roti tererosi → memory bank 6513 patch dari 17 roti normal train → jarak 1-NN, skor gambar = patch maks; skor 0–100 = clip(50·mentah/ref), ref = p95 skor leave-one-out normal train; heatmap F10), kalibrasi threshold F9 (`calibrate.py`, dari CV out-of-fold train+val, bukan test), perbandingan A vs B (`compare.py` → [`docs/results_m3.md`](results_m3.md)), `segment_all` (banyak roti per frame, ROI, buang objek di tepi, abaikan bayangan gelap) + domain check foto konveyor asli `train-data/domain-check/conveyor_good_01.webp`. Dependency baru: `torch 2.15.0` + `torchvision 0.29.2` CPU (index CPU PyTorch via `uv --torch-backend cpu`; Linux +cpu tanpa CUDA, macOS arm64 wheel biasa; `.venv` +±0.9 GB). Evaluasi refactor: kedua scorer berbagi kode CV/laporan (`represent`/`fit_repr`/`score_repr`); laporan scorer A kini `docs/results_m3_baseline.md` (snapshot M2 tetap), angka A identik dengan M2.
  Hasil (48 foto studio — indikasi kelayakan): **A** test AUROC 1.000, CV AUROC 0.988 ± 0.025, ≈ 4 ms/roti, model 3 kB; **B** test AUROC 1.000, CV AUROC 1.000 ± 0.000 (gabungan OOF 0.995), ≈ 20–28 ms/roti, model 10 MB (+ bobot ResNet18 47 MB di cache). Test 8/8 benar untuk keduanya pada threshold config 50 dan terkalibrasi. B memisahkan urutan dengan baik tetapi marginnya sempit (normal 36–51 vs kempes 50–64, CV). **Threshold terkalibrasi (FPR ≤ 0.05, CV):** A **48.9**, B **52.1** (val saja: 48.3 / 52.7) — config tidak diubah, keputusan user. **Domain check konveyor (15 roti terdeteksi, semua bagus):** A **0/15** REJECT (config & kalibrasi); B **11/15** @50 dan **5/15** @52.1 — heatmap menunjuk kilap atas roti (tampilan yang tidak ada di normal studio) → domain gap. Rekomendasi: A sebagai keputusan utama demo M4, heatmap B sebagai diagnostik; B perlu memory bank dari roti bagus kamera konveyor sebelum dipakai.
  Catatan: (1) hold-out sangat kecil, AUROC 1.000 tidak membedakan scorer; (2) A bersandar pada kilap/warna — 0/15 di konveyor bisa "benar karena alasan kurang tepat", recall di konveyor belum terukur (belum ada foto kempes konveyor); (3) domain check hanya 1 foto roti bagus (hanya FPR); (4) langkah FPR kalibrasi kasar (CV 0.05, val 0.33); (5) aturan kalibrasi diganti dari "tepat di atas normal ke-(k+1)" ke "tengah celah ke kempes terendah" setelah melihat A = 26.4 (di bawah semua kempes tanpa tambahan recall CV) — perubahan dibuat setelah domain check versi pertama terlihat; versi pertama: A 26.4 → 1/15, B 51.0 → 7/15 (dilaporkan di `docs/results_m3.md`); (6) parameter segmentasi multi-objek (`obj_*`) disetel dengan mata pada foto konveyor (bukan metrik); parameter scorer B adalah default awal, tidak di-tuning.
- 2026-11-08 · M4 selesai (uji file video): pipeline real-time `src/loaflens/app.py` (`capture.py` sumber `video:`/`camera:`, `segment_all` + ROI, `track.py` tracker centroid tetangga terdekat + garis keputusan = satu keputusan per roti saat centroid melintas, skor **scorer A**, overlay F7 kotak/id/skor/OK-REJECT/garis/FPS/counter/banner SYNTHETIC, F8 snapshot + `outputs/rejects/rejects.csv`, `outputs/app/decisions.csv`, log `DECISION ...` per roti sebagai hook F11 sederhana), section config `app:` + `track.max_distance/max_missed`, `track.direction` → `top_to_bottom` (sesuai video demo). Video demo **SYNTHETIC** `src/loaflens/synth.py` (`make demo-video`): 2 foto `train-data/domain-check/` di-scroll (941x900, 30 fps, 949 frame, 8 px/frame, 2 putaran) + ground truth posisi (`labels.json`). Dependency: `opencv-python-headless` → `opencv-python` (GUI; keputusan user); `make lock` gagal karena `download-r2.pytorch.org` diblokir di sandbox → pin diganti manual (versi sama 5.0.0.93). Laporan [`docs/results_m4.md`](results_m4.md).
  Hasil (SYNTHETIC — bukan untuk evaluasi akurasi): **70/70 roti diputuskan tepat sekali, miscount 0%** (0 duplikat, 0 miss; juga 35/35 pada 4/16/24 px/frame); kempes **5/5** di-REJECT per putaran, roti OK ikut REJECT **0/30**, borderline #12/#19 OK (skor 6, 29–30); kempes terendah 59 vs OK tertinggi 36. **±80 FPS** loop penuh (64–87), segment+track ≈ 6 ms/frame, skor ≈ 6 ms/roti (CPU 4 vCPU).
  Catatan: (1) scroll foto diam (tanpa blur/cahaya/kamera) → hanya menguji logika pipeline & penghitungan; (2) skor memakai 2 foto yang sama dengan domain check M3 → bukan bukti akurasi baru; putaran 2 = konten sama; (3) **demo kamera-ke-layar (≥ 5 menit, akurasi dilaporkan terpisah) belum dijalankan — menunggu user** (langkah di README); jalur kamera & jendela tidak bisa diuji penuh di container (jendela hanya di bawah Xvfb, 90 frame).
- 2026-11-08 · Uji user: app berjalan di MacBook user dengan kamera langsung. **Uji video konveyor 10 detik** dari user (`samples/conveyor_real_10s.mp4`, 360×640, 24 fps; tidak di-commit), scorer A, config apa adanya: 9 roti diputuskan tepat sekali (0 duplikat); kempes 1/1 REJECT, tetapi **roti OK ikut REJECT 7/8** → gagal di video ini. Diagnosis: resolusi (lebar roti ±55–75 px vs ±130–190 px di foto); foto konveyor yang diperkecil ke skala video menaikkan salah tolak 0/15 → 8/15 dan 0/15 → 5/15. Rekomendasi: lebar roti di frame ≥ ±130 px dan/atau data latih dari kamera konveyor + augmentasi resolusi. Laporan [`docs/results_real_video.md`](results_real_video.md); README ditulis ulang dengan screenshot bukti (`docs/images/`).
- 2026-11-08 · **Perbaikan resolusi scorer A** (Run 2 video). Metode: fitur dari crop kanonik **64 px** (dulu 192; parameter piksel fitur diskalakan 1/3: `erode_px` 3, `crease_px` 2, `texture_blur_sigma` 0.5) + **augmentasi resolusi saat fit** (`model.params.augment`: 4 salinan per foto latih, roti diperkecil ke 50–200 px log-uniform + JPEG 40–90, RNG = seed + CRC gambar; di CV hanya di fold latih). Dipilih **tanpa melihat video**, hanya dari CV studio train+val (bersih dan fold uji yang diperkecil + JPEG) dan dua foto domain check yang diperkecil ×1.0/0.6/0.45/0.383/0.3 (foto ini sudah pernah dilihat → berfungsi sebagai validasi, bukan uji independen). Validasi @50, skala video ×0.383: foto bagus OK ter-reject 8/15 → **0/15**, foto campuran OK ter-reject 5/15 → **0/15**, kempes 5/5 → **5/5**; ×0.3 (roti ~40–57 px) masih 2/16 dan 1/15. Studio: CV AUROC 0.988 ± 0.025 → **1.000 ± 0.000**, test (dievaluasi sekali) 8/8 → 8/8. Kalibrasi CV menyarankan **40.4** (dulu 48.9); config tetap **50** (di 50 CV FPR 0/20, recall 20/20; config tidak diubah otomatis — keputusan user). **Video dijalankan sekali setelah metode dibekukan:** 9 keputusan, kempes **1/1** (skor 99), roti OK ter-reject **0/8** @50 (Run 1: 7/8); catatan: roti #1 skor 47.7 → akan REJECT pada saran 40.4 (1/8). n kecil, label video visual (Claude). Baru: `--domain-scale` di `loaflens.evaluate` (+ `DOMAIN_SCALE` Makefile, banding `labels.json`). Laporan [`docs/results_real_video.md`](results_real_video.md) (Run 1 dipertahankan); gambar `docs/images/` dibuat ulang dari Run 2.
