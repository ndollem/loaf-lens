# CLAUDE.md — LoafLens

Nama proyek: **LoafLens** · repo: `loaf-lens` · package Python: `loaflens`.

POC visi komputer low-cost untuk mendeteksi roti **kempes** (afkir) di konveyor.
Detail tujuan & requirement: `docs/PRD.md`. Baca itu dulu.

## Prinsip
- **Lokal & murah**: inferensi harus jalan di CPU (laptop / mini PC / Raspberry Pi 5 class). Tanpa LLM, tanpa API cloud berbayar saat inferensi.
- **Model dilatih dari data kita sendiri** di `data/`. Pretrained backbone ImageNet boleh sebagai feature extractor (dijelaskan di README); tidak boleh memanggil layanan eksternal saat runtime.
- **Skor kontinu** (0–100, makin tinggi = makin kempes/anomali), keputusan reject = skor ≥ threshold di config.
- **Reproducible**: seed tetap, config YAML, hasil eksperimen tersimpan di `outputs/` (gitignored).
- Jujur soal hasil: laporkan metrik apa adanya; jangan tuning threshold di test set.

## Stack (default, boleh diusulkan alternatif dengan alasan)
- Python 3.14 (lihat `.python-version`), `uv` atau `pip` + `requirements.txt` terkunci
- OpenCV (capture, segmentasi, tracking sederhana)
- Baseline klasik: scikit-learn (fitur bentuk/warna → SVM/RandomForest/IsolationForest)
- Anomaly detection: PatchCore-lite sendiri (`src/loaflens/anomaly.py`) di atas `torch` + `torchvision` **CPU** (index CPU PyTorch via `uv --torch-backend cpu`); `anomalib` tidak dipakai (disetujui user, M3). Export ONNX/OpenVINO bila perlu (M5)
- UI demo: jendela OpenCV (`opencv-python`, build GUI; bukan `-headless`), `--headless` tanpa display
- Test: `pytest`; lint: `ruff`

## Struktur repo
```
loaf-lens/
├── CLAUDE.md
├── README.md
├── docs/            # PRD.md, USAGE.md (panduan pengguna), KICKOFF_PROMPT.md, catatan eksperimen
├── configs/         # *.yaml (threshold, ROI, model, source)
├── data/            # LIHAT ATURAN DATA di bawah
│   ├── raw/         # foto asli dari client (gitignored)
│   ├── normal/      # roti OK untuk training (gitignored)
│   ├── defect/      # roti kempes: train (scorer A) + val/test, split per batch (gitignored)
│   └── README.md    # cara mengisi, konvensi penamaan, ukuran dataset
├── train-data/      # foto dari user, DI-COMMIT atas keputusan user (~6.6 MB)
│   ├── good-loaf/   # label normal
│   ├── bad-loaf/    # label defect (kempes)
│   ├── domain-check/ # foto konveyor asli (+ labels.json posisi roti kempes) — HANYA domain check/demo, tidak dilatih
│   └── manifest.csv # split beku (python -m loaflens.data)
├── src/loaflens/
│   ├── config.py    # load + validasi configs/*.yaml
│   ├── data.py      # scan → manifest + split per grup, loader gambar
│   ├── audit.py     # laporan audit dataset (outputs/audit.md, docs/audit_m1.md)
│   ├── capture.py   # sumber: camera / video file
│   ├── segment.py   # segment (1 roti/foto) + segment_all (banyak roti/frame, ROI, buang tepi)
│   ├── track.py     # satu roti = satu keputusan (garis keputusan / centroid tracking)
│   ├── features.py  # fitur bentuk/warna untuk baseline
│   ├── score.py     # antarmuka Scorer + scorer A (baseline) + new_scorer/load_scorer
│   ├── anomaly.py   # scorer B: PatchCore-lite (ResNet18 beku + memory bank + heatmap)
│   ├── train.py     # CLI training
│   ├── evaluate.py  # CLI evaluasi (+ --domain-check foto konveyor)
│   ├── calibrate.py # CLI saran threshold dari CV train+val (F9)
│   ├── compare.py   # CLI tabel A vs B → docs/results_m3.md
│   ├── synth.py     # CLI video demo SYNTHETIC dari foto konveyor + ground truth (M4)
│   └── app.py       # CLI/UI demo real-time (+ --gt banding ground truth)
├── tests/
├── samples/         # video demo (gitignored; buat ulang: make demo-video)
└── outputs/         # model, laporan, snapshot reject (gitignored)
```

## Aturan data
- **Jangan commit** gambar/video asli, model besar, atau output eksperimen. Pastikan `.gitignore` mencakup `data/*`, `outputs/`, `*.onnx`, `*.pt`, kecuali `data/README.md`.
- **Pengecualian:** `train-data/` berisi foto dari user yang sengaja di-commit atas keputusan user (kecil, ~6.6 MB). Data/video asli lain dan model tetap tidak di-commit. Folder aktif diatur di `data.normal_dir`/`data.defect_dir` config.
- Pisahkan split **per batch/sesi pengambilan**, bukan acak per gambar, untuk menghindari kebocoran (frame mirip di train & test).
- Training memakai **kedua** folder: `normal/` dan `defect/` di-split train/val/test per batch. Scorer A (baseline) dilatih supervised dari normal + kempes; scorer B (anomaly) `fit` dari normal saja, kempes hanya untuk val/test.
- Folder normal (`train-data/good-loaf/`, atau `data/normal/` bila dipakai) hanya berisi roti yang benar-benar normal (konfirmasi manusia). Satu roti kempes yang nyelip ke normal merusak model anomaly.
- Catat metadata (tanggal, kamera, pencahayaan, jenis roti) di manifest (`data.manifest` di config, saat ini `train-data/manifest.csv`).

## Konvensi kode
- Type hints, docstring singkat, fungsi kecil; I/O (kamera/file) dipisah dari logika skor agar bisa di-test.
- Semua parameter di `configs/*.yaml`; CLI memakai `argparse`/`typer`.
- Setiap modul skor mengimplementasikan interface yang sama: `fit(images)`, `score(image) -> float`, `save/load`.
- Log dengan `logging`, bukan `print`.

## Cara kerja dengan saya
- Mulai dari rencana, bukan kode. Kerjakan satu milestone per giliran, tunggu konfirmasi.
- Sebelum menambah dependency besar, jelaskan alasan + alternatif ringan.
- Setelah perubahan: jalankan `pytest` & `ruff`, tampilkan hasil, update status di `docs/PRD.md`.
- Jika data belum ada, buat generator data sintetis/dummy HANYA untuk menguji pipeline dan beri label jelas "SYNTHETIC — bukan untuk evaluasi akurasi".
- Bila ragu soal asumsi domain (jenis roti, kamera, kecepatan konveyor), tanyakan; jangan menebak diam-diam.

## Perintah (diisi saat dibuat)
```
make setup | make test | make lint | make format | make lock
make manifest   # python -m loaflens.data --config configs/baseline.yaml (tidak menimpa; --force = split ulang)
make audit      # python -m loaflens.audit --config configs/baseline.yaml
make train      # python -m loaflens.train --config configs/baseline.yaml → outputs/baseline/
make evaluate   # python -m loaflens.evaluate --config configs/baseline.yaml (+ docs/results_m3_baseline.md)
make train-anomaly     # python -m loaflens.train --config configs/anomaly.yaml → outputs/anomaly/
make evaluate-anomaly  # python -m loaflens.evaluate --config configs/anomaly.yaml (+ docs/results_m3_anomaly.md)
make calibrate [CONFIG=configs/<x>.yaml]   # default baseline; python -m loaflens.calibrate --config ... → calibration.yaml
make compare           # python -m loaflens.compare configs/baseline.yaml configs/anomaly.yaml → docs/results_m3.md
make domain-check [CONFIG=configs/<x>.yaml] # default baseline; python -m loaflens.evaluate --config ... --domain-check <foto>
make demo-video        # python -m loaflens.synth --config configs/baseline.yaml --out samples/demo.mp4 (SYNTHETIC + demo.gt.json)
make demo              # python -m loaflens.app --config configs/baseline.yaml --source video:samples/demo.mp4 --headless --out outputs/demo_out.mp4 --gt samples/demo.gt.json
make demo-live         # python -m loaflens.app --config configs/baseline.yaml --source video:samples/demo.mp4 (jendela; q/ESC)
make camera CAMERA=0   # python -m loaflens.app --config configs/baseline.yaml --source camera:0
```
