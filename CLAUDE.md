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
- Python 3.11, `uv` atau `pip` + `requirements.txt` terkunci
- OpenCV (capture, segmentasi, tracking sederhana)
- Baseline klasik: scikit-learn (fitur bentuk/warna → SVM/RandomForest/IsolationForest)
- Anomaly detection: `anomalib` (PatchCore / EfficientAd), export ONNX/OpenVINO bila perlu
- UI demo: OpenCV window atau Gradio/Streamlit (pilih yang paling ringan)
- Test: `pytest`; lint: `ruff`

## Struktur repo
```
loaf-lens/
├── CLAUDE.md
├── README.md
├── docs/            # PRD.md, KICKOFF_PROMPT.md, catatan eksperimen
├── configs/         # *.yaml (threshold, ROI, model, source)
├── data/            # LIHAT ATURAN DATA di bawah
│   ├── raw/         # foto asli dari client (gitignored)
│   ├── normal/      # roti OK untuk training (gitignored)
│   ├── defect/      # roti kempes, HANYA untuk validasi/test (gitignored)
│   └── README.md    # cara mengisi, konvensi penamaan, ukuran dataset
├── src/loaflens/
│   ├── capture.py   # sumber: camera / video file
│   ├── segment.py   # ekstrak roti per frame (ROI, thresholding/background subtraction)
│   ├── track.py     # satu roti = satu keputusan (garis keputusan / centroid tracking)
│   ├── features.py  # fitur bentuk/warna untuk baseline
│   ├── score.py     # antarmuka scorer: baseline & anomaly
│   ├── train.py     # CLI training
│   ├── evaluate.py  # CLI evaluasi
│   └── app.py       # CLI/UI demo real-time
├── tests/
└── outputs/         # model, laporan, snapshot reject (gitignored)
```

## Aturan data
- **Jangan commit** gambar/video asli, model besar, atau output eksperimen. Pastikan `.gitignore` mencakup `data/*`, `outputs/`, `*.onnx`, `*.pt`, kecuali `data/README.md`.
- Pisahkan split **per batch/sesi pengambilan**, bukan acak per gambar, untuk menghindari kebocoran (frame mirip di train & test).
- `data/normal/` hanya berisi roti yang benar-benar normal (konfirmasi manusia). Satu roti kempes yang nyelip ke normal merusak model anomaly.
- Catat metadata (tanggal, kamera, pencahayaan, jenis roti) di `data/manifest.csv`.

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
make setup | make test | make lint
python -m loaflens.train --config configs/anomaly.yaml
python -m loaflens.evaluate --config configs/anomaly.yaml
python -m loaflens.app --source video:samples/demo.mp4
python -m loaflens.app --source camera:0
```
