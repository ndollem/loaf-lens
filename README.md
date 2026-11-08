# LoafLens

POC visi komputer low-cost untuk mendeteksi roti **kempes** (afkir) di konveyor.
Kamera + model ML lokal (CPU-only, tanpa LLM/API cloud) memberi **skor kempes 0–100** per roti;
roti di-REJECT jika skor ≥ `threshold` di config.

Detail tujuan & requirement: [`docs/PRD.md`](docs/PRD.md). Aturan kerja: [`CLAUDE.md`](CLAUDE.md).

## Setup

Butuh [`uv`](https://docs.astral.sh/uv/) dan `make`. Versi Python ada di `.python-version` (3.14).

```bash
make setup   # buat .venv, pasang dependency terkunci (requirements-dev.txt) + paket editable
make test    # pytest
make lint    # ruff check + ruff format --check
make format  # perbaiki format otomatis
make lock    # perbarui requirements*.txt setelah mengubah pyproject.toml
```

**OpenCV:** sejak M4 dependency-nya `opencv-python` (build dengan GUI untuk jendela demo
`cv2.imshow`), menggantikan `opencv-python-headless`. Keduanya memakai modul `cv2` yang sama dan
bentrok bila terpasang bersamaan. `.venv` lama (sebelum M4): jalankan
`uv pip uninstall --python .venv/bin/python opencv-python-headless` lalu `make setup`.
Di Linux tanpa display, pakai `--headless`.

### PyTorch (CPU-only) untuk scorer B

Scorer B memakai `torch` + `torchvision` (bukan anomalib). Di Linux, `torch` dari PyPI menarik paket
CUDA berukuran beberapa GB — karena itu `make setup` dan `make lock` memakai
`uv ... --torch-backend cpu`, yang mengambil **hanya** paket ekosistem PyTorch dari index CPU
`https://download.pytorch.org/whl/cpu` (paket lain tetap dari PyPI):

- **Linux x86_64/aarch64:** terpasang `torch==2.15.0+cpu`, `torchvision==0.29.2+cpu` (tanpa paket
  `nvidia-*`/`triton`). Ukuran `.venv` naik ±0.9 GB (torch ±750 MB).
- **macOS Apple Silicon:** wheel macOS memang CPU/MPS-only; index CPU menyediakan versi yang sama
  (`torch==2.15.0`, butuh macOS ≥ 14). `make setup` sama persis.
- `requirements*.txt` mem-pin `torch==2.15.0` / `torchvision==0.29.2` **tanpa** label `+cpu`
  (`make lock` membuangnya) sehingga satu file berlaku di Linux dan macOS; pin tanpa label lokal
  cocok dengan wheel `+cpu` (PEP 440).
- Tanpa uv: `pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu`
  (pip memilih `+cpu` karena dianggap versi lebih tinggi). Jangan `pip install torch` polos di Linux.
- Wheel dari index CPU di-host di `download-r2.pytorch.org`; host itu harus bisa diakses saat setup.

## Data

Foto dari user ada di `train-data/` (di-commit atas keputusan user): `good-loaf/` = normal,
`bad-loaf/` = kempes, split beku di `train-data/manifest.csv`. Data client lain **tidak** masuk git.
Lihat [`data/README.md`](data/README.md) untuk cara menambah foto, `batch_id`, manifest, dan aturan
split; hasil audit di [`docs/audit_m1.md`](docs/audit_m1.md).

## Config

Semua parameter ada di `configs/*.yaml`:

- `configs/baseline.yaml` — scorer A (fitur bentuk/warna + scikit-learn, latih dari normal + kempes)
- `configs/anomaly.yaml` — scorer B (anomaly detection "PatchCore-lite", latih dari normal saja)

## Backbone pretrained (scorer B)

Scorer B memakai **ResNet18 pretrained ImageNet (torchvision) hanya sebagai feature extractor beku**:
tidak di-fine-tune, tidak ada kepala klasifikasi ImageNet yang dipakai. Yang "dilatih dari data kita"
adalah memory bank patch roti normal dari `train-data/` + kalibrasi skor. Bobot (±47 MB,
`resnet18-f37072fd.pth`, hash diverifikasi) diunduh **sekali saat training** ke cache torch
(`~/.cache/torch/hub/checkpoints/`, atau `model.params.weights_dir`). Saat inferensi/evaluasi tidak
ada panggilan jaringan: bila file bobot tidak ada, scorer berhenti dengan pesan jelas. Bobot dan
`model.pt` (memory bank ±10 MB) tidak di-commit (`outputs/` di-gitignore).

Skor B: jarak kNN patch roti terjauh ke memory bank → `skor = clip(50 · mentah / ref, 0, 100)`, dengan
`ref` = persentil `ref_percentile` (p95) skor *leave-one-image-out* roti normal train. Jadi **50 ≈
seaneh roti normal train yang paling aneh**; di atas 50 = di luar variasi normal yang pernah dilihat.

## Perintah

| Perintah | Status |
|---|---|
| `make manifest` = `python -m loaflens.data --config configs/baseline.yaml` | M1; tidak menimpa manifest, `--force` = split ulang |
| `make audit` = `python -m loaflens.audit --config configs/baseline.yaml` | M1; → `outputs/audit.md` + `docs/audit_m1.md` |
| `make train` = `python -m loaflens.train --config configs/baseline.yaml` | M2; fit dari split `train` → `outputs/baseline/model.joblib` + `segment_debug.jpg` |
| `make evaluate` = `python -m loaflens.evaluate --config configs/baseline.yaml` | M2/M3; val/test + CV + kalibrasi + domain check → `outputs/baseline/` + `docs/results_m3_baseline.md` |
| `make train-anomaly` = `python -m loaflens.train --config configs/anomaly.yaml` | M3; normal saja → `outputs/anomaly/model.pt` (unduh bobot ResNet18 sekali) |
| `make evaluate-anomaly` = `python -m loaflens.evaluate --config configs/anomaly.yaml` | M3; + heatmap → `outputs/anomaly/heatmaps/` + `docs/results_m3_anomaly.md` |
| `make calibrate [CONFIG=...]` = `python -m loaflens.calibrate --config configs/<x>.yaml` | M3; saran threshold dari CV train+val (bukan test) → `outputs/<run>/calibration.yaml` (config tidak diubah) |
| `make compare` = `python -m loaflens.compare configs/baseline.yaml configs/anomaly.yaml` | M3; tabel A vs B dari `metrics.json` → `outputs/compare.md` + `docs/results_m3.md` |
| `make domain-check [CONFIG=...]` = `python -m loaflens.evaluate --config configs/<x>.yaml --domain-check <foto>` | M3; skor tiap roti di foto konveyor → `outputs/<run>/domain_check.jpg` |
| `make demo-video` = `python -m loaflens.synth --config configs/baseline.yaml --out samples/demo.mp4` | M4; video demo **SYNTHETIC** dari foto konveyor + ground truth `samples/demo.gt.json` |
| `make demo` = `python -m loaflens.app --config configs/baseline.yaml --source video:samples/demo.mp4 --headless --out outputs/demo_out.mp4 --gt samples/demo.gt.json` | M4; tanpa jendela, video beranotasi + banding ground truth → `outputs/app/gt_compare.md` |
| `make demo-live` = `python -m loaflens.app --config configs/baseline.yaml --source video:samples/demo.mp4` | M4; jendela OpenCV (`q`/ESC keluar), tempo = fps video |
| `make camera [CAMERA=0]` = `python -m loaflens.app --config configs/baseline.yaml --source camera:0` | M4; kamera langsung (demo kamera-ke-layar, lihat di bawah) |

Urutan lengkap M3: `make train train-anomaly evaluate evaluate-anomaly compare` (`evaluate` sudah
menjalankan kalibrasi + domain check; `calibrate`/`domain-check` untuk dijalankan terpisah).

## Demo real-time (M4)

Pipeline: `capture → segment_all (ROI) → track (garis keputusan) → skor scorer A → threshold → overlay/log`.
Demo memakai **scorer A** (`configs/baseline.yaml`, model `outputs/baseline/model.joblib` dari
`make train`); scorer B gagal di domain konveyor (lihat `docs/results_m3.md`).

```bash
make train        # sekali, bila outputs/baseline/model.joblib belum ada
make demo-video   # samples/demo.mp4 + samples/demo.gt.json (SYNTHETIC, tidak di-commit)
make demo         # headless + banding ground truth → outputs/demo_out.mp4, outputs/app/gt_compare.md
make demo-live    # sama, dengan jendela (butuh display; OpenCV build GUI `opencv-python`)
```

Opsi app: `--source video:<path>|camera:<id>`, `--headless`, `--out <video>`, `--max-frames N`,
`--gt <sidecar.gt.json>`, `--synthetic` (paksa banner). Semua parameter lain di config:
`track.*` (arah, `line_pos`, `max_distance`, `max_missed`), `camera.roi`, `threshold`, `app.*`.

- **Satu roti = satu keputusan:** roti dilacak (centroid, tetangga terdekat) dan diputuskan sekali saat
  titik tengahnya melewati garis keputusan kuning searah `track.direction`; skor dari segmentasi
  frame itu.
- **Overlay:** kotak abu = belum diputuskan, hijau = OK, merah = REJECT (+ id, skor); FPS dan counter
  total/OK/REJECT; banner merah "SYNTHETIC - bukan untuk evaluasi akurasi" bila sumbernya video
  sintetis (sidecar `.gt.json` ditemukan, atau `--synthetic`).
- **Output:** tiap REJECT → crop + frame beranotasi di `outputs/rejects/` dan satu baris di
  `outputs/rejects/rejects.csv` (ditambahkan antar run); semua keputusan run → `outputs/app/decisions.csv`;
  satu baris log `DECISION id=.. frame=.. score=.. OK|REJECT` per roti (hook sinyal sederhana, belum
  serial/PLC); ringkasan FPS & latency di akhir.

### Video demo SYNTHETIC

`make demo-video` menumpuk foto `train-data/domain-check/conveyor_mixed_01.webp` (5 roti kempes +
15 OK) lalu `conveyor_good_01.webp` (15 OK), diulang 2x, dan menggeser jendela 941x900 px sehingga
roti bergerak ke bawah 8 px/frame @ 30 fps. Roti terpotong di sambungan foto ditutup warna konveyor.
Ground truth (`samples/demo.gt.json`) berisi posisi tiap roti per frame dan label acuan dari
`train-data/domain-check/labels.json`. **SYNTHETIC — bukan untuk evaluasi akurasi:** ini scroll foto
diam (tanpa motion blur, perubahan cahaya, atau efek kamera); hanya menguji pipeline & penghitungan.
`samples/` tidak di-commit — buat ulang kapan saja dengan `make demo-video` (opsi `--speed`, `--fps`,
`--height`, `--direction`, `--repeat`, `--images`). Hasil: [`docs/results_m4.md`](docs/results_m4.md).

### Demo kamera-ke-layar (belum diuji di repo ini)

1. Putar `samples/demo.mp4` **layar penuh** di device lain (laptop/tablet/TV), kecerahan layar tinggi.
2. Arahkan kamera tegak lurus ke layar, seluruh area konveyor terlihat; kunci fokus bila bisa.
3. Sesuaikan config: `camera.roi` = area layar di frame kamera (fraksi `[x, y, w, h]`; buang bingkai
   layar & tepi ruangan), `track.direction` = arah gerak roti **di gambar kamera** (bila kamera
   diputar, arahnya ikut berputar), `track.line_pos` di tengah area gerak. Bila roti di kamera tampak
   jauh lebih kecil/besar, periksa `segment.obj_min_area_frac`/`obj_max_area_frac` (fraksi luas ROI).
4. Jalankan `make camera` (atau `CAMERA=1`), pantau overlay; `q`/ESC untuk berhenti. Tambah
   `--out outputs/camera_out.mp4` untuk merekam hasil, `--synthetic` agar banner tetap tampil.
5. Tips: matikan auto-exposure/auto white balance bila kamera mendukung, hindari pantulan lampu di
   layar; **moiré** berkurang dengan sedikit memiringkan kamera, menjauhkan kamera + zoom, atau
   sedikit defocus; refresh rate layar bisa menimbulkan garis/flicker — coba exposure lebih lama.
6. Sesuai PRD §9: jalankan **≥ 5 menit** stabil (video diputar berulang) dan laporkan hasil kamera
   **terpisah** dari uji file video (`make demo`): jumlah keputusan vs roti yang lewat, duplikat,
   kempes tertangkap, OK ikut ter-reject. Ground truth `--gt` hanya berlaku untuk file video, bukan
   kamera (posisi/waktu berbeda).

## Status

Lihat `docs/PRD.md` §13. Saat ini: **M4** selesai untuk uji file video — pipeline real-time
(capture/segment/track/score/overlay, snapshot REJECT) pada video demo SYNTHETIC: 70/70 roti
diputuskan tepat sekali (miscount 0%), kempes 5/5 per putaran, roti OK ter-reject 0/30
([`docs/results_m4.md`](docs/results_m4.md)). Demo kamera-ke-layar masih harus dijalankan user.
M3: scorer B (PatchCore-lite) + kalibrasi threshold + perbandingan A vs B + domain check foto konveyor:
[`docs/results_m3.md`](docs/results_m3.md) (rinci: [`results_m3_baseline.md`](docs/results_m3_baseline.md),
[`results_m3_anomaly.md`](docs/results_m3_anomaly.md); snapshot M2: [`results_m2.md`](docs/results_m2.md)).
Di 48 foto studio kedua scorer memisahkan kelas hampir sempurna (CV AUROC A 0.988, B 1.000), tetapi di
foto konveyor A jauh lebih stabil → A dipakai untuk demo M4, B butuh memory bank dari kamera konveyor.
