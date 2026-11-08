# Panduan Penggunaan LoafLens

Panduan praktis untuk anggota tim yang baru pertama kali membuka repo ini: instalasi, data, config,
alur kerja lengkap (latih → evaluasi → kalibrasi → demo), arti output, dan troubleshooting.
Semua perintah dijalankan dari **root repo** (`loaf-lens/`).

Dokumen terkait: [`PRD.md`](PRD.md) (tujuan & requirement), [`../CLAUDE.md`](../CLAUDE.md) (aturan
kerja & konvensi kode), [`../data/README.md`](../data/README.md) (aturan data), hasil:
[`results_m3.md`](results_m3.md), [`results_m4.md`](results_m4.md),
[`domain_check_mixed.md`](domain_check_mixed.md).

## Daftar isi

1. [Ringkasan](#1-ringkasan)
2. [Persyaratan & instalasi](#2-persyaratan--instalasi)
3. [Struktur repo](#3-struktur-repo)
4. [Data](#4-data)
5. [Referensi konfigurasi](#5-referensi-konfigurasi)
6. [Alur kerja](#6-alur-kerja)
7. [Output & cara membaca metrik](#7-output--cara-membaca-metrik)
8. [Referensi Makefile](#8-referensi-makefile)
9. [Referensi CLI](#9-referensi-cli)
10. [Pengembangan](#10-pengembangan)
11. [Troubleshooting / FAQ](#11-troubleshooting--faq)
12. [Langkah berikutnya](#12-langkah-berikutnya)

---

## 1. Ringkasan

**LoafLens** adalah POC visi komputer low-cost untuk mendeteksi roti **kempes** (afkir) di konveyor.
Kamera atau file video → segmentasi tiap roti → tracking (satu roti = satu keputusan) → model ML
lokal memberi **skor kempes 0–100** (makin tinggi = makin kempes) → roti di-**REJECT** bila
`skor ≥ threshold` di config. Semua inferensi jalan di **CPU**, tanpa LLM dan tanpa API cloud. Ada
dua scorer dengan antarmuka sama:

| | Scorer A — baseline (`configs/baseline.yaml`) | Scorer B — anomaly (`configs/anomaly.yaml`) |
|---|---|---|
| Metode | segmentasi → 25 fitur tekstur/warna/bentuk → StandardScaler + logistic regression | "PatchCore-lite": ResNet18 ImageNet **beku** sebagai feature extractor + memory bank patch roti normal |
| Data latih | roti normal **dan** kempes | roti normal saja |
| Skor | `100 · P(kempes)` | jarak kNN patch terjauh, dinormalisasi: 50 ≈ roti normal train yang paling aneh |
| Kecepatan / ukuran | ≈ 4 ms/roti, model 3 kB | ≈ 25 ms/roti, model 10 MB (+ bobot ResNet18 47 MB) |
| Status | **direkomendasikan, dipakai demo** | belum layak di konveyor tanpa refit |

**Status milestone** (detail: [`PRD.md` §13](PRD.md#13-status)):

| M | Isi | Status |
|---|---|---|
| M0 | skeleton repo, config + validasi, Makefile, test/lint | selesai |
| M1 | manifest + split beku, audit dataset | selesai |
| M2 | scorer A + evaluasi | selesai |
| M3 | scorer B, kalibrasi threshold, perbandingan A vs B, domain check foto konveyor | selesai |
| M4 | pipeline real-time (capture/segment/track/overlay, snapshot REJECT), video demo SYNTHETIC | selesai untuk uji file video; **demo kamera-ke-layar belum dijalankan** |
| M5 | uji Raspberry Pi / export ONNX, hook sinyal | opsional, belum |

> **Catatan jujur — baca sebelum memakai angka apa pun**
> - Model dilatih dari **48 foto studio** (24 normal + 24 kempes, satu roti per foto, bukan foto
>   konveyor). Test set hanya 4+4 foto → AUROC 1.000 di test tidak berarti banyak; acuan utama CV
>   (A 0.988, B 1.000). Semua angka = **indikasi kelayakan, bukan akurasi produksi**.
> - **Scorer A direkomendasikan.** Di 2 foto konveyor asli A: 0/15 roti bagus ter-reject dan 5/5
>   kempes tertangkap ([`domain_check_mixed.md`](domain_check_mixed.md)). A bersandar pada
>   kilap/warna, jadi hasil baik ini bisa sebagian "benar karena alasan kurang tepat".
> - **Scorer B gagal di domain konveyor tanpa refit:** memory bank-nya berisi roti studio, sehingga
>   5–11/15 roti bagus (foto 1) dan 20/20 roti (foto 2) di-REJECT. B perlu memory bank dari roti bagus
>   yang direkam kamera konveyor.
> - **Video demo adalah SYNTHETIC** (foto diam yang digeser). Hasil 70/70 roti diputuskan tepat
>   sekali hanya membuktikan logika pipeline, **bukan** akurasi deteksi.

---

## 2. Persyaratan & instalasi

### Persyaratan

| Kebutuhan | Keterangan |
|---|---|
| OS | Linux x86_64/aarch64, atau macOS Apple Silicon **≥ 14** (wheel torch 2.15 butuh macOS 14+) |
| [`uv`](https://docs.astral.sh/uv/) | pengelola venv/paket; harus versi yang mendukung `--torch-backend` (repo diuji dengan uv 0.11) |
| `make` | macOS: `xcode-select --install`; Debian/Ubuntu: `sudo apt install make` |
| Python 3.14 | versi di `.python-version`; `uv` mengunduh Python 3.14 sendiri bila belum ada |
| Disk | ±1 GB untuk `.venv` (torch CPU ±750 MB) + 47 MB bobot ResNet18 (hanya scorer B) |
| Jaringan saat setup | PyPI + `download.pytorch.org` / `download-r2.pytorch.org` (wheel torch CPU) |
| Kamera/display | hanya untuk `make demo-live` / `make camera`; tanpa display pakai `--headless` |

Pasang `uv` bila belum ada (pilih salah satu):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
brew install uv
```

### Instalasi (macOS dan Linux sama)

```bash
make setup
```

`make setup` membuat `.venv`, memasang dependency terkunci dari `requirements-dev.txt`, lalu memasang
paket `loaflens` sebagai editable. Target lain (`make test`, `make train`, ...) selalu memakai
`.venv/bin/...`, jadi venv tidak perlu di-*activate*. Untuk menjalankan `python -m loaflens.<modul>`
langsung, aktifkan dulu venv-nya atau pakai `.venv/bin/python -m ...`.

```bash
source .venv/bin/activate
```

Cek instalasi:

```bash
make test
```

### PyTorch CPU-only

Scorer B memakai `torch` + `torchvision` (bukan anomalib). Di Linux, `torch` dari PyPI menarik paket
CUDA berukuran beberapa GB, karena itu `make setup` dan `make lock` memakai
`uv pip ... --torch-backend cpu`: **hanya** paket ekosistem PyTorch diambil dari index CPU
`https://download.pytorch.org/whl/cpu`, paket lain tetap dari PyPI.

- **Linux:** terpasang `torch==2.15.0+cpu`, `torchvision==0.29.2+cpu`, tanpa paket `nvidia-*`/`triton`.
- **macOS Apple Silicon:** wheel macOS memang CPU/MPS-only; versi sama (`torch==2.15.0`).
- `requirements*.txt` mem-pin `torch==2.15.0` / `torchvision==0.29.2` **tanpa** label `+cpu`
  (`make lock` membuangnya) sehingga satu file berlaku di Linux dan macOS.
- Wheel index CPU di-host di `download-r2.pytorch.org`; host itu harus bisa diakses saat setup/lock.

**Alternatif tanpa uv** (pip; pip memilih `+cpu` karena dianggap versi lebih tinggi). Jangan
`pip install torch` polos di Linux (akan menarik CUDA). Python 3.14 harus sudah terpasang:

```bash
python3.14 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/bin/pip install --no-deps -e .
```

### Bobot ResNet18 (hanya scorer B)

Bobot ImageNet ResNet18 (`resnet18-f37072fd.pth`, ±47 MB, hash diverifikasi) diunduh **sekali saat
`make train-anomaly`** ke cache torch `~/.cache/torch/hub/checkpoints/` (atau ke
`model.params.weights_dir` bila diisi). Evaluasi/inferensi **tidak pernah** mengunduh: bila file tidak
ada, scorer berhenti dengan pesan jelas. Untuk mesin offline, salin file itu ke folder cache atau ke
`weights_dir`. Scorer A tidak butuh bobot ini.

### Upgrade `.venv` lama (sebelum M4)

Sejak M4 dependency OpenCV adalah `opencv-python` (build GUI untuk jendela demo), menggantikan
`opencv-python-headless`. Keduanya memakai modul `cv2` yang sama dan bentrok bila terpasang bersamaan.
Untuk `.venv` lama:

```bash
uv pip uninstall --python .venv/bin/python opencv-python-headless
```

```bash
make setup
```

Atau mulai bersih dengan `make clean` (menghapus `.venv`, cache pytest/ruff) lalu `make setup`.

---

## 3. Struktur repo

```
loaf-lens/
├── CLAUDE.md, README.md     # aturan kerja / ringkasan + quickstart
├── Makefile                 # semua perintah (lihat §8)
├── pyproject.toml           # dependency + config ruff/pytest
├── requirements*.txt        # dependency terkunci (make lock)
├── configs/
│   ├── baseline.yaml        # scorer A + parameter app/demo
│   └── anomaly.yaml         # scorer B
├── docs/                    # PRD, USAGE (dokumen ini), audit & hasil per milestone
├── data/                    # data client lain (gitignored kecuali README.md, manifest.example.csv)
├── train-data/              # foto dari user, DI-COMMIT (keputusan user, ~6.6 MB)
│   ├── good-loaf/           # label normal
│   ├── bad-loaf/            # label defect (kempes)
│   ├── domain-check/        # foto konveyor asli + labels.json — hanya cek domain/demo, tidak dilatih
│   └── manifest.csv         # split beku train/val/test
├── src/loaflens/
│   ├── config.py            # load + validasi YAML
│   ├── data.py              # CLI manifest + split per grup, loader gambar
│   ├── audit.py             # CLI audit dataset
│   ├── segment.py           # segment (1 roti/foto) + segment_all (banyak roti/frame, ROI)
│   ├── features.py          # fitur tekstur/warna/bentuk scorer A
│   ├── score.py             # antarmuka Scorer + scorer A
│   ├── anomaly.py           # scorer B (PatchCore-lite + heatmap)
│   ├── train.py / evaluate.py / calibrate.py / compare.py   # CLI ML
│   ├── capture.py           # sumber video:/camera:
│   ├── track.py             # centroid tracker + garis keputusan
│   ├── synth.py             # CLI video demo SYNTHETIC + ground truth
│   └── app.py               # CLI demo real-time
├── tests/                   # pytest (gambar sintetis, tanpa kamera)
├── samples/                 # video demo (gitignored; buat ulang: make demo-video)
└── outputs/                 # model, laporan, snapshot (gitignored)
```

---

## 4. Data

### Layout `train-data/`

| Folder | Label | Isi saat ini |
|---|---|---|
| `train-data/good-loaf/` | `normal` | 24 PNG 312x417 (`bagus (0..11)`, `medium (0..11)`) |
| `train-data/bad-loaf/` | `defect` (kempes) | 24 PNG 360x355 (`bad (0..11)`, `medium`, `medium (1..11)`) |
| `train-data/domain-check/` | — (tidak dilatih) | `conveyor_good_01.webp` (semua roti OK), `conveyor_mixed_01.webp` (5 kempes + 15 OK), `labels.json` (posisi roti kempes/borderline) |
| `train-data/manifest.csv` | — | split beku per gambar |

Folder aktif diatur di config (`data.normal_dir`, `data.defect_dir`, `data.manifest`). Format yang
dibaca untuk training: `data.extensions` (default `.png`, `.jpg`, `.jpeg`). Isi `data/` lainnya dan
semua output **tidak** di-commit.

### Arti prefix nama file

| Folder | Prefix | Arti | Keputusan | Kelompok di laporan |
|---|---|---|---|---|
| `good-loaf/` | `bagus (n)` | jelas layak | terima | `bagus` |
| `good-loaf/` | `medium (n)` | masih layak walau permukaan tidak sempurna | terima | `medium-good` |
| `bad-loaf/` | `bad (n)` | kempes parah | reject | `bad` |
| `bad-loaf/` | `medium (n)`, `medium.png` | tampak lumayan, tapi permukaan terlalu kempes | reject | `medium-bad` |

Prefix **hanya** dipakai untuk rincian per kelompok di laporan, bukan fitur dan bukan batch. Nomor
sama dengan prefix berbeda = roti berbeda. Label ditentukan oleh **folder**, bukan nama file.

### Menambah foto baru

1. Taruh foto di folder label, **satu subfolder per batch/sesi pengambilan** (kamera, posisi, dan
   pencahayaan yang sama). Normal dan kempes dari sesi yang sama memakai nama subfolder yang sama:

   ```
   train-data/good-loaf/20261110_s1/<nama>.jpg
   train-data/bad-loaf/20261110_s1/<nama>.jpg
   ```

2. Folder normal **hanya** berisi roti yang benar-benar normal (konfirmasi manusia). Satu roti kempes
   yang nyelip merusak scorer B. Kalau ragu, jangan masukkan.
3. Jalankan audit; bagian "Konsistensi manifest" melaporkan file yang belum masuk manifest.
4. Bangun ulang manifest dengan `--force` (lihat di bawah), lalu latih ulang (§6.9).

Aturan `batch_id`: subfolder langsung di bawah folder label → `batch_id` = nama subfolder; file
langsung di folder label → satu grup per gambar (`img_<label>_<nama>`, audit memberi WARNING).
Gambar sebatch dan pasangan near-duplicate (dHash, jarak ≤ `data.near_dup_hamming`) selalu masuk
split yang sama. Detail: [`data/README.md`](../data/README.md).

### Manifest & split beku

```bash
make manifest
```

- Membuat `train-data/manifest.csv` (kolom `path, label, batch_id, date, camera, camera_view,
  lighting, bread_type, split`). Metadata (tanggal, kamera, ...) diisi manual di CSV.
- **Manifest = split beku.** Bila file sudah ada, CLI **tidak** menimpanya (log WARNING, exit 0).
  Tidak ada mode "tambah file baru saja": foto baru baru masuk lewat split ulang.
- Split ulang (`--force`) membuat ulang **seluruh** split (test set ikut berubah, hasil lama tidak
  bisa dibandingkan lagi). Metadata manual dipertahankan per `path`. Lakukan hanya sebelum
  tuning/evaluasi pada data baru, dan catat alasannya di `docs/PRD.md`:

```bash
.venv/bin/python -m loaflens.data --config configs/baseline.yaml --force
```

### Audit dataset

```bash
make audit
```

Menulis `outputs/audit.md` dan salinan ber-versi `docs/audit_m1.md` (`audit.docs_copy`). Isinya:
jumlah per label/split, ukuran gambar, risiko *shortcut* (fitur trivial seperti ukuran gambar yang
memisahkan kelas), pasangan nama mirip, near-duplicate, file tak terbaca, dan konsistensi manifest.
Jalankan setiap kali data berubah dan baca bagian WARNING.

---

## 5. Referensi konfigurasi

Semua parameter ada di `configs/*.yaml`, divalidasi oleh `src/loaflens/config.py`. Kunci top-level
yang tidak dikenal, atau kunci tak dikenal di dalam section (kecuali `model.params`), menghasilkan
`ConfigError`. Kunci **wajib**: `seed`, `data.{normal_dir, defect_dir, manifest, split}`,
`model.{type, train_labels}`, `threshold`, `output.dir`. Section lain opsional (pakai default).

Kolom "Default" = nilai bila kunci tidak ditulis (default kode); bila config repo memakai nilai lain,
nilainya ditulis di kolom keterangan.

### Section umum (sama di kedua config)

| Kunci | Default | Nilai valid | Arti |
|---|---|---|---|
| `seed` | wajib | int | seed split, CV, subsample (config repo: 42) |
| `threshold` | wajib | 0–100 | REJECT bila skor ≥ threshold (config repo: 50) |
| `output.dir` | wajib | path | folder artefak: `outputs/baseline` / `outputs/anomaly` |

**`data`**

| Kunci | Default | Nilai valid | Arti |
|---|---|---|---|
| `normal_dir` | wajib | path | folder roti normal (`train-data/good-loaf`) |
| `defect_dir` | wajib | path | folder roti kempes (`train-data/bad-loaf`) |
| `manifest` | wajib | path | CSV split beku (`train-data/manifest.csv`) |
| `split` | wajib | kunci tepat `train/val/test`, tiap 0–1, jumlah 1.0 | rasio split (repo: 0.70/0.15/0.15) |
| `split_by` | `batch_id` | `batch_id` | split per batch, bukan acak per gambar |
| `extensions` | `[.png, .jpg, .jpeg]` | list tak kosong berawalan `.` | ekstensi yang dipindai |
| `near_dup_hamming` | 5 | 0–64 | jarak dHash ≤ ini = near-duplicate (dipaksa satu split) |

**`audit`** (hanya `make audit`)

| Kunci | Default | Nilai valid | Arti |
|---|---|---|---|
| `report` | `outputs/audit.md` | path | laporan audit |
| `docs_copy` | null | path / null | salinan ber-versi (repo: `docs/audit_m1.md`) |
| `border_frac` | 0.05 | (0, 0.5) | lebar tepi untuk statistik warna latar |
| `center_frac` | 0.5 | (0, 1] | kotak tengah untuk statistik warna roti |
| `shortcut_acc` | 0.9 | (0.5, 1] | WARNING bila 1 fitur trivial memisahkan kelas ≥ akurasi ini |
| `pair_warn_ratio` | 0.6 | (0, 1] | WARNING kemiripan pasangan nama bernomor sama |
| `min_eval_per_class` | 30 | ≥ 1 | WARNING bila val/test per label < ini |

**`camera`**

| Kunci | Default | Nilai valid | Arti |
|---|---|---|---|
| `view` | `top` | `top` \| `oblique` | sudut kamera (tegak dari atas / miring ~45°). Saat ini hanya divalidasi & dicatat, **belum** mengubah logika |
| `roi` | null | null atau `[x, y, w, h]` fraksi 0–1, `x+w ≤ 1`, `y+h ≤ 1`, `w,h > 0` | area deteksi roti di frame (app, domain check). null = seluruh frame. Digambar sebagai kotak biru muda di overlay |

**`segment`** — isolasi roti dari latar (kanal saturasi HSV)

| Kunci | Default | Nilai valid | Arti |
|---|---|---|---|
| `blur` | 5 | ganjil ≥ 1 | kernel Gaussian sebelum threshold |
| `sat_threshold` | null | null atau 0–255 | null = Otsu per gambar; angka = threshold tetap |
| `morph_kernel` | 7 | ganjil ≥ 1 | kernel elips open/close |
| `min_area_frac` | 0.02 | (0, 1) | 1 roti/foto: kontur terbesar < fraksi frame ini = gagal (fallback seluruh gambar) |
| `crop_margin` | 0.05 | [0, 1) | margin crop di sekitar bbox (fraksi sisi bbox) |
| `obj_min_area_frac` | 0.003 | `0 < min < max ≤ 1` | banyak roti/frame: luas objek minimum, fraksi luas ROI/frame |
| `obj_max_area_frac` | 0.15 | (lihat atas) | luas objek maksimum |
| `drop_border` | true | bool | buang roti yang menyentuh tepi ROI/frame (terpotong) |
| `obj_min_value` | 0 | 0–255 | piksel V < ini bukan roti (bayangan gelap); 0 = mati. Config repo: **40** |

**`track`** — satu roti = satu keputusan (app)

| Kunci | Default | Nilai valid | Arti |
|---|---|---|---|
| `direction` | `left_to_right` | `left_to_right` \| `right_to_left` \| `top_to_bottom` \| `bottom_to_top` | arah gerak roti **di gambar**. Config repo: **`top_to_bottom`** (sesuai video demo) |
| `line_pos` | 0.5 | (0, 1) | garis keputusan (kuning), fraksi **seluruh frame** searah gerak dari tepi masuk (bukan fraksi ROI) |
| `max_distance` | 0.08 | (0, 1] | jarak maks prediksi→deteksi, fraksi sisi terpanjang frame |
| `max_missed` | 5 | ≥ 0 | track dihapus setelah sekian frame berturut-turut tanpa deteksi |

**`eval`** (evaluate / calibrate)

| Kunci | Default | Nilai valid | Arti |
|---|---|---|---|
| `fpr_target` | 0.05 | (0, 1) | target FPR untuk recall@FPR dan kalibrasi threshold |
| `cv_folds` | 5 | ≥ 2 | StratifiedGroupKFold pada train+val (test tidak ikut) |
| `top_errors` | 5 | ≥ 0 | jumlah kesalahan terbesar per split di laporan |
| `docs_copy` | null | path / null | salinan laporan ber-versi (`docs/results_m3_baseline.md` / `_anomaly.md`) |
| `domain_check` | `[]` | list path foto | foto konveyor yang dinilai saat `evaluate` (repo: `conveyor_good_01.webp`) |
| `heatmaps` | true | bool | scorer B: simpan overlay heatmap val/test (diabaikan scorer A) |

**`app`** (demo real-time; hanya ada di `baseline.yaml`, `anomaly.yaml` memakai default yang sama)

| Kunci | Default | Arti |
|---|---|---|
| `window_name` | `LoafLens` | judul jendela OpenCV |
| `out_fps` | null | fps video `--out`; null = fps sumber (kamera tanpa fps → 30). Bila diisi harus > 0 |
| `snapshot_dir` | `outputs/rejects` | crop + frame beranotasi tiap REJECT |
| `rejects_csv` | `outputs/rejects/rejects.csv` | log REJECT, **ditambahkan** antar run |
| `decisions_csv` | `outputs/app/decisions.csv` | semua keputusan satu run (ditimpa per run) |
| `snapshot_margin` | 0.15 | margin crop snapshot, [0, 1) |
| `synthetic_banner` | `SYNTHETIC - bukan untuk evaluasi akurasi` | teks banner (ASCII) |
| `gt_match_dist` | 0.05 | `--gt`: jarak maks keputusan→roti ground truth, (0, 1] fraksi sisi terpanjang |
| `gt_report` | `outputs/app/gt_compare.md` | `--gt`: laporan perbandingan |

### `model` — scorer A (`baseline.yaml`)

| Kunci | Nilai repo | Validasi | Arti |
|---|---|---|---|
| `type` | `baseline` | `baseline` \| `anomaly` | jenis scorer |
| `train_labels` | `[normal, defect]` | subset tak kosong dari `normal`, `defect` | label yang dipakai fit |
| `params.crop_size` | 192 | int ≥ 32 | crop bbox → sisi terpanjang px (aspek dijaga, pad) |
| `params.classifier` | `logistic_regression` | `logistic_regression` \| `random_forest` \| `svm` | classifier sklearn |
| `params.classifier_params.<nama>` | lihat config | bebas (hyperparameter sklearn) | hanya milik classifier terpilih yang dipakai |
| `params.features.erode_px` | 9 | — | erosi mask sebelum statistik tekstur |
| `params.features.texture_blur_sigma` | 1.5 | — | blur sebelum tekstur (0 = mati; default kode 0) |
| `params.features.canny` | `[50, 150]` | — | threshold Canny (kepadatan tepi/kerutan) |
| `params.features.crease_px` | 6 | — | radius kernel black-hat untuk alur lipatan |
| `params.features.specular_v` / `specular_s` | 230 / 60 | — | piksel kilap: V ≥ v dan S ≤ s (HSV 0–255) |

### `model` — scorer B (`anomaly.yaml`)

| Kunci | Nilai repo | Validasi | Arti |
|---|---|---|---|
| `type` | `anomaly` | — | |
| `train_labels` | `[normal]` | **harus** `[normal]` | B hanya di-fit dari roti normal |
| `params.weights` | `imagenet` | `imagenet` \| null | ResNet18 pretrained beku; null = bobot acak (hanya untuk test) |
| `params.weights_dir` | null | — | folder bobot; null = cache torch `~/.cache/torch/hub/checkpoints` |
| `params.image_size` | 224 | int ≥ 32 | crop roti → persegi px |
| `params.fill_background` | true | — | piksel di luar mask diisi abu rata-rata ImageNet |
| `params.mask_erode_px` | 8 | — | erosi mask sebelum memilih patch (default kode 0) |
| `params.pool` | 3 | ganjil | avg-pool lokal pada feature map |
| `params.bank_ratio` | 1.0 | (0, 1] | subsample acak memory bank |
| `params.bank_max` | 20000 | ≥ 1 | batas jumlah patch di bank |
| `params.knn` | 1 | ≥ 1 | jarak rata-rata ke k tetangga terdekat |
| `params.image_score` | `max` | `max` \| `topk_mean` | skor gambar = patch terjauh / rata-rata `topk_frac` patch terjauh |
| `params.topk_frac` | 0.01 | (0, 1] | fraksi patch untuk `topk_mean` |
| `params.ref_percentile` | 95 | (0, 100], **wajib** | skor 50 = persentil ini dari skor leave-one-out roti normal train |
| `params.heatmap_sigma` | 4 | — | blur heatmap (px pada crop) |
| `params.num_threads` | 4 | — | `torch.set_num_threads` |

### Kapan perubahan berlaku

| Perubahan | Berlaku |
|---|---|
| `threshold`, `camera.roi`, `track.*`, `segment.obj_*`, `segment.drop_border`, `app.*` | langsung pada run berikutnya (tanpa train ulang) |
| `model.params.*`, `data.*`, `seed` | wajib `make train` (dan `make evaluate`) ulang |
| `segment.blur`, `sat_threshold`, `morph_kernel`, `min_area_frac`, `crop_margin` | dipakai deteksi multi-roti langsung, tetapi juga disimpan di model saat training → train ulang agar konsisten |

### Yang perlu diubah untuk setup kamera baru

Ubah di `configs/baseline.yaml` (config yang dipakai app):

| Kunci | Apa yang dilakukan |
|---|---|
| `camera.view` | catat `top` / `oblique` sesuai pemasangan (metadata; belum mengubah logika) |
| `camera.roi` | batasi ke area konveyor/layar di frame: `[x, y, w, h]` fraksi, buang tepi ruangan, bingkai layar, garis lantai |
| `track.direction` | arah gerak roti **di gambar kamera** (bila kamera diputar, arahnya ikut berputar) |
| `track.line_pos` | taruh garis keputusan di tengah area gerak, di dalam ROI, tempat roti terlihat utuh |
| `segment.obj_min_area_frac` / `obj_max_area_frac` | sesuaikan bila roti tampak jauh lebih kecil/besar (fraksi luas ROI; satu roti di video demo ≈ 1.3–1.9% luas frame, rentang default 0.3–15%) |
| `segment.obj_min_value` | naikkan bila bayangan gelap ikut terdeteksi, turunkan bila roti gelap hilang |
| `threshold` | setel hanya dari kalibrasi non-test (§6.2); jangan dari test set |

Bila kamera menghasilkan frame lebih besar/kecil, `track.max_distance` (fraksi sisi terpanjang)
ikut terskala; naikkan bila roti bergerak cepat dan sering dapat id baru.

---

## 6. Alur kerja

Semua alur memakai model di `outputs/` (gitignored). Repo yang baru di-clone **belum** punya model;
jalankan training dulu.

### 6.1 Latih & evaluasi scorer A (baseline)

```bash
make train
```

Fit dari split `train` (normal + kempes) → `outputs/baseline/model.joblib`, `config.yaml` (salinan),
`segment_debug.jpg` (contact sheet segmentasi semua gambar — cek dengan mata bahwa mask hijau menutup
roti).

```bash
make evaluate
```

Val/test dengan threshold config, CV per grup pada train+val, kalibrasi threshold, domain check foto
di `eval.domain_check` → `outputs/baseline/` + salinan `docs/results_m3_baseline.md`. Test dievaluasi
dengan threshold dari config; jangan setel apa pun dari angka test.

### 6.2 Kalibrasi threshold

`make evaluate` sudah menulis `calibration.yaml`. Untuk menghitung ulang saja (butuh model terlatih).
**Selalu tulis `CONFIG=`** — default Makefile adalah `configs/anomaly.yaml`:

```bash
make calibrate CONFIG=configs/baseline.yaml
```

Hasil di `outputs/baseline/calibration.yaml`: `cv.threshold` (dari skor CV out-of-fold train+val,
saran utama), `val.threshold` (pembanding), FPR/recall pada threshold itu dan pada threshold config,
serta `suggested_threshold`. Aturan: threshold terendah dengan FPR ≤ `eval.fpr_target` pada roti
normal, lalu digeser ke tengah celah menuju skor kempes terendah di atasnya.

**Config tidak diubah otomatis.** Untuk menerapkan: buka `configs/baseline.yaml`, ganti `threshold:`
dengan `suggested_threshold` (mis. `threshold: 48.9`), catat keputusannya di `docs/PRD.md`, lalu
jalankan ulang `make evaluate` / demo. Saat ini config tetap 50 (saran A 48.9, B 52.1; praktis sama
untuk A).

### 6.3 Scorer B: latih, evaluasi, bandingkan

```bash
make train-anomaly
```

Fit memory bank dari roti **normal** split train → `outputs/anomaly/model.pt`. Run pertama mengunduh
bobot ResNet18 (butuh internet sekali).

```bash
make evaluate-anomaly
```

Seperti `make evaluate`, plus heatmap val/test di `outputs/anomaly/heatmaps/` dan
`domain_check_heatmaps.jpg` → salinan `docs/results_m3_anomaly.md`.

```bash
make compare
```

Membaca `metrics.json` kedua run (tidak menghitung ulang; jalankan `evaluate` keduanya dulu) →
`outputs/compare.md` + `docs/results_m3.md`. Teks di bawah penanda `REKOMENDASI` di
`docs/results_m3.md` ditulis manual dan dipertahankan.

### 6.4 Domain check pada foto konveyor

Menilai setiap roti yang terdeteksi di satu foto konveyor (tanpa training/tuning). Label tiap roti
tidak diketahui tool: pada foto yang semua rotinya OK, setiap REJECT = false positive.

```bash
make domain-check CONFIG=configs/baseline.yaml
```

```bash
make domain-check CONFIG=configs/baseline.yaml DOMAIN_IMG=train-data/domain-check/conveyor_mixed_01.webp
```

Default `DOMAIN_IMG` = `conveyor_good_01.webp`, default `CONFIG` = `configs/anomaly.yaml`. Output:
`outputs/<run>/domain_check.jpg` (kontur + `#nomor skor OK/REJECT`, plus keputusan pada threshold
kalibrasi bila `calibration.yaml` ada) dan `domain_check.md` (tabel skor). File ini menimpa domain
check dari `make evaluate`. Foto domain check baru: taruh di `train-data/domain-check/` dengan nama
deskriptif dan catat labelnya di `data/README.md` (dan `labels.json` bila dipakai untuk video demo).

### 6.5 Buat video demo SYNTHETIC

```bash
make demo-video
```

Menumpuk `conveyor_mixed_01.webp` lalu `conveyor_good_01.webp`, diulang 2x, digeser 8 px/frame @ 30
fps (941x900, ±32 detik) → `samples/demo.mp4` + ground truth `samples/demo.gt.json` (`"synthetic":
true`). Arah gerak dan garis keputusan diambil dari `track.*` config. Opsi lain (mis. kecepatan):

```bash
.venv/bin/python -m loaflens.synth --config configs/baseline.yaml --out samples/demo_fast.mp4 --speed 16
```

**SYNTHETIC — bukan untuk evaluasi akurasi:** tidak ada motion blur, perubahan cahaya, atau efek
kamera; hanya menguji pipeline & penghitungan.

### 6.6 Demo headless dengan ground truth

```bash
make demo
```

Alur: `capture → segment_all (ROI) → track → skor scorer A → threshold → overlay/log`. Tiap roti dilacak (centroid, tetangga terdekat) dan diputuskan **sekali** saat titik tengahnya melewati garis keputusan searah `track.direction`; skor dari segmentasi frame itu.

Tanpa jendela. Menulis `outputs/demo_out.mp4` (video beranotasi), `outputs/app/decisions.csv`,
snapshot REJECT di `outputs/rejects/`, dan karena `--gt` dipakai: `outputs/app/gt_compare.md`
(roti diharapkan vs diputuskan, duplikat, miss, miscount, kempes tertangkap, OK ikut REJECT). Hasil
acuan: 70/70 diputuskan tepat sekali, kempes 5/5 per putaran, OK ter-reject 0/30
([`results_m4.md`](results_m4.md)). Bila `track.direction`/`line_pos` config berbeda dari saat
video dibuat, app memberi WARNING bahwa perbandingan tidak valid — buat ulang videonya.

Butuh `outputs/baseline/model.joblib` (`make train`) dan `samples/demo.mp4` (`make demo-video`).

### 6.7 Demo jendela (live)

```bash
make demo-live
```

Jendela OpenCV, tempo mengikuti fps video; `q` atau ESC untuk keluar. Butuh display (macOS
langsung; Linux butuh `DISPLAY`/`WAYLAND_DISPLAY`). Overlay: kotak abu = belum diputuskan, hijau =
OK, merah = REJECT (+ id dan skor); garis kuning = garis keputusan; HUD FPS + counter
total/OK/REJECT; banner merah SYNTHETIC bila sumbernya video sintetis (sidecar `.gt.json` di sebelah
video, atau `--synthetic`).

### 6.8 Kamera & demo kamera-ke-layar

```bash
make camera
```

```bash
make camera CAMERA=1
```

Demo kamera-ke-layar (PRD §4) **belum diuji di repo ini** — langkahnya:

1. Putar `samples/demo.mp4` **layar penuh** di device lain (laptop/tablet/TV), kecerahan tinggi,
   diulang (loop).
2. Arahkan kamera tegak lurus ke layar, seluruh area konveyor terlihat; kunci fokus bila bisa.
3. Sesuaikan `configs/baseline.yaml` (§5 "setup kamera baru"): `camera.roi` = area layar di frame
   kamera, `track.direction` = arah gerak roti di gambar kamera, `track.line_pos` di tengah area
   gerak; periksa `segment.obj_min_area_frac`/`obj_max_area_frac` bila roti tampak jauh lebih
   kecil/besar.
4. Jalankan `make camera`, pantau overlay; `q`/ESC untuk berhenti. Untuk merekam hasil dan memaksa
   banner SYNTHETIC tetap tampil:

   ```bash
   .venv/bin/python -m loaflens.app --config configs/baseline.yaml --source camera:0 --out outputs/camera_out.mp4 --synthetic
   ```

5. Tips: matikan auto-exposure/auto white balance bila kamera mendukung; hindari pantulan lampu di
   layar; **moiré** berkurang dengan sedikit memiringkan kamera, menjauhkan kamera + zoom, atau sedikit
   defocus; refresh rate layar bisa menimbulkan garis/flicker — coba exposure lebih lama.
6. Sesuai PRD §9: jalankan **≥ 5 menit** stabil dan laporkan hasil kamera **terpisah** dari uji file
   video (`make demo`): jumlah keputusan vs roti yang lewat, duplikat, kempes tertangkap, OK ikut
   ter-reject. `--gt` hanya berlaku untuk file video, bukan kamera.

Kamera konveyor sungguhan: langkah 2–3 sama (ROI = area belt, arah = arah belt di gambar).

### 6.9 Latih ulang dengan data baru (end-to-end)

1. Tambah foto ke `train-data/good-loaf/<batch_id>/` dan `train-data/bad-loaf/<batch_id>/` (§4).
2. Audit dulu (lihat file baru & WARNING):

   ```bash
   make audit
   ```

3. Split ulang (test set berubah — catat alasannya di `docs/PRD.md`):

   ```bash
   .venv/bin/python -m loaflens.data --config configs/baseline.yaml --force
   ```

4. Audit lagi untuk memastikan semua file masuk manifest dan rasio split masuk akal:

   ```bash
   make audit
   ```

5. Latih dan evaluasi scorer A:

   ```bash
   make train
   ```

   ```bash
   make evaluate
   ```

6. Tinjau `outputs/baseline/report.md` dan `calibration.yaml`; bila disetujui, salin
   `suggested_threshold` ke `threshold:` secara manual (§6.2).
7. (Opsional) scorer B dan perbandingan:

   ```bash
   make train-anomaly
   ```

   ```bash
   make evaluate-anomaly
   ```

   ```bash
   make compare
   ```

8. Cek di domain konveyor dan demo:

   ```bash
   make domain-check CONFIG=configs/baseline.yaml DOMAIN_IMG=train-data/domain-check/conveyor_mixed_01.webp
   ```

   ```bash
   make demo
   ```

9. `make audit`, `make evaluate*`, dan `make compare` menimpa file ber-versi di `docs/`
   (`audit_m1.md`, `results_m3*.md`). Periksa `git diff docs/` sebelum commit; snapshot
   `docs/results_m2.md` tidak disentuh.

Urutan lengkap M3 sekaligus: `make train train-anomaly evaluate evaluate-anomaly compare`.

---

## 7. Output & cara membaca metrik

### File di `outputs/` (gitignored)

| File | Dibuat oleh | Isi |
|---|---|---|
| `outputs/audit.md` | `make audit` | laporan audit dataset (salinan: `docs/audit_m1.md`) |
| `outputs/<run>/model.joblib` / `model.pt` | `make train` / `train-anomaly` | model A / memory bank B (+ params & config segmentasi) |
| `outputs/<run>/config.yaml` | train | salinan config saat training |
| `outputs/<run>/segment_debug.jpg` | train | contact sheet: mask hijau + bbox merah per gambar (`N`/`D` + nama) |
| `outputs/<run>/report.md` | evaluate | laporan lengkap dengan gambar (salinan tanpa gambar: `docs/results_m3_<run>.md`) |
| `outputs/<run>/metrics.json` | evaluate | angka mesin-baca untuk `compare` |
| `outputs/<run>/scores.csv` | evaluate | `split, path, label, prefix, score, pred` per gambar; `split=cv_oof` = skor out-of-fold CV |
| `outputs/<run>/calibration.yaml` | evaluate / calibrate | saran threshold (tidak diterapkan otomatis) |
| `outputs/<run>/roc_test.png` | evaluate | kurva ROC test |
| `outputs/<run>/domain_check.jpg` / `.md` | evaluate / domain-check | foto konveyor beranotasi + tabel skor (`domain_check1.jpg`, ... untuk foto berikutnya di `eval.domain_check`) |
| `outputs/anomaly/domain_check_heatmaps.jpg` | evaluate-anomaly | heatmap B per roti foto konveyor |
| `outputs/anomaly/heatmaps/` | evaluate-anomaly | overlay heatmap val/test per gambar + `sheet.jpg` |
| `outputs/compare.md` | compare | tabel A vs B (salinan: `docs/results_m3.md`) |
| `outputs/demo_out.mp4` | `make demo` (`--out`) | video beranotasi |
| `outputs/app/decisions.csv` | app | satu baris per keputusan run ini: `frame, track_id, cx, cy, score, decision` |
| `outputs/app/gt_compare.md` | app `--gt` | penghitungan vs ground truth + tabel per roti |
| `outputs/rejects/<waktu>_f<frame>_id<id>_crop.jpg` / `_frame.jpg` | app | crop roti REJECT + frame beranotasi |
| `outputs/rejects/rejects.csv` | app | `time, frame, track_id, score, crop, frame_image` — **ditambahkan** antar run (hapus manual untuk mulai bersih) |

App juga menulis satu baris log `DECISION id=.. frame=.. score=.. OK|REJECT` per roti (hook sinyal
sederhana; belum serial/PLC) dan ringkasan FPS + latency di akhir run.

### Dokumen hasil di `docs/` (di-commit)

| File | Isi | Ditimpa oleh |
|---|---|---|
| `audit_m1.md` | audit dataset | `make audit` |
| `results_m2.md` | snapshot hasil scorer A di M2 (beku) | — |
| `results_m3_baseline.md` / `results_m3_anomaly.md` | laporan evaluasi A / B | `make evaluate` / `make evaluate-anomaly` |
| `results_m3.md` | perbandingan A vs B + rekomendasi manual | `make compare` (bagian rekomendasi dipertahankan) |
| `domain_check_mixed.md` | domain check foto konveyor campuran (manual) | — |
| `results_m4.md` | hasil pipeline real-time pada video SYNTHETIC (manual) | — |

### Cara membaca metrik

| Metrik | Arti | Catatan |
|---|---|---|
| **AUROC** | peluang roti kempes acak diberi skor lebih tinggi dari roti normal acak; 0.5 = acak, 1.0 = sempurna. Tidak bergantung threshold | target PRD ≥ 0.90. Pada 3–4 roti per kelas, 1.000 mudah tercapai — utamakan CV |
| **AUROC CV** | rata-rata ± std per fold (5-fold per grup, train+val), dan "gabungan out-of-fold" dari semua skor CV | lebih bisa dipercaya daripada hold-out kecil |
| **recall @ FPR ≤ target** | fraksi kempes tertangkap bila roti normal yang ikut ter-reject ≤ `fpr_target` (5%); FPR aktual di kurung | target PRD ≥ 0.80. FPR kasar: 1/n normal (val 0.33, CV 0.05) |
| **TN FP FN TP** | confusion matrix pada threshold: FP = roti bagus ter-reject, FN = kempes lolos | dilaporkan untuk threshold config dan kalibrasi |
| **threshold terkalibrasi** | `suggested_threshold` dari CV train+val, bukan test | saran; terapkan manual (§6.2) |
| **rincian per kelompok** | skor rata-rata/min–max/salah untuk `bagus`, `medium-good`, `medium-bad`, `bad` | kelompok `medium-*` = borderline, paling penting |
| **domain check** | jumlah REJECT di foto konveyor pada threshold config / kalibrasi | foto semua-OK → REJECT = false positive |
| **latency** | ms per roti (CPU, setelah pemanasan) | target PRD ≤ 100 ms |
| **gt_compare: miscount** | selisih mutlak jumlah keputusan dan roti diharapkan, dibagi roti diharapkan; duplikat, miss, dan keputusan tak cocok dilaporkan terpisah | target PRD < 2%; hanya untuk video SYNTHETIC |

Skor A = `100 · P(kempes)` dari logistic regression. Skor B = `clip(50 · mentah / ref, 0, 100)` dengan
`ref` = persentil `ref_percentile` skor leave-one-out roti normal train; jadi B ≈ 50 berarti "seaneh
roti normal train yang paling aneh", dan margin normal vs kempes B sempit (studio: 36–51 vs 50–64).

---

## 8. Referensi Makefile

Diverifikasi dengan `make -n <target>`. Variabel yang bisa di-override: `CONFIG` (default
`configs/anomaly.yaml`), `DOMAIN_IMG` (default `train-data/domain-check/conveyor_good_01.webp`),
`APP_CONFIG` (default `configs/baseline.yaml`), `DEMO_VIDEO` (default `samples/demo.mp4`),
`CAMERA` (default `0`), `UV` (default `uv`).

| Target | Menjalankan | Hasil |
|---|---|---|
| `setup` | `uv venv --python 3.14 .venv`; `uv pip install --python .venv/bin/python --torch-backend cpu -r requirements-dev.txt`; `uv pip install --python .venv/bin/python --no-deps -e .` | `.venv` siap |
| `lock` | `uv pip compile pyproject.toml --python-version 3.14 --torch-backend cpu ... -o requirements.txt` (+ `--extra dev -o requirements-dev.txt`), lalu `sed` membuang label `+cpu` | `requirements*.txt` terkunci |
| `test` | `.venv/bin/pytest` | test |
| `lint` | `.venv/bin/ruff check .`; `.venv/bin/ruff format --check .` | lint + cek format |
| `format` | `.venv/bin/ruff check --fix .`; `.venv/bin/ruff format .` | perbaiki otomatis |
| `manifest` | `python -m loaflens.data --config configs/baseline.yaml` | `train-data/manifest.csv` (tidak menimpa) |
| `audit` | `python -m loaflens.audit --config configs/baseline.yaml` | `outputs/audit.md`, `docs/audit_m1.md` |
| `train` | `python -m loaflens.train --config configs/baseline.yaml` | `outputs/baseline/` |
| `evaluate` | `python -m loaflens.evaluate --config configs/baseline.yaml` | `outputs/baseline/`, `docs/results_m3_baseline.md` |
| `train-anomaly` | `python -m loaflens.train --config configs/anomaly.yaml` | `outputs/anomaly/` |
| `evaluate-anomaly` | `python -m loaflens.evaluate --config configs/anomaly.yaml` | `outputs/anomaly/`, `docs/results_m3_anomaly.md` |
| `calibrate` | `python -m loaflens.calibrate --config $(CONFIG)` | `outputs/<run>/calibration.yaml` |
| `compare` | `python -m loaflens.compare configs/baseline.yaml configs/anomaly.yaml` | `outputs/compare.md`, `docs/results_m3.md` |
| `domain-check` | `python -m loaflens.evaluate --config $(CONFIG) --domain-check "$(DOMAIN_IMG)"` | `outputs/<run>/domain_check.{jpg,md}` |
| `demo-video` | `python -m loaflens.synth --config $(APP_CONFIG) --out $(DEMO_VIDEO)` | `samples/demo.mp4` + `demo.gt.json` |
| `demo` | `python -m loaflens.app --config $(APP_CONFIG) --source video:$(DEMO_VIDEO) --headless --out outputs/demo_out.mp4 --gt samples/demo.gt.json` | video beranotasi + `outputs/app/` |
| `demo-live` | `python -m loaflens.app --config $(APP_CONFIG) --source video:$(DEMO_VIDEO)` | jendela |
| `camera` | `python -m loaflens.app --config $(APP_CONFIG) --source camera:$(CAMERA)` | jendela dari kamera |
| `clean` | `rm -rf .venv .pytest_cache .ruff_cache` | hapus venv & cache (tidak menyentuh `outputs/`) |

Semua `python` di atas = `.venv/bin/python`. Ground truth `demo` mengikuti `DEMO_VIDEO`
(`<video>.gt.json`).

---

## 9. Referensi CLI

Semua CLI: `python -m loaflens.<modul>` (pakai `.venv/bin/python` atau venv aktif), dijalankan dari
root repo (path di config relatif terhadap direktori kerja). Opsi di bawah sesuai `--help`.

| Modul | Opsi | Keterangan |
|---|---|---|
| `loaflens.data` | `--config CONFIG` (wajib), `--force` | bangun manifest + split; `--force` = timpa manifest (split ulang!) |
| `loaflens.audit` | `--config CONFIG` (wajib) | audit dataset → `audit.report` (+ `audit.docs_copy`) |
| `loaflens.train` | `--config CONFIG` (wajib) | latih scorer dari split train → `output.dir` |
| `loaflens.evaluate` | `--config CONFIG` (wajib), `--domain-check FOTO` | evaluasi lengkap; dengan `--domain-check` **hanya** domain check satu foto |
| `loaflens.calibrate` | `--config CONFIG` (wajib) | saran threshold dari CV train+val → `calibration.yaml` |
| `loaflens.compare` | `configs [configs ...]` (posisional) | tabel dari `metrics.json` config yang sudah di-evaluate → `outputs/compare.md` + `docs/results_m3.md` |
| `loaflens.synth` | lihat tabel di bawah | video demo SYNTHETIC + ground truth |
| `loaflens.app` | lihat tabel di bawah | demo real-time |

**`python -m loaflens.synth`**

| Opsi | Default | Arti |
|---|---|---|
| `--config` | `configs/baseline.yaml` | sumber parameter `segment` + `track` (`camera.roi` diabaikan untuk ground truth) |
| `--out` | `samples/demo.mp4` | video keluaran; ground truth ditulis ke `<out>.gt.json` |
| `--images` | `conveyor_mixed_01.webp conveyor_good_01.webp` (di `train-data/domain-check/`) | urutan foto yang ditampilkan |
| `--direction` | `track.direction` config | `bottom_to_top` \| `left_to_right` \| `right_to_left` \| `top_to_bottom` |
| `--speed` | 8.0 | px/frame (> 0) |
| `--fps` | 30.0 | fps video |
| `--height` | 900 | panjang jendela searah scroll (px), harus < tinggi foto |
| `--repeat` | 2 | ulang urutan foto (≥ 1) |

**`python -m loaflens.app`**

| Opsi | Default | Arti |
|---|---|---|
| `--config` | `configs/baseline.yaml` | config (scorer + `segment`/`track`/`camera`/`app`) |
| `--source` | wajib | `video:<path>` atau `camera:<id>` (id angka ≥ 0) |
| `--headless` | mati | tanpa jendela (wajib di Linux tanpa display) |
| `--out` | — | tulis video beranotasi (`.mp4` = mp4v, lainnya MJPG) |
| `--max-frames` | — | berhenti setelah N frame |
| `--gt` | — | sidecar ground truth video SYNTHETIC → `app.gt_report` |
| `--synthetic` | mati | paksa banner SYNTHETIC (mis. kamera yang merekam layar) |

App bisa dijalankan dengan `--config configs/anomaly.yaml` (memakai `outputs/anomaly/model.pt`),
tetapi tidak disarankan di konveyor karena domain gap (§1).

---

## 10. Pengembangan

```bash
make test
```

```bash
make lint
```

```bash
make format
```

- `make test`: pytest pada `tests/` (100 test, gambar sintetis berlabel "SYNTHETIC", tanpa
  kamera/display, ±20 detik).
- `make lint`: `ruff check` + `ruff format --check` (aturan di `pyproject.toml`, panjang baris 100).
- `make lock`: perbarui `requirements*.txt` setelah mengubah `pyproject.toml`. Butuh akses ke
  `download-r2.pytorch.org` (index torch CPU). Di sandbox yang memblokir host itu `make lock` gagal;
  di M4 pin diubah manual sebagai gantinya. Setelah lock, jalankan `make setup` lagi.

```bash
make lock
```

Konvensi kode, aturan data, dan cara kerja (satu milestone per giliran, jalankan `pytest` + `ruff`
setelah perubahan, update status di `docs/PRD.md`) ada di [`CLAUDE.md`](../CLAUDE.md). Ringkas: type
hints, fungsi kecil, I/O dipisah dari logika skor, semua parameter di `configs/*.yaml`, log pakai
`logging`, setiap scorer mengimplementasikan `fit` / `score` / `save` / `load`.

---

## 11. Troubleshooting / FAQ

| Gejala | Penyebab & solusi |
|---|---|
| `ERROR ... Tidak ada display (DISPLAY kosong) — pakai --headless` | Linux tanpa display (server/SSH/container). Pakai `make demo` atau tambah `--headless --out outputs/x.mp4`. |
| `ERROR ... Model outputs/baseline/model.joblib tidak ada — jalankan make train dulu` | Model belum dilatih (repo baru di-clone; `outputs/` tidak di-commit). Jalankan `make train` (scorer B: `make train-anomaly`). `evaluate`/`calibrate`/`domain-check` tanpa model berhenti dengan `FileNotFoundError ... model.joblib` — solusinya sama. |
| `FileNotFoundError: Bobot backbone tidak ditemukan ...` (scorer B) | Bobot ResNet18 belum ada dan inferensi tidak mengunduh. Jalankan `make train-anomaly` sekali dengan internet, atau salin `resnet18-f37072fd.pth` ke `~/.cache/torch/hub/checkpoints/` atau ke `model.params.weights_dir`. |
| `ERROR ... sumber tidak bisa dibuka: camera:0` | Indeks kamera salah atau dipakai aplikasi lain: coba `make camera CAMERA=1`, tutup Zoom/browser. macOS: beri izin kamera untuk aplikasi terminal (System Settings → Privacy & Security → Camera). Linux: cek `/dev/video*` dan izin grup `video`. |
| `file video tidak ditemukan: samples/demo.mp4` | `samples/` tidak di-commit. Jalankan `make demo-video`. |
| Terlalu banyak objek terdeteksi (bayangan, garis lantai, bingkai layar) | Persempit `camera.roi`; naikkan `segment.obj_min_area_frac`; naikkan `segment.obj_min_value` untuk bayangan gelap; biarkan `drop_border: true`. |
| Terlalu sedikit roti terdeteksi / roti berdempetan jadi satu | Turunkan `obj_min_area_frac` atau naikkan `obj_max_area_frac` (fraksi luas ROI); turunkan `obj_min_value` bila roti gelap hilang; roti yang menempel memang di luar asumsi POC (PRD §11). Cek visual dengan `make domain-check` pada frame/foto dari kamera itu. |
| Satu roti diputuskan dua kali / dapat id baru di tengah jalan | Roti bergerak lebih jauh dari `track.max_distance` per frame → naikkan (mis. 0.12); deteksi berkedip → naikkan `track.max_missed`; pastikan `track.direction` benar dan garis (`line_pos`) berada di area roti terlihat utuh. |
| Roti lewat tanpa keputusan | Arah salah (`track.direction`), garis di luar ROI, atau roti muncul pertama kali sudah di seberang garis (sengaja tidak diputuskan untuk menghindari hitung ganda) → geser `line_pos` lebih ke tengah. |
| Semua/banyak roti REJECT dengan scorer B | Domain gap: memory bank B berisi roti studio. Pakai scorer A (`configs/baseline.yaml`) untuk keputusan; B perlu refit dari roti bagus kamera konveyor (§12). |
| `WARNING ... Manifest sudah ada, TIDAK ditimpa` | Disengaja: split beku. Foto baru masuk hanya lewat `python -m loaflens.data --config configs/baseline.yaml --force` (split ulang; §4). |
| `WARNING ... GT dibuat untuk ... perbandingan tidak valid` | `track.direction`/`line_pos` berubah sejak video dibuat. Jalankan ulang `make demo-video`. |
| `ConfigError: ...` | Nilai config di luar rentang atau kunci salah ketik; pesan menyebut kunci & nilai valid (§5). |
| `ImportError: libGL.so.1` (Linux minimal) | `opencv-python` (build GUI) butuh library sistem: `sudo apt install libgl1 libglib2.0-0`. |
| `cv2` aneh / jendela tidak muncul setelah upgrade | `opencv-python-headless` lama masih terpasang; lihat §2 "Upgrade `.venv` lama". |
| `make setup` gagal mengunduh torch | Host `download.pytorch.org` / `download-r2.pytorch.org` diblokir; buka akses jaringan, atau jalankan dari jaringan lain. |
| `make calibrate` / `make domain-check` menghasilkan file di `outputs/anomaly/` | Default `CONFIG` Makefile = `configs/anomaly.yaml`; tambahkan `CONFIG=configs/baseline.yaml`. |

---

## 12. Langkah berikutnya

1. **Kumpulkan data domain konveyor:** foto/video dari kamera & sudut konveyor yang sebenarnya —
   roti bagus (≥ 50–100, untuk refit B dan mengecek FPR A) dan roti kempes (untuk mengukur recall
   kedua scorer). Simpan per batch/sesi, label dikonfirmasi manusia.
2. **Refit scorer B** dengan memory bank dari roti bagus kamera konveyor (cukup roti normal, tanpa
   label kempes), lalu ulangi evaluasi + `make compare`. Perlu folder normal/config baru (mis.
   `data.normal_dir` ke foto konveyor yang di-crop per roti) — usulkan dulu sebelum mengubah kode.
3. **Jalankan demo kamera-ke-layar** ≥ 5 menit dan laporkan terpisah dari uji file video (§6.8).
4. **Jawab pertanyaan terbuka PRD §10** bersama client: definisi kempes resmi, SKU, kecepatan
   konveyor, posisi kamera, toleransi false positive vs false negative.
5. **M5 (opsional):** benchmark di Raspberry Pi 5 / mini PC, export ONNX/OpenVINO untuk scorer B,
   hook sinyal (serial/file) untuk aktuator.
