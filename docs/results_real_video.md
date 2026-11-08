# Uji video konveyor 10 detik (2026-11-08)

**Video:** `Golden_Buns_on_the_Factory_Conveyor.mp4` dari user, disimpan lokal sebagai
`samples/conveyor_real_10s.mp4` (gitignored, tidak di-commit). Spesifikasinya:

- 360×640 px, 24 fps, 240 frame
- roti bergerak dari atas ke bawah
- konveyor mesh gelap, kamera dari atas dengan perspektif

Video ini diperlakukan sebagai **data uji hold-out**: tidak dipakai untuk memilih metode, fitur,
parameter, maupun threshold. Ada dua run:

| run | model | kempes tertangkap | roti OK ikut ter-reject |
|---|---|---|---|
| [Run 1](#run-1-model-lama) | scorer A lama (crop 192 px, tanpa augmentasi) | 1/1 | **7/8** (gagal) |
| [Run 2](#run-2-setelah-perbaikan-resolusi) | scorer A tahan resolusi (crop kanonik 64 px + augmentasi resolusi) | 1/1 | **0/8** @ threshold 50 |

Untuk membuat ulang (memakai model yang sedang ada di `outputs/baseline/`):

```bash
.venv/bin/python -m loaflens.app --config configs/baseline.yaml --source video:samples/conveyor_real_10s.mp4 --headless --out outputs/real_out.mp4
```

## Run 1 (model lama)

**Pipeline:** scorer A (baseline), `configs/baseline.yaml` versi saat itu (`crop_size: 192`):

- threshold 50
- `track.direction: top_to_bottom`
- `track.line_pos: 0.5`

Model **tidak** dilatih ulang, dan config maupun threshold **tidak** disetel dari video ini.

### Hasil

9 roti melintasi garis keputusan dalam 10 detik, dan tiap roti diputuskan tepat sekali. Roti yang
sudah berada di bawah garis sejak frame pertama tidak pernah melintas, jadi tidak diputuskan.
Contohnya roti kempes #10 di frame awal.

| track | frame | skor | keputusan app | penilaian visual |
|---|---|---|---|---|
| #9 | 8 | 60 | REJECT | OK |
| #8 | 46 | 77 | REJECT | OK |
| #7 | 77 | 100 | REJECT | **kempes** (puncak cekung berbentuk bintang) |
| #6 | 93 | 80 | REJECT | OK (lekukan kecil)* |
| #5 | 124 | 75 | REJECT | OK |
| #4 | 150 | 60 | REJECT | OK |
| #3 | 167 | 49 | OK | OK |
| #2 | 205 | 74 | REJECT | OK (lekukan kecil)* |
| #1 | 235 | 94 | REJECT | OK |

\* Lekukan kecil dinilai OK, mengikuti keputusan user untuk roti #12/#19 di
`docs/domain_check_mixed.md`. Penilaian visual dibuat oleh Claude dan belum dikonfirmasi user.

| ringkasan | nilai |
|---|---|
| Roti diputuskan / duplikat | 9 / 0 |
| Kempes tertangkap | 1/1 |
| **Roti OK ikut ter-reject** | **7/8** |
| Kecepatan | sekitar 147 FPS, segment+track 2 ms/frame |

**Kesimpulan jujur: di video ini scorer A gagal.** Roti kempes tertangkap, tapi hampir semua roti
OK ikut ter-reject. Tracking dan penghitungan berjalan benar.

### Diagnosis: resolusi terlalu rendah

Lebar roti di video ini hanya sekitar 55–75 px, sedangkan di foto konveyor sebelumnya sekitar
130–190 px. Untuk menguji dugaan ini, foto konveyor lama diperkecil ke skala video (×0.383) lalu
diskor ulang dengan model yang sama. Hanya resolusinya yang berubah:

| foto | resolusi asli: OK ter-reject | diperkecil ×0.383: OK ter-reject | kempes tertangkap (asli → kecil) |
|---|---|---|---|
| `conveyor_good_01.webp` (15 OK) | 0/15 | **8/15** | – |
| `conveyor_mixed_01.webp` (15 OK + 5 kempes) | 0/15 | **5/15** | 5/5 → 5/5 |

Penurunan resolusi saja sudah cukup menimbulkan salah tolak yang sama seperti di video. Penyebabnya:
fitur tekstur dan kilap scorer A dihitung dari crop yang di-resize ke 192 px. Roti kecil yang
diperbesar dari sekitar 60 px menjadi buram, dan tekstur kompresi video berubah, sehingga fitur
bergeser ke arah "kempes". Model dilatih dari foto studio beresolusi tinggi saja.

### Rekomendasi (saat itu)
1. **Syarat kamera (paling murah):** pastikan lebar roti di frame **≥ sekitar 130 px**. Caranya:
   kamera lebih dekat, zoom, atau resolusi 1080p atau lebih.
2. **Perbaikan model:**
   - tambah data latih dari kamera konveyor yang sebenarnya;
   - atau augmentasi resolusi saat training (foto latih diperkecil lalu diperbesar lagi);
   - lalu kalibrasi ulang threshold di data validasi, bukan di video uji ini.
3. Tambah batas minimum ukuran roti di `segment.obj_min_area_frac` supaya roti yang terlalu kecil
   tidak diskor. Batas ini harus diturunkan dari syarat kamera di atas, bukan dari hasil video ini.

Gambar Run 1 sudah diganti dengan hasil Run 2 di `docs/images/` (versi lama ada di riwayat git,
commit `a6e5245`).

## Run 2 (setelah perbaikan resolusi)

### Metode
Masalahnya: fitur tekstur/kilap bergeser saat resolusi roti turun. Perbaikan dipilih dengan dua
prinsip: buat fitur **tidak peka resolusi secara konstruksi**, lalu latih classifier agar **terbiasa
dengan citra beresolusi rendah dan terkompresi**.

1. **Resolusi kanonik kecil.** Fitur dihitung dari crop roti dengan sisi terpanjang **64 px**
   (dulu 192 px). Roti studio (~250 px), foto konveyor (130–190 px), dan video (55–75 px) semuanya
   diperkecil atau hampir tidak diubah ke ukuran yang sama, sehingga tampak mirip bagi fitur.
   Semua parameter piksel fitur diskalakan dengan faktor yang sama, 64/192 = 1/3, tanpa disetel:
   `erode_px` 9 → 3, `crease_px` 6 → 2, `texture_blur_sigma` 1.5 → 0.5.
2. **Augmentasi resolusi saat fit.** Tiap foto latih ditambah 4 salinan: gambar diperkecil
   (INTER_AREA) sampai sisi terpanjang roti 50–200 px (log-uniform), lalu dikompres JPEG kualitas
   40–90, lalu disegmentasi ulang. Salinan berlabel sama dengan aslinya. RNG = seed config + CRC
   isi gambar (deterministik). Di cross-validation, salinan hanya dipakai di fold latih; skor fold uji
   selalu dari gambar asli.

Parameter ada di `configs/baseline.yaml` (`model.params.crop_size`, `model.params.augment`,
`model.params.features`). Tidak ada dependency baru.

### Validasi (tanpa video)
Metode dipilih dan dibekukan **sebelum** video dijalankan, hanya dengan:

- (a) CV 5-fold per grup pada train+val studio (test tidak ikut);
- (b) dua foto konveyor domain check, resolusi asli dan diperkecil (INTER_AREA). **Catatan jujur:**
  kedua foto ini sudah pernah dilihat (dipakai untuk domain check sebelumnya), jadi di sini mereka
  berfungsi sebagai **set validasi**, bukan uji independen. Label: foto `conveyor_good_01` semua OK;
  foto `conveyor_mixed_01` kempes di posisi `labels.json`, sisanya OK (#12/#19 dikonfirmasi user OK);
- (c) simulasi degradasi foto studio fold uji CV (roti diperkecil ke 60/75 px + JPEG kualitas 50).

Tabel (b), threshold config 50. Lebar roti di foto asli sekitar 130–190 px; di video sekitar 55–75 px
(≈ skala ×0.38–0.45). Dibuat ulang dengan
`make domain-check DOMAIN_IMG=train-data/domain-check/<foto>.webp DOMAIN_SCALE=<s>`:

| skala foto | lebar roti ≈ | foto bagus: OK ter-reject (lama → baru) | foto campuran: kempes tertangkap (lama → baru) | foto campuran: OK ter-reject (lama → baru) |
|---|---|---|---|---|
| ×1.0 | 130–190 px | 0/15 → **0/15** | 5/5 → **5/5** | 0/15 → **0/15** |
| ×0.6 | 80–115 px | 1/15 → **0/15** | 5/5 → **5/5** | 0/15 → **0/15** |
| ×0.45 | 60–85 px | 6/15 → **0/15** | 5/5 → **5/5** | 5/15 → **0/15** |
| ×0.383 | 50–73 px | 8/15 → **0/15** | 5/5 → **5/5** | 5/15 → **0/15** |
| ×0.3 | 40–57 px | 14/16 → **2/16** | 4/5 → **5/5** | 7/15 → **1/15** |

Di ×0.3 segmentasi foto bagus mendeteksi 16 objek; roti < ~55 px tetap menjadi titik lemah.

Uji tambahan (harness eksperimen, di luar repo): foto yang diperkecil **dan** dikompres JPEG
kualitas 50, meniru kompresi video:

| skala + JPEG 50 | foto bagus: OK ter-reject (lama → baru) | foto campuran: OK ter-reject (lama → baru) |
|---|---|---|
| ×0.45 | 11/15 → **0/15** | 4/15 → **1/15*** |
| ×0.383 | 11/15 → **0/15** | 7/15 → **1/15*** |

\* Satu-satunya REJECT ini adalah blob gabungan: roti kempes #6 dan roti OK #7 menyatu saat
segmentasi pada citra terkompresi. Keempat roti kempes lain yang terdeteksi terpisah tertangkap.

Tabel (a) dan (c), CV out-of-fold train+val (20 normal / 20 kempes), threshold 50:

| | AUROC (lama → baru) | normal ter-reject (lama → baru) | kempes lolos (lama → baru) |
|---|---|---|---|
| CV, foto asli | 0.992 → **1.000** | 1/20 → **0/20** | 0/20 → **0/20** |
| CV, fold uji diperkecil ke roti 60 px + JPEG 50 | 0.992 → **1.000** | 5/20 → **0/20** | 0/20 → **0/20** |
| CV, fold uji diperkecil ke roti 75 px + JPEG 50 | 0.982 → **1.000** | 4/20 → **0/20** | 0/20 → **0/20** |

**Ablasi** (harness, sebelum dibekukan): crop 64 saja sudah memperbaiki tabel (b) tanpa kompresi,
tetapi dengan JPEG 50 di ×0.383 masih me-reject 2/15 roti foto bagus. Augmentasi saja (crop 192)
melewatkan 1 roti kempes di ×0.6 dan ×0.45. Gabungan keduanya paling stabil, jadi itu yang dipilih.

**Studio (`make evaluate`, test dievaluasi sekali dengan model baru):** test 8/8 benar (AUROC 1.000)
di threshold 50 maupun terkalibrasi, sama seperti model lama. CV AUROC 0.988 ± 0.025 → **1.000 ± 0.000**.
Satu-satunya salah CV lama (`bagus (10)`, skor 84) kini 42.8.

**Threshold.** Config tetap **50**. Aturan kalibrasi (`make calibrate`, CV train+val, FPR ≤ 0.05)
menyarankan **40.4** (lama 48.9). Saran ini memakai jatah 1 false positive dari 20 roti normal CV
(`bagus (10)`, 42.8), sedangkan di 50 CV sudah FPR 0/20 dengan recall 20/20. Sesuai keputusan user
sebelumnya, config tidak diubah otomatis.

### Hasil video (dijalankan sekali, setelah metode dibekukan)
Video dijalankan **satu kali** dengan model dan config final. Tidak ada iterasi setelahnya.

| track | frame | skor Run 1 → **Run 2** | keputusan Run 2 (thr 50) | penilaian visual |
|---|---|---|---|---|
| #9 | 8 | 60 → **0** | OK | OK |
| #8 | 46 | 77 → **5** | OK | OK |
| #7 | 77 | 100 → **99** | REJECT | **kempes** (puncak cekung berbentuk bintang) |
| #6 | 93 | 80 → **3** | OK | OK (lekukan kecil)* |
| #5 | 124 | 75 → **4** | OK | OK |
| #4 | 150 | 60 → **7** | OK | OK** |
| #3 | 167 | 49 → **2** | OK | OK |
| #2 | 205 | 74 → **4** | OK | OK (lekukan kecil)* |
| #1 | 235 | 94 → **48** | OK | OK |

Track id dan frame keputusan sama persis dengan Run 1 (segmentasi dan tracking tidak berubah).

| ringkasan | Run 1 | **Run 2** |
|---|---|---|
| Roti diputuskan / duplikat | 9 / 0 | 9 / 0 |
| Kempes tertangkap | 1/1 | **1/1** (skor 99) |
| Roti OK ikut ter-reject @ threshold 50 | 7/8 | **0/8** |
| Roti OK ikut ter-reject @ saran kalibrasi (48.9 / 40.4) | 7/8 | **1/8** (#1, skor 47.7) |
| Kecepatan (headless) | ~147 FPS | ~203 FPS; segment+track 1.9 ms/frame, skor ~10 ms/roti rata-rata |

\* Lekukan kecil dinilai OK, mengikuti keputusan user untuk roti #12/#19 di
`docs/domain_check_mixed.md`. Penilaian visual dibuat oleh Claude dan belum dikonfirmasi user.

\*\* Roti #4 tampak mulus di frame keputusan (150). Setelah lewat garis (sekitar frame 200–239)
puncaknya tampak berubah menjadi lekukan berbentuk bintang. Ini teramati di videonya sendiri
(kemungkinan artefak video hasil edit/generate); keputusan diambil di garis, saat roti tampak OK.

**Kesimpulan jujur:** setelah perbaikan resolusi, scorer A memisahkan roti kempes dari roti OK di video
ini: kempes 99, roti OK 0–7, kecuali #1 = 47.7. Skor #1 hanya 2.3 poin di bawah threshold 50
dan akan REJECT pada saran kalibrasi 40.4, jadi marginnya tipis untuk roti itu.

### Caveat
- **n kecil:** 9 keputusan, 1 roti kempes. Ini indikasi, bukan evaluasi akurasi.
- **Foto domain check dipakai sebagai validasi**, karena sudah pernah dilihat. Satu-satunya uji yang
  benar-benar baru untuk metode ini adalah video, dan video hanya dijalankan sekali.
- **Label video** adalah penilaian visual Claude dari crop 55–75 px, belum dikonfirmasi user.
- Model masih dilatih dari 48 foto studio. Roti < ~55 px (skala ×0.3) masih menghasilkan salah tolak
  di validasi. Validasi bersih sampai lebar roti ~50–60 px (×0.383); untuk kamera produksi, lebih
  lebar lebih aman (mis. ≥ ~80 px).
- Rekomendasi Run 1 tetap berlaku: data latih dari kamera konveyor asli, lalu kalibrasi threshold di
  data validasi kamera itu.

Gambar Run 2: `docs/images/real_video_frames.jpg`, `docs/images/real_video_decisions.jpg`,
`docs/images/real_video_demo.gif`.
