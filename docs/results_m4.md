# Hasil M4: pipeline real-time pada video demo SYNTHETIC (2026-11-08)

> **SYNTHETIC — bukan untuk evaluasi akurasi.** Video dibuat dengan menggeser dua foto konveyor diam.
> Angka di bawah menguji pipeline (segment → track → satu keputusan per roti → overlay/log). Angka ini
> bukan akurasi deteksi kempes di lini produksi.

## Setup

- **Pipeline:** `python -m loaflens.app` (`make demo`) dengan alur `capture → segment_all → CentroidTracker`
  `→ skor scorer A → threshold 50 → overlay/CSV/log`. Model `outputs/baseline/model.joblib` tidak
  dilatih ulang, threshold tidak diubah.
- **Video** (`make demo-video`, `samples/demo.mp4`, tidak di-commit):
  - foto `conveyor_mixed_01.webp` (5 kempes + 15 OK) lalu `conveyor_good_01.webp` (15 OK), diulang 2 kali;
  - 941x900 px, 30 fps, 949 frame (±32 detik);
  - roti bergerak `top_to_bottom` 8 px/frame;
  - bantalan polos di kedua ujung, dan roti yang terpotong sambungan foto ditutup warna konveyor;
  - label `SYNTHETIC` kecil tertulis di sudut bawah video.
- **Config:** `track.direction: top_to_bottom`, `line_pos 0.5`, `max_distance 0.08` (75 px),
  `max_missed 5`, `camera.roi: null`. Parameter segmentasi sama dengan M3.
- **Ground truth** (`samples/demo.gt.json`):
  - roti utuh per foto dihitung dengan `segment_all` (batas luas disetarakan ke luas frame);
  - posisi tiap roti per frame dihitung dari kecepatan scroll;
  - label diambil dari posisi di `train-data/domain-check/labels.json`: kempes = #1, #3, #6, #17, #20,
    sedangkan #12 dan #19 = OK (dikonfirmasi user), lihat [`domain_check_mixed.md`](domain_check_mixed.md);
  - keputusan app dicocokkan ke roti GT terdekat pada frame keputusan (≤ 47 px).
- **CPU:** Intel Xeon 4 vCPU (container), tanpa GPU.

## Penghitungan (PRD §9: tiap roti diputuskan tepat sekali, miscount < 2%)

| roti diharapkan | diputuskan | duplikat | miss | keputusan tak cocok | miscount |
|---|---|---|---|---|---|
| 70 | 70 | 0 | 0 | 0 | **0.0%** |

- Keputusan terjadi pada frame lintas yang diprediksi GT, atau 1 frame sesudahnya. Selisih 1 frame
  muncul karena segmentasi per frame sedikit berbeda dari segmentasi foto.
- Ada 71 track. Satu track tambahan adalah potongan roti di bawah label `SYNTHETIC` yang tertulis di
  sisi keluar video. Track itu lahir setelah garis, jadi tidak pernah diputuskan.
- **Uji kecepatan** (bukan tuning, 1 putaran = 35 roti per video): kecepatan 4, 16, dan 24 px/frame
  masing-masing menghasilkan 35/35 keputusan, 0 duplikat, dan 0 miss.

## Keputusan vs label acuan

| putaran | kempes di-REJECT | roti OK ikut REJECT | skor kempes (#1, #3, #6, #17, #20) | skor OK maks / median | borderline #12, #19 |
|---|---|---|---|---|---|
| 1 | **5/5** | **0/30** | 91, 74, 98, 59, 96 | 35 / 6 | 6, 29 (OK) |
| 2 | **5/5** | **0/30** | 86, 73, 98, 59, 96 | 36 / 6 | 6, 30 (OK) |

Bandingkan dengan skor di foto diam (M3 domain check): kempes 85, 71, 95, 66, 94; OK ≤ 31; #12 = 15,
#19 = 27. Skor di video sedikit bergeser karena dua hal: kompresi mp4 dan segmentasi per frame (Otsu
per frame, posisi roti saat melintas). Kempes terendah (#17, skor 59) dan OK tertinggi (36) masih
berada di sisi yang benar dari threshold 50. Tabel per roti lengkap ada di
`outputs/app/gt_compare.md` (dibuat ulang oleh `make demo`).

## Kecepatan (CPU)

| ukuran | nilai (run terakhir) |
|---|---|
| FPS loop penuh (decode + segment + track + skor + overlay + tulis mp4) | **80** (4 run: 64–87) |
| segment + track per frame | 6.2 ms (p95 8.0) |
| skor scorer A per roti | 5.6 ms (p95 6.0) (run pertama 7.7 ms) |

Jauh di dalam target ≤ 100 ms/roti. Mode jendela (`make demo-live`, diuji di bawah Xvfb selama 90
frame) mengikuti tempo video ±28–30 fps.

## Output yang diperiksa

- `outputs/demo_out.mp4` (video beranotasi), `outputs/app/decisions.csv` (70 baris).
- `outputs/rejects/` berisi 10 crop + 10 frame, dan `rejects.csv` mendapat 10 baris per run (file ini
  ditambahkan antar run).
- 70 baris log `DECISION id=.. frame=.. score=.. OK|REJECT`.
- Beberapa frame sudah diperiksa visual (`outputs/app/frame_*.jpg`):
  - kotak abu sebelum garis, hijau/merah sesudah keputusan;
  - id + skor tampil, label tetap menempel saat roti terus bergerak;
  - counter naik tepat sekali per roti;
  - banner SYNTHETIC tampil.

## Catatan jujur / batasan

1. **SYNTHETIC: scroll foto diam ≠ konveyor nyata.** Video ini tidak punya motion blur, perubahan
   cahaya, roti berputar atau bersentuhan, maupun noise sensor. Sambungan foto dan tambalan warna
   konveyor juga tidak alami. Penghitungan 0% miscount hanya membuktikan logika tracker pada gerak
   lurus yang mulus dengan roti yang terpisah.
2. **Skor bukan bukti baru.** Video memakai **2 foto yang sama** dengan domain check M3, jadi 5/5
   kempes dan 0/30 OK mengulang hasil foto tersebut. Putaran 2 juga konten yang sama, sehingga hanya
   menguji konsistensi. Scorer A di konveyor baru divalidasi pada 2 foto (20 + 15 roti) dengan label
   visual.
3. **Belum diuji di repo ini:**
   - demo **kamera-ke-layar** (`make camera`), karena tidak ada kamera/display di container. Langkah
     dan tips ada di README. PRD meminta ≥ 5 menit stabil dengan akurasi dilaporkan terpisah dari uji
     file video. **Masih harus dijalankan user**; moiré, exposure, dan sudut kamera bisa menggeser
     skor maupun segmentasi;
   - jalur `camera:<id>` hanya diperiksa sampai pesan error yang jelas saat kamera tidak ada.
4. Tracker mengasumsikan roti terpisah dan bergerak searah. Roti yang menempel bisa tersegmentasi
   jadi satu objek, yang menghasilkan miss atau skor salah (risiko PRD §11). Roti yang hilang tepat
   di garis lebih lama dari `max_missed` frame akan terlewat, bukan dihitung dua kali.
5. `track.direction` di config diubah ke `top_to_bottom` agar sesuai video demo. Untuk kamera
   sungguhan, arah dan `line_pos` wajib disesuaikan dengan gambar kamera.

## Update 2026-11-08: scorer A setelah perbaikan resolusi
Diulang dengan scorer A baru (crop kanonik 64 px + augmentasi resolusi; lihat
[`results_real_video.md`](results_real_video.md#run-2-setelah-perbaikan-resolusi)), `make demo`:
70/70 diputuskan tepat sekali, miscount 0%, kempes 10/10 REJECT, roti OK ikut REJECT 0/60
(sama seperti sebelumnya). Skor kempes terendah 66 vs roti OK tertinggi 15 (dulu 59 vs 36); ±94 FPS,
skor ≈ 3 ms/roti. Tetap SYNTHETIC — bukan untuk evaluasi akurasi.
