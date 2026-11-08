# Uji video konveyor 10 detik (2026-11-08)

**Video:** `Golden_Buns_on_the_Factory_Conveyor.mp4` dari user, disimpan lokal sebagai
`samples/conveyor_real_10s.mp4` (gitignored, tidak di-commit). Spesifikasinya:

- 360×640 px, 24 fps, 240 frame
- roti bergerak dari atas ke bawah
- konveyor mesh gelap, kamera dari atas dengan perspektif

**Pipeline:** scorer A (baseline), `configs/baseline.yaml` apa adanya:

- threshold 50
- `track.direction: top_to_bottom`
- `track.line_pos: 0.5`

Model **tidak** dilatih ulang, dan config maupun threshold **tidak** disetel dari video ini.

Untuk membuat ulang:

```bash
.venv/bin/python -m loaflens.app --config configs/baseline.yaml --source video:samples/conveyor_real_10s.mp4 --headless --out outputs/real_out.mp4
```

## Hasil

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

## Diagnosis: resolusi terlalu rendah

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

## Rekomendasi
1. **Syarat kamera (paling murah):** pastikan lebar roti di frame **≥ sekitar 130 px**. Caranya:
   kamera lebih dekat, zoom, atau resolusi 1080p atau lebih.
2. **Perbaikan model:**
   - tambah data latih dari kamera konveyor yang sebenarnya;
   - atau augmentasi resolusi saat training (foto latih diperkecil lalu diperbesar lagi);
   - lalu kalibrasi ulang threshold di data validasi, bukan di video uji ini.
3. Tambah batas minimum ukuran roti di `segment.obj_min_area_frac` supaya roti yang terlalu kecil
   tidak diskor. Batas ini harus diturunkan dari syarat kamera di atas, bukan dari hasil video ini.

Gambar: `docs/images/real_video_frames.jpg`, `docs/images/real_video_decisions.jpg`,
`docs/images/real_video_demo.gif`.
