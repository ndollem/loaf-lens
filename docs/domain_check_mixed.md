# Domain check: foto konveyor campuran (2026-11-08)

Foto: `train-data/domain-check/conveyor_mixed_01.webp` (dari user). Isinya roti bagus dan kempes di konveyor mesh gelap, kamera dari atas agak miring. Model **tidak** dilatih ulang dan threshold **tidak** disetel dari foto ini. Kedua scorer masih dilatih dari 48 foto studio (`train-data/good-loaf`, `train-data/bad-loaf`).

`segment_all` mendeteksi **20 roti**. Dua roti yang terpotong di tepi atas dan bawah frame dibuang.

## Label acuan (visual oleh Claude; #12 & #19 dikonfirmasi user = OK)
Nomor roti mengikuti gambar anotasi, diurutkan dari atas ke bawah.

- **Kempes jelas** (puncak cekung, berkerut): #1, #3, #6, #17, #20
- **Borderline** (lekukan kecil di puncak): #12, #19 — **dikonfirmasi user: masih bisa diterima (OK)**
- **Bagus**: sisanya (13 roti)

## Hasil

| roti | label acuan | skor A (baseline) | skor B (anomaly) |
|---|---|---|---|
| #1 | kempes | 85 | 63 |
| #3 | kempes | 71 | 64 |
| #6 | kempes | 95 | 69 |
| #17 | kempes | 66 | 75 |
| #20 | kempes | 94 | 70 |
| #12 | borderline | 15 | 59 |
| #19 | borderline | 27 | 57 |
| 13 roti bagus | bagus | 2–31 (median 5) | 55–69 |

| | A (thr 50 / kalibrasi 48.9) | B (thr 50 / kalibrasi 52.1) |
|---|---|---|
| REJECT | 5/20 | 20/20 |
| Kempes tertangkap | 5/5 | 5/5 |
| Roti OK ikut ter-reject | 0/15 | 15/15 |
| AUROC (label acuan, 5 kempes vs 15 OK) | 1.00 | 0.86 |

## Catatan
- **A memisahkan roti kempes dengan jelas** pada foto ini. Skor kempes 66–95, skor roti OK ≤ 31. Roti borderline (#12 = 15, #19 = 27) mendapat skor di atas sebagian besar roti bagus (median 5), tapi masih di bawah threshold.
- **B tidak bisa dipakai untuk keputusan di domain ini.** Semua skornya di atas threshold karena memory bank berisi roti studio, bukan roti konveyor. Urutan skornya masih ada sinyal (AUROC 0.86), tapi butuh refit memory bank dari roti bagus di kamera konveyor.
- **Satu foto, 20 roti:** ini indikasi, bukan evaluasi akurasi. Label acuan dari pengamatan visual; status #12 & #19 sudah dikonfirmasi user (OK).
- Gambar anotasi (tidak di-commit): `outputs/baseline/domain_check.jpg`, `outputs/anomaly/domain_check.jpg`. Untuk membuat ulang: `make domain-check CONFIG=configs/baseline.yaml DOMAIN_IMG=train-data/domain-check/conveyor_mixed_01.webp`.
