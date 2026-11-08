# Kickoff Prompt untuk Claude Code (LoafLens / repo `loaf-lens`)

> Cara pakai: taruh `CLAUDE.md` di root repo dan `docs/PRD.md` di folder docs, buka Claude Code
> di root repo (idealnya mulai dengan Plan Mode / Shift+Tab 2x), lalu tempel prompt di bawah.

---

## PROMPT (tempel apa adanya)

```
Kamu akan membantu saya membangun POC bernama LoafLens (repo: loaf-lens, package Python: loaflens): sistem visi komputer low-cost
untuk mendeteksi roti KEMPES (afkir) di konveyor lewat kamera, memakai ML yang dilatih dari
gambar kita sendiri (BUKAN LLM / API AI eksternal, harus jalan lokal di CPU).

Konteks lengkap ada di:
- CLAUDE.md  (aturan kerja & konvensi repo)
- docs/PRD.md (tujuan, scope, requirement, acceptance criteria, risiko, pertanyaan terbuka)

Baca kedua file itu dulu sampai selesai, lalu JANGAN langsung menulis kode. Lakukan berurutan:

1. Ringkas pemahamanmu tentang tujuan, batasan, dan definisi sukses dalam 8 baris maksimal.
2. Daftar pertanyaan terbuka/asumsi yang menurutmu berisiko (maks 8, urut dari paling berdampak),
   sertakan asumsi default yang akan kamu pakai kalau saya tidak menjawab.
3. Usulkan rencana implementasi per milestone (M0–M4 di PRD), tiap milestone berisi:
   deliverable, file yang akan dibuat, cara saya memverifikasinya (perintah yang bisa saya jalankan).
4. Tunggu persetujuan saya sebelum mengerjakan milestone apa pun.

Setelah saya setuju, kerjakan SATU milestone per giliran. Di akhir tiap milestone:
- jalankan test/lint yang relevan dan tampilkan hasilnya,
- tulis ringkasan apa yang berubah + cara menjalankannya,
- update bagian "Status" di docs/PRD.md,
- berhenti dan minta konfirmasi sebelum lanjut.

Aturan keras:
- Jangan menambah dependency berat tanpa menjelaskan alasan dan alternatif yang lebih ringan.
- Jangan menaruh data/gambar asli ke git (lihat CLAUDE.md soal data/).
- Semua threshold dan parameter di config YAML, bukan hardcode.
- Kalau hasil evaluasi buruk, laporkan apa adanya beserta hipotesis penyebabnya; jangan
  mengubah metrik atau threshold hanya agar angka terlihat bagus.
```

---

## Prompt lanjutan yang berguna (siapkan untuk nanti)

**Sebelum mulai M1 (data):**
```
Saya sudah menaruh foto di data/raw/. Buat skrip audit dataset: jumlah gambar, resolusi,
distribusi kecerahan, duplikat/near-duplicate, dan gambar rusak. Jangan ubah file asli.
Beri saya laporan apa yang kurang untuk training yang andal.
```

**Setelah baseline jalan (M2/M3):**
```
Bandingkan baseline fitur-bentuk+scikit-learn vs anomaly detection pada hold-out set yang sama.
Tampilkan tabel: AUROC, recall roti kempes pada false-positive rate 5%, latency per roti (CPU),
ukuran model. Lalu sebutkan 5 gambar yang paling sering salah diklasifikasi dan hipotesis kenapa.
```

**Saat demo kamera/layar (M4):**
```
Tambahkan mode --source camera dan --source video:file.mp4. Pastikan satu roti hanya diputuskan
sekali (tracking + garis keputusan), dan simpan snapshot tiap roti yang di-reject ke outputs/rejects/
beserta skornya.
```

**Review mandiri sebelum demo ke client:**
```
Review repo ini seolah kamu reviewer skeptis. Cari: kebocoran data train->test, threshold yang
dituning di test set, asumsi domain (layar vs roti asli) yang belum diuji, dan klaim di README
yang tidak didukung hasil eksperimen.
```
