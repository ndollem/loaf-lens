# samples/

Video demo **SYNTHETIC — bukan untuk evaluasi akurasi**. Isi folder ini tidak di-commit (kecuali file ini).

```bash
make demo-video   # → samples/demo.mp4 + samples/demo.gt.json (ground truth, ditandai "synthetic": true)
make demo         # app headless + banding ground truth → outputs/demo_out.mp4, outputs/app/gt_compare.md
```

Video dibuat dari foto diam `train-data/domain-check/` yang digeser (lihat README → "Video demo
SYNTHETIC"). Video konveyor asli dari client, bila ada, juga diletakkan di sini (tetap tidak di-commit), mis.
`samples/conveyor_real_10s.mp4` (uji 10 detik, lihat [`docs/results_real_video.md`](../docs/results_real_video.md)).
