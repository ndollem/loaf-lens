"""Perbandingan A vs B: ``python -m loaflens.compare configs/baseline.yaml configs/anomaly.yaml``.

Membaca ``metrics.json`` hasil ``evaluate`` tiap config (tidak menghitung ulang) → tabel markdown
``outputs/compare.md`` + salinan ``docs/results_m3.md``. Teks di bawah penanda ``REKOMENDASI`` pada
salinan docs ditulis manual dan dipertahankan saat tabel dibuat ulang.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from loaflens.config import load_config
from loaflens.evaluate import GROUP_ORDER, HEADLINE, fmt

log = logging.getLogger(__name__)

OUT = Path("outputs/compare.md")
DOCS = Path("docs/results_m3.md")
MARKER = "<!-- REKOMENDASI: teks di bawah baris ini ditulis manual, dipertahankan compare.py -->"
NAMES = {"baseline": "A (baseline)", "anomaly": "B (anomaly)"}


def cm(c: dict[str, int]) -> str:
    return f"{c['tn']} {c['fp']} {c['fn']} {c['tp']}"


def rows(ms: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """(nama metrik, nilai per scorer)."""

    def each(f: Any) -> list[str]:
        return [f(m) for m in ms]

    out = [
        ("AUROC val (3+3)", each(lambda m: fmt(m["val"]["auroc"]))),
        ("AUROC test (4+4)", each(lambda m: fmt(m["test"]["auroc"]))),
        (
            "AUROC CV per fold (mean ± std)",
            each(lambda m: f"{m['cv']['auroc_mean']:.3f} ± {m['cv']['auroc_std']:.3f}"),
        ),
        ("AUROC CV gabungan out-of-fold (n=40)", each(lambda m: fmt(m["cv"]["auroc_pooled"]))),
        (
            "recall @ FPR ≤ target, CV out-of-fold (FPR)",
            each(lambda m: f"{m['cv']['recall_at_fpr']:.2f} ({m['cv']['fpr_at']:.2f})"),
        ),
        (
            "recall @ FPR ≤ target, test (FPR)",
            each(lambda m: f"{m['test']['recall_at_fpr']:.2f} ({m['test']['fpr_at']:.2f})"),
        ),
        ("threshold config", each(lambda m: f"{m['threshold_config']:g}")),
        ("threshold terkalibrasi (CV, FPR ≤ target)", each(lambda m: f"{m['threshold_cal']:g}")),
    ]
    for key, label in (("config", "config"), ("cal", "kalibrasi")):
        out += [
            (
                f"akurasi val / test / CV @ {label}",
                each(
                    lambda m, k=key: (
                        f"{m['val'][f'acc_{k}']:.2f} / {m['test'][f'acc_{k}']:.2f} / "
                        f"{m['cv'][f'acc_{k}']:.2f}"
                    )
                ),
            ),
            (f"test TN FP FN TP @ {label}", each(lambda m, k=key: cm(m["test"][f"cm_{k}"]))),
        ]
    out += [
        (
            "latency per roti (ms, CPU, setelah pemanasan)",
            each(lambda m: f"{m['latency_ms']['mean']:.1f}"),
        ),
        ("ukuran file model", each(lambda m: size(m["model_bytes"]))),
    ]
    for i, dc in enumerate(ms[0]["domain_check"]):
        out.append(
            (
                f"domain check konveyor: REJECT config / kalibrasi (dari {dc['n']} roti OK)",
                each(
                    lambda m, i=i: " / ".join(
                        str(v) for v in m["domain_check"][i]["n_reject"].values()
                    )
                ),
            )
        )
        out.append(
            (
                "domain check: skor min / median / maks",
                each(lambda m, i=i: spread(m["domain_check"][i]["scores"])),
            )
        )
    for split, label in (("cv", "CV out-of-fold"), ("test", "test")):
        for g in GROUP_ORDER:
            out.append(
                (
                    f"skor rata-rata {g} ({label})",
                    each(
                        lambda m, s=split, g=g: f"{m[s]['prefix_means'].get(g, float('nan')):.1f}"
                    ),
                )
            )
    return out


def size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def spread(scores: list[float]) -> str:
    s = sorted(scores)
    return f"{s[0]:.0f} / {s[len(s) // 2]:.0f} / {s[-1]:.0f}" if s else "n/a"


def render(ms: list[dict[str, Any]], docs: dict[str, str]) -> str:
    """Tabel perbandingan markdown."""
    head = [NAMES.get(m["model_type"], m["model_type"]) for m in ms]
    out = [
        "# Perbandingan scorer A vs B — LoafLens M3",
        "",
        f"> **{HEADLINE}**",
        "",
        f"- Tanggal: {date.today().isoformat()}. Target FPR: {ms[0]['fpr_target']:g}. Sumber: "
        + ", ".join(f"`{m['output_dir']}/metrics.json`" for m in ms)
        + ". Laporan lengkap: "
        + ", ".join(f"[`{Path(d).name}`]({Path(d).name})" for d in docs.values())
        + ".",
        "- Kedua scorer memakai split beku dan fold CV yang sama. Threshold config dan threshold "
        "terkalibrasi keduanya TIDAK dipilih dari test. Hold-out sangat kecil → acuan utama CV; "
        "sinyal paling realistis = domain check foto konveyor (semua roti OK, setiap REJECT = "
        "false positive).",
        "",
        "| metrik | " + " | ".join(head) + " |",
        "|---|" + "---|" * len(head),
    ]
    out += [f"| {name} | " + " | ".join(vals) + " |" for name, vals in rows(ms)]
    return "\n".join(out) + "\n"


def compare(config_paths: list[str], root: Path) -> str:
    """Bangun outputs/compare.md + docs/results_m3.md dari metrics.json tiap config."""
    ms, docs = [], {}
    for cp in config_paths:
        cfg = load_config(cp)
        path = root / cfg.output_dir / "metrics.json"
        if not path.exists():
            raise SystemExit(f"{path} tidak ada — jalankan dulu evaluate --config {cp}")
        ms.append(json.loads(path.read_text(encoding="utf-8")))
        if cfg.eval.docs_copy:
            docs[cfg.model.type] = cfg.eval.docs_copy
    table = render(ms, docs)
    tail = ""
    if (root / DOCS).exists() and MARKER in (old := (root / DOCS).read_text(encoding="utf-8")):
        tail = old.split(MARKER, 1)[1]
    text = f"{table}\n{MARKER}{tail or chr(10)}"
    (root / OUT).parent.mkdir(parents=True, exist_ok=True)
    (root / OUT).write_text(text, encoding="utf-8")
    (root / DOCS).write_text(text, encoding="utf-8")
    log.info("Perbandingan: %s (salinan %s)", root / OUT, DOCS)
    return text


def main(argv: list[str] | None = None) -> int:
    """CLI perbandingan."""
    parser = argparse.ArgumentParser(description="Tabel perbandingan scorer dari metrics.json.")
    parser.add_argument("configs", nargs="+", help="configs/*.yaml (sudah di-evaluate)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    compare(args.configs, Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
