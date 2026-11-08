"""Audit dataset → laporan Markdown (jumlah, split, ukuran, warna, duplikat, kebocoran, WARNING).

Audit hanya melaporkan; tidak mengubah data atau split.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

from loaflens.config import AuditConfig, Config, load_config
from loaflens.data import (
    FALLBACK_PREFIX,
    SPLITS,
    Row,
    dhash,
    load_image,
    near_duplicate_pairs,
    read_manifest,
    scan_dataset,
)

log = logging.getLogger(__name__)

# Fitur "trivial" yang seharusnya TIDAK memisahkan kelas (bukan bentuk roti).
TRIVIAL_FEATURES = [
    "width",
    "height",
    "aspect",
    "border_gray",
    "border_r_minus_b",
    "center_gray",
    "center_r_minus_b",
]
_NUM_SUFFIX = re.compile(r"[\s_\-]*\(?(\d+)\)?$")


# --- fungsi murni (mudah di-test) ---------------------------------------------------------


def name_prefix(path: str) -> str:
    """Prefix nama file tanpa nomor urut: 'bagus (3).png' → 'bagus'."""
    return _NUM_SUFFIX.sub("", Path(path).stem) or Path(path).stem


def name_number(path: str) -> int | None:
    """Nomor urut di akhir nama file ('bagus (3).png' → 3), None bila tidak ada."""
    m = _NUM_SUFFIX.search(Path(path).stem)
    return int(m.group(1)) if m else None


def image_stats(img: np.ndarray, border_frac: float, center_frac: float) -> dict[str, float]:
    """Ukuran, aspek, dan rata-rata warna tepi (≈ latar) & tengah (≈ roti) dari gambar BGR."""
    h, w = img.shape[:2]
    bh, bw = max(1, round(h * border_frac)), max(1, round(w * border_frac))
    mask = np.ones((h, w), bool)
    mask[bh : h - bh, bw : w - bw] = False
    ch, cw = round(h * (1 - center_frac) / 2), round(w * (1 - center_frac) / 2)
    center = img[ch : h - ch, cw : w - cw].reshape(-1, 3).astype(float)
    border = img[mask].astype(float)
    stats: dict[str, float] = {"width": w, "height": h, "aspect": w / h}
    for name, px in (("border", border), ("center", center)):
        b, g, r = px.mean(axis=0)
        stats |= {f"{name}_r": r, f"{name}_g": g, f"{name}_b": b}
        stats[f"{name}_gray"] = 0.299 * r + 0.587 * g + 0.114 * b
        stats[f"{name}_r_minus_b"] = r - b
    return stats


def stump_accuracy(values: list[float], is_pos: list[bool]) -> float:
    """Akurasi terbaik memisahkan dua kelas dengan SATU threshold pada satu fitur."""
    v, y = np.asarray(values, float), np.asarray(is_pos, int)
    order = np.argsort(v, kind="stable")
    v, y = v[order], y[order]
    n = len(y)
    k = np.arange(n + 1)
    left_pos = np.concatenate([[0], np.cumsum(y)])
    acc = (k - left_pos + (y.sum() - left_pos)) / n  # kiri=negatif, kanan=positif
    valid = np.ones(n + 1, bool)
    valid[1:n] = v[:-1] < v[1:]  # potong hanya di antara nilai berbeda
    return float(max(acc[valid].max(), (1 - acc[valid]).max()))


def batch_leaks(rows: list[Row]) -> dict[str, set[str]]:
    """batch_id yang muncul di lebih dari satu split."""
    splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("split"):
            splits[r["batch_id"]].add(r["split"])
    return {b: s for b, s in splits.items() if len(s) > 1}


def shortcut_warnings(
    stats: list[dict[str, float]], labels: list[str], min_acc: float
) -> tuple[dict[str, float], list[str]]:
    """Akurasi 1-threshold per fitur trivial + WARNING bila fitur itu memisahkan kelas."""
    if len(set(labels)) < 2:
        return {}, []
    is_pos = [lab == "defect" for lab in labels]
    accs = {f: stump_accuracy([s[f] for s in stats], is_pos) for f in TRIVIAL_FEATURES}
    warnings = []
    sizes: dict[str, set[tuple[float, float]]] = defaultdict(set)
    for s, lab in zip(stats, labels, strict=True):
        sizes[lab].add((s["width"], s["height"]))
    if not sizes["normal"] & sizes["defect"]:
        warnings.append(
            "Ukuran gambar TIDAK overlap antar kelas "
            f"(normal {_fmt_sizes(sizes['normal'])}; defect {_fmt_sizes(sizes['defect'])}): "
            "model bisa belajar ukuran/rasio gambar, bukan bentuk roti. Perbaikan di M2: "
            "crop hasil segmentasi + resize ke ukuran tetap, tanpa fitur ukuran gambar."
        )
    for f, acc in accs.items():
        if acc >= min_acc:
            warnings.append(
                f"Fitur trivial `{f}` memisahkan kelas dengan akurasi {acc:.0%} (satu threshold) "
                "— risiko shortcut (latar/pencahayaan/ukuran berbeda per kelas)."
            )
    return accs, warnings


def _mean_std(stats: list[dict[str, float]], key: str) -> str:
    v = [s[key] for s in stats]
    return f"{np.mean(v):.0f}±{np.std(v):.0f}"


def _fmt_sizes(sizes: set[tuple[float, float]]) -> str:
    return ", ".join(f"{int(w)}x{int(h)}" for w, h in sorted(sizes))


# --- audit utama --------------------------------------------------------------------------


def run_audit(cfg: Config, root: Path) -> tuple[str, list[str]]:
    """Jalankan audit penuh → (laporan Markdown, daftar WARNING)."""
    a: AuditConfig = cfg.audit
    rows = scan_dataset(cfg.data, root)
    manifest_path = root / cfg.data.manifest
    manifest = read_manifest(manifest_path) if manifest_path.exists() else []
    split_of = {r["path"]: r["split"] for r in manifest}
    warnings: list[str] = []

    ok_rows, stats, hashes, sha, unreadable = [], [], [], [], []
    for r in rows:
        try:
            img = load_image(r["path"], root)
        except ValueError:
            unreadable.append(r["path"])
            continue
        ok_rows.append(r)
        stats.append(image_stats(img, a.border_frac, a.center_frac))
        hashes.append(dhash(img))
        sha.append(hashlib.sha1((root / r["path"]).read_bytes()).hexdigest())
    labels = [r["label"] for r in ok_rows]
    label_set = sorted(set(labels))

    # Batch.
    n_fallback = sum(r["batch_id"].startswith(FALLBACK_PREFIX) for r in rows)
    if n_fallback:
        warnings.append(
            f"{n_fallback}/{len(rows)} gambar tanpa subfolder batch → satu grup per gambar. "
            "Split per batch/sesi TIDAK mungkin; gambar dari sesi yang sama bisa tersebar ke "
            "train/val/test. Minta info batch/sesi, lalu taruh di train-data/<label>/<batch_id>/."
        )
    used = manifest or rows
    leaks = batch_leaks(used)
    if leaks:
        warnings.append(f"Kebocoran batch: {len(leaks)} batch_id muncul di >1 split.")

    # Manifest vs disk.
    disk = {r["path"] for r in rows}
    if not manifest:
        warnings.append("Manifest belum ada → jalankan `make manifest`; info split dilewati.")
    not_in_manifest = sorted(disk - set(split_of)) if manifest else []
    missing_on_disk = sorted(set(split_of) - disk)
    if not_in_manifest or missing_on_disk:
        warnings.append(
            f"Manifest tidak sinkron: {len(not_in_manifest)} file baru belum di manifest, "
            f"{len(missing_on_disk)} entri manifest tidak ada di disk."
        )
    if unreadable:
        warnings.append(f"{len(unreadable)} file tidak terbaca (lihat daftar).")

    # Prefix nama file.
    prefix_counts = Counter((r["label"], name_prefix(r["path"])) for r in rows)
    prefix_labels: dict[str, set[str]] = defaultdict(set)
    for lab, p in prefix_counts:
        prefix_labels[p].add(lab)
    for p, labs in sorted(prefix_labels.items()):
        if len(labs) > 1:
            warnings.append(
                f"Prefix nama `{p}` muncul di kedua label ({', '.join(sorted(labs))}) — "
                "arti prefix perlu dikonfirmasi (mis. tingkat kempes?). Prefix TIDAK dipakai "
                "sebagai batch."
            )

    # Pasangan bernomor sama dengan prefix berbeda (mis. 'bagus (3)' vs 'medium (3)').
    pair_rows = []
    for lab in label_set:
        idx = [i for i, r in enumerate(ok_rows) if r["label"] == lab]
        same, other, crossed = [], [], 0
        for i, j in combinations(idx, 2):
            pi, pj = ok_rows[i]["path"], ok_rows[j]["path"]
            if name_prefix(pi) == name_prefix(pj):
                continue
            d = (hashes[i] ^ hashes[j]).bit_count()
            ni, nj = name_number(pi), name_number(pj)
            if ni is not None and ni == nj:
                same.append(d)
                crossed += split_of.get(pi) != split_of.get(pj)
            else:
                other.append(d)
        if same and other:
            ms, mo = statistics.median(same), statistics.median(other)
            pair_rows.append((lab, len(same), ms, mo, crossed if manifest else "-"))
            if ms <= a.pair_warn_ratio * mo:
                warnings.append(
                    f"[{lab}] File bernomor sama dengan prefix berbeda (mis. `x (3)`/`y (3)`) "
                    f"jauh lebih mirip (median dHash {ms:g} vs {mo:g}) — kemungkinan varian roti/"
                    f"foto yang sama; {crossed if manifest else '?'}/{len(same)} pasangan "
                    "terpisah ke split berbeda. Jika memang varian, pasangan harus satu grup "
                    "(risiko bocor); konfirmasi ke user. Saat ini TIDAK digabung otomatis."
                )

    # Duplikat.
    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(sha):
        by_sha[h].append(i)
    exact = [g for g in by_sha.values() if len(g) > 1]
    near = near_duplicate_pairs(hashes, cfg.data.near_dup_hamming)
    if exact:
        warnings.append(f"{len(exact)} grup duplikat persis (hash file sama).")
    for i, j, _ in near:
        pi, pj = ok_rows[i]["path"], ok_rows[j]["path"]
        if ok_rows[i]["label"] != ok_rows[j]["label"]:
            warnings.append(f"Near-duplicate beda label: `{pi}` vs `{pj}` — cek label.")
        if manifest and split_of.get(pi) != split_of.get(pj):
            warnings.append(f"Near-duplicate beda split: `{pi}` vs `{pj}` — bocor.")

    # Shortcut ukuran/warna.
    accs, sc_warn = shortcut_warnings(stats, labels, a.shortcut_acc)
    warnings += sc_warn

    # Ukuran dataset & split.
    split_counts = Counter((r["split"], r["label"]) for r in manifest)
    if manifest:
        small = [
            f"{s}/{lab}={split_counts[(s, lab)]}"
            for s in ("val", "test")
            for lab in label_set
            if split_counts[(s, lab)] < a.min_eval_per_class
        ]
        if small:
            warnings.append(
                f"Val/test sangat kecil ({', '.join(small)}; < {a.min_eval_per_class} per label): "
                "metrik (AUROC, recall) akan sangat noisy — satu gambar bisa menggeser hasil "
                "beberapa persen. Rekomendasi M2: cross-validation per grup selain hold-out."
            )
    label_counts = Counter(labels)
    if label_counts["normal"] < 200 or label_counts["defect"] < 30:
        warnings.append(
            f"Dataset di bawah perkiraan PRD §8 (normal {label_counts['normal']}/≥200, "
            f"defect {label_counts['defect']}/≥30–50)."
        )

    # --- laporan ---
    out = [
        "# Audit dataset — LoafLens (M1)",
        "",
        f"- Tanggal: {date.today().isoformat()}",
        f"- Folder: normal=`{cfg.data.normal_dir}`, defect=`{cfg.data.defect_dir}`",
        f"- Manifest: `{cfg.data.manifest}` ({'ada' if manifest else 'belum ada'})",
        f"- Near-duplicate: dHash 64-bit, jarak ≤ {cfg.data.near_dup_hamming}",
        "",
        f"## Peringatan ({len(warnings)})",
        "",
        *([f"- **WARNING:** {w}" for w in warnings] or ["- (tidak ada)"]),
        "",
        "## Jumlah per label",
        "",
        "| label | file | terbaca | batch_id unik | fallback per-gambar |",
        "|---|---|---|---|---|",
    ]
    for lab in sorted({r["label"] for r in rows}):
        lr = [r for r in rows if r["label"] == lab]
        nb = len({r["batch_id"] for r in lr})
        nf = sum(r["batch_id"].startswith(FALLBACK_PREFIX) for r in lr)
        out.append(f"| {lab} | {len(lr)} | {label_counts[lab]} | {nb} | {nf} |")
    out += ["", "## Prefix nama file", "", "| label | prefix | jumlah |", "|---|---|---|"]
    out += [f"| {lab} | `{p}` | {n} |" for (lab, p), n in sorted(prefix_counts.items())]
    if pair_rows:
        out += [
            "",
            "Kemiripan pasangan bernomor sama, prefix berbeda (median jarak dHash; kecil = mirip):",
            "",
            "| label | pasangan | median nomor sama | median lainnya | pasangan beda split |",
            "|---|---|---|---|---|",
            *[f"| {lab} | {n} | {ms:g} | {mo:g} | {c} |" for lab, n, ms, mo, c in pair_rows],
        ]
    out += ["", "## Split (dari manifest)", ""]
    if manifest:
        out += ["| split | normal | defect | total |", "|---|---|---|---|"]
        for s in SPLITS:
            n, d = split_counts[(s, "normal")], split_counts[(s, "defect")]
            out.append(f"| {s} | {n} | {d} | {n + d} |")
        leak_txt = "tidak ada" if not leaks else ", ".join(f"`{b}`" for b in sorted(leaks))
        out += ["", f"Kebocoran batch (batch_id di >1 split): {leak_txt}."]
    else:
        out.append("(manifest belum ada)")

    out += ["", "## Ukuran & rasio aspek per kelas", ""]
    out += ["| label | ukuran WxH (jumlah) | aspek W/H min–max |", "|---|---|---|"]
    for lab in label_set:
        ls = [s for s, x in zip(stats, labels, strict=True) if x == lab]
        sizes = Counter((int(s["width"]), int(s["height"])) for s in ls)
        size_txt = ", ".join(f"{w}x{h} ({n})" for (w, h), n in sorted(sizes.items()))
        asp = [s["aspect"] for s in ls]
        out.append(f"| {lab} | {size_txt} | {min(asp):.3f}–{max(asp):.3f} |")

    out += [
        "",
        "## Warna per kelas (rata-rata RGB 0–255, ± std antar gambar)",
        "",
        f"Tepi = {a.border_frac:.0%} pinggir gambar (≈ latar); "
        f"tengah = kotak {a.center_frac:.0%} sisi (≈ roti).",
        "",
        "| label | tepi R,G,B | tepi gray | tengah R,G,B | tengah gray |",
        "|---|---|---|---|---|",
    ]
    for lab in label_set:
        ls = [s for s, x in zip(stats, labels, strict=True) if x == lab]
        cols = [
            ", ".join(_mean_std(ls, f"{part}_{c}") for c in "rgb")
            + " | "
            + _mean_std(ls, f"{part}_gray")
            for part in ("border", "center")
        ]
        out.append(f"| {lab} | {' | '.join(cols)} |")
    if accs:
        out += [
            "",
            "## Pemisahan kelas oleh fitur trivial",
            "",
            "Akurasi terbaik dengan satu threshold (50% = tak berguna, 100% = pemisah sempurna).",
            "",
            "| fitur | akurasi |",
            "|---|---|",
            *[f"| `{f}` | {acc:.0%} |" for f, acc in accs.items()],
        ]

    out += ["", "## Duplikat", ""]
    out.append(f"Duplikat persis: {len(exact)} grup.")
    for g in exact:
        out.append("- " + ", ".join(f"`{ok_rows[i]['path']}`" for i in g))
    out += ["", f"Near-duplicate (dHash ≤ {cfg.data.near_dup_hamming}): {len(near)} pasangan."]
    for i, j, d in near:
        pi, pj = ok_rows[i]["path"], ok_rows[j]["path"]
        out.append(
            f"- `{pi}` ({split_of.get(pi, '-')}) ↔ `{pj}` ({split_of.get(pj, '-')}), jarak {d}"
        )
    out += ["", "## File tak terbaca", ""]
    out += [f"- `{p}`" for p in unreadable] or ["(tidak ada)"]
    out += ["", "## Konsistensi manifest", ""]
    out.append(f"- File di disk belum di manifest: {len(not_in_manifest)}")
    out += [f"  - `{p}`" for p in not_in_manifest]
    out.append(f"- Entri manifest tidak ada di disk: {len(missing_on_disk)}")
    out += [f"  - `{p}`" for p in missing_on_disk]
    return "\n".join(out) + "\n", warnings


def main(argv: list[str] | None = None) -> int:
    """CLI: tulis laporan audit ke outputs/ (+ salinan di docs/ bila diset)."""
    parser = argparse.ArgumentParser(description="Audit dataset → laporan Markdown.")
    parser.add_argument("--config", required=True, help="path configs/*.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    root = Path.cwd()
    report, warnings = run_audit(cfg, root)
    out = root / cfg.audit.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log.info("Laporan audit: %s", cfg.audit.report)
    if cfg.audit.docs_copy:
        (root / cfg.audit.docs_copy).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, root / cfg.audit.docs_copy)
        log.info("Salinan: %s", cfg.audit.docs_copy)
    for w in warnings:
        log.warning("%s", w)
    log.info("%d WARNING", len(warnings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
