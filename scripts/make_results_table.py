#!/usr/bin/env python3
"""Build a fusion-comparison LaTeX table (Prec./Rec./F1) from finished runs.

Scans an experiment root for the final-eval prediction TSVs written by
`refun.train` (one per run, under ``<root>/<dataset>/<fusion>/``) and computes
token-level precision/recall/F1 with the same token-overlap basis used during
training (strip name tags, whitespace-split, set overlap). The MoE fusion is the
reference row; the other fusions are shown with percentage deltas vs MoE.

Usage:
    python scripts/make_results_table.py --root experiment_runs [--out table.tex]
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

START_TAG = "<function Name>"
END_TAG = "</function Name>"

FUSIONS = ["moe", "simple_gating", "cross_attention", "concat"]
FUSION_LABEL = {
    "moe": r"\system\ (MoE)",
    "simple_gating": "Gating",
    "cross_attention": "Cross-Modal Attn.",
    "concat": "Concatenation",
}


def strip_tags(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text.startswith(START_TAG):
        text = text[len(START_TAG):]
    if text.endswith(END_TAG):
        text = text[: -len(END_TAG)]
    return text.strip()


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def token_prf_from_tsv(path):
    tp = fp = fn = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            g = set(strip_tags(row.get("ground_truth", "")).split())
            p = set(strip_tags(row.get("prediction", "")).split())
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
    return prf(tp, fp, fn)


def classify(path):
    """fusion = the path component matching a known fusion; dataset = its parent dir."""
    parts = os.path.normpath(path).split(os.sep)
    for i, part in enumerate(parts):
        if part in FUSIONS:
            dataset = parts[i - 1] if i >= 1 else "dataset"
            return dataset, part
    return None, None


def fmt_delta(val, ref):
    if not ref or val is None:
        return ""
    d = (val - ref) / ref * 100.0
    color = "red" if d < 0 else "green!50!black"
    return f" (\\textcolor{{{color}}}{{{d:+.1f}\\%}})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiment_runs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.root, "**", "*_inference_*.tsv"), recursive=True)
    results = defaultdict(dict)  # results[dataset][fusion] = (p, r, f)
    for path in files:
        dataset, fusion = classify(path)
        if dataset and fusion:
            results[dataset][fusion] = token_prf_from_tsv(path)

    datasets = sorted(results)
    found = sum(len(v) for v in results.values())
    print(f"[scan] {len(files)} TSVs; {found} (dataset x fusion) cells across {len(datasets)} datasets.")

    lines = [
        r"\begin{table}[t]", r"\centering", r"\scriptsize",
        r"\caption{Fusion-strategy comparison. Percentage changes are relative to "
        r"the full {\system} (MoE) model.}",
        r"\label{tab:fusion_comparison}",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l l c c c}", r"\toprule",
        r"\textbf{Dataset} & \textbf{Fusion} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\",
        r"\midrule",
    ]
    for di, dataset in enumerate(datasets):
        cells = results[dataset]
        ref = cells.get("moe")
        rp, rr, rf = ref if ref else (None, None, None)
        lines.append(rf"\multirow{{4}}{{*}}{{{dataset}}}")
        for fusion in FUSIONS:
            label = FUSION_LABEL[fusion]
            if fusion in cells:
                p, r, f = cells[fusion]
                if fusion == "moe":
                    lines.append(f" & {label} & {p:.4f} & {r:.4f} & {f:.4f} \\\\")
                else:
                    lines.append(f" & {label} & {p:.4f}{fmt_delta(p, rp)} "
                                 f"& {r:.4f}{fmt_delta(r, rr)} & {f:.4f}{fmt_delta(f, rf)} \\\\")
            else:
                lines.append(f" & {label} & -- & -- & -- \\\\  % pending")
        lines.append(r"\midrule" if di < len(datasets) - 1 else r"\bottomrule")
    if not datasets:
        lines.append(r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    print("\n" + tex + "\n")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(tex + "\n")
        print(f"[write] -> {args.out}")


if __name__ == "__main__":
    main()
