"""UniFuN: one model across every architecture and optimisation level.

ReFuN trains one model per (arch, opt) config. UniFuN asks whether a single
model can serve all of them -- the practically useful setting, since an analyst
facing an unknown binary does not get to pick a checkpoint per target.

Architecturally UniFuN and ReFuN are the *same* model; only the training corpus
differs. That is deliberate, and it is why this module is thin: any measured
difference between the two is attributable to data, not to an architecture
change. Concretely, UniFuN is

    python -m refun.train --fusion moe --datasets unifun ...

and this entry point exists to make that explicit, to set the defaults the
unified runs used, and to emit the per-config breakdown that the pooled numbers
would otherwise hide.

Pooled F1 over 16 configs is dominated by the largest (x64_O0 and x86_O0
contribute far more rows than mips_O3), so it can improve while every hard
config degrades. Alongside the per-corpus `metrics_<dataset>.json` that
`refun.train` writes, this module emits `unifun_breakdown.json` with the
per-config table and both a macro and a row-weighted average.

USAGE
-----
    export REFUN_HF_NAMESPACE=<account>
    python -m refun.unifun --output_dir runs/unifun_moe --epochs 500

    # a subset, e.g. cross-architecture generalisation
    python -m refun.unifun --datasets x64 x86 --output_dir runs/unifun_x64x86

Every `refun.train` flag is accepted and forwarded unchanged.
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


def _breakdown(output_dir: Path) -> Optional[dict]:
    """Collect per-config metrics that `refun.train` wrote, into one table."""
    results_dir = output_dir / "inference_results"
    search = results_dir if results_dir.is_dir() else output_dir
    rows = {}
    for p in sorted(search.rglob("metrics_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                rows[p.stem[len("metrics_"):]] = json.load(fh)
        except Exception as e:  # noqa: BLE001
            print(f"[unifun] could not read {p}: {e}")
    if not rows:
        return None

    def _f(rec, *keys):
        for k in keys:
            if k in rec and isinstance(rec[k], (int, float)):
                return float(rec[k])
        return None

    table, weighted_num, weighted_den, macro = {}, 0.0, 0.0, []
    for cfg, rec in sorted(rows.items()):
        f1 = _f(rec, "token_f1", "f1", "eval_token_f1")
        n = _f(rec, "num_samples", "n", "count") or 0.0
        table[cfg] = {
            "token_precision": _f(rec, "token_precision", "precision"),
            "token_recall": _f(rec, "token_recall", "recall"),
            "token_f1": f1,
            "exact_match": _f(rec, "exact_match", "exact"),
            "cwordnet_f1": _f(rec, "cwordnet_f1", "cword_f1"),
            "num_samples": n or None,
        }
        if f1 is not None:
            macro.append(f1)
            if n:
                weighted_num += f1 * n
                weighted_den += n

    return {
        "per_config": table,
        "macro_avg_token_f1": round(sum(macro) / len(macro), 6) if macro else None,
        "row_weighted_token_f1": (
            round(weighted_num / weighted_den, 6) if weighted_den else None
        ),
        "note": (
            "macro_avg weights every config equally; row_weighted follows corpus "
            "size and is dominated by the O0 configs."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Default to the full compiler sweep unless the caller narrowed it.
    if not any(a == "--datasets" or a.startswith("--datasets=") for a in argv):
        argv += ["--datasets", "unifun"]
    # MoE is the proposed fusion; still overridable.
    if not any(a == "--fusion" or a.startswith("--fusion=") for a in argv):
        argv += ["--fusion", "moe"]

    out = None
    for i, a in enumerate(argv):
        if a == "--output_dir" and i + 1 < len(argv):
            out = argv[i + 1]
        elif a.startswith("--output_dir="):
            out = a.split("=", 1)[1]

    from . import train as train_mod

    saved = sys.argv
    sys.argv = ["refun.unifun"] + argv
    try:
        rc = train_mod.main()
    finally:
        sys.argv = saved

    if out:
        bd = _breakdown(Path(out))
        if bd:
            path = Path(out) / "unifun_breakdown.json"
            path.write_text(json.dumps(bd, indent=2), encoding="utf-8")
            print(f"\n[unifun] per-config breakdown -> {path}")
            print(f"[unifun] macro token-F1        : {bd['macro_avg_token_f1']}")
            print(f"[unifun] row-weighted token-F1 : {bd['row_weighted_token_f1']}")
            for cfg, r in bd["per_config"].items():
                print(f"           {cfg:<10} F1={r['token_f1']}  n={r['num_samples']}")
        else:
            print("[unifun] no metrics_*.json found; breakdown skipped")

    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
