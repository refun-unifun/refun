"""Measure ground-truth function-name leakage in the input views.

A function-name model is only interesting if the name is not already in its
input. On this corpus that is often false, via two mechanisms with different
implications. Measured on x64_O0 test (n=794, names >= 4 chars):

  1. NOT ACTUALLY STRIPPED (28.2%). Dynamically-linked and thunk functions keep
     their names in .dynsym, which survives strip. Ghidra recovers them, so the
     "stripped" body reads `void abort(void)`. Predicting the name is a copy,
     not an inference. This is a dataset-construction defect.

  2. RESIDUAL STRING LITERALS (8.6%). Assertion macros embed the enclosing
     function name as a constant:
         FUN_0017c650("../../gold/object.h", 0x3c7, "do_output_section_offset")
     This is a property of the domain, present in every binary corpus.

Per-view rates: decompiled C 36.9%, S-expression 36.9% (it quotes the source),
assembly 0.0%, eval reasoning trace 22.3%, training trace 3.0%.

The reasoning views differ by design. `model_generated_description_test` is
produced without showing the teacher the gold name, so its 22.3% is the teacher
succeeding. `reasoning_1` is generated *with* the gold name in the prompt as a
distillation target and is never used at evaluation time -- which is why
--train_desc_field and --eval_desc_field are separate flags.

`is_self_named` is the predicate behind --drop_selfnamed, which excludes
category 1 from training and evaluation.

    python -m refun.audit_leakage --configs x64_O0 --split test --n 2000
    python -m refun.audit_leakage --configs all --split test --out leakage.json
"""
import argparse
import json
import random
import re
import sys
from typing import Dict, List, Optional

VIEW_COLUMNS = [
    "decompiled_code_stripped",
    "S-Expression_of_decompiled_code_stripped",
    "assembly_code",
    "model_generated_description_test",
    "reasoning_1",
]
NAME_COL = "original_function_name"
CODE_COL = "decompiled_code_stripped"

# 2-3 character names ("cp", "add") match incidentally everywhere, so counting
# them would report coincidence as leakage.
MIN_NAME_LEN = 4


def _word_re(name: str) -> "re.Pattern":
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")


def classify(code: str, gold: str) -> str:
    """How the gold name reached this record's decompiled body: 'clean',
    'signature' (in its own declarator, i.e. not effectively stripped),
    'string_literal', or 'other'."""
    if not code or not gold:
        return "clean"
    pat = _word_re(gold)
    if not pat.search(code):
        return "clean"
    head = code.split("{", 1)[0] if "{" in code else code
    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(gold) + r"\s*\(", head):
        return "signature"
    if re.search(r'"[^"]*(?<![A-Za-z0-9_])' + re.escape(gold) + r'(?![A-Za-z0-9_])[^"]*"', code):
        return "string_literal"
    return "other"


def is_self_named(record: dict) -> bool:
    """True when the record's own decompiled signature carries the gold name."""
    gold = (record.get(NAME_COL) or "").strip()
    if len(gold) < MIN_NAME_LEN:
        return False
    return classify(record.get(CODE_COL) or "", gold) == "signature"


def audit_split(ds, n: Optional[int] = None, seed: int = 0) -> Dict:
    """Leakage rates for one split; `n` subsamples, None scores every row."""
    total = len(ds)
    idx = range(total)
    if n and n < total:
        idx = random.Random(seed).sample(range(total), n)

    cols = [c for c in VIEW_COLUMNS if c in ds.column_names]
    hits = {c: 0 for c in cols}
    kinds = {"clean": 0, "signature": 0, "string_literal": 0, "other": 0}
    scored = skipped = 0

    for i in idx:
        r = ds[i]
        gold = (r.get(NAME_COL) or "").strip()
        if len(gold) < MIN_NAME_LEN:
            skipped += 1
            continue
        scored += 1
        pat = _word_re(gold)
        for c in cols:
            v = r.get(c)
            if v and pat.search(str(v)):
                hits[c] += 1
        kinds[classify(r.get(CODE_COL) or "", gold)] += 1

    def pct(x):
        return round(100.0 * x / scored, 2) if scored else 0.0

    return {
        "rows_total": total,
        "rows_scored": scored,
        "rows_skipped_short_name": skipped,
        "view_leak_pct": {c: pct(hits[c]) for c in cols},
        "mechanism_pct": {k: pct(v) for k, v in kinds.items()},
        "mechanism_counts": kinds,
    }


def main(argv: Optional[List[str]] = None) -> int:
    from . import datasets as ds_registry

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["x64_O0"],
                    help="config names, group aliases ('all'), or full repo IDs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000,
                    help="subsample size per config; 0 scores every row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    report = {}
    for cfg in args.configs:
        for repo in ds_registry.resolve([cfg]):
            name = ds_registry.config_of_repo(repo)
            print(f"[audit] {name}", flush=True)
            d = load_dataset(repo, cache_dir=args.cache_dir)
            if args.split not in d:
                print(f"  split {args.split!r} absent (have {list(d)}), skipping")
                continue
            rec = audit_split(d[args.split], n=args.n or None, seed=args.seed)
            report[name] = rec
            print(f"  scored {rec['rows_scored']} of {rec['rows_total']}")
            for c, p in rec["view_leak_pct"].items():
                print(f"    {c:<45} {p:5.1f}%")
            m = rec["mechanism_pct"]
            print(f"    -> not-stripped (own signature):  {m['signature']:5.1f}%  "
                  f"(--drop_selfnamed excludes these)")
            print(f"    -> assert/string literal only:    {m['string_literal']:5.1f}%")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"[audit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
