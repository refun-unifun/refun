"""Measure ground-truth function-name leakage in the input views.

WHY THIS EXISTS
---------------
A function-name model is only interesting if the name is not already in its
input. For decompiled binaries that is not automatic, and on this corpus it is
frequently false. Two distinct mechanisms put the gold name into the input:

  1. NOT ACTUALLY STRIPPED. Dynamically-linked and thunk functions keep their
     names in `.dynsym`, which survives `strip`. Ghidra recovers them, so the
     "stripped" decompiled body reads `void abort(void)` or
     `int fputs_unlocked(char *__s, FILE *__stream)`. The task is then a copy,
     not an inference. Measured on x64_O0 test: 27.5% of records.

  2. RESIDUAL STRING LITERALS. Assertion macros embed the enclosing function
     name as a string constant:
         FUN_0017c650("../../gold/object.h", 0x3c7, "do_output_section_offset")
     This is genuine signal that a real analyst would also exploit, and it is
     present in every published binary corpus. Measured on x64_O0 test: 7.8%.

Category 1 inflates scores and should be excluded from headline numbers, or at
minimum reported separately. Category 2 is defensible but must be disclosed.
The distinction matters because they have opposite implications: (1) is a
dataset construction defect, (2) is a property of the domain.

The reasoning views leak too, at lower rates: the *evaluation* description
(`model_generated_description_test`, produced without showing the LLM the gold
name) contains it in 22.5% of records -- that is the teacher model succeeding,
not a defect. The *training* trace (`reasoning_1`) is generated WITH the gold
name in the prompt by design, as a distillation target; it is never used at
evaluation time. Keeping those two fields distinct is why `--train_desc_field`
and `--eval_desc_field` exist.

USAGE
-----
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

# Names shorter than this are excluded: 2-3 character names ("cp", "add") match
# incidentally everywhere and would report leakage that is really coincidence.
MIN_NAME_LEN = 4


def _word_re(name: str) -> "re.Pattern":
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")


def classify(code: str, gold: str) -> str:
    """How the gold name reached this record's decompiled body.

    Returns one of: 'clean', 'signature', 'string_literal', 'other'.
    'signature' means the name appears in the function's own declarator -- the
    binary was not effectively stripped for this function.
    """
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
    """True when the record's own decompiled signature carries the gold name.

    This is the predicate behind `--drop_selfnamed` in the training script: the
    single filter that removes the not-actually-stripped functions.
    """
    gold = (record.get(NAME_COL) or "").strip()
    if len(gold) < MIN_NAME_LEN:
        return False
    return classify(record.get(CODE_COL) or "", gold) == "signature"


def audit_split(ds, n: Optional[int] = None, seed: int = 0) -> Dict:
    """Leakage rates for one split. `n` subsamples for speed; None scores all."""
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
            print(f"    -> not-stripped (own signature):  {m['signature']:5.1f}%   "
                  f"[exclude with --drop_selfnamed]")
            print(f"    -> assert/string literal only:    {m['string_literal']:5.1f}%   "
                  f"[disclose, do not exclude]")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"[audit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
