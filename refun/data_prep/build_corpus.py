"""Assemble a ReFuN corpus from Ghidra JSON exports.

Takes the `<binary>.orig.json` / `<binary>.stripped.json` pairs produced by
`run_ghidra.py` and emits a HuggingFace dataset with the columns `refun.train`
expects:

    assembly_code                              stripped disassembly
    decompiled_code_stripped                   stripped decompiled C
    S-Expression_of_decompiled_code_stripped   tree-sitter AST of the above
    original_function_name                     label, from the unstripped side
    reasoning_1                                training trace  (added separately)
    model_generated_description_test           eval trace      (added separately)

The reasoning columns are *not* produced here -- they need an LLM. Build the
structural corpus first, then run `refun.data_prep.reasoning` over it and merge.

PAIRING AND SPLITS
------------------
Functions are paired across the two exports by entry-point address, which
`strip` preserves. Splitting is by *binary*, never by function: two
compilations of the same source, or two functions from one binary, share far
too much for a function-level split to measure generalisation. A function-level
split is the single most common way this task gets accidentally inflated.

    python -m refun.data_prep.build_corpus \\
        --ghidra_json ghidra_out/ --out corpus_x64_O0/ \\
        --arch x64 --opt O0 --test_frac 0.15
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Ghidra names unrecovered functions FUN_<addr>; those have no label to learn.
_PLACEHOLDER = re.compile(r"^(FUN|SUB|UndefinedFunction|thunk_FUN)_[0-9a-fA-F]+$")

MIN_ASM_LINES = 3
MIN_BODY_CHARS = 32


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def _usable_name(name: str) -> bool:
    if not name or _PLACEHOLDER.match(name):
        return False
    # Ghidra's operator/mangled leftovers make poor targets and are not names a
    # reverse engineer would be asked to recover.
    return not name.startswith(("_GLOBAL__", "__cxx_", "operator."))


def pair_functions(orig: dict, stripped: dict) -> Iterator[Tuple[dict, dict]]:
    """Match functions across the two exports by entry-point address."""
    by_addr = {}
    for f in stripped.get("functions", []):
        a = f.get("address")
        if a:
            by_addr[str(a)] = f
    for f in orig.get("functions", []):
        a = str(f.get("address") or "")
        if a in by_addr:
            yield f, by_addr[a]


def build_record(orig_fn: dict, strip_fn: dict, binary: str, package: str,
                 arch: str, opt: str, with_sexpr: bool = True) -> Optional[dict]:
    """One corpus row, or None if the function is not usable as an example."""
    name = str(orig_fn.get("original_function_name") or "").strip()
    if not _usable_name(name):
        return None

    body = str(strip_fn.get("decompiled_code") or "").strip()
    asm = str(strip_fn.get("assembly") or "").strip()
    if len(body) < MIN_BODY_CHARS or asm.count("\n") + 1 < MIN_ASM_LINES:
        return None
    if body.startswith("ERROR:") or asm.startswith("ERROR:"):
        return None

    rec = {
        "original_function_name": name,
        "stripped_function_name": str(strip_fn.get("stripped_function_name") or ""),
        "decompiled_code_stripped": body,
        "decompiled_code_original": str(orig_fn.get("decompiled_code") or ""),
        "assembly_code": asm,
        "address": str(orig_fn.get("address") or ""),
        "binary_name": binary,
        "package": package,
        "arch": arch,
        "opt_level": opt,
    }
    if with_sexpr:
        from ..sexpr import sexpr_with_text, sexpr_clean, sexpr_fields
        rec["S-Expression_of_decompiled_code_stripped"] = sexpr_with_text(body)
        rec["S-Expression_decompiled_code_stripped_clean"] = sexpr_clean(body)
        rec["Root Node"] = sexpr_fields(body)
    return rec


def split_by_binary(records: List[dict], test_frac: float, seed: int
                    ) -> Tuple[List[dict], List[dict]]:
    """Binary-level split. Every function of a binary lands on one side."""
    by_bin = defaultdict(list)
    for r in records:
        by_bin[(r["package"], r["binary_name"])].append(r)
    keys = sorted(by_bin)
    random.Random(seed).shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_frac))) if keys else 0
    test_keys = set(keys[:n_test])
    train = [r for k in keys if k not in test_keys for r in by_bin[k]]
    test = [r for k in keys if k in test_keys for r in by_bin[k]]
    return train, test


def dedup_exact(records: List[dict]) -> List[dict]:
    """Drop functions with an identical (name, normalised body) pair.

    The same static library is linked into many binaries, so without this a
    handful of libc helpers dominate both splits.
    """
    seen, out = set(), []
    for r in records:
        body = re.sub(r"\s+", " ", r["decompiled_code_stripped"]).strip()
        key = hashlib.sha256(
            (r["original_function_name"] + "\x00" + body).encode("utf-8")
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghidra_json", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--opt", required=True)
    ap.add_argument("--package", default="",
                    help="package label; defaults to the input directory name")
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_sexpr", action="store_true",
                    help="skip AST columns (much faster; view unavailable)")
    ap.add_argument("--jsonl", action="store_true",
                    help="write JSONL instead of a HF dataset directory")
    args = ap.parse_args(argv)

    package = args.package or args.ghidra_json.name
    pairs = {}
    for p in sorted(args.ghidra_json.glob("*.orig.json")):
        stem = p.name[: -len(".orig.json")]
        s = args.ghidra_json / f"{stem}.stripped.json"
        if s.exists():
            pairs[stem] = (p, s)
    if not pairs:
        print(f"no .orig/.stripped JSON pairs under {args.ghidra_json}",
              file=sys.stderr)
        return 1
    print(f"[corpus] {len(pairs)} binary pairs")

    records, skipped = [], 0
    for i, (stem, (op, sp)) in enumerate(sorted(pairs.items()), 1):
        try:
            orig, strip = _load(op), _load(sp)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(pairs)}] {stem}: unreadable ({e})")
            continue
        n_before = len(records)
        for of, sf in pair_functions(orig, strip):
            rec = build_record(of, sf, stem, package, args.arch, args.opt,
                               with_sexpr=not args.no_sexpr)
            if rec:
                records.append(rec)
            else:
                skipped += 1
        print(f"  [{i}/{len(pairs)}] {stem}: +{len(records) - n_before}", flush=True)

    print(f"[corpus] {len(records)} usable functions ({skipped} skipped)")
    records = dedup_exact(records)
    print(f"[corpus] {len(records)} after exact dedup")

    train, test = split_by_binary(records, args.test_frac, args.seed)
    print(f"[corpus] split by binary -> train={len(train)} test={len(test)}")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        for name, rows in (("train", train), ("test", test)):
            path = args.out / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[corpus] wrote {path}")
    else:
        from datasets import Dataset, DatasetDict
        DatasetDict({
            "train": Dataset.from_list(train),
            "test": Dataset.from_list(test),
        }).save_to_disk(str(args.out))
        print(f"[corpus] wrote HF dataset to {args.out}")

    (args.out / "build_manifest.json").write_text(json.dumps({
        "arch": args.arch, "opt_level": args.opt, "package": package,
        "binaries": len(pairs), "functions_train": len(train),
        "functions_test": len(test), "test_frac": args.test_frac,
        "seed": args.seed, "split_unit": "binary",
        "sexpr": not args.no_sexpr,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
