"""Record per-config row counts into `refun/datasets.py`'s ROW_COUNTS.

Recorded counts turn a silent short read into a visible mismatch: if a corpus
is later re-uploaded with fewer rows, every downstream split shrinks quietly
unless something remembers what the size was supposed to be.

    export REFUN_HF_NAMESPACE=<account>
    python scripts/record_dataset_stats.py --configs all
    python scripts/record_dataset_stats.py --configs x64_O0 --write

Without --write it prints a paste-ready ROW_COUNTS block; with --write it
rewrites the block in refun/datasets.py in place.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TARGET = Path(__file__).resolve().parents[1] / "refun" / "datasets.py"
BLOCK = re.compile(r"^ROW_COUNTS: Dict\[str, Dict\[str, int\]\] = \{.*?^\}\n",
                   re.S | re.M)


def render(counts: dict) -> str:
    lines = ["ROW_COUNTS: Dict[str, Dict[str, int]] = {"]
    for cfg in sorted(counts):
        c = counts[cfg]
        lines.append(f'    "{cfg}": {{"train": {c["train"]}, "test": {c["test"]}}},')
    lines.append("}\n")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["all"])
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--write", action="store_true",
                    help="rewrite ROW_COUNTS in refun/datasets.py")
    args = ap.parse_args()

    from datasets import load_dataset
    from refun import datasets as reg

    counts = dict(reg.ROW_COUNTS)
    for cfg in args.configs:
        for repo in reg.resolve([cfg]):
            name = reg.config_of_repo(repo)
            try:
                d = load_dataset(repo, cache_dir=args.cache_dir)
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: FAILED ({type(e).__name__}: {e})")
                continue
            rec = {"train": len(d["train"]) if "train" in d else 0,
                   "test": len(d["test"]) if "test" in d else 0}
            prev = reg.ROW_COUNTS.get(name)
            flag = ""
            if prev and prev != rec:
                flag = f"   CHANGED from {prev}"
            counts[name] = rec
            print(f"  {name}: train={rec['train']:,} test={rec['test']:,}{flag}")

    block = render(counts)
    if args.write:
        src = TARGET.read_text(encoding="utf-8")
        if not BLOCK.search(src):
            print("could not locate the ROW_COUNTS block; not writing",
                  file=sys.stderr)
            return 1
        TARGET.write_text(BLOCK.sub(block, src), encoding="utf-8")
        print(f"\nwrote {len(counts)} entries to {TARGET}")
    else:
        print("\n" + block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
