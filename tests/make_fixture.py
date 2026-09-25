"""Build a tiny synthetic corpus with the exact schema `refun.train` expects.

The published corpora are tens of gigabytes and gated behind an account. The
smoke test needs neither: it needs a handful of rows with the right column
names and plausible content, so that every code path -- preprocessing, the four
fusions, the metric callbacks, inference dump -- actually executes.

    python tests/make_fixture.py --out /tmp/refun_fixture --n 64

Writes a `datasets.save_to_disk` directory with `train` and `test` splits.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VERBS = ["read", "write", "parse", "alloc", "free", "init", "close", "hash",
         "encode", "decode", "lookup", "insert", "remove", "compare", "flush",
         "resize", "append", "validate", "serialize", "connect"]
NOUNS = ["buffer", "header", "entry", "node", "stream", "table", "record",
         "socket", "index", "config", "block", "chunk", "packet", "symbol"]

BODY = """
{ret} FUN_{addr:08x}({params})

{{
  long lVar1;
  int iVar2;
  undefined8 uVar3;

  lVar1 = *(long *)(param_1 + {off1});
  if (lVar1 == 0) {{
    return {zero};
  }}
  iVar2 = FUN_{callee:08x}(lVar1,{k});
  if (iVar2 < 0) {{
    uVar3 = FUN_{callee2:08x}("{msg}",{k});
    return {zero};
  }}
  *(int *)(param_1 + {off2}) = iVar2;
  return {nonzero};
}}
"""

ASM = """PUSH RBP
MOV RBP,RSP
SUB RSP,0x{s:x}
MOV RAX,qword ptr [RDI + 0x{o:x}]
TEST RAX,RAX
JZ LAB_{a:08x}
MOV ESI,0x{k:x}
MOV RDI,RAX
CALL FUN_{c:08x}
TEST EAX,EAX
JS LAB_{a2:08x}
MOV dword ptr [RBX + 0x{o2:x}],EAX
MOV EAX,0x1
LEAVE
RET"""

REASON_TRAIN = (
    "The routine dereferences a pointer field from its first argument and "
    "guards against a null result before delegating to a helper. The return "
    "value of that helper is range-checked and, on success, stored back into a "
    "field of the same structure. The control flow is a two-stage validation "
    "with an early exit on each failure, and the operation it performs on the "
    "{noun} is a {verb}. The name is appropriate because it names the action "
    "and the object acted upon, at the level of abstraction the surrounding "
    "code uses."
)
REASON_EVAL = (
    "This function retrieves a pointer from an offset in the structure passed "
    "as its first parameter, returns early if it is null, calls a helper with "
    "that pointer and a constant, and writes the helper's result into another "
    "field when it is non-negative. It appears to {verb} a {noun}."
)


def make_row(rng: random.Random, idx: int, split: str) -> dict:
    from refun.sexpr import sexpr_with_text

    verb, noun = rng.choice(VERBS), rng.choice(NOUNS)
    style = rng.random()
    if style < 0.45:
        name = f"{verb}_{noun}"
    elif style < 0.8:
        name = verb + noun.capitalize()
    else:
        name = f"{verb}_{noun}_{rng.choice(['ex', 'internal', 'locked'])}"

    addr = 0x100000 + idx * 0x40
    body = BODY.format(
        ret=rng.choice(["int", "long", "undefined8", "bool"]),
        addr=addr,
        params=rng.choice(["long param_1", "long param_1,int param_2",
                           "undefined8 *param_1"]),
        off1=rng.choice([8, 0x10, 0x18, 0x28]),
        off2=rng.choice([0x30, 0x38, 0x40]),
        callee=addr + 0x2000, callee2=addr + 0x3000,
        k=rng.choice([0, 1, 4, 0x10, 0x68]),
        zero=rng.choice(["0", "0xffffffff"]),
        nonzero=rng.choice(["1", "iVar2"]),
        msg=f"{verb} failed for {noun}",
    )
    asm = ASM.format(s=rng.choice([0x18, 0x28, 0x38]),
                     o=rng.choice([8, 0x10]), a=addr + 0x20,
                     a2=addr + 0x30, k=rng.choice([1, 4, 0x10]),
                     c=addr + 0x2000, o2=rng.choice([0x30, 0x38]))

    return {
        "original_function_name": name,
        "stripped_function_name": f"FUN_{addr:08x}",
        "decompiled_code_stripped": body,
        "assembly_code": asm,
        "S-Expression_of_decompiled_code_stripped": sexpr_with_text(body),
        "Root Node": "",
        "reasoning_1": REASON_TRAIN.format(verb=verb, noun=noun),
        "model_generated_description_test": REASON_EVAL.format(verb=verb, noun=noun),
        "arch": "x64",
        "opt_level": "O0",
        "package": "synthetic",
        "binary_name": f"bin_{idx // 8:03d}",
        "address": f"{addr:08x}",
        "original_split": split,
    }


def build(n: int, seed: int = 0):
    from datasets import Dataset, DatasetDict

    rng = random.Random(seed)
    n_test = max(4, n // 4)
    # Distinct index ranges so no function name or body is shared across splits;
    # the trainer's cross-repo dedup would otherwise delete most of the fixture.
    train = [make_row(rng, i, "train") for i in range(n)]
    test = [make_row(rng, 10_000 + i, "test") for i in range(n_test)]
    return DatasetDict({"train": Dataset.from_list(train),
                        "test": Dataset.from_list(test)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dd = build(args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(args.out)
    print(f"[fixture] train={len(dd['train'])} test={len(dd['test'])} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
