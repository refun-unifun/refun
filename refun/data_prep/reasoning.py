"""Reasoning-trace distillation: the fourth encoder view.

The fourth view is a natural-language rationale about what a function does,
used asymmetrically:

  reasoning_1 (training, PROMPT_TRAIN)
      The teacher sees the code *and the ground-truth name* and justifies that
      name without stating it. A distillation target encoding why the name
      fits; valid only as a training signal.

  model_generated_description_test (evaluation, PROMPT_EVAL)
      The teacher sees only the code. Nothing about the gold name enters the
      prompt, so this is available at inference on an unseen binary.

Pointing --train_desc_field and --eval_desc_field at the same column leaks the
answer. Both templates below are reproduced exactly as used, including their
original typography, so regenerated traces match the published distribution.

    python -m refun.data_prep.reasoning --show
    python -m refun.data_prep.reasoning --in functions.jsonl --out traces.jsonl \\
        --mode eval --model <model-id> --base_url <openai-compatible-endpoint>

The generator speaks the OpenAI chat-completions protocol, which every vendor
and local server (vLLM, llama.cpp, Ollama) exposes. No provider is hardcoded.
"""
import argparse
import json
import os
import re
import sys
import time
from typing import Dict, Iterable, Iterator, List, Optional

# --- Prompt templates, verbatim as used to build the corpus ---------------

PROMPT_TRAIN = """You are an expert in software reverse engineering and semantic function analysis.
You will be provided with:
- Decompiled C-like code of a function
- A function name assigned by the original developer
Your task is to produce abstract reasoning about naming decisions, *without repeating, referencing, or generating any specific function names*:
1. Justify why the provided function name is appropriate in approximately 100 words. Base your reasoning on the function’s semantic intent and purpose, control flow, recursion, input/output behavior, memory or buffer operations, safety checks, escaping logic, data transformation, and the overall behavior flow.
2. Construct a counterfactual rationale describing why some hypothetical, less appropriate name (not shown or stated) might initially seem reasonable but would be misleading, vague, overly specific, too broad, or inaccurate. Do not reference any names, and only briefly explain why these names misrepresent or fail to fully capture the function's actual purpose or level of abstraction, and how it might confuse future readers or maintainers.
3. Compare the original developer-assigned name to the alternatives.
   - Discuss how the original name better aligns with the function’s behavior.
   - Consider what level of abstraction or context the developer might have intended.

Please do not mention the function name in your response. Please output exactly in the following structure, with no additional text:
Reasoning: <... Do not mention any kind of function name in it though the function is provided>
Counterfactual Justification: <…>
decompile c-code: "{code}"
developer function name: {name}
Response"""

PROMPT_EVAL = """You are an expert in software reverse engineering and program analysis.
Given the following decompiled C-like function, your task is:
1. Propose a concise function name.
2. Explain (~100 words).
3. Justify why other names would mislead.
Please output exactly in the following structure, with no additional text:
Reasoning: <…Do not mention any kind of function name in it though the function is there >
Counterfactual Justification: <Please do not mention the function name in here…>
Function Name: <…>
Code:
{code}

Response:"""


def build_prompt(code: str, name: Optional[str] = None, mode: str = "eval") -> str:
    """Render one prompt. `mode='train'` requires the gold name."""
    if mode == "train":
        if not name:
            raise ValueError("mode='train' needs the ground-truth function name")
        return PROMPT_TRAIN.format(code=code, name=name)
    if mode == "eval":
        return PROMPT_EVAL.format(code=code)
    raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")


# --- Response parsing ------------------------------------------------------

_SEC = re.compile(
    r"Reasoning:\s*(?P<reasoning>.*?)"
    r"(?:Counterfactual\s+Justification:\s*(?P<counterfactual>.*?))?"
    r"(?:Function\s+Name:\s*(?P<name>.*?))?$",
    re.S | re.I,
)


def parse_response(raw: str) -> Dict[str, str]:
    """Split a teacher response into its three labelled sections.

    Absent sections come back empty rather than raising, so one malformed
    generation does not abort a corpus build.
    """
    if not raw:
        return {"reasoning": "", "counterfactual": "", "function_name": ""}
    m = _SEC.search(raw.strip())
    if not m:
        return {"reasoning": raw.strip(), "counterfactual": "", "function_name": ""}
    name = (m.group("name") or "").strip()
    # Teachers wrap the name in backticks or underscores; strip decoration but
    # keep the identifier itself untouched.
    name = name.strip("`*_ \t\r\n").split("\n")[0].strip()
    return {
        "reasoning": (m.group("reasoning") or "").strip(),
        "counterfactual": (m.group("counterfactual") or "").strip(),
        "function_name": name,
    }


def scrub_name(text: str, gold: Optional[str]) -> str:
    """Remove the gold name from a trace where the teacher stated it anyway
    (~3% on x64_O0), so the model cannot learn to copy a span it will never
    see at inference."""
    if not text or not gold or len(gold) < 4:
        return text or ""
    return re.sub(r"(?<![A-Za-z0-9_])" + re.escape(gold) + r"(?![A-Za-z0-9_])",
                  "<NAME>", text)


# --- Generation ------------------------------------------------------------

def generate(records: Iterable[Dict], model: str, mode: str = "eval",
             base_url: Optional[str] = None, api_key: Optional[str] = None,
             code_field: str = "decompiled_code_stripped",
             name_field: str = "original_function_name",
             max_tokens: int = 512, temperature: float = 0.2,
             retries: int = 3, sleep: float = 2.0) -> Iterator[Dict]:
    """Yield one parsed trace per input record, in order.

    Failures carry an `error` key rather than being dropped, so output length
    matches input and a partial run resumes by filtering on that key.
    """
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError("trace generation needs `pip install openai`") from e

    client = OpenAI(
        base_url=base_url or os.environ.get("REFUN_LLM_BASE_URL") or None,
        api_key=api_key or os.environ.get("REFUN_LLM_API_KEY") or "EMPTY",
    )

    for rec in records:
        code = rec.get(code_field) or ""
        gold = rec.get(name_field) or ""
        if not code.strip():
            yield {**rec, "error": "empty_code"}
            continue
        prompt = build_prompt(code, gold, mode)
        raw, err = "", None
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw = resp.choices[0].message.content or ""
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                if attempt < retries - 1:
                    time.sleep(sleep * (attempt + 1))
        if err:
            yield {**rec, "error": err}
            continue
        parsed = parse_response(raw)
        out = {**rec, "raw_output": raw}
        if mode == "train":
            out["reasoning_1"] = scrub_name(parsed["reasoning"], gold)
            out["counterfactual_justification_1"] = scrub_name(
                parsed["counterfactual"], gold)
        else:
            out["model_generated_description_test"] = parsed["reasoning"]
            out["model_generated_counterfactual_test"] = parsed["counterfactual"]
            out["model_generated_model_function_name"] = parsed["function_name"]
        yield out


def _read_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true",
                    help="print both prompt templates and exit")
    ap.add_argument("--in", dest="inp", help="input JSONL of function records")
    ap.add_argument("--out", help="output JSONL with traces")
    ap.add_argument("--mode", choices=["train", "eval"], default="eval")
    ap.add_argument("--model", help="teacher model id")
    ap.add_argument("--base_url", default=None,
                    help="OpenAI-compatible endpoint ($REFUN_LLM_BASE_URL)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args(argv)

    if args.show:
        print("=" * 30, "PROMPT_TRAIN (-> reasoning_1)", "=" * 30)
        print(PROMPT_TRAIN)
        print()
        print("=" * 30, "PROMPT_EVAL (-> model_generated_description_test)", "=" * 30)
        print(PROMPT_EVAL)
        return 0

    if not (args.inp and args.out and args.model):
        ap.error("--in, --out and --model are required unless --show is given")

    records = _read_jsonl(args.inp)
    if args.limit:
        records = (r for i, r in enumerate(records) if i < args.limit)

    n = failed = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in generate(records, model=args.model, mode=args.mode,
                            base_url=args.base_url, max_tokens=args.max_tokens,
                            temperature=args.temperature):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            failed += 1 if rec.get("error") else 0
            if n % 50 == 0:
                print(f"  {n} done ({failed} failed)", flush=True)
    print(f"[reasoning] wrote {n} records to {args.out} ({failed} failed)")
    return 1 if failed == n and n else 0


if __name__ == "__main__":
    sys.exit(main())
