"""Fast unit tests for the pieces that do not need a model or a GPU.

    python tests/test_units.py        # or: pytest tests/test_units.py

Covers the dataset registry, S-expression generation, leakage classification
and reasoning-response parsing. The full training path is covered separately by
`scripts/smoke_test.sh`, which is slow because it actually trains.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


# --------------------------------------------------------------- registry ---
def test_registry():
    print("registry")
    os.environ["REFUN_HF_NAMESPACE"] = "testns"
    import importlib
    from refun import datasets as d
    importlib.reload(d)

    check("16 compiler configs", len(d.COMPILER_CONFIGS) == 16,
          f"got {len(d.COMPILER_CONFIGS)}")
    check("4 obfuscation configs", len(d.OBFUSCATION_CONFIGS) == 4)
    check("all 4 arches x 4 opts present",
          {f"{a}_{o}" for a in d.ARCHES for o in d.OPT_LEVELS}
          == set(d.COMPILER_CONFIGS))
    check("repo_id joins namespace",
          d.repo_id("x64_O0").startswith("testns/"))
    check("explicit repo id passes through",
          d.repo_id("someone/else") == "someone/else")
    check("alias: arch", len(d.resolve(["x64"])) == 4)
    check("alias: opt level", len(d.resolve(["O0"])) == 4)
    check("alias: unifun", len(d.resolve(["unifun"])) == 16)
    check("alias: all", len(d.resolve(["all"])) == 20)
    check("resolve dedupes", len(d.resolve(["x64_O0", "x64_O0"])) == 1)
    check("config_of_repo inverts repo_id",
          d.config_of_repo(d.repo_id("mips_O3")) == "mips_O3")
    check("unknown config raises", _raises(lambda: d.repo_id("nope_O9"), KeyError))

    os.environ.pop("REFUN_HF_NAMESPACE")
    importlib.reload(d)
    check("missing namespace raises explanatory error",
          _raises(lambda: d.repo_id("x64_O0"), RuntimeError))
    os.environ["REFUN_HF_NAMESPACE"] = "testns"


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ------------------------------------------------------------------ sexpr ---
SAMPLE = """
void FUN_00157910(undefined8 *param_1)

{
  *param_1 = &PTR_FUN_00426068;
  if ((undefined8 *)param_1[1] != param_1 + 3) {
    FUN_002dac70();
  }
  return;
}
"""


def test_sexpr():
    print("sexpr")
    from refun.sexpr import (sexpr_with_text, sexpr_clean, sexpr_fields,
                             parse_status)

    wt = sexpr_with_text(SAMPLE)
    check("with_text starts at translation_unit",
          wt.startswith("(translation_unit "))
    check("with_text keeps source span", "FUN_00157910" in wt)
    check("with_text balanced parens", wt.count("(") >= wt.count(")") > 0)

    cl = sexpr_clean(SAMPLE)
    check("clean drops identifiers", "FUN_00157910" not in cl)
    check("clean uses IDENT placeholder", "IDENT" in cl)
    check("clean uses TYPE placeholder", "TYPE" in cl)
    check("clean keeps structure", "function_definition" in cl)

    fl = sexpr_fields(SAMPLE)
    check("fields has field labels", "declarator:" in fl)
    check("fields drops source text", "FUN_00157910" not in fl)

    check("empty input -> empty output", sexpr_with_text("") == "")
    check("whitespace input -> empty output", sexpr_clean("   \n ") == "")

    st = parse_status(SAMPLE)
    check("parse_status reports node count", st["nodes"] > 10)
    check("parse_status ok on valid C", st["ok"] is True, str(st))
    check("parse_status flags empty",
          parse_status("")["reason"] == "empty")

    # Truncation budget is honoured rather than ignored.
    short = sexpr_with_text(SAMPLE, max_nodes=5)
    check("max_nodes truncates", len(short) < len(wt))


# ---------------------------------------------------------------- leakage ---
def test_leakage():
    print("leakage")
    from refun.audit_leakage import classify, is_self_named

    thunk = "\nvoid abort(void)\n\n{\n  (*(code *)PTR_abort_0012b048)();\n  return;\n}\n"
    check("thunk -> signature", classify(thunk, "abort") == "signature")

    assertion = ('\nvoid FUN_001(void)\n\n{\n  FUN_0017c650("../../gold/object.h",'
                 '0x3c7,"do_output_section_offset");\n}\n')
    check("assert string -> string_literal",
          classify(assertion, "do_output_section_offset") == "string_literal")

    clean = "\nbool FUN_0012ef30(long param_1)\n\n{\n  return param_1 != 0;\n}\n"
    check("no occurrence -> clean", classify(clean, "write_build_id") == "clean")

    check("is_self_named true for thunk",
          is_self_named({"original_function_name": "abort",
                         "decompiled_code_stripped": thunk}))
    check("is_self_named false for clean",
          not is_self_named({"original_function_name": "write_build_id",
                             "decompiled_code_stripped": clean}))
    check("is_self_named false for assert-only",
          not is_self_named({"original_function_name": "do_output_section_offset",
                             "decompiled_code_stripped": assertion}))
    check("short names ignored",
          not is_self_named({"original_function_name": "cp",
                             "decompiled_code_stripped": "int cp(void){return 0;}"}))
    # Substring must not count: 'read' inside 'read_config' is not a leak.
    check("word boundary respected",
          classify("void FUN_1(void){ read_config(); }", "read") == "clean")


# -------------------------------------------------------------- reasoning ---
def test_reasoning():
    print("reasoning")
    from refun.data_prep.reasoning import (build_prompt, parse_response,
                                           scrub_name, PROMPT_TRAIN, PROMPT_EVAL)

    p = build_prompt("int f(void){return 0;}", "do_thing", mode="train")
    check("train prompt embeds code", "int f(void)" in p)
    check("train prompt embeds gold name", "do_thing" in p)
    e = build_prompt("int f(void){return 0;}", mode="eval")
    check("eval prompt embeds code", "int f(void)" in e)
    check("eval prompt withholds gold name", "do_thing" not in e)
    check("train mode needs a name",
          _raises(lambda: build_prompt("x", None, "train"), ValueError))
    check("bad mode raises",
          _raises(lambda: build_prompt("x", "n", "nope"), ValueError))

    raw = ("Reasoning: It writes a build id.\n\n"
           "Counterfactual Justification: Other names mislead.\n\n"
           "Function Name: `HandleBuildID`")
    got = parse_response(raw)
    check("parses reasoning", got["reasoning"] == "It writes a build id.")
    check("parses counterfactual",
          got["counterfactual"] == "Other names mislead.")
    check("strips name decoration", got["function_name"] == "HandleBuildID")

    partial = parse_response("Reasoning: only this section")
    check("tolerates missing sections",
          partial["reasoning"] == "only this section"
          and partial["counterfactual"] == "")
    check("tolerates empty input", parse_response("")["reasoning"] == "")

    check("scrub removes gold name",
          "write_build_id" not in scrub_name("calls write_build_id here",
                                             "write_build_id"))
    check("scrub respects word boundary",
          scrub_name("write_build_idx", "write_build_id") == "write_build_idx")
    check("scrub ignores short names",
          scrub_name("a cp b", "cp") == "a cp b")


def main():
    for fn in (test_registry, test_sexpr, test_leakage, test_reasoning):
        fn()
        print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("all unit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
