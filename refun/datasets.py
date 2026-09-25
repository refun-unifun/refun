"""Dataset registry for ReFuN / UniFuN.

DOUBLE-BLIND NOTE
-----------------
The corpora live on the HuggingFace Hub under a single account. The account
name is *not* committed to this repository, because it identifies the authors.
Every repo ID is therefore stored here as a bare name and joined to a namespace
read from the environment:

    export REFUN_HF_NAMESPACE=<the account>
    python -m refun.train --configs x64_O0 ...

Reviewers who have been given the namespace out of band can run everything
unchanged. Without it, `repo_id()` raises with an explanatory message rather
than silently trying to download `<anonymized>/...`.

Config naming is `<arch>_<opt>` for the compiler corpora and `obf_<pass>` for
the obfuscation study. The upstream repo names are historical and inconsistent
(several were rebuilt mid-project and kept a `recreated`/`final` prefix); the
mapping below is the single source of truth that hides that inconsistency from
the rest of the code.
"""
import os
from typing import Dict, List, Optional

NAMESPACE_ENV = "REFUN_HF_NAMESPACE"
_ANON = "<anonymized>"

ARCHES = ("x64", "x86", "arm", "mips")
OPT_LEVELS = ("O0", "O1", "O2", "O3")

# --- The 16 compiler configs (4 architectures x 4 optimisation levels) ------
COMPILER_DATASETS: Dict[str, str] = {
    "x64_O0": "prompt_reverse_engineering_code_reverse_engineering_code_dataset_O0_x64_O0",
    "x64_O1": "prompt_rreverse_engineering_code_dataset_O1_x64_O1",
    "x64_O2": "prompt_reverse_engineering_code_reverse_engineering_code_dataset_O2_x64_O2",
    "x64_O3": "final_reverse_engineering_code_dataset_O3_x64_O3",

    "x86_O0": "prompt_reverse_engineering_code_dataset_O0_x86_O0",
    "x86_O1": "prompt_reverse_engineering_code_dataset_O1_x86_O1",
    "x86_O2": "final_recreated_reverse_engineering_code_dataset_O2_x86_O2",
    "x86_O3": "final_reverse_engineering_dataset_O3_x86_O3",

    "arm_O0": "prompt_reverse_engineering_code_dataset_O0_arm_O0_advanced_custom_test",
    "arm_O1": "prompt_reverse_engineering_code_dataset_O1_arm_O1_advanced_custom_test",
    "arm_O2": "prompt_reverse_engineering_code_dataset_O2_arm_O2_advanced_custom_test",
    "arm_O3": "prompt_reverse_engineering_code_dataset_O3_arm_O3_advanced_custom_test",

    "mips_O0": "prompt_reverse_engineering_code_reverse_engineering_code_dataset_O0_mips_O0",
    "mips_O1": "prompt_reverse_engineering_code_reverse_engineering_code_dataset_O1_mips_O1",
    "mips_O2": "prompt_reverse_engineering_code_reverse_engineering_code_dataset_O2_mips_O2",
    "mips_O3": "prompt_reverse_engineering_code_reverse_engineering_code_dataset_O3_mips_O3",
}

# --- The obfuscation study (Obfuscator-LLVM passes, x64/O0) ----------------
OBFUSCATION_DATASETS: Dict[str, str] = {
    "obf_orig":   "prompt_obfuscated_binaries_orig",     # unobfuscated control
    "obf_bcfobf": "prompt_obfuscated_binaries_bcfobf",   # bogus control flow
    "obf_cffobf": "All_prompt_obfuscated_binaries_cffobf",  # control-flow flattening
    "obf_subobf": "All_prompt_obfuscated_binaries_subobf",  # instruction substitution
}

# Superseded rebuilds kept only so an older run log can be resolved. Not part of
# any reported result -- `ALL_CONFIGS` deliberately excludes them.
SUPERSEDED: Dict[str, str] = {
    "arm_O1__v1": "prompt_reverse_engineering_code_dataset_O1_arm_O1",
    "x86_O1__v1": "recreated_reverse_engineering_code_dataset_O1_x86_O1",
    "x86_O2__v1": "recreated_reverse_engineering_code_dataset_O2_x86_O2",
    "x86_O3__v1": "recreated_reverse_engineering_code_dataset_O3_x86_O3",
}

DATASETS: Dict[str, str] = {**COMPILER_DATASETS, **OBFUSCATION_DATASETS}
_RESOLVABLE: Dict[str, str] = {**DATASETS, **SUPERSEDED}

ALL_CONFIGS = tuple(DATASETS)
COMPILER_CONFIGS = tuple(COMPILER_DATASETS)
OBFUSCATION_CONFIGS = tuple(OBFUSCATION_DATASETS)

# UniFuN: one model over every compiler config at once. ReFuN trains one model
# per config. This tuple is the only thing that differs between the two.
UNIFUN_CONFIGS = COMPILER_CONFIGS

# Row counts as loaded (train/test), recorded so a later mismatch is detectable
# rather than silent. Populated by `scripts/record_dataset_stats.py`; a config
# absent here has simply not been measured on this machine yet.
ROW_COUNTS: Dict[str, Dict[str, int]] = {
    "x64_O0": {"train": 62415, "test": 10585},
}


def namespace(required: bool = True) -> Optional[str]:
    """The HuggingFace account holding the corpora, from $REFUN_HF_NAMESPACE."""
    ns = os.environ.get(NAMESPACE_ENV, "").strip().strip("/")
    if ns:
        return ns
    if not required:
        return None
    raise RuntimeError(
        f"{NAMESPACE_ENV} is not set.\n"
        "This is an anonymised artifact: the HuggingFace account that hosts the\n"
        "corpora is not committed to the repository. Set it before running:\n"
        f"    export {NAMESPACE_ENV}=<account>\n"
        "or pass fully-qualified repo IDs directly with --datasets."
    )


def repo_id(config: str) -> str:
    """Fully-qualified HF repo ID for a config name.

    A value that already contains '/' is treated as an explicit repo ID and
    returned untouched, so --datasets accepts config names and raw IDs alike.
    """
    if "/" in config:
        return config
    if config not in _RESOLVABLE:
        raise KeyError(
            f"unknown config {config!r}; known configs: {', '.join(sorted(DATASETS))}"
        )
    return f"{namespace()}/{_RESOLVABLE[config]}"


def resolve(configs: List[str]) -> List[str]:
    """Map a mixed list of config names / repo IDs / group aliases to repo IDs."""
    out: List[str] = []
    for c in configs:
        for expanded in _expand_alias(c):
            rid = repo_id(expanded)
            if rid not in out:
                out.append(rid)
    return out


def _expand_alias(token: str) -> List[str]:
    """Group aliases: 'all', 'compiler', 'obfuscation', 'unifun', an arch name
    ('x64'), or an opt level ('O0')."""
    t = token.strip()
    if "/" in t:
        return [t]
    low = t.lower()
    if low == "all":
        return list(ALL_CONFIGS)
    if low in ("compiler", "unifun"):
        return list(COMPILER_CONFIGS)
    if low in ("obfuscation", "obf"):
        return list(OBFUSCATION_CONFIGS)
    if low in ARCHES:
        return [c for c in COMPILER_CONFIGS if c.split("_")[0] == low]
    if t.upper() in OPT_LEVELS:
        return [c for c in COMPILER_CONFIGS if c.split("_")[1] == t.upper()]
    return [t]


def arch_of(config: str) -> str:
    return config.split("_")[0]


def opt_of(config: str) -> str:
    parts = config.split("_")
    return parts[1] if len(parts) > 1 else ""


def config_of_repo(rid: str) -> str:
    """Inverse of `repo_id` -- config name for a repo ID, for labelling outputs."""
    bare = rid.split("/")[-1]
    for cfg, name in _RESOLVABLE.items():
        if name == bare:
            return cfg
    return bare


def describe() -> str:
    """Human-readable inventory; used by `python -m refun.datasets`."""
    ns = namespace(required=False) or _ANON
    lines = [f"namespace: {ns}  (from ${NAMESPACE_ENV})", ""]
    lines.append(f"compiler configs ({len(COMPILER_CONFIGS)}):")
    for c in COMPILER_CONFIGS:
        rc = ROW_COUNTS.get(c)
        cnt = f"  train={rc['train']:,} test={rc['test']:,}" if rc else "  (counts not recorded)"
        lines.append(f"  {c:<9} {ns}/{COMPILER_DATASETS[c]}{cnt}")
    lines.append("")
    lines.append(f"obfuscation configs ({len(OBFUSCATION_CONFIGS)}):")
    for c in OBFUSCATION_CONFIGS:
        lines.append(f"  {c:<9} {ns}/{OBFUSCATION_DATASETS[c]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
