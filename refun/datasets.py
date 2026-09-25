"""Dataset registry.

Archival source: https://zenodo.org/records/15530083

The corpora are also mirrored on the HuggingFace Hub. Repo IDs are stored here
as bare names and joined to $REFUN_HF_NAMESPACE at load time, so the account
that hosts them is not committed to this repository:

    export REFUN_HF_NAMESPACE=<account>
    python -m refun.train --datasets x64_O0 ...

Without it, `repo_id()` raises rather than silently trying to download from a
placeholder namespace. Fully-qualified repo IDs and local paths bypass it.

Upstream repo names are historical and inconsistent; the mapping below is the
single source of truth that keeps that out of the rest of the code.
"""
import os
from typing import Dict, List, Optional

NAMESPACE_ENV = "REFUN_HF_NAMESPACE"
ZENODO_RECORD = "https://zenodo.org/records/15530083"
_ANON = "<anonymized>"

ARCHES = ("x64", "x86", "arm", "mips")
OPT_LEVELS = ("O0", "O1", "O2", "O3")

# 4 architectures x 4 optimisation levels.
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

# Obfuscator-LLVM passes, x64/O0.
OBFUSCATION_DATASETS: Dict[str, str] = {
    "obf_orig":   "prompt_obfuscated_binaries_orig",        # unobfuscated control
    "obf_bcfobf": "prompt_obfuscated_binaries_bcfobf",      # bogus control flow
    "obf_cffobf": "All_prompt_obfuscated_binaries_cffobf",  # control-flow flattening
    "obf_subobf": "All_prompt_obfuscated_binaries_subobf",  # instruction substitution
}

DATASETS: Dict[str, str] = {**COMPILER_DATASETS, **OBFUSCATION_DATASETS}

ALL_CONFIGS = tuple(DATASETS)
COMPILER_CONFIGS = tuple(COMPILER_DATASETS)
OBFUSCATION_CONFIGS = tuple(OBFUSCATION_DATASETS)

# UniFuN trains one model over every compiler config; ReFuN trains one per
# config. This tuple is the only difference between them.
UNIFUN_CONFIGS = COMPILER_CONFIGS

# Row counts as loaded, so a later mismatch is detectable rather than silent.
# Refresh with scripts/record_dataset_stats.py.
ROW_COUNTS: Dict[str, Dict[str, int]] = {
    "x64_O0": {"train": 62415, "test": 10585},
}


def namespace(required: bool = True) -> Optional[str]:
    """The HuggingFace account hosting the corpora, from $REFUN_HF_NAMESPACE."""
    ns = os.environ.get(NAMESPACE_ENV, "").strip().strip("/")
    if ns:
        return ns
    if not required:
        return None
    raise RuntimeError(
        f"{NAMESPACE_ENV} is not set.\n"
        f"Set it to the account hosting the Hub mirror:\n"
        f"    export {NAMESPACE_ENV}=<account>\n"
        f"or pass a fully-qualified repo ID or a local path to --datasets.\n"
        f"The corpora are also archived at {ZENODO_RECORD}."
    )


def repo_id(config: str) -> str:
    """Fully-qualified repo ID for a config name.

    Values containing '/' are treated as explicit repo IDs and returned
    unchanged, so --datasets accepts config names and repo IDs alike.
    """
    if "/" in config:
        return config
    if config not in DATASETS:
        raise KeyError(
            f"unknown config {config!r}; known configs: {', '.join(sorted(DATASETS))}"
        )
    return f"{namespace()}/{DATASETS[config]}"


def resolve(configs: List[str]) -> List[str]:
    """Map config names, aliases and repo IDs to a deduplicated list of IDs."""
    out: List[str] = []
    for c in configs:
        for expanded in _expand_alias(c):
            rid = repo_id(expanded)
            if rid not in out:
                out.append(rid)
    return out


def _expand_alias(token: str) -> List[str]:
    """Group aliases: 'all', 'compiler'/'unifun', 'obfuscation', an arch name,
    or an optimisation level."""
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
    """Inverse of `repo_id`, for labelling output files."""
    bare = rid.split("/")[-1]
    for cfg, name in DATASETS.items():
        if name == bare:
            return cfg
    return bare


def describe() -> str:
    ns = namespace(required=False) or _ANON
    lines = [f"archival source: {ZENODO_RECORD}",
             f"namespace: {ns}  (from ${NAMESPACE_ENV})", ""]
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
