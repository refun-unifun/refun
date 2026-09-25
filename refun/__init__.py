"""ReFuN / UniFuN - reasoning-guided multi-view function name inference.

Modules
-------
train          all four fusion strategies, one code path (--fusion)
unifun         UniFuN: one model over every config, with a per-config breakdown
predict        load a trained checkpoint and run inference
datasets       corpus registry; repo IDs joined to $REFUN_HF_NAMESPACE
sexpr          tree-sitter S-expression views of decompiled C
audit_leakage  measures how often the gold name is already in the input
data_prep      corpus construction: Ghidra -> views -> reasoning traces

Entry points
------------
    python -m refun.train --fusion moe --datasets x64_O0 --output_dir runs/moe
    python -m refun.unifun --output_dir runs/unifun
    python -m refun.predict --model runs/moe/final_model --input x64_O0
    python -m refun.datasets
    python -m refun.audit_leakage --configs x64_O0
"""

__version__ = "1.0.0"
