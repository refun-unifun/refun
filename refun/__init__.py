"""ReFuN / UniFuN — reasoning-guided multi-view function name inference.

Modules
-------
train          all four fusion strategies, one code path (`--fusion`)
unifun         UniFuN entry point: one model over every config, + breakdown
datasets       corpus registry; repo IDs joined to $REFUN_HF_NAMESPACE
sexpr          tree-sitter S-expression views of decompiled C
audit_leakage  how often the gold name is already in the input (read this)
data_prep      corpus construction: Ghidra -> views -> reasoning traces

Entry points
------------
    python -m refun.train --fusion moe --datasets x64_O0 --output_dir runs/moe
    python -m refun.unifun --output_dir runs/unifun
    python -m refun.datasets            # print the corpus inventory
    python -m refun.audit_leakage --configs x64_O0
"""

__version__ = "1.0.0"
