"""Corpus construction for ReFuN / UniFuN.

Four stages, each a module with its own CLI so a rebuild can resume anywhere:

    1. run_ghidra      binaries    -> per-binary Ghidra JSON (orig + stripped)
    2. build_corpus    Ghidra JSON -> HF dataset with assembly, C and AST views
    3. reasoning       dataset     -> the fourth view, LLM reasoning traces
    4. (merge)         traces      -> the published dataset

Stage 3 is separate because it is the only one needing a foundation model and
the only one whose cost scales with an external API; 1-2 are deterministic and
run offline. The published corpora already contain all four views, so training
does not require any of this.
"""
