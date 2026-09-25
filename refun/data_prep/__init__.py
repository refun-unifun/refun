"""Corpus construction for ReFuN / UniFuN.

The pipeline runs in four stages; each is a standalone module with a CLI, so a
rebuild can resume at any point.

    1. run_ghidra      binaries        -> per-binary Ghidra JSON (orig + stripped)
    2. build_corpus    Ghidra JSON     -> HF dataset with the three code views
                                          (assembly, decompiled C, AST)
    3. reasoning       HF dataset      -> the fourth view: LLM reasoning traces
    4. (merge)         traces + corpus -> the published dataset

Stage 3 is separate because it is the only stage that needs a foundation model
and the only one whose cost scales with an external API. Stages 1-2 are
deterministic and reproducible offline.

The published corpora already contain all four views -- these modules exist so
the corpus can be rebuilt or extended to new architectures, not because
training requires them.
"""
