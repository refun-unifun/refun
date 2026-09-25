"""Load a trained ReFuN / UniFuN checkpoint and predict function names.

    python -m refun.predict --model runs/moe_x64_O0/final_model --asm f.asm --code f.c
    python -m refun.predict --model runs/moe_x64_O0/final_model \\
        --input x64_O0 --split test --limit 500 --out preds.tsv

`--input` accepts a JSONL file, a local dataset directory, or a registry
config name. Batch mode reports exact match and token P/R/F1.

`final_model/` cannot be loaded from the weights alone: the fusion strategy
picks the class, and LABEL_LEN (computed dynamically during training) shapes
the fusion projection layers, so a mismatch fails on every fusion weight.
Training writes both to run_config.json, which `load_model` reads. For
checkpoints predating that file, pass --fusion and --label_len.

The model consumes four views; missing ones may be empty, and the encoder then
sees only its marker token. Accuracy drops accordingly, most sharply without
the reasoning view. The AST view is derived from --code when not supplied.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch


def load_model(model_dir: str, fusion: Optional[str] = None,
               label_len: Optional[int] = None,
               tokens_per_encoder: Optional[int] = None,
               device: Optional[str] = None):
    """Rebuild the model and tokenizer from `final_model/`.

    Returns (model, tokenizer, run_config), the model in eval mode.
    """
    from transformers import AutoTokenizer
    from . import train as T

    cfg_path = os.path.join(model_dir, "run_config.json")
    cfg: Dict = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    elif not (fusion and label_len):
        raise FileNotFoundError(
            f"{cfg_path} not found. This checkpoint predates run_config.json, "
            "so the fusion strategy and label length cannot be recovered. "
            "Pass --fusion and --label_len; --label_len must match training or "
            "the state dict will not load."
        )

    fusion = fusion or cfg.get("fusion")
    label_len = label_len or cfg.get("label_len")
    tokens = tokens_per_encoder or cfg.get("tokens_per_encoder") or T.CHUNK_LEN
    if not fusion:
        raise ValueError("fusion strategy unknown; pass --fusion")
    if not label_len:
        raise ValueError("label_len unknown; pass --label_len")

    # Module-level globals read by the model classes at construction time.
    T.LABEL_LEN = int(label_len)
    T.CHUNK_LEN = int(tokens)
    T.TOKENS_PER_ENCODER = int(tokens)
    T.ENCODER_MODE = cfg.get("encoder_mode")
    if cfg.get("num_experts"):
        T.NUM_EXPERTS = cfg["num_experts"]
        T.NUM_SELECTED_EXPERTS = cfg.get("num_selected_experts",
                                         T.NUM_SELECTED_EXPERTS)

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = T.build_model(fusion, tok)

    # Training adds seven special tokens and resizes embeddings to match. The
    # saved tokenizer carries them but build_model() starts from a stock CodeT5
    # at 32100, so without this the state dict fails on every embedding.
    if len(tok) != model.config.vocab_size:
        print(f"[predict] resizing embeddings {model.config.vocab_size} -> {len(tok)}")
        model.resize_token_embeddings(len(tok))

    state_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"no pytorch_model.bin in {model_dir}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[predict] {len(missing)} missing key(s), first few: {missing[:5]}")
    if unexpected:
        print(f"[predict] {len(unexpected)} unexpected key(s), first few: {unexpected[:5]}")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    print(f"[predict] loaded {fusion} (label_len={label_len}, "
          f"tokens_per_encoder={tokens}, encoder_mode={T.ENCODER_MODE}) on {dev}")
    return model, tok, cfg


def _encode_views(tok, asm: str, code: str, sexpr: str, desc: str,
                  tokens_per_encoder: int, device: str):
    """Tokenise the four views as training does: marker token, then truncate."""
    markers = ["<ASM>", "<DEC>", "<SEXP>", "<DESC>"]
    batch = {}
    for i, (marker, text) in enumerate(zip(markers, [asm, code, sexpr, desc]), 1):
        enc = tok(marker + " " + (text or ""), truncation=True,
                  max_length=tokens_per_encoder, return_tensors="pt")
        batch[f"input_ids{i}"] = enc.input_ids.to(device)
        batch[f"attention_mask{i}"] = enc.attention_mask.to(device)
    return batch


@torch.no_grad()
def predict_one(model, tok, asm: str = "", code: str = "", sexpr: str = "",
                desc: str = "", tokens_per_encoder: int = 512,
                max_new_tokens: Optional[int] = None) -> str:
    """Predict a name for one function; missing views may be empty."""
    from . import train as T

    if code and not sexpr:
        try:
            from .sexpr import sexpr_with_text
            sexpr = sexpr_with_text(code)
        except ImportError:
            pass  # tree-sitter absent; proceed without the AST view

    device = next(model.parameters()).device
    batch = _encode_views(tok, asm, code, sexpr, desc, tokens_per_encoder, device)
    out = model.generate(**batch,
                         max_length=max_new_tokens or T.LABEL_LEN,
                         num_beams=1)
    text = tok.decode(out[0], skip_special_tokens=True)
    return T.strip_tags(text).strip()


def _iter_records(spec: str, split: str, limit: int, cache_dir: Optional[str]):
    """Records from a JSONL file, a dataset directory, or a registry config."""
    if spec.endswith(".jsonl") and os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    return
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    from .train import _load_dataset_any
    from . import datasets as reg

    repo = spec if (os.path.isdir(spec) or "/" in spec) else reg.repo_id(spec)
    d = _load_dataset_any(repo, cache_dir)
    ds = d[split] if split in d else d[list(d)[0]]
    n = min(limit, len(ds)) if limit else len(ds)
    for i in range(n):
        yield ds[i]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to final_model/")
    ap.add_argument("--fusion", default=None,
                    help="only needed for checkpoints without run_config.json")
    ap.add_argument("--label_len", type=int, default=None)
    ap.add_argument("--tokens_per_encoder", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (auto by default)")

    ap.add_argument("--input", default=None,
                    help="JSONL file, local dataset dir, or registry config name")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", default=None, help="write predictions as TSV")

    ap.add_argument("--asm", default=None, help="file with the assembly view")
    ap.add_argument("--code", default=None, help="file with decompiled C")
    ap.add_argument("--sexpr", default=None,
                    help="file with the AST view (derived from --code if omitted)")
    ap.add_argument("--desc", default=None, help="file with the reasoning trace")
    args = ap.parse_args(argv)

    model, tok, cfg = load_model(args.model, args.fusion, args.label_len,
                                 args.tokens_per_encoder, args.device)
    tpe = args.tokens_per_encoder or cfg.get("tokens_per_encoder") or 512

    def _read(p):
        if not p:
            return ""
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    # --- single function mode -------------------------------------------
    if not args.input:
        if not (args.asm or args.code):
            ap.error("give --input, or at least one of --asm / --code")
        name = predict_one(model, tok, _read(args.asm), _read(args.code),
                           _read(args.sexpr), _read(args.desc), tpe)
        print(f"\npredicted function name: {name}")
        return 0

    # --- batch mode ------------------------------------------------------
    desc_field = cfg.get("eval_desc_field", "model_generated_description_test")
    rows, correct, scored = [], 0, 0
    for i, r in enumerate(_iter_records(args.input, args.split, args.limit,
                                        args.cache_dir), 1):
        pred = predict_one(
            model, tok,
            str(r.get("assembly_code") or ""),
            str(r.get("decompiled_code_stripped") or ""),
            str(r.get("S-Expression_of_decompiled_code_stripped") or ""),
            str(r.get(desc_field) or ""),
            tpe,
        )
        gold = str(r.get("original_function_name") or "")
        rows.append((gold, pred))
        if gold:
            scored += 1
            correct += int(gold.strip() == pred.strip())
        if i % 25 == 0:
            print(f"  {i} predicted", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("ground_truth\tprediction\n")
            for g, p in rows:
                fh.write(f"{g}\t{p}\n")
        print(f"[predict] wrote {len(rows)} predictions to {args.out}")

    if scored:
        from .train import token_prf, load_sp, normalise
        sp = load_sp(os.environ.get("REFUN_SP_MODEL", "./assets/segmentation.model"))
        refs = [normalise(g, sp) for g, _ in rows]
        hyps = [normalise(p, sp) for _, p in rows]
        pr, rc, f1 = token_prf(refs, hyps)
        print(f"\nexact match : {correct}/{scored} = {100.0 * correct / scored:.2f}%")
        print(f"token P/R/F1: {pr:.4f} / {rc:.4f} / {f1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
