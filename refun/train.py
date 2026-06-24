# train.py
# Unified multi-view CodeT5 training script for four fusion strategies:
#   concat, cross_attention, simple_gating, moe
# Single self-contained file (module-level globals such as LABEL_LEN are mutated
# with `global` and must remain in one module for the original behavior to hold).
import os
import sys
import json
import math
import random
import collections
import re
import hashlib
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sentencepiece as spm
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    EarlyStoppingCallback,
)
from transformers.optimization import get_scheduler
from transformers.trainer_utils import SchedulerType
from datasets import load_dataset, concatenate_datasets

# --- Environment and Constants ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_DEBUG", "ERROR")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---- Monkey-patch: transformers 5.x passes transformers.AddedToken to the
# tokenizers Rust backend which now requires tokenizers.AddedToken. ----
def _patch_tokenizer_add_tokens():
    try:
        from tokenizers import AddedToken as TokAddedToken
        from transformers.tokenization_utils_base import AddedToken as HFAddedToken
        import transformers.tokenization_utils_tokenizers as _tut

        def _fixed_add_tokens(self, new_tokens, special_tokens=False):
            converted = []
            for t in new_tokens:
                if isinstance(t, HFAddedToken):
                    converted.append(TokAddedToken(
                        str(t),
                        single_word=t.single_word,
                        lstrip=t.lstrip,
                        rstrip=t.rstrip,
                        normalized=t.normalized,
                        special=getattr(t, 'special', False),
                    ))
                elif not isinstance(t, (str, TokAddedToken)):
                    converted.append(str(t))
                else:
                    converted.append(t)
            if special_tokens:
                return self._tokenizer.add_special_tokens(converted)
            return self._tokenizer.add_tokens(converted)

        _tut.PreTrainedTokenizerFast._add_tokens = _fixed_add_tokens
        print("[Patch] Tokenizer add_tokens patch applied.")
    except Exception as e:
        print(f"[Patch] Could not apply tokenizer patch: {e}")

_patch_tokenizer_add_tokens()
# ---- End monkey-patch ----


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

MODEL_NAME = "Salesforce/codet5-base"

CACHE_DIR = "./cache_arrow_10"
# Anonymized: paths now come from CLI args (--sp_model_path / --word_cluster_path),
# which default to env vars REFUN_SP_MODEL / REFUN_WORD_CLUSTER, else repo-relative.
SP_MODEL_PATH = os.environ.get("REFUN_SP_MODEL", "./assets/segmentation.model")
WORD_CLUSTER_PATH = os.environ.get("REFUN_WORD_CLUSTER", "./assets/word_cluster.json")
CHUNK_LEN = 512
LABEL_LEN = 16
BATCH_GPU = 16
GRAD_ACC = 8
MAX_EPOCHS = 2000
WARMUP = 2000
WEIGHT_DECAY = 0.01
LR_DECODER = 5e-4

LR_ENCODERS = [5e-4, 5e-4, 5e-4, 5e-4]
LABEL_SMOOTH = 0.1
LENGTH_PENALTY = 1.0

NUM_EXPERTS = 4
NUM_SELECTED_EXPERTS = 2
MOE_LOSS_WEIGHT = 0.01

USE_CONCAT_LINEAR_FUSION = False
TOKENS_PER_ENCODER = 512

EVAL_SUBSET_CB = 64
CB_BATCH = 4
COSINE_SIM = True
LOG_STEPS = 30
SAVE_DIR = "./4enc_moe_checkpoint_seq_tokens_all_fixes_symgen"
METRIC_KEY = "token_f1"
MAX_ALLOWED_LABEL_LEN = 15
DEBUG_FUSION = False
DEBUG_LOSS = True

START_TAG = "<function Name>"
END_TAG = "</function Name>"

# Anonymized: datasets must be supplied via --datasets on the CLI. No default
# dataset repositories are bundled with this release.
GENERIC_DATASET_REPOS: List[str] = []

GLOBAL_DUP_NAMES = set()  # after cross-repo name dedup

# --- SymGen-style dedup constants (column names) ---
BODY_COL = "decompiled_code_stripped"
FUNCNAME_COL = "original_function_name"
ASM_COL = "assembly_code"
ROOT_NODE_COL = "Root Node"

# --- Required columns for preprocessing ---
REQUIRED_BASE_COLS = [
    "assembly_code",
    "decompiled_code_stripped",
    "S-Expression_of_decompiled_code_stripped",
    "original_function_name",
]

# Regex helpers for dedup
_WS = re.compile(r"\s+")


def compact(s: Optional[str]) -> str:
    if not s:
        return ""
    return _WS.sub(" ", str(s)).strip()


def sha256_str(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


# asm normalization (similar to user script)
_HEX_ADDR = re.compile(r"\b0x[0-9a-fA-F]+\b")
_LINE_ADDR_PREFIX = re.compile(r"^\s*(?:[0-9a-fA-F]+:)?\s*", re.MULTILINE)
_LABEL_LINE = re.compile(r"^[.\w$]+:\s*$", re.MULTILINE)
_SEMI_COMMENT = re.compile(r"(^|\s);.*?$", re.MULTILINE)


def normalize_asm(s: Optional[str]) -> str:
    if not s:
        return ""
    x = str(s)
    x = _SEMI_COMMENT.sub(r"\1 ", x)
    x = _LABEL_LINE.sub(" ", x)
    x = _LINE_ADDR_PREFIX.sub("", x)
    x = _HEX_ADDR.sub(" <ADDR> ", x)
    x = compact(x).lower()
    return x


# callee-name normalization
_USE_TS = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_c

    C_LANGUAGE = Language(tree_sitter_c.language())
    _USE_TS = True
except Exception:
    _USE_TS = False

_CALLEE_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_EXCLUDE = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "case",
    "break",
    "do",
    "else",
    "static",
    "const",
    "volatile",
    "inline",
    "struct",
    "union",
    "enum",
    "typedef",
    "unsigned",
    "signed",
    "short",
    "long",
    "int",
    "char",
    "float",
    "double",
    "void",
    "restrict",
    "goto",
    "continue",
    "default",
}


def callee_norm_regex(code: str) -> str:
    mapping: Dict[str, str] = {}
    counter = 0

    def repl(m):
        nonlocal counter
        name = m.group(1)
        if name.lower() in _EXCLUDE:
            return f"{name}("
        if name not in mapping:
            counter += 1
            mapping[name] = f"CALLEE_{counter}"
        return f"{mapping[name]}("

    return _CALLEE_CALL_RE.sub(repl, code)


def callee_norm_treesitter(code: str) -> str:
    parser = Parser(C_LANGUAGE)
    tree = parser.parse(bytes(code, "utf-8"))
    root = tree.root_node
    b = bytearray(code, "utf-8")
    spans: List[Tuple[int, int]] = []

    def walk(n):
        if n.type == "call_expression" and len(n.children) >= 2:
            callee = n.children[0]
            if callee.type == "identifier":
                spans.append((callee.start_byte, callee.end_byte))
            elif callee.type == "field_expression":
                ids = [c for c in callee.children if c.type == "identifier"]
                if ids:
                    last = ids[-1]
                    spans.append((last.start_byte, last.end_byte))
        for c in n.children:
            walk(c)

    walk(root)
    spans.sort(key=lambda x: x[0])
    mapping: Dict[str, str] = {}
    counter = 0
    for s, e in reversed(spans):
        ident = b[s:e].decode("utf-8", "ignore")
        if ident.lower() in _EXCLUDE:
            continue
        if ident not in mapping:
            counter += 1
            mapping[ident] = f"CALLEE_{counter}"
        b[s:e] = mapping[ident].encode("utf-8")
    return b.decode("utf-8", "ignore")


def callee_normalize(code: str) -> str:
    code = compact(code)
    if not code:
        return ""
    if _USE_TS:
        try:
            return callee_norm_treesitter(code)
        except Exception:
            return callee_norm_regex(code)
    return callee_norm_regex(code)


def is_rank0():
    return int(os.environ.get("RANK", 0)) == 0


def load_sp(path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def strip_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text.startswith(START_TAG):
        text = text[len(START_TAG) :]
    if text.endswith(END_TAG):
        text = text[: -len(END_TAG)]
    return text.strip()


def normalise(name: str, sp=None) -> str:
    name = strip_tags(name)
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    if not name:
        return ""
    parts = []
    for c in name:
        if c in {"_", ".", " ", "-"}:
            parts.append(" ")
        elif c.isdigit():
            parts.append(" ")
        elif c.isupper():
            parts.append(" ")
            parts.append(c.lower())
        else:
            parts.append(c)
    spaced = "".join(parts).strip()
    while "  " in spaced:
        spaced = spaced.replace("  ", " ")
    words = spaced.split(" ")
    words = [w for w in words if w]
    if not words:
        return ""
    if sp is not None:
        subwords = []
        for w in words:
            subwords.extend(sp.encode_as_pieces(w))
        words = [s for s in subwords if s]
    return " ".join(words)


def safe_name(value_or_entry):
    if isinstance(value_or_entry, dict):
        name = value_or_entry.get("original_function_name", "")
    else:
        name = value_or_entry
    if name is None:
        return ""
    return str(name).strip()


def wordcluster_pr_counts(gold_toks: list, pred_toks: list, word_cluster: dict):
    if not gold_toks and not pred_toks:
        return 0, 0, 0
    if not gold_toks:
        return 0, len(set(pred_toks)), 0
    if not pred_toks:
        return 0, 0, len(set(gold_toks))
    repl, skip = {}, set()
    for j, p in enumerate(pred_toks):
        if p in gold_toks:
            skip.add(j)
    for g in gold_toks:
        for j, p in enumerate(pred_toks):
            if j in skip or j in repl or p == g:
                continue
            g_cl = [k for k, syns in word_cluster.items() if g in syns or g == k]
            p_cl = [k for k, syns in word_cluster.items() if p in syns or p == k]
            if g_cl and p_cl and g_cl[0] == p_cl[0]:
                repl[j] = g
    pred_toks_modified = list(pred_toks)
    for j, v in repl.items():
        pred_toks_modified[j] = v
    gold_set = set(gold_toks)
    pred_set = set(pred_toks_modified)
    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set.difference(gold_set))
    fn = len(gold_set.difference(pred_set))
    return tp, fp, fn


def prf(tp, fp, fn):
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def token_metrics(refs: List[str], hyps: List[str]):
    tp = fp = fn = 0
    for g, p in zip(refs, hyps):
        g_stripped = strip_tags(g)
        p_stripped = strip_tags(p)
        g_toks = g_stripped.split()
        p_toks = p_stripped.split()
        g_set = set(g_toks)
        p_set = set(p_toks)
        tp += len(g_set.intersection(p_set))
        fp += len(p_set.difference(g_set))
        fn += len(g_set.difference(p_set))
    _, _, f1 = prf(tp, fp, fn)
    return f1


def token_prf(refs: List[str], hyps: List[str]):
    """Token-level (precision, recall, F1) on the same token-overlap basis as token_metrics."""
    tp = fp = fn = 0
    for g, p in zip(refs, hyps):
        g_set = set(strip_tags(g).split())
        p_set = set(strip_tags(p).split())
        tp += len(g_set.intersection(p_set))
        fp += len(p_set.difference(g_set))
        fn += len(g_set.difference(p_set))
    return prf(tp, fp, fn)


def build_preprocess(tok, spm_obj, desc_field: str):
    def preprocess(examples):
        cols = [
            "assembly_code",
            "decompiled_code_stripped",
            "S-Expression_of_decompiled_code_stripped",
            desc_field,
            "original_function_name",
        ]
        asm, dec, sexpr, desc, tgt = [examples[c] for c in cols]
        out = {f"input_ids{i}": [] for i in range(1, 5)}
        out.update({f"attention_mask{i}": [] for i in range(1, 5)})
        out["labels"] = []
        for a, d, s, de, t in zip(asm, dec, sexpr, desc, tgt):
            a, d, s, de, t = map(lambda x: str(x or "").strip(), [a, d, s, de, t])
            for idx, txt in enumerate([a, d, s, de], 1):
                prefix = ["<ASM>", "<DEC>", "<SEXP>", "<DESC>"][idx - 1] + " "
                enc = tok(prefix + txt, truncation=True, max_length=CHUNK_LEN)
                out[f"input_ids{idx}"].append(enc.input_ids)
                out[f"attention_mask{idx}"].append(enc.attention_mask)
            tagged_target = f"{START_TAG}{t}{END_TAG}"
            lab = tok(tagged_target, truncation=True, max_length=LABEL_LEN).input_ids
            out["labels"].append(lab)
        return out

    return preprocess


def symgen_style_dedup_in_memory(
    repo_id: str,
    raw_splits,
    min_body_len_tokens: int = 0,
    drop_empty: bool = False,
):
    """
    In-memory SymGen-style dedup for one repo:
      1) body_exact
      2) callee_norm
      3) asm_norm
      4) root_node
      5) funcname
    Keep-priority: test > validation > eval > train
    """
    order = ["test", "validation", "eval", "train"]
    dsdict = {sp: raw_splits[sp] for sp in order if sp in raw_splits}
    if not dsdict:
        return raw_splits, {}, {"total_before": 0, "total_after": 0}, {}

    print(f"[SymGenDedup] Repo {repo_id}: running SymGen-style dedup across splits {list(dsdict.keys())}")

    seen_body: Dict[str, Tuple[str, int]] = {}
    seen_call: Dict[str, Tuple[str, int]] = {}
    seen_asm: Dict[str, Tuple[str, int]] = {}
    seen_root: Dict[str, Tuple[str, int]] = {}
    seen_name: Dict[str, Tuple[str, int]] = {}

    keep_masks: Dict[str, List[bool]] = {sp: [True] * len(dsdict[sp]) for sp in dsdict}
    stats = {
        sp: {
            "total": len(dsdict[sp]),
            "removed_body_exact": 0,
            "removed_callee_norm": 0,
            "removed_asm_norm": 0,
            "removed_root_node": 0,
            "removed_funcname": 0,
            "removed_filtered": 0,
            "elapsed_s": 0.0,
        }
        for sp in dsdict
    }

    for sp in order:
        if sp not in dsdict:
            continue
        ds = dsdict[sp]
        t0 = time.time()
        for i in range(len(ds)):
            row = ds[i]
            body = compact(row.get(BODY_COL, ""))

            # guard
            if (drop_empty and not body) or (
                min_body_len_tokens > 0 and len(body.split()) < min_body_len_tokens
            ):
                keep_masks[sp][i] = False
                stats[sp]["removed_filtered"] += 1
                continue

            # (1) body_exact
            h_body = sha256_str(body)
            if h_body in seen_body:
                keep_masks[sp][i] = False
                stats[sp]["removed_body_exact"] += 1
                continue

            # (2) callee_norm
            h_call = sha256_str(callee_normalize(body))
            if h_call in seen_call:
                keep_masks[sp][i] = False
                stats[sp]["removed_callee_norm"] += 1
                continue

            # (3) asm_norm
            asm_norm = normalize_asm(row.get(ASM_COL, ""))
            if asm_norm:
                h_asm = sha256_str(asm_norm)
                if h_asm in seen_asm:
                    keep_masks[sp][i] = False
                    stats[sp]["removed_asm_norm"] += 1
                    continue

            # (4) root_node
            root_str = compact(row.get(ROOT_NODE_COL, ""))
            if root_str:
                h_root = sha256_str(root_str)
                if h_root in seen_root:
                    keep_masks[sp][i] = False
                    stats[sp]["removed_root_node"] += 1
                    continue

            # (5) funcname
            fname = compact(row.get(FUNCNAME_COL, ""))
            if fname:
                h_name = sha256_str(fname)
                if h_name in seen_name:
                    keep_masks[sp][i] = False
                    stats[sp]["removed_funcname"] += 1
                    continue

            # first time seeing fingerprints
            seen_body[h_body] = (sp, i)
            seen_call[h_call] = (sp, i)
            if asm_norm:
                seen_asm[sha256_str(asm_norm)] = (sp, i)
            if root_str:
                seen_root[sha256_str(root_str)] = (sp, i)
            if fname:
                seen_name[sha256_str(fname)] = (sp, i)

        stats[sp]["elapsed_s"] = time.time() - t0
        print(
            f"[SymGenDedup][{repo_id}::{sp}] total={stats[sp]['total']} "
            f"removed: body={stats[sp]['removed_body_exact']}, "
            f"callee={stats[sp]['removed_callee_norm']}, asm={stats[sp]['removed_asm_norm']}, "
            f"root={stats[sp]['removed_root_node']}, name={stats[sp]['removed_funcname']}, "
            f"filtered={stats[sp]['removed_filtered']} | elapsed={stats[sp]['elapsed_s']:.2f}s"
        )

    kept_splits: Dict[str, object] = {}
    total_before = total_after = 0
    for sp in dsdict:
        mask = keep_masks[sp]
        kept = dsdict[sp].filter(lambda _, idx: mask[idx], with_indices=True)
        kept_splits[sp] = kept
        total_before += stats[sp]["total"]
        total_after += len(kept)

    totals = {
        "total_before": total_before,
        "total_after": total_after,
        "total_removed": total_before - total_after,
    }
    counts = {
        "body_exact": sum(stats[sp]["removed_body_exact"] for sp in stats),
        "callee_norm": sum(stats[sp]["removed_callee_norm"] for sp in stats),
        "asm_norm": sum(stats[sp]["removed_asm_norm"] for sp in stats),
        "root_node": sum(stats[sp]["removed_root_node"] for sp in stats),
        "funcname": sum(stats[sp]["removed_funcname"] for sp in stats),
        "filtered": sum(stats[sp]["removed_filtered"] for sp in stats),
    }
    print(
        f"[SymGenDedup][{repo_id}] total_before={totals['total_before']}, "
        f"total_after={totals['total_after']}, total_removed={totals['total_removed']}"
    )
    print(
        f"[SymGenDedup][{repo_id}] stage counts: "
        f"body={counts['body_exact']}, callee={counts['callee_norm']}, "
        f"asm={counts['asm_norm']}, root={counts['root_node']}, "
        f"name={counts['funcname']}, filtered={counts['filtered']}"
    )

    return kept_splits, counts, totals, stats


def load_and_clean(
    tok,
    spm_obj,
    train_desc_field: str,
    eval_desc_field: str,
    dataset_repos: Optional[List[str]] = None,
    calculate_dynamic_label_len: bool = True,
):
    global LABEL_LEN, GLOBAL_DUP_NAMES

    if not dataset_repos:
        raise ValueError("This script is designed for multi-dataset training only.")

    print(f"[Data] Multi-dataset mode with SymGen-style dedup: {len(dataset_repos)} repos.")
    raw_multi = {}
    all_names = []
    total_train_raw = 0
    total_test_raw = 0

    # 1) Per-repo SymGen-style dedup
    for repo in dataset_repos:
        print(f"[Data] Loading repo: {repo}")
        raw = load_dataset(repo, cache_dir=CACHE_DIR)
        # SymGen dedup across splits
        kept_splits, counts, totals, _stats = symgen_style_dedup_in_memory(repo, raw)
        raw_multi[repo] = kept_splits

        train_size_repo = len(kept_splits["train"]) if "train" in kept_splits else 0
        test_size_repo = len(kept_splits["test"]) if "test" in kept_splits else 0
        total_train_raw += train_size_repo
        total_test_raw += test_size_repo
        print(
            f"[Data][{repo}] After SymGen-dedup -> train={train_size_repo}, test={test_size_repo}"
        )

        for split_name, ds_split in kept_splits.items():
            if split_name in ("train", "test"):
                all_names.extend(
                    [safe_name(n) for n in ds_split[FUNCNAME_COL]]
                )

    print(
        f"[Data] Combined raw (after SymGen) functions across repos -> "
        f"train: {total_train_raw}, test: {total_test_raw}"
    )

    # 2) Cross-repo duplicate function-name removal (global funcname dedup)
    dup_counter = collections.Counter(all_names)
    dup_names = {n for n, c in dup_counter.items() if c > 1}
    GLOBAL_DUP_NAMES = dup_names
    print(
        f"[Data] Global funcname dedup: found {len(dup_names)} duplicated function names across repos; filtering..."
    )

    def fn_filter(entry):
        return safe_name(entry) not in dup_names

    train_splits = []
    eval_splits = []

    for repo, kept_splits in raw_multi.items():
        if "train" in kept_splits:
            before = len(kept_splits["train"])
            filtered_train = kept_splits["train"].filter(fn_filter, num_proc=8)
            after = len(filtered_train)
            print(
                f"[Data][{repo}] train after global funcname-dedup {before} -> {after}"
            )
            train_splits.append(filtered_train)
        if "test" in kept_splits:
            before = len(kept_splits["test"])
            filtered_test = kept_splits["test"].filter(fn_filter, num_proc=8)
            after = len(filtered_test)
            print(
                f"[Data][{repo}] test after global funcname-dedup {before} -> {after}"
            )
            eval_splits.append(filtered_test)

    if not train_splits:
        raise ValueError("[Data] No train splits found after dedup.")

    train_raw = concatenate_datasets(train_splits)
    if eval_splits:
        eval_raw = concatenate_datasets(eval_splits)
    else:
        eval_raw = None
        print("[Data] Warning: no test splits found after dedup; eval_ds=None")

    print(
        f"[Data] Combined after cross-repo funcname-dedup -> "
        f"train: {len(train_raw)}, test: {len(eval_raw) if eval_raw is not None else 0}"
    )

    # 3) Dynamic label length
    if calculate_dynamic_label_len:
        print(
            "Calculating dynamic LABEL_LEN based on combined multi-dataset training data..."
        )
        sample_size = min(5000, len(train_raw))
        if sample_size > 0:
            sample_indices = random.sample(range(len(train_raw)), sample_size)
            train_sample = train_raw.select(sample_indices)

            def get_temp_labels_batched(examples):
                temp_labels_list = []
                for original_name in examples["original_function_name"]:
                    tagged_name = (
                        f"{START_TAG}{str(original_name or '').strip()}{END_TAG}"
                    )
                    tokenized = tok(tagged_name, truncation=False)
                    ids = tokenized.input_ids
                    temp_labels_list.append(ids)
                return {"temp_labels": temp_labels_list}

            processed_samples = train_sample.map(
                get_temp_labels_batched,
                batched=True,
                num_proc=4,
                remove_columns=train_sample.column_names,
            )
            processed_sample_labels = processed_samples["temp_labels"]
            max_label_len_tokens = (
                max(len(label_ids) for label_ids in processed_sample_labels)
                if processed_sample_labels
                else 0
            )
            print(
                f"  Max label length (tokens) found in sample: {max_label_len_tokens}"
            )
            LABEL_LEN = min(max_label_len_tokens, MAX_ALLOWED_LABEL_LEN)
            print(
                f"  Setting LABEL_LEN to: {LABEL_LEN} (capped at {MAX_ALLOWED_LABEL_LEN})"
            )
        else:
            print(
                "  Warning: Combined training dataset is empty. Using default LABEL_LEN."
            )
            LABEL_LEN = min(16, MAX_ALLOWED_LABEL_LEN)

    train_prep = build_preprocess(tok, spm_obj, train_desc_field)
    eval_prep = build_preprocess(tok, spm_obj, eval_desc_field)

    train_ds = train_raw.map(
        train_prep,
        batched=True,
        num_proc=8,
        remove_columns=train_raw.column_names,
    )

    if eval_raw is not None:
        eval_ds = eval_raw.map(
            eval_prep,
            batched=True,
            num_proc=8,
            remove_columns=eval_raw.column_names,
        )
    else:
        eval_ds = None

    print(
        f"[Data] Preprocessed multi-dataset sizes -> "
        f"train: {len(train_ds)}, eval: {len(eval_ds) if eval_ds is not None else 0}"
    )
    return train_ds, eval_ds


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        keys = [f"input_ids{i}" for i in range(1, 5)] + [
            f"attention_mask{i}" for i in range(1, 5)
        ] + ["labels"]
        out = {}
        for k in keys:
            pad_val = -100 if k == "labels" else self.pad_id
            L = max(len(x[k]) for x in feats)
            arr = [x[k] + [pad_val] * (L - len(x[k])) for x in feats]
            out[k] = torch.tensor(arr, dtype=torch.long)
        return out


# ===========================================================================
# Model 1: Concat fusion (from main_concat.py)
# ===========================================================================
class MultiViewCodeT5Concat(nn.Module):
    def __init__(self, tok):
        super().__init__()
        base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.encs = nn.ModuleList([base_model.get_encoder() for _ in range(4)])
        self.dec = base_model.get_decoder()
        self.lm_head = base_model.lm_head
        self.config = base_model.config
        self.length_penalty = LENGTH_PENALTY
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self._tie_weights()

        self.encoder_hidden_size = self.encs[0].config.hidden_size
        self.decoder_hidden_size = self.dec.config.hidden_size
        self.target_seq_len = LABEL_LEN

    def _tie_weights(self):
        for enc in self.encs:
            self.lm_head.weight = enc.embed_tokens.weight

    def resize_token_embeddings(self, new_num_tokens: int):
        for enc in self.encs:
            enc.resize_token_embeddings(new_num_tokens)
        self.dec.resize_token_embeddings(new_num_tokens)
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(
            old_lm_head.in_features,
            new_num_tokens,
            bias=old_lm_head.bias is not None,
        )
        if old_lm_head.weight.shape[0] < new_num_tokens:
            with torch.no_grad():
                new_lm_head.weight[: old_lm_head.weight.shape[0]] = old_lm_head.weight
                if old_lm_head.bias is not None:
                    new_lm_head.bias[: old_lm_head.bias.shape[0]] = old_lm_head.bias
        else:
            with torch.no_grad():
                new_lm_head.weight = nn.Parameter(old_lm_head.weight[:new_num_tokens])
                if old_lm_head.bias is not None:
                    new_lm_head.bias = nn.Parameter(old_lm_head.bias[:new_num_tokens])
        self.lm_head = new_lm_head
        self._tie_weights()

    def _encode(self, **kwargs):
        encoder_sequences = []
        encoder_attention_masks = []
        batch_size = None
        device = None
        for i in range(4):
            key_ids = f"input_ids{i+1}"
            key_mask = f"attention_mask{i+1}"
            if key_ids in kwargs:
                input_ids = kwargs[key_ids]
                attention_mask = kwargs.get(key_mask)
                if batch_size is None:
                    batch_size = input_ids.size(0)
                    device = input_ids.device
                encoder_output = self.encs[i](input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = encoder_output.last_hidden_state
                seq_len_actual = last_hidden_state.size(1)
                tokens_to_take = min(TOKENS_PER_ENCODER, seq_len_actual)
                sliced_sequence = last_hidden_state[:, :tokens_to_take, :]
                if tokens_to_take < TOKENS_PER_ENCODER:
                    padding_size = TOKENS_PER_ENCODER - tokens_to_take
                    padding = torch.zeros(
                        (batch_size, padding_size, self.encoder_hidden_size), device=device
                    )
                    sliced_sequence = torch.cat([sliced_sequence, padding], dim=1)
                    if attention_mask is not None:
                        sliced_mask = attention_mask[:, :tokens_to_take]
                        mask_padding = torch.zeros((batch_size, padding_size), dtype=torch.long, device=device)
                        sliced_mask = torch.cat([sliced_mask, mask_padding], dim=1)
                    else:
                        sliced_mask = torch.cat([
                            torch.ones((batch_size, tokens_to_take), dtype=torch.long, device=device),
                            torch.zeros((batch_size, padding_size), dtype=torch.long, device=device),
                        ], dim=1)
                else:
                    if attention_mask is not None:
                        sliced_mask = attention_mask[:, :TOKENS_PER_ENCODER]
                    else:
                        sliced_mask = torch.ones((batch_size, TOKENS_PER_ENCODER), dtype=torch.long, device=device)
                encoder_sequences.append(sliced_sequence)
                encoder_attention_masks.append(sliced_mask)

        if not encoder_sequences:
            raise ValueError("No encoder inputs provided")

        # Simply concatenate all encoder sequences along the sequence dimension
        concat_seq = torch.cat(encoder_sequences, dim=1)   # (B, 4*T, H)
        concat_mask = torch.cat(encoder_attention_masks, dim=1)  # (B, 4*T)

        if DEBUG_FUSION:
            print(f"[Concat] concat_seq shape: {concat_seq.shape}, active tokens: {concat_mask.sum(dim=1).float().mean().item():.1f}")

        aux_loss = torch.zeros(1, device=device)
        return concat_seq, concat_mask, aux_loss

    def forward(self, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        enc_kw = {
            k: v for k, v in kwargs.items()
            if k.startswith("input_ids") or k.startswith("attention_mask")
        }

        fused_output_sequence, fused_attention_mask, _ = self._encode(**enc_kw)

        dec_input_ids = kwargs.get("decoder_input_ids")
        dec_attn_mask = kwargs.get("decoder_attention_mask")
        if dec_input_ids is None and labels is not None:
            dec_input_ids = self.dec._shift_right(labels)

        dec_out = self.dec(
            input_ids=dec_input_ids,
            attention_mask=dec_attn_mask,
            encoder_hidden_states=fused_output_sequence,
            encoder_attention_mask=fused_attention_mask,
        )

        lm_logits = self.lm_head(dec_out.last_hidden_state)
        total_loss = torch.zeros((), device=lm_logits.device)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH, ignore_index=-100)
            ce_loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            total_loss = ce_loss

        return Seq2SeqLMOutput(loss=total_loss, logits=lm_logits)

    @torch.no_grad()
    def generate(self, **kwargs):
        enc_kw = {
            k: v for k, v in kwargs.items()
            if k.startswith("input_ids") or k.startswith("attention_mask")
        }
        fused_output_sequence, fused_attention_mask, _ = self._encode(**enc_kw)

        generation_config = GenerationConfig.from_model_config(self.config)
        generation_config.update(**{k: v for k, v in kwargs.items() if hasattr(generation_config, k)})
        if generation_config.max_length is None or generation_config.max_length <= 0:
            generation_config.max_length = LABEL_LEN
        if generation_config.decoder_start_token_id is None:
            generation_config.decoder_start_token_id = self.config.decoder_start_token_id
        if generation_config.decoder_start_token_id is None:
            generation_config.decoder_start_token_id = self.config.pad_token_id

        device = next(self.parameters()).device
        input_ids = torch.full(
            (fused_output_sequence.shape[0], 1),
            generation_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=device)

        for _ in range(generation_config.max_length - 1):
            decoder_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": fused_output_sequence,
                "encoder_attention_mask": fused_attention_mask,
            }
            decoder_outputs = self.dec(**decoder_inputs, return_dict=True)
            logits = self.lm_head(decoder_outputs[0])
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            next_tokens = next_tokens * unfinished_sequences + self.config.pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            eos_token_id = (
                generation_config.eos_token_id
                if generation_config.eos_token_id is not None
                else self.config.eos_token_id
            )
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break

        return input_ids

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        for idx, enc in enumerate(self.encs):
            torch.save(enc.state_dict(), os.path.join(save_dir, f"encoder_{idx}.bin"))
        torch.save(self.dec.state_dict(), os.path.join(save_dir, "decoder.bin"))
        torch.save(self.lm_head.state_dict(), os.path.join(save_dir, "lm_head.bin"))
        # No extra fusion layer to save for concatenation model
        self.config.save_pretrained(save_dir)


# ===========================================================================
# Model 2: Cross-attention fusion (from main_cross_attention.py)
# ===========================================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize FFN
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        encoder_sequences: List[torch.Tensor],
        encoder_masks: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Query: first encoder (ASM), shape (B, T, H)
        query = encoder_sequences[0]
        query_mask = encoder_masks[0]

        # Key/Value: all 4 encoders concatenated, shape (B, 4T, H)
        kv = torch.cat(encoder_sequences, dim=1)       # (B, 4T, H)
        kv_mask = torch.cat(encoder_masks, dim=1)      # (B, 4T)

        # key_padding_mask: True = position to IGNORE (padding)
        key_padding_mask = (kv_mask == 0)              # (B, 4T)

        attn_out, _ = self.cross_attn(
            query=query,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
        )

        # Residual + LayerNorm
        out = self.layer_norm_1(query + self.dropout(attn_out))

        # FFN + Residual + LayerNorm
        out = self.layer_norm_2(out + self.dropout(self.ffn(out)))

        if DEBUG_FUSION:
            print(f"[CrossAttn] output shape: {out.shape}, query_mask sum: {query_mask.sum()}")

        # Output mask: use the query (ASM encoder) mask
        return out, query_mask


class MultiViewCodeT5CrossAttn(nn.Module):
    def __init__(self, tok):
        super().__init__()
        base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.encs = nn.ModuleList([base_model.get_encoder() for _ in range(4)])
        self.dec = base_model.get_decoder()
        self.lm_head = base_model.lm_head
        self.config = base_model.config
        self.length_penalty = LENGTH_PENALTY
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self._tie_weights()

        self.encoder_hidden_size = self.encs[0].config.hidden_size
        self.decoder_hidden_size = self.dec.config.hidden_size
        self.target_seq_len = LABEL_LEN

        self.cross_attn_layer = CrossAttentionFusion(
            hidden_size=self.encoder_hidden_size,
            num_heads=8,
            dropout=0.1,
        )

    def _tie_weights(self):
        for enc in self.encs:
            self.lm_head.weight = enc.embed_tokens.weight

    def resize_token_embeddings(self, new_num_tokens: int):
        for enc in self.encs:
            enc.resize_token_embeddings(new_num_tokens)
        self.dec.resize_token_embeddings(new_num_tokens)
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(
            old_lm_head.in_features,
            new_num_tokens,
            bias=old_lm_head.bias is not None,
        )
        if old_lm_head.weight.shape[0] < new_num_tokens:
            with torch.no_grad():
                new_lm_head.weight[: old_lm_head.weight.shape[0]] = old_lm_head.weight
                if old_lm_head.bias is not None:
                    new_lm_head.bias[: old_lm_head.bias.shape[0]] = old_lm_head.bias
        else:
            with torch.no_grad():
                new_lm_head.weight = nn.Parameter(old_lm_head.weight[:new_num_tokens])
                if old_lm_head.bias is not None:
                    new_lm_head.bias = nn.Parameter(old_lm_head.bias[:new_num_tokens])
        self.lm_head = new_lm_head
        self._tie_weights()

    def _encode(self, **kwargs):
        encoder_sequences = []
        encoder_attention_masks = []
        batch_size = None
        device = None
        for i in range(4):
            key_ids = f"input_ids{i+1}"
            key_mask = f"attention_mask{i+1}"
            if key_ids in kwargs:
                input_ids = kwargs[key_ids]
                attention_mask = kwargs.get(key_mask)
                if batch_size is None:
                    batch_size = input_ids.size(0)
                    device = input_ids.device
                encoder_output = self.encs[i](input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = encoder_output.last_hidden_state
                seq_len_actual = last_hidden_state.size(1)
                tokens_to_take = min(TOKENS_PER_ENCODER, seq_len_actual)
                sliced_sequence = last_hidden_state[:, :tokens_to_take, :]
                if tokens_to_take < TOKENS_PER_ENCODER:
                    padding_size = TOKENS_PER_ENCODER - tokens_to_take
                    padding = torch.zeros(
                        (batch_size, padding_size, self.encoder_hidden_size), device=device
                    )
                    sliced_sequence = torch.cat([sliced_sequence, padding], dim=1)
                    if attention_mask is not None:
                        sliced_mask = attention_mask[:, :tokens_to_take]
                        mask_padding = torch.zeros((batch_size, padding_size), dtype=torch.long, device=device)
                        sliced_mask = torch.cat([sliced_mask, mask_padding], dim=1)
                    else:
                        sliced_mask = torch.cat([
                            torch.ones((batch_size, tokens_to_take), dtype=torch.long, device=device),
                            torch.zeros((batch_size, padding_size), dtype=torch.long, device=device),
                        ], dim=1)
                else:
                    if attention_mask is not None:
                        sliced_mask = attention_mask[:, :TOKENS_PER_ENCODER]
                    else:
                        sliced_mask = torch.ones((batch_size, TOKENS_PER_ENCODER), dtype=torch.long, device=device)
                encoder_sequences.append(sliced_sequence)
                encoder_attention_masks.append(sliced_mask)

        if not encoder_sequences:
            raise ValueError("No encoder inputs provided")

        # Apply cross-attention fusion
        fused_seq, fused_mask = self.cross_attn_layer(encoder_sequences, encoder_attention_masks)
        aux_loss = torch.zeros(1, device=device)

        if DEBUG_FUSION:
            print(f"[CrossAttn _encode] fused_seq shape: {fused_seq.shape}")

        return fused_seq, fused_mask, aux_loss

    def forward(self, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        enc_kw = {
            k: v for k, v in kwargs.items()
            if k.startswith("input_ids") or k.startswith("attention_mask")
        }

        fused_output_sequence, fused_attention_mask, _ = self._encode(**enc_kw)

        dec_input_ids = kwargs.get("decoder_input_ids")
        dec_attn_mask = kwargs.get("decoder_attention_mask")
        if dec_input_ids is None and labels is not None:
            dec_input_ids = self.dec._shift_right(labels)

        dec_out = self.dec(
            input_ids=dec_input_ids,
            attention_mask=dec_attn_mask,
            encoder_hidden_states=fused_output_sequence,
            encoder_attention_mask=fused_attention_mask,
        )

        lm_logits = self.lm_head(dec_out.last_hidden_state)
        total_loss = torch.zeros((), device=lm_logits.device)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH, ignore_index=-100)
            ce_loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            total_loss = ce_loss

        return Seq2SeqLMOutput(loss=total_loss, logits=lm_logits)

    @torch.no_grad()
    def generate(self, **kwargs):
        enc_kw = {
            k: v for k, v in kwargs.items()
            if k.startswith("input_ids") or k.startswith("attention_mask")
        }
        fused_output_sequence, fused_attention_mask, _ = self._encode(**enc_kw)

        generation_config = GenerationConfig.from_model_config(self.config)
        generation_config.update(**{k: v for k, v in kwargs.items() if hasattr(generation_config, k)})
        if generation_config.max_length is None or generation_config.max_length <= 0:
            generation_config.max_length = LABEL_LEN
        if generation_config.decoder_start_token_id is None:
            generation_config.decoder_start_token_id = self.config.decoder_start_token_id
        if generation_config.decoder_start_token_id is None:
            generation_config.decoder_start_token_id = self.config.pad_token_id

        device = next(self.parameters()).device
        input_ids = torch.full(
            (fused_output_sequence.shape[0], 1),
            generation_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=device)

        for _ in range(generation_config.max_length - 1):
            decoder_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": fused_output_sequence,
                "encoder_attention_mask": fused_attention_mask,
            }
            decoder_outputs = self.dec(**decoder_inputs, return_dict=True)
            logits = self.lm_head(decoder_outputs[0])
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            next_tokens = next_tokens * unfinished_sequences + self.config.pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            eos_token_id = (
                generation_config.eos_token_id
                if generation_config.eos_token_id is not None
                else self.config.eos_token_id
            )
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break

        return input_ids

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        for idx, enc in enumerate(self.encs):
            torch.save(enc.state_dict(), os.path.join(save_dir, f"encoder_{idx}.bin"))
        torch.save(self.dec.state_dict(), os.path.join(save_dir, "decoder.bin"))
        torch.save(self.lm_head.state_dict(), os.path.join(save_dir, "lm_head.bin"))
        torch.save(self.cross_attn_layer.state_dict(), os.path.join(save_dir, "cross_attn_layer.bin"))
        self.config.save_pretrained(save_dir)


# ===========================================================================
# Model 3: Simple gating fusion (from main_simple_gating.py)
# ===========================================================================
class SimpleGatingFusion(nn.Module):
    def __init__(self, hidden_size: int, num_encoders: int = 4):
        super().__init__()
        self.num_encoders = num_encoders
        self.hidden_size = hidden_size
        # Gate: maps from hidden_size to num_encoders (one weight per encoder, per position)
        self.gate = nn.Linear(hidden_size, num_encoders)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        encoder_sequences: List[torch.Tensor],
        encoder_masks: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_sequences: list of (B, T, H), all same T = TOKENS_PER_ENCODER
        stacked = torch.stack(encoder_sequences, dim=2)          # (B, T, num_enc, H)
        batch_size, seq_len, num_enc, hidden_size = stacked.shape

        # Compute gate input as mean of all encoder representations at each position
        mean_h = stacked.mean(dim=2)                              # (B, T, H)
        gate_logits = self.gate(mean_h)                           # (B, T, num_enc)
        gate_weights = torch.softmax(gate_logits, dim=-1)         # (B, T, num_enc)
        gate_weights = gate_weights.unsqueeze(-1)                 # (B, T, num_enc, 1)

        # Weighted sum across encoders
        fused = (gate_weights * stacked).sum(dim=2)               # (B, T, H)

        # Combined mask: a token position is valid if any encoder marks it as valid
        stacked_masks = torch.stack(encoder_masks, dim=-1)        # (B, T, num_enc)
        combined_mask = (stacked_masks.sum(dim=-1) > 0).long()    # (B, T)

        if DEBUG_FUSION:
            print(f"[SimpleGating] fused shape: {fused.shape}, gate_weights sample: {gate_weights[0, 0, :, 0].tolist()}")

        return fused, combined_mask


class MultiViewCodeT5SimpleGating(nn.Module):
    def __init__(self, tok):
        super().__init__()
        base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.encs = nn.ModuleList([base_model.get_encoder() for _ in range(4)])
        self.dec = base_model.get_decoder()
        self.lm_head = base_model.lm_head
        self.config = base_model.config
        self.length_penalty = LENGTH_PENALTY
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self._tie_weights()

        self.encoder_hidden_size = self.encs[0].config.hidden_size
        self.decoder_hidden_size = self.dec.config.hidden_size
        self.target_seq_len = LABEL_LEN

        self.gating_layer = SimpleGatingFusion(
            hidden_size=self.encoder_hidden_size,
            num_encoders=4,
        )

    def _tie_weights(self):
        for enc in self.encs:
            self.lm_head.weight = enc.embed_tokens.weight

    def resize_token_embeddings(self, new_num_tokens: int):
        for enc in self.encs:
            enc.resize_token_embeddings(new_num_tokens)
        self.dec.resize_token_embeddings(new_num_tokens)
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(
            old_lm_head.in_features,
            new_num_tokens,
            bias=old_lm_head.bias is not None,
        )
        if old_lm_head.weight.shape[0] < new_num_tokens:
            with torch.no_grad():
                new_lm_head.weight[: old_lm_head.weight.shape[0]] = old_lm_head.weight
                if old_lm_head.bias is not None:
                    new_lm_head.bias[: old_lm_head.bias.shape[0]] = old_lm_head.bias
        else:
            with torch.no_grad():
                new_lm_head.weight = nn.Parameter(old_lm_head.weight[:new_num_tokens])
                if old_lm_head.bias is not None:
                    new_lm_head.bias = nn.Parameter(old_lm_head.bias[:new_num_tokens])
        self.lm_head = new_lm_head
        self._tie_weights()

    def _encode(self, **kwargs):
        encoder_sequences = []
        encoder_attention_masks = []
        batch_size = None
        device = None
        for i in range(4):
            key_ids = f"input_ids{i+1}"
            key_mask = f"attention_mask{i+1}"
            if key_ids in kwargs:
                input_ids = kwargs[key_ids]
                attention_mask = kwargs.get(key_mask)
                if batch_size is None:
                    batch_size = input_ids.size(0)
                    device = input_ids.device
                encoder_output = self.encs[i](input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = encoder_output.last_hidden_state
                seq_len_actual = last_hidden_state.size(1)
                tokens_to_take = min(TOKENS_PER_ENCODER, seq_len_actual)
                sliced_sequence = last_hidden_state[:, :tokens_to_take, :]
                if tokens_to_take < TOKENS_PER_ENCODER:
                    padding_size = TOKENS_PER_ENCODER - tokens_to_take
                    padding = torch.zeros(
                        (batch_size, padding_size, self.encoder_hidden_size), device=device
                    )
                    sliced_sequence = torch.cat([sliced_sequence, padding], dim=1)
                    if attention_mask is not None:
                        sliced_mask = attention_mask[:, :tokens_to_take]
                        mask_padding = torch.zeros((batch_size, padding_size), dtype=torch.long, device=device)
                        sliced_mask = torch.cat([sliced_mask, mask_padding], dim=1)
                    else:
                        sliced_mask = torch.cat([
                            torch.ones((batch_size, tokens_to_take), dtype=torch.long, device=device),
                            torch.zeros((batch_size, padding_size), dtype=torch.long, device=device),
                        ], dim=1)
                else:
                    if attention_mask is not None:
                        sliced_mask = attention_mask[:, :TOKENS_PER_ENCODER]
                    else:
                        sliced_mask = torch.ones((batch_size, TOKENS_PER_ENCODER), dtype=torch.long, device=device)
                encoder_sequences.append(sliced_sequence)
                encoder_attention_masks.append(sliced_mask)

        if not encoder_sequences:
            raise ValueError("No encoder inputs provided")

        # Apply simple gating fusion
        fused_seq, fused_mask = self.gating_layer(encoder_sequences, encoder_attention_masks)
        aux_loss = torch.zeros(1, device=device)

        if DEBUG_FUSION:
            print(f"[SimpleGating _encode] fused_seq shape: {fused_seq.shape}")

        return fused_seq, fused_mask, aux_loss

    def forward(self, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        enc_kw = {
            k: v for k, v in kwargs.items()
            if k.startswith("input_ids") or k.startswith("attention_mask")
        }

        fused_output_sequence, fused_attention_mask, _ = self._encode(**enc_kw)

        dec_input_ids = kwargs.get("decoder_input_ids")
        dec_attn_mask = kwargs.get("decoder_attention_mask")
        if dec_input_ids is None and labels is not None:
            dec_input_ids = self.dec._shift_right(labels)

        dec_out = self.dec(
            input_ids=dec_input_ids,
            attention_mask=dec_attn_mask,
            encoder_hidden_states=fused_output_sequence,
            encoder_attention_mask=fused_attention_mask,
        )

        lm_logits = self.lm_head(dec_out.last_hidden_state)
        total_loss = torch.zeros((), device=lm_logits.device)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH, ignore_index=-100)
            ce_loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            total_loss = ce_loss

        return Seq2SeqLMOutput(loss=total_loss, logits=lm_logits)

    @torch.no_grad()
    def generate(self, **kwargs):
        enc_kw = {
            k: v for k, v in kwargs.items()
            if k.startswith("input_ids") or k.startswith("attention_mask")
        }
        fused_output_sequence, fused_attention_mask, _ = self._encode(**enc_kw)

        generation_config = GenerationConfig.from_model_config(self.config)
        generation_config.update(**{k: v for k, v in kwargs.items() if hasattr(generation_config, k)})
        if generation_config.max_length is None or generation_config.max_length <= 0:
            generation_config.max_length = LABEL_LEN
        if generation_config.decoder_start_token_id is None:
            generation_config.decoder_start_token_id = self.config.decoder_start_token_id
        if generation_config.decoder_start_token_id is None:
            generation_config.decoder_start_token_id = self.config.pad_token_id

        device = next(self.parameters()).device
        input_ids = torch.full(
            (fused_output_sequence.shape[0], 1),
            generation_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=device)

        for _ in range(generation_config.max_length - 1):
            decoder_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": fused_output_sequence,
                "encoder_attention_mask": fused_attention_mask,
            }
            decoder_outputs = self.dec(**decoder_inputs, return_dict=True)
            logits = self.lm_head(decoder_outputs[0])
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            next_tokens = next_tokens * unfinished_sequences + self.config.pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            eos_token_id = (
                generation_config.eos_token_id
                if generation_config.eos_token_id is not None
                else self.config.eos_token_id
            )
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break

        return input_ids

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        for idx, enc in enumerate(self.encs):
            torch.save(enc.state_dict(), os.path.join(save_dir, f"encoder_{idx}.bin"))
        torch.save(self.dec.state_dict(), os.path.join(save_dir, "decoder.bin"))
        torch.save(self.lm_head.state_dict(), os.path.join(save_dir, "lm_head.bin"))
        torch.save(self.gating_layer.state_dict(), os.path.join(save_dir, "gating_layer.bin"))
        self.config.save_pretrained(save_dir)


# ===========================================================================
# Model 4: Mixture-of-Experts fusion (from main.py)
# ===========================================================================
class ConcatAndLinearFusion(nn.Module):
    def __init__(
        self,
        encoder_hidden_size,
        target_hidden_size,
        target_seq_len,
        use_mlp=False,
        mlp_hidden_factor=4,
    ):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.target_hidden_size = target_hidden_size
        self.target_seq_len = target_seq_len
        self.use_mlp = use_mlp
        input_dim = 4 * encoder_hidden_size
        if self.use_mlp:
            mlp_hidden_dim = int(input_dim * mlp_hidden_factor)
            self.fusion_layer = nn.Sequential(
                nn.Linear(input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, target_hidden_size),
            )
        else:
            self.fusion_layer = nn.Linear(input_dim, target_hidden_size)
        self.to_decoder_proj = nn.Linear(
            target_hidden_size, target_seq_len * target_hidden_size
        )
        self._init_weights()

    def _init_weights(self):
        layers_to_init = [self.to_decoder_proj]
        if self.use_mlp:
            layers_to_init.extend([self.fusion_layer[0], self.fusion_layer[2]])
        else:
            layers_to_init.append(self.fusion_layer)
        for layer in layers_to_init:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, pooled_encodings_list):
        batch_size = pooled_encodings_list[0].size(0)
        device = pooled_encodings_list[0].device
        concatenated_pooled = torch.cat(pooled_encodings_list, dim=1)
        fused_output = self.fusion_layer(concatenated_pooled)
        projected = self.to_decoder_proj(fused_output)
        output_sequence = projected.view(
            batch_size, self.target_seq_len, self.target_hidden_size
        )
        attention_mask = torch.ones(
            (batch_size, self.target_seq_len), dtype=torch.long, device=device
        )
        return output_sequence, attention_mask


class MoELayerForSequences(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        num_selected_experts: int,
        activation=nn.ReLU,
    ):
        super(MoELayerForSequences, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts

        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(num_experts)
            ]
        )

        nn.init.xavier_uniform_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)

        gate_logits = self.gate(x_flat)
        topk_logits, topk_indices = torch.topk(
            gate_logits, self.num_selected_experts, dim=1
        )
        gate_weights = torch.softmax(topk_logits, dim=1)

        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x_flat)
            expert_outputs.append(expert_out)
        stacked_expert_outputs = torch.stack(expert_outputs, dim=1)

        expanded_gate_weights = gate_weights.unsqueeze(-1)
        selected_expert_outputs = torch.gather(
            stacked_expert_outputs,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.output_dim),
        )
        weighted_selected_outputs = expanded_gate_weights * selected_expert_outputs
        output_flat = torch.sum(weighted_selected_outputs, dim=1)

        output = output_flat.view(batch_size, seq_len, self.output_dim)

        gate_weights_softmax = torch.softmax(gate_logits, dim=1)
        importance = torch.mean(gate_weights_softmax, dim=0)
        load = importance
        uniform_load = 1.0 / self.num_experts
        moe_loss = torch.sum((load - uniform_load) ** 2) * self.num_experts
        return output, moe_loss


class MultiViewCodeT5MoE(nn.Module):
    def __init__(self, tok):
        super().__init__()
        base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.encs = nn.ModuleList([base_model.get_encoder() for _ in range(4)])
        self.dec = base_model.get_decoder()
        self.lm_head = base_model.lm_head
        self.config = base_model.config
        self.length_penalty = LENGTH_PENALTY
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self._tie_weights()

        self.encoder_hidden_size = self.encs[0].config.hidden_size
        self.decoder_hidden_size = self.dec.config.hidden_size
        self.target_seq_len = LABEL_LEN

        if USE_CONCAT_LINEAR_FUSION:
            self.fusion_hidden_size = self.decoder_hidden_size
            USE_MLP_FUSION = False
            self.fusion_layer = ConcatAndLinearFusion(
                encoder_hidden_size=self.encoder_hidden_size,
                target_hidden_size=self.fusion_hidden_size,
                target_seq_len=self.target_seq_len,
                use_mlp=USE_MLP_FUSION,
            )
        else:
            moe_input_dim = self.encoder_hidden_size
            moe_hidden_dim = self.encoder_hidden_size
            moe_output_dim = self.decoder_hidden_size

            self.moe_layer = MoELayerForSequences(
                input_dim=moe_input_dim,
                hidden_dim=moe_hidden_dim,
                output_dim=moe_output_dim,
                num_experts=NUM_EXPERTS,
                num_selected_experts=NUM_SELECTED_EXPERTS,
            )

    def _tie_weights(self):
        for enc in self.encs:
            self.lm_head.weight = enc.embed_tokens.weight

    def resize_token_embeddings(self, new_num_tokens: int):
        for enc in self.encs:
            enc.resize_token_embeddings(new_num_tokens)
        self.dec.resize_token_embeddings(new_num_tokens)
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(
            old_lm_head.in_features,
            new_num_tokens,
            bias=old_lm_head.bias is not None,
        )
        if old_lm_head.weight.shape[0] < new_num_tokens:
            with torch.no_grad():
                new_lm_head.weight[: old_lm_head.weight.shape[0]] = old_lm_head.weight
                if old_lm_head.bias is not None:
                    new_lm_head.bias[: old_lm_head.bias.shape[0]] = old_lm_head.bias
        else:
            with torch.no_grad():
                new_lm_head.weight = nn.Parameter(
                    old_lm_head.weight[:new_num_tokens]
                )
                if old_lm_head.bias is not None:
                    new_lm_head.bias = nn.Parameter(
                        old_lm_head.bias[:new_num_tokens]
                    )
        self.lm_head = new_lm_head
        self._tie_weights()

    def _encode(self, **kwargs):
        encoder_sequences = []
        encoder_attention_masks = []
        batch_size = None
        device = None
        for i in range(4):
            key_ids = f"input_ids{i+1}"
            key_mask = f"attention_mask{i+1}"
            if key_ids in kwargs:
                input_ids = kwargs[key_ids]
                attention_mask = kwargs.get(key_mask)
                if batch_size is None:
                    batch_size = input_ids.size(0)
                    device = input_ids.device
                encoder_output = self.encs[i](
                    input_ids=input_ids, attention_mask=attention_mask
                )
                last_hidden_state = encoder_output.last_hidden_state
                seq_len_actual = last_hidden_state.size(1)
                tokens_to_take = min(TOKENS_PER_ENCODER, seq_len_actual)
                sliced_sequence = last_hidden_state[:, :tokens_to_take, :]
                if tokens_to_take < TOKENS_PER_ENCODER:
                    padding_size = TOKENS_PER_ENCODER - tokens_to_take
                    padding = torch.zeros(
                        (batch_size, padding_size, self.encoder_hidden_size),
                        device=device,
                    )
                    sliced_sequence = torch.cat([sliced_sequence, padding], dim=1)
                    if attention_mask is not None:
                        sliced_mask = attention_mask[:, :tokens_to_take]
                        mask_padding = torch.zeros(
                            (batch_size, padding_size),
                            dtype=torch.long,
                            device=device,
                        )
                        sliced_mask = torch.cat([sliced_mask, mask_padding], dim=1)
                    else:
                        sliced_mask = torch.cat(
                            [
                                torch.ones(
                                    (batch_size, tokens_to_take),
                                    dtype=torch.long,
                                    device=device,
                                ),
                                torch.zeros(
                                    (batch_size, padding_size),
                                    dtype=torch.long,
                                    device=device,
                                ),
                            ],
                            dim=1,
                        )
                else:
                    if attention_mask is not None:
                        sliced_mask = attention_mask[:, :TOKENS_PER_ENCODER]
                    else:
                        sliced_mask = torch.ones(
                            (batch_size, TOKENS_PER_ENCODER),
                            dtype=torch.long,
                            device=device,
                        )
                encoder_sequences.append(sliced_sequence)
                encoder_attention_masks.append(sliced_mask)

        if not encoder_sequences:
            raise ValueError("No encoder inputs provided")

        if USE_CONCAT_LINEAR_FUSION:
            raise NotImplementedError(
                "ConcatAndLinearFusion not adapted for sequence inputs."
            )
        else:
            concatenated_sequences = torch.cat(encoder_sequences, dim=1)
            concatenated_masks = torch.cat(encoder_attention_masks, dim=1)

            if DEBUG_FUSION:
                print(
                    f"[MoE Fusion Debug] Concatenated sequences shape: {concatenated_sequences.shape}"
                )

            moe_output_sequence, moe_loss = self.moe_layer(concatenated_sequences)

            if DEBUG_FUSION:
                print(
                    f"[MoE Fusion Debug] MoE output sequence shape: {moe_output_sequence.shape}, MoE Loss: {moe_loss.item()}"
                )

            moe_attention_mask = concatenated_masks

            return moe_output_sequence, moe_attention_mask, moe_loss

    def forward(self, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        enc_kw = {
            k: v
            for k, v in kwargs.items()
            if k.startswith("input_ids") or k.startswith("attention_mask")
        }

        fused_output_sequence, fused_attention_mask, moe_loss_component = self._encode(
            **enc_kw
        )

        dec_input_ids = kwargs.get("decoder_input_ids")
        dec_attn_mask = kwargs.get("decoder_attention_mask")
        if dec_input_ids is None and labels is not None:
            dec_input_ids = self.dec._shift_right(labels)

        dec_out = self.dec(
            input_ids=dec_input_ids,
            attention_mask=dec_attn_mask,
            encoder_hidden_states=fused_output_sequence,
            encoder_attention_mask=fused_attention_mask,
        )

        lm_logits = self.lm_head(dec_out.last_hidden_state)
        ce_loss = None
        total_loss = torch.zeros((), device=lm_logits.device)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                label_smoothing=LABEL_SMOOTH, ignore_index=-100
            )
            ce_loss = loss_fn(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )
            total_loss = ce_loss if ce_loss is not None else total_loss
            if self.training and not USE_CONCAT_LINEAR_FUSION:
                total_loss = total_loss + MOE_LOSS_WEIGHT * moe_loss_component

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=lm_logits,
        )

    @torch.no_grad()
    def generate(self, **kwargs):
        enc_kw = {
            k: v
            for k, v in kwargs.items()
            if k.startswith("input_ids") or k.startswith("attention_mask")
        }
        fused_output_sequence, fused_attention_mask, _ = self._encode(**enc_kw)

        generation_config = GenerationConfig.from_model_config(self.config)
        generation_config.update(
            **{k: v for k, v in kwargs.items() if hasattr(generation_config, k)}
        )
        if generation_config.max_length is None or generation_config.max_length <= 0:
            generation_config.max_length = LABEL_LEN
        if generation_config.decoder_start_token_id is None:
            generation_config.decoder_start_token_id = (
                self.config.decoder_start_token_id
            )
        if generation_config.decoder_start_token_id is None:
            generation_config.decoder_start_token_id = self.config.pad_token_id

        encoder_outputs = BaseModelOutput(last_hidden_state=fused_output_sequence)
        device = next(self.parameters()).device

        input_ids = torch.full(
            (fused_output_sequence.shape[0], 1),
            generation_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=device
        )

        for _ in range(generation_config.max_length - 1):
            decoder_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_outputs.last_hidden_state,
                "encoder_attention_mask": fused_attention_mask,
            }
            decoder_outputs = self.dec(**decoder_inputs, return_dict=True)
            sequence_output = decoder_outputs[0]
            logits = self.lm_head(sequence_output)
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            next_tokens = next_tokens * unfinished_sequences + self.config.pad_token_id * (
                1 - unfinished_sequences
            )
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            eos_token_id = (
                generation_config.eos_token_id
                if generation_config.eos_token_id is not None
                else self.config.eos_token_id
            )
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )
            if unfinished_sequences.max() == 0:
                break

        return input_ids

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        # Full state dict (all parameters)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        # Encoders individually
        for idx, enc in enumerate(self.encs):
            torch.save(
                enc.state_dict(), os.path.join(save_dir, f"encoder_{idx}.bin")
            )
        # Decoder, lm_head, and MoE layer (if present)
        torch.save(self.dec.state_dict(), os.path.join(save_dir, "decoder.bin"))
        torch.save(self.lm_head.state_dict(), os.path.join(save_dir, "lm_head.bin"))
        if hasattr(self, "moe_layer"):
            torch.save(
                self.moe_layer.state_dict(),
                os.path.join(save_dir, "moe_layer.bin"),
            )
        # Config
        self.config.save_pretrained(save_dir)


# ===========================================================================
# Model factory
# ===========================================================================
FUSION_CHOICES = ["concat", "cross_attention", "simple_gating", "moe"]


def build_model(fusion: str, tok):
    """Return the multi-view CodeT5 model for the requested fusion strategy."""
    if fusion == "concat":
        return MultiViewCodeT5Concat(tok)
    if fusion == "cross_attention":
        return MultiViewCodeT5CrossAttn(tok)
    if fusion == "simple_gating":
        return MultiViewCodeT5SimpleGating(tok)
    if fusion == "moe":
        return MultiViewCodeT5MoE(tok)
    raise ValueError(f"Unknown fusion type: {fusion!r}. Choose from {FUSION_CHOICES}.")


# ===========================================================================
# Unified trainer
#
# The MoE load-balance loss (paper's L_bal) is produced inside
# MoELayerForSequences.forward as `moe_loss = sum((load - 1/E)^2) * E`, and the
# MoE model's own forward() adds it to the cross-entropy as
#   total_loss = ce_loss + MOE_LOSS_WEIGHT * moe_loss_component
# but ONLY when `self.training and not USE_CONCAT_LINEAR_FUSION`. The other three
# fusion models return CE-only loss from their forward(). Therefore the trainer
# simply reads `outputs.loss`, and the aux term is intrinsically applied only for
# the moe model. We additionally pass `fusion` so the debug logging label is
# correct and, defensively, so the aux path is gated to moe-only at the trainer
# level too.
# ===========================================================================
class MyMoETrainer(Seq2SeqTrainer):
    def __init__(self, *args, pad_token_id=None, fusion="moe", **kwargs):
        super().__init__(*args, **kwargs)
        self._pad_id = pad_token_id if pad_token_id is not None else self.processing_class.pad_token_id
        self.fusion = fusion

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        # outputs.loss already includes the MoE load-balance term (only for the
        # moe model, only during training). For the other fusions it is CE-only.
        total_loss = outputs.loss

        if DEBUG_LOSS and self.state.global_step % LOG_STEPS == 0:
            loss_label = "CE + MoE" if self.fusion == "moe" else "CE"
            print(f"[Loss Debug] Step {self.state.global_step}:")
            print(
                f"  - Labels Shape: {labels.shape}, Num Valid Tokens: {(labels != -100).sum().item()}"
            )
            print(
                f"  - Logits Shape: {logits.shape}, Logits Norm: {torch.norm(logits).item():.4f}"
            )
            total_loss_for_logging = (
                total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            )
            print(
                f"  - Total Combined Loss ({loss_label}): {total_loss_for_logging:.6f}"
            )

            try:
                pred_ids = torch.argmax(logits, dim=-1)
                if is_rank0():
                    num_examples_to_log = min(2, labels.size(0))
                    for ex_idx in range(num_examples_to_log):
                        pred_ids_ex = pred_ids[ex_idx]
                        label_ids_ex = labels[ex_idx]
                        valid_label_mask = label_ids_ex != -100
                        valid_label_ids = label_ids_ex.masked_select(
                            valid_label_mask
                        )

                        if self.processing_class is not None and hasattr(
                            self.processing_class, "decode"
                        ):
                            pred_text = self.processing_class.decode(
                                pred_ids_ex, skip_special_tokens=True
                            )
                            label_text = self.processing_class.decode(
                                valid_label_ids, skip_special_tokens=True
                            )
                            print(f"  - Example {ex_idx} Pred:  '{pred_text}'")
                            print(f"  - Example {ex_idx} Label: '{label_text}'")
                        else:
                            print(
                                f"  - Example {ex_idx}: tokenizer missing, cannot decode."
                            )

            except Exception as e:
                print(f"[Loss Debug] Error decoding example: {e}")

        return (total_loss, outputs) if return_outputs else total_loss


class MyCallback:
    def __init__(
        self,
        eval_ds,
        collator,
        tok,
        spm_obj,
        word_cluster,
        subset=EVAL_SUBSET_CB,
        batch_size=CB_BATCH,
    ):
        self.eval_ds = eval_ds
        self.collator = collator
        self.tok = tok
        self.spm_obj = spm_obj
        self.word_cluster = word_cluster
        self.subset = subset
        self.batch_size = batch_size

    def on_init_end(self, args, state, control, **kwargs):
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        if is_rank0():
            print("[MyCallback] Training is starting.")
            print(
                f"[MyCallback] Eval dataset size (functions): {len(self.eval_ds) if self.eval_ds is not None else 0}"
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if is_rank0():
            print("[MyCallback] Training has ended.")
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        return control

    def on_save(self, args, state, control, **kwargs):
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control

    def on_epoch_end(self, args, state, control, model, **kwargs):
        if not is_rank0():
            return control
        epoch = int(state.epoch) if state.epoch is not None else 0
        if epoch % 5 != 0:  # diagnostic eval only every 5 epochs (final metrics come from post-training inference)
            return control
        eval_size = len(self.eval_ds) if self.eval_ds is not None else 0
        print(f"[Callback] Epoch {epoch} — running evaluation callback on subset.")
        print(
            f"[Callback] Eval dataset size (functions): {eval_size}, subset size: {min(self.subset, eval_size)}"
        )

        if eval_size == 0:
            return control

        idxs = random.sample(
            range(len(self.eval_ds)), min(self.subset, len(self.eval_ds))
        )
        subset = self.eval_ds.select(idxs)
        dl = DataLoader(subset, batch_size=self.batch_size, collate_fn=self.collator)
        preds_all, refs_all = [], []
        with torch.no_grad():
            for batch in dl:
                device = next(model.parameters()).device
                enc_batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k.startswith("input_ids") or k.startswith("attention_mask")
                }
                pred_ids = model.generate(**enc_batch)
                preds = self.tok.batch_decode(pred_ids, skip_special_tokens=True)
                labels_np = np.where(
                    batch["labels"].cpu().numpy() != -100,
                    batch["labels"].cpu().numpy(),
                    self.tok.pad_token_id,
                )
                refs = self.tok.batch_decode(labels_np, skip_special_tokens=True)
                preds_all.extend(preds)
                refs_all.extend(refs)
        pred_norm = [normalise(p, self.spm_obj) for p in preds_all]
        ref_norm = [normalise(r, self.spm_obj) for r in refs_all]
        exact = float(np.mean([p == g for p, g in zip(pred_norm, ref_norm)]))
        tok_f = token_metrics(refs_all, preds_all)
        tp_cw = fp_cw = fn_cw = 0
        for g_norm, p_norm in zip(ref_norm, pred_norm):
            g_toks = g_norm.split()
            p_toks = p_norm.split()
            tpc, fpc, fnc = wordcluster_pr_counts(
                g_toks, p_toks, self.word_cluster
            )
            tp_cw += tpc
            fp_cw += fpc
            fn_cw += fnc
        _, _, cw_f = prf(tp_cw, fp_cw, fn_cw)
        print(
            f"[Callback] Epoch {epoch} metrics on subset -> "
            f"exact={exact:.3f}, token_F1={tok_f:.3f}, CWordNet_F1={cw_f:.3f}"
        )
        df = pd.DataFrame(
            {
                "ground_truth": refs_all,
                "prediction": preds_all,
                "gt_norm": ref_norm,
                "pred_norm": pred_norm,
            }
        )
        df.to_csv(
            f"all_fixes_Fungen_{epoch:03d}_gt_pred.tsv", sep="\t", index=False
        )
        print(
            f"[Callback] Saved epoch {epoch} predictions to all_fixes_symgen_{epoch:03d}_gt_pred.tsv"
        )
        return control


def build_compute_metrics(tok, spm_obj, word_cluster, emb_matrix=None):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        vocab_size = len(tok)
        filtered_preds = []
        for pred_seq in preds:
            filtered_seq = [
                token_id for token_id in pred_seq if 0 <= token_id < vocab_size
            ]
            filtered_preds.append(filtered_seq)
        preds_text = tok.batch_decode(filtered_preds, skip_special_tokens=True)
        labels_np = np.where(labels != -100, labels, tok.pad_token_id)
        refs_text = tok.batch_decode(labels_np, skip_special_tokens=True)
        pred_norm = [normalise(p, spm_obj) for p in preds_text]
        ref_norm = [normalise(r, spm_obj) for r in refs_text]
        exact = float(np.mean([p == g for p, g in zip(pred_norm, ref_norm)]))
        tok_f = token_metrics(refs_text, preds_text)
        tp_cw = fp_cw = fn_cw = 0
        for g_norm, p_norm in zip(ref_norm, pred_norm):
            g_toks = g_norm.split()
            p_toks = p_norm.split()
            tpc, fpc, fnc = wordcluster_pr_counts(g_toks, p_toks, word_cluster)
            tp_cw += tpc
            fp_cw += fpc
            fn_cw += fnc
        _, _, cw_f = prf(tp_cw, fp_cw, fn_cw)
        cos = 0.0
        metrics = {
            "token_f1": tok_f,
            "cwordnet_f1": cw_f,
            "exact": exact,
            "cosine": cos,
        }
        print(
            f"[Metrics] Eval metrics (full eval set) -> "
            f"exact={exact:.3f}, token_F1={tok_f:.3f}, CWordNet_F1={cw_f:.3f}"
        )
        return metrics

    return compute_metrics


def check_required_columns_for_split(ds_split, repo: str, split_name: str, desc_field: str) -> bool:
    """
    Check that all required columns for preprocessing exist in the given split.
    If any are missing, print a helpful message (including HF link and available columns)
    and return False. Otherwise return True.
    """
    required = set(REQUIRED_BASE_COLS + [desc_field])
    available = set(ds_split.column_names)
    missing = sorted(list(required - available))

    if not missing:
        return True

    print(f"[ColumnCheck][{repo}::{split_name}] Missing required columns: {missing}")
    print(f"[ColumnCheck] HuggingFace dataset page: https://huggingface.co/datasets/{repo}")
    print(f"[ColumnCheck] Available columns in this split: {sorted(available)}")
    print(
        "[ColumnCheck] This split will be skipped for preprocessing/inference. "
        "Consider fixing the dataset or using a different --eval_desc_field."
    )
    return False


def run_inference_on_all_datasets(
    model: nn.Module,
    tok,
    spm_obj,
    word_cluster: dict,
    dataset_repos: List[str],
    eval_desc_field: str,
    output_dir: str,
    batch_size: int = CB_BATCH,
):
    """
    For each dataset repo:
      - load 'test' split
      - apply global funcname dedup (GLOBAL_DUP_NAMES)
      - check required columns
      - preprocess
      - run model.generate()
      - compute metrics
      - save per-sample GT + prediction
      - record inference timing in blocks of 100 samples
    """
    device = next(model.parameters()).device
    model.eval()

    print(f"[Inference] Running inference on {len(dataset_repos)} dataset(s).")
    for repo in dataset_repos:
        print(f"[Inference] Dataset: {repo}")
        raw = load_dataset(repo, cache_dir=CACHE_DIR)

        if "test" not in raw:
            print("  - No 'test' split found; skipping.")
            continue

        ds_test = raw["test"]
        orig_test_size = len(ds_test)
        print(f"  - Original test functions: {orig_test_size}")

        global GLOBAL_DUP_NAMES
        if GLOBAL_DUP_NAMES:
            num_before = len(ds_test)
            ds_test = ds_test.filter(
                lambda e: safe_name(e) not in GLOBAL_DUP_NAMES,
                num_proc=8,
            )
            print(
                f"  - After global funcname filter: {num_before} -> {len(ds_test)} test functions"
            )

        # Check required columns before preprocessing
        if not check_required_columns_for_split(
            ds_test, repo=repo, split_name="test", desc_field=eval_desc_field
        ):
            # Skip this dataset if columns are missing
            continue

        eval_prep = build_preprocess(tok, spm_obj, eval_desc_field)
        test_ds = ds_test.map(
            eval_prep,
            batched=True,
            num_proc=8,
            remove_columns=ds_test.column_names,
        )

        print(
            f"  - Preprocessed test dataset size (functions): {len(test_ds)}"
        )

        collator = PadCollator(tok.pad_token_id)
        dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=collator)

        preds_all, refs_all = [], []

        timings_per_sample: List[float] = []
        timing_blocks: List[dict] = []
        sample_idx = 0

        with torch.no_grad():
            for batch in dl:
                enc_batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k.startswith("input_ids") or k.startswith("attention_mask")
                }
                t_start = time.time()
                pred_ids = model.generate(**enc_batch)
                t_end = time.time()
                batch_time = t_end - t_start
                batch_size_actual = pred_ids.size(0)
                time_per_sample = batch_time / max(batch_size_actual, 1)

                preds = tok.batch_decode(pred_ids, skip_special_tokens=True)
                labels_np = np.where(
                    batch["labels"].cpu().numpy() != -100,
                    batch["labels"].cpu().numpy(),
                    tok.pad_token_id,
                )
                refs = tok.batch_decode(labels_np, skip_special_tokens=True)

                preds_all.extend(preds)
                refs_all.extend(refs)

                for _ in range(batch_size_actual):
                    timings_per_sample.append(time_per_sample)
                    sample_idx += 1
                    if sample_idx % 100 == 0:
                        block = timings_per_sample[-100:]
                        avg_block = float(sum(block) / len(block))
                        total_block = float(sum(block))
                        block_index = sample_idx // 100
                        timing_blocks.append(
                            {
                                "dataset": repo,
                                "block_index": block_index,
                                "start_sample": sample_idx - 99,
                                "end_sample": sample_idx,
                                "num_samples": len(block),
                                "avg_time_s": avg_block,
                                "total_time_s": total_block,
                            }
                        )

        pred_norm = [normalise(p, spm_obj) for p in preds_all]
        ref_norm = [normalise(r, spm_obj) for r in refs_all]
        exact = float(np.mean([p == g for p, g in zip(pred_norm, ref_norm)]))
        tok_p, tok_r, tok_f = token_prf(refs_all, preds_all)

        tp_cw = fp_cw = fn_cw = 0
        for g_norm, p_norm in zip(ref_norm, pred_norm):
            g_toks = g_norm.split()
            p_toks = p_norm.split()
            tpc, fpc, fnc = wordcluster_pr_counts(g_toks, p_toks, word_cluster)
            tp_cw += tpc
            fp_cw += fpc
            fn_cw += fnc
        _, _, cw_f = prf(tp_cw, fp_cw, fn_cw)

        print(
            f"  - Test metrics (full test set) -> "
            f"exact={exact:.3f}, token_P={tok_p:.3f}, token_R={tok_r:.3f}, "
            f"token_F1={tok_f:.3f}, CWordNet_F1={cw_f:.3f}, "
            f"num_samples={len(refs_all)}"
        )

        dataset_id = repo.split("/")[-1]
        os.makedirs(output_dir, exist_ok=True)
        try:
            with open(os.path.join(output_dir, f"metrics_{dataset_id}.json"), "w") as _mf:
                json.dump({
                    "dataset": repo,
                    "num_samples": len(refs_all),
                    "exact": exact,
                    "token_precision": tok_p,
                    "token_recall": tok_r,
                    "token_f1": tok_f,
                    "cwordnet_f1": cw_f,
                }, _mf, indent=2)
        except Exception as _e:
            print(f"  - [warn] could not write metrics json: {_e}")
        out_path = os.path.join(output_dir, f"generic_inference_{dataset_id}.tsv")

        df = pd.DataFrame(
            {
                "ground_truth": refs_all,
                "prediction": preds_all,
                "gt_norm": ref_norm,
                "pred_norm": pred_norm,
            }
        )
        df.to_csv(out_path, sep="\t", index=False)
        print(f"  - Saved per-sample predictions to {out_path}")

        # Timing CSV (per 100-sample block)
        timing_path = os.path.join(
            output_dir, f"generic_inference_{dataset_id}_timing_blocks.csv"
        )
        if timing_blocks:
            df_t = pd.DataFrame(timing_blocks)
            df_t.to_csv(timing_path, index=False)
            print(
                f"  - Saved timing blocks (per 100 samples) to {timing_path}"
            )
        else:
            print("  - No timing blocks to save (less than 100 samples total).")


def main():
    global NUM_EXPERTS, NUM_SELECTED_EXPERTS, MOE_LOSS_WEIGHT
    global SP_MODEL_PATH, WORD_CLUSTER_PATH

    import argparse

    parser = argparse.ArgumentParser(
        description="Unified multi-view CodeT5 training for four fusion strategies."
    )
    parser.add_argument(
        "--fusion",
        type=str,
        required=True,
        choices=FUSION_CHOICES,
        help="Fusion strategy to train.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="HuggingFace dataset repo IDs to use for training/eval.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_desc_field", type=str, default="reasoning_1")
    parser.add_argument(
        "--eval_desc_field",
        type=str,
        default="model_generated_description_test",
    )
    parser.add_argument("--eval_subset_cb", type=int, default=EVAL_SUBSET_CB)
    parser.add_argument("--cb_batch", type=int, default=CB_BATCH)
    parser.add_argument(
        "--use_bf16",
        dest="use_bf16",
        action="store_true",
        default=True,
        help="Use BF16 mixed precision (if available). On by default.",
    )
    parser.add_argument(
        "--no_bf16",
        dest="use_bf16",
        action="store_false",
        help="Disable BF16 mixed precision.",
    )
    parser.add_argument(
        "--use_fp16", action="store_true", help="Use FP16 mixed precision."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sp_model_path",
        type=str,
        default=os.environ.get("REFUN_SP_MODEL", "./assets/segmentation.model"),
        help="Path to the SentencePiece segmentation model.",
    )
    parser.add_argument(
        "--word_cluster_path",
        type=str,
        default=os.environ.get("REFUN_WORD_CLUSTER", "./assets/word_cluster.json"),
        help="Path to the word-cluster JSON (CodeWordNet).",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs). <=0 disables early stopping.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=NUM_EXPERTS,
        help="Number of experts in the MoE layer (moe fusion only).",
    )
    parser.add_argument(
        "--num_selected_experts",
        type=int,
        default=NUM_SELECTED_EXPERTS,
        help="Number of top experts to select per sample (moe fusion only).",
    )
    parser.add_argument(
        "--moe_loss_weight",
        type=float,
        default=MOE_LOSS_WEIGHT,
        help="Weight for the MoE load-balancing loss (moe fusion only).",
    )
    args = parser.parse_args()

    if not args.datasets:
        raise ValueError(
            "No datasets provided. Pass one or more HuggingFace dataset repo IDs via --datasets."
        )

    set_seed(args.seed)

    NUM_EXPERTS = args.num_experts
    NUM_SELECTED_EXPERTS = args.num_selected_experts
    MOE_LOSS_WEIGHT = args.moe_loss_weight
    SP_MODEL_PATH = args.sp_model_path
    WORD_CLUSTER_PATH = args.word_cluster_path

    num_gpus = torch.cuda.device_count()
    print(f"[Init] Detected {num_gpus} GPU(s).")
    if num_gpus == 0:
        print("[Init] Warning: No GPUs detected. Training will be very slow on CPU.")

    BASE_BATCH_PER_GPU = 16
    adjusted_batch_per_gpu = BASE_BATCH_PER_GPU if num_gpus > 0 else BATCH_GPU
    final_batch_per_gpu = adjusted_batch_per_gpu
    final_grad_acc = GRAD_ACC
    print(
        f"[Init] Adjusted training batch size: {final_batch_per_gpu} per device, {final_grad_acc} gradient accumulation steps."
    )

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tok.add_special_tokens(
        {
            "additional_special_tokens": [
                "<ASM>",
                "<DEC>",
                "<SEXP>",
                "<DESC>",
                "FUN",
                START_TAG,
                END_TAG,
            ]
        }
    )
    print(
        f"[Init] Tokenizer vocab size after adding special tokens: {len(tok)}"
    )

    spm_obj = load_sp(SP_MODEL_PATH)
    with open(WORD_CLUSTER_PATH) as f:
        word_cluster = json.load(f)

    train_ds, eval_ds = load_and_clean(
        tok,
        spm_obj,
        args.train_desc_field,
        args.eval_desc_field,
        dataset_repos=args.datasets,
        calculate_dynamic_label_len=True,
    )
    collator = PadCollator(tok.pad_token_id)

    print(
        f"[Data] Final preprocessed dataset sizes (functions) -> "
        f"train: {len(train_ds)}, eval: {len(eval_ds) if eval_ds is not None else 0}"
    )

    model = build_model(args.fusion, tok)
    model.resize_token_embeddings(len(tok))
    print(
        f"[Init] Model embeddings resized to match tokenizer (size: {len(tok)})."
    )

    emb_w = None
    if COSINE_SIM:
        try:
            emb_w = model.dec.get_input_embeddings().weight.detach().cpu()
        except Exception as e:
            print(
                f"[Main] Warning: Could not retrieve embedding matrix for cosine sim: {e}"
            )
            emb_w = None

    compute_metrics = build_compute_metrics(
        tok, spm_obj, word_cluster, emb_matrix=emb_w
    )

    grouped_params = {
        "encoder_0": {
            "params": [],
            "lr": LR_ENCODERS[0],
            "weight_decay": WEIGHT_DECAY,
        },
        "encoder_1": {
            "params": [],
            "lr": LR_ENCODERS[1],
            "weight_decay": WEIGHT_DECAY,
        },
        "encoder_2": {
            "params": [],
            "lr": LR_ENCODERS[2],
            "weight_decay": WEIGHT_DECAY,
        },
        "encoder_3": {
            "params": [],
            "lr": LR_ENCODERS[3],
            "weight_decay": WEIGHT_DECAY,
        },
        "decoder_and_head": {
            "params": [],
            "lr": LR_DECODER,
            "weight_decay": WEIGHT_DECAY,
        },
        "fusion_and_projection": {
            "params": [],
            "lr": LR_DECODER,
            "weight_decay": WEIGHT_DECAY,
        },
    }

    print("\n[DEBUG] Inspecting model parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
    print("[DEBUG] End of parameter name inspection.\n")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "encs.0." in name:
            grouped_params["encoder_0"]["params"].append(param)
        elif "encs.1." in name:
            grouped_params["encoder_1"]["params"].append(param)
        elif "encs.2." in name:
            grouped_params["encoder_2"]["params"].append(param)
        elif "encs.3." in name:
            grouped_params["encoder_3"]["params"].append(param)
        elif name.startswith("dec.") or name.startswith("lm_head."):
            grouped_params["decoder_and_head"]["params"].append(param)
        elif (
            name.startswith("moe_layer.")
            or name.startswith("fusion_layer.")
            or name.startswith("cross_attn_layer.")
            or name.startswith("gating_layer.")
        ):
            grouped_params["fusion_and_projection"]["params"].append(param)
        else:
            print(
                f"[Init Debug] Parameter not explicitly grouped, adding to decoder_and_head: {name}"
            )
            grouped_params["decoder_and_head"]["params"].append(param)

    groups = [group for group in grouped_params.values() if len(group["params"]) > 0]
    for group_name, group in grouped_params.items():
        print(
            f"[Init Debug] Optimizer group '{group_name}': {len(group['params'])} parameters"
        )

    try:
        optimizer = AdamW(groups)
        print("[Init] AdamW optimizer created successfully.")
    except ValueError as e:
        print(f"[Init] Error creating AdamW optimizer: {e}")
        print("[Init] Dumping group info for debugging:")
        for i, group in enumerate(groups):
            print(
                f"  Group {i}: {len(group['params'])} params, LR={group['lr']}, WD={group['weight_decay']}"
            )
        raise

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (final_batch_per_gpu * world_size * final_grad_acc) > 0:
        steps_per_epoch = math.ceil(
            len(train_ds) / (final_batch_per_gpu * world_size * final_grad_acc)
        )
    else:
        steps_per_epoch = 1
    total_steps = steps_per_epoch * args.epochs
    lr_scheduler = get_scheduler(
        SchedulerType.COSINE,
        optimizer=optimizer,
        num_warmup_steps=WARMUP,
        num_training_steps=total_steps,
    )

    callbacks = [
        MyCallback(
            eval_ds,
            collator,
            tok,
            spm_obj,
            word_cluster,
            subset=args.eval_subset_cb,
            batch_size=args.cb_batch,
        )
    ]
    # REQUIRED CHANGE A: early stopping (paper claims it; originals lacked it).
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )
        print(
            f"[Init] EarlyStoppingCallback enabled with patience={args.early_stopping_patience}."
        )
    else:
        print("[Init] Early stopping disabled (--early_stopping_patience <= 0).")

    last_ckpt = None
    if args.resume and os.path.exists(args.output_dir):
        final_model_path = os.path.join(args.output_dir, "final_model")
        if os.path.exists(final_model_path):
            print(
                f"[Main] Found final model at {final_model_path}. Resuming might not be standard."
            )
        else:
            ckpts = [
                os.path.join(args.output_dir, d)
                for d in os.listdir(args.output_dir)
                if d.startswith("checkpoint-")
            ]
            if ckpts:
                last_ckpt = sorted(
                    ckpts, key=lambda x: int(x.split("-")[-1])
                )[-1]
                print(f"[Main] Resuming training from checkpoint: {last_ckpt}")
            else:
                print(
                    f"[Main] Resume flag set, but no checkpoints found in {args.output_dir}. Starting from scratch."
                )

    use_bf16 = args.use_bf16 and torch.cuda.is_bf16_supported()
    use_fp16 = args.use_fp16 or (args.use_bf16 and not torch.cuda.is_bf16_supported())
    if use_bf16:
        print("[Init] Using BF16 mixed precision.")
    elif use_fp16:
        print("[Init] Using FP16 mixed precision.")
    else:
        print("[Init] Using FP32 precision.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=final_batch_per_gpu,
        per_device_eval_batch_size=final_batch_per_gpu,
        gradient_accumulation_steps=final_grad_acc,
        auto_find_batch_size=True,
        learning_rate=LR_DECODER,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP,
        lr_scheduler_type=SchedulerType.COSINE,
        predict_with_generate=True,
        generation_max_length=LABEL_LEN,
        generation_num_beams=1,
        ddp_find_unused_parameters=True,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_KEY,
        greater_is_better=True,
        dataloader_pin_memory=False,
        logging_steps=LOG_STEPS,
        remove_unused_columns=False,
        report_to="none",
        label_smoothing_factor=0.0,
        eval_accumulation_steps=16,
    )

    trainer = MyMoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=(optimizer, lr_scheduler),
        pad_token_id=tok.pad_token_id,
        processing_class=tok,
        fusion=args.fusion,
    )

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Using dynamic LABEL_LEN: {LABEL_LEN}")
    print(f"Fusion Configuration: {args.fusion}")
    print(
        "Effective Batch Size: "
        f"{final_batch_per_gpu} (per GPU) * {num_gpus} (GPUs) * {final_grad_acc} (Grad Acc) = "
        f"{final_batch_per_gpu * num_gpus * final_grad_acc if num_gpus > 0 else final_batch_per_gpu * final_grad_acc}"
    )
    if args.fusion == "moe":
        print(
            f"  MoE: {NUM_EXPERTS} experts, selecting top {NUM_SELECTED_EXPERTS}, loss weight {MOE_LOSS_WEIGHT}"
        )

    trainer.train(resume_from_checkpoint=last_ckpt)

    if is_rank0():
        print(
            "[Main] Running generic inference (testing) on all datasets with the best model..."
        )
        run_inference_on_all_datasets(
            model,
            tok,
            spm_obj,
            word_cluster,
            args.datasets,
            args.eval_desc_field,
            args.output_dir,
            batch_size=args.cb_batch,
        )

    if is_rank0():
        final_model_path = os.path.join(args.output_dir, "final_model")
        print(f"Saving final model to {final_model_path}")
        model.save_pretrained(final_model_path)
        tok.save_pretrained(final_model_path)
        # save training args as JSON
        with open(os.path.join(final_model_path, "training_args.json"), "w") as f:
            json.dump(training_args.to_dict(), f, indent=2)
        print("Final model (encoders, decoder, fusion), tokenizer, and training args saved.")


if __name__ == "__main__":
    main()
