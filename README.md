# ReFuN / UniFuN — Reasoning-Guided Multi-View Function Name Inference for Stripped Binaries

> **Anonymous artifact for double-blind review.** This repository intentionally
> contains no author, institution, or account-identifying information — including
> the HuggingFace account that hosts the corpora (see §4). Please do not add any
> before the review process completes.

Given a function from a **stripped** binary, predict the name the original
developer gave it. Each function is represented by four complementary views —
**assembly**, **decompiled C**, an **AST s-expression**, and an **LLM reasoning
trace** — encoded by four CodeT5 encoders and fused before a shared decoder
emits the name.

Two models share this codebase:

| | training corpus | entry point |
|---|---|---|
| **ReFuN** | one model per `(arch, opt)` config | `python -m refun.train` |
| **UniFuN** | one model across all 16 configs | `python -m refun.unifun` |

They are architecturally identical; only the corpus differs, so any measured
difference is attributable to data rather than to a model change.

Four fusion strategies are implemented. The proposed model is
**mixture-of-experts (`moe`)**; `concat`, `cross_attention` and `simple_gating`
are the ablation baselines.

---

## 1. Repository layout

```
refun-release/
├── refun/
│   ├── train.py              # all four fusions, one code path (--fusion)
│   ├── unifun.py             # UniFuN entry point + per-config breakdown
│   ├── datasets.py           # corpus registry (anonymised namespace)
│   ├── sexpr.py              # tree-sitter AST view generation
│   ├── predict.py            # load a checkpoint, run inference (§8)
│   ├── audit_leakage.py      # ground-truth leakage measurement (§9)
│   └── data_prep/
│       ├── run_ghidra.py     # binaries      -> Ghidra JSON
│       ├── build_corpus.py   # Ghidra JSON   -> HF dataset (3 code views)
│       ├── reasoning.py      # + the 4th view: LLM reasoning traces
│       └── ghidra/           # Jython scripts run inside Ghidra headless
├── scripts/
│   ├── run_experiments.py    # (dataset × fusion) sweeps on local GPUs
│   ├── submit_slurm.py       # one SLURM job per (dataset × fusion)
│   ├── make_results_table.py # LaTeX fusion-comparison table
│   ├── smoke_test.sh         # trains all four fusions on a fixture (§7)
│   └── record_dataset_stats.py # refresh ROW_COUNTS from the live corpora
├── assets/                   # bundled eval assets (segmentation + clusters)
├── tests/
│   ├── make_fixture.py       # tiny synthetic corpus, real schema
│   └── test_units.py         # fast unit checks, no model needed
├── configs/.env.example
└── requirements.txt
```

---

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. The model builds on `Salesforce/codet5-base`, pulled from the Hub
on first run.

---

## 3. Quick start

```bash
export REFUN_HF_NAMESPACE=<hf-account>      # see §4

# Proposed model, one config
python -m refun.train --fusion moe --datasets x64_O0 \
    --output_dir runs/moe_x64_O0 --epochs 500

# The three ablation baselines
python -m refun.train --fusion concat          --datasets x64_O0 --output_dir runs/concat
python -m refun.train --fusion cross_attention --datasets x64_O0 --output_dir runs/xattn
python -m refun.train --fusion simple_gating   --datasets x64_O0 --output_dir runs/gating

# UniFuN: one model over all 16 configs, with a per-config breakdown
python -m refun.unifun --output_dir runs/unifun_moe --epochs 500
```

`--datasets` accepts config names (`x64_O0`), group aliases (`all`, `compiler`,
`unifun`, `obfuscation`, an arch like `x64`, an opt level like `O0`), a local
dataset directory, or fully-qualified Hub repo IDs — mixed freely.

Outputs land under `--output_dir`: `checkpoints/`, per-epoch dumps in
`eval_outputs/`, the early-stopping best model in `final_model/`, and
`metrics_<dataset>.json` + per-sample TSVs in `inference_results/`.

---

## 4. Datasets

### Archival source

**https://zenodo.org/records/15530083**

The corpora are archived on Zenodo — a citable, versioned copy that does not
depend on any account remaining live. Use this as the reference source; it is
what the paper cites.

### Hub mirror (for streaming)

The same corpora are mirrored on the HuggingFace Hub under a **single account,
deliberately not committed here** — naming it would break double-blind review.
Every repo ID is stored in `refun/datasets.py` as a bare name and joined to
`$REFUN_HF_NAMESPACE` at load time. Reviewers given the namespace out of band
can stream everything unchanged; without it the code raises an explanatory
error rather than silently failing.

```bash
export REFUN_HF_NAMESPACE=<hf-account>
python -m refun.datasets          # print the full resolved inventory
```

Either source works. To train from a Zenodo download instead of the Hub, point
`--datasets` at the extracted directory — `--datasets /data/x64_O0` — which
loads it with `load_from_disk` and skips the namespace entirely.

**16 compiler configs** — 4 architectures × 4 optimisation levels, built from
the same package set so cross-config comparisons are like-for-like:

| | O0 | O1 | O2 | O3 |
|---|---|---|---|---|
| **x64** | ✓ | ✓ | ✓ | ✓ |
| **x86** | ✓ | ✓ | ✓ | ✓ |
| **arm** | ✓ | ✓ | ✓ | ✓ |
| **mips** | ✓ | ✓ | ✓ | ✓ |

**4 obfuscation configs** (Obfuscator-LLVM, x64/O0): `obf_orig` (unobfuscated
control), `obf_bcfobf` (bogus control flow), `obf_cffobf` (control-flow
flattening), `obf_subobf` (instruction substitution).

Measured size, x64_O0: **62,415 train / 10,585 test** functions after the
pipeline's own deduplication.

### Required columns

| Column | Role |
|---|---|
| `assembly_code` | view 1 — stripped disassembly |
| `decompiled_code_stripped` | view 2 — stripped decompiled C |
| `S-Expression_of_decompiled_code_stripped` | view 3 — tree-sitter AST |
| `reasoning_1` | view 4 at **training** time (`--train_desc_field`) |
| `model_generated_description_test` | view 4 at **eval** time (`--eval_desc_field`) |
| `original_function_name` | label |

The two reasoning fields are **not interchangeable** — see §6.

---

## 5. Building a corpus from scratch

Only needed to extend the work to new binaries; the published corpora already
contain all four views.

```bash
# 1. Binaries -> Ghidra JSON (unstripped + stripped, analysed separately)
python -m refun.data_prep.run_ghidra \
    --binaries ./binaries --out ./ghidra_json \
    --ghidra $GHIDRA_INSTALL_DIR --jobs 8

# 2. Ghidra JSON -> HF dataset with the three code views
python -m refun.data_prep.build_corpus \
    --ghidra_json ./ghidra_json --out ./corpus_x64_O0 \
    --arch x64 --opt O0 --test_frac 0.15

# 3. Add the fourth view (needs an LLM endpoint)
python -m refun.data_prep.reasoning --show          # inspect the exact prompts
python -m refun.data_prep.reasoning \
    --in corpus.jsonl --out traces.jsonl \
    --mode eval --model <teacher-model> --base_url <openai-compatible-endpoint>
```

**Splits are by binary, never by function.** Two functions from one binary — and
the same function compiled at two optimisation levels — share too much for a
function-level split to measure generalisation.

### The AST view

`refun/sexpr.py` produces three linearisations of the tree-sitter C parse:

- `sexpr_with_text` — `(node_type "source span" children…)`; **this is the model
  input**, giving the encoder structure and surface form together.
- `sexpr_clean` — identical tree with every identifier, type and literal
  collapsed to `IDENT` / `TYPE` / `LIT`; the input for a "structure without
  surface tokens" ablation.
- `sexpr_fields` — tree-sitter's canonical printer with field labels; used only
  for deduplication.

With `tree_sitter` 0.26 / `tree_sitter_c` 0.24, `sexpr_with_text` reproduces
the published x64_O0 column **byte-for-byte on 98.5% of records** (n=200). The
residual 1.5% is error-recovery drift between grammar versions on decompiler
output that does not parse cleanly.

---

## 6. The two reasoning traces

The fourth view is an LLM rationale, used asymmetrically **by design**:

- **`reasoning_1` (training only).** The teacher sees the decompiled code *and
  the ground-truth name*, and justifies that name without stating it. This is a
  distillation target: it encodes *why* the name fits.
- **`model_generated_description_test` (evaluation).** The teacher sees only the
  decompiled code. Nothing about the gold name enters the prompt, so this is
  available at inference on a genuinely unseen binary.

Pointing `--train_desc_field` and `--eval_desc_field` at the same column — in
particular pointing eval at `reasoning_1` — leaks the answer. Both prompt
templates are reproduced verbatim in `refun/data_prep/reasoning.py`; print them
with `--show`.

---

## 7. Verifying the code runs

No GPU, no Hub access, and no corpus required:

```bash
bash scripts/smoke_test.sh
```

It builds a 64-row synthetic corpus with the real schema
(`tests/make_fixture.py`) and trains **all four fusions** end-to-end for one
epoch each — exercising preprocessing, deduplication, every fusion module, the
metric callbacks, early stopping, and the inference dump.

Both evaluation assets are **bundled in `assets/`**, so nothing needs
downloading:

| Asset | Size | Purpose | Env var | Flag |
|---|---|---|---|---|
| `assets/segmentation.model` | 507 KB | identifier subtoken splitting | `REFUN_SP_MODEL` | `--sp_model_path` |
| `assets/word_cluster.json` | 423 KB | synonym (CWordNet) F1, 18,379 clusters | `REFUN_WORD_CLUSTER` | `--word_cluster_path` |

They affect scoring only — the model never sees them. See
[`assets/README.md`](assets/README.md). If you point the paths elsewhere and a
file is missing, the metric degrades with a printed warning rather than
aborting. `--require_assets` makes absence a hard error instead.

---

## 8. Loading a trained model and running inference

### What a run writes

```
runs/moe_x64_O0/
├── checkpoints/…            per-epoch checkpoints (save_total_limit=2)
├── eval_outputs/…           per-epoch prediction dumps
├── inference_results/
│   ├── metrics_<dataset>.json        precision, recall, F1, exact, CWordNet-F1
│   └── *_inference_<dataset>.tsv     per-sample gold vs prediction
└── final_model/             ← the best checkpoint, restored by early stopping
    ├── pytorch_model.bin    full state dict
    ├── encoder_{0..3}.bin   per-encoder weights
    ├── decoder.bin, lm_head.bin, moe_layer.bin
    ├── run_config.json      ← how to rebuild the model (see below)
    ├── training_args.json   HuggingFace TrainingArguments dump
    └── vocab.json, merges.txt, added_tokens.json, …   tokenizer
```

### Why `run_config.json` matters

`final_model/` cannot be loaded from the weights alone. Two parameters decide
the *shape* of the model:

- **`fusion`** — picks which of the four classes to instantiate.
- **`label_len`** — the fusion projection layers are `label_len ×
  hidden_size`, and it is computed *dynamically* from the training data
  (longest tokenised name, capped at 15). Rebuilding with the wrong value fails
  with a shape mismatch on every fusion weight.

Training writes both, plus `encoder_mode`, `tokens_per_encoder`, the MoE
settings and the dataset list, into `run_config.json`. `refun.predict` reads it
automatically:

```json
{
  "fusion": "moe",
  "label_len": 9,
  "tokens_per_encoder": 512,
  "encoder_mode": "independent",
  "num_experts": 4,
  "num_selected_experts": 2,
  "eval_desc_field": "model_generated_description_test"
}
```

Training also resizes the embedding matrix from 32,100 to **32,107** for the
seven added special tokens (`<ASM>`, `<DEC>`, `<SEXP>`, `<DESC>`, `FUN`, and
the two name tags). `refun.predict` re-applies the resize before loading the
state dict; if you write your own loader, do the same or nothing will load.

### Predict

```bash
# One function from files. The AST view is derived from --code automatically.
python -m refun.predict --model runs/moe_x64_O0/final_model \
    --asm func.asm --code func.c

# A whole split, scored against ground truth
python -m refun.predict --model runs/moe_x64_O0/final_model \
    --input x64_O0 --split test --limit 500 --out preds.tsv

# A local JSONL or a Zenodo download
python -m refun.predict --model runs/moe_x64_O0/final_model \
    --input test.jsonl --out preds.tsv
python -m refun.predict --model runs/moe_x64_O0/final_model \
    --input /data/x64_O0 --split test --out preds.tsv
```

Batch mode prints exact-match and token P/R/F1 at the end. Views you do not
have may be omitted — the corresponding encoder sees only its marker token —
but expect a real accuracy drop, particularly without the reasoning view.

For a checkpoint predating `run_config.json`, pass the two shape parameters by
hand; `--label_len` must match what training used:

```bash
python -m refun.predict --model old_run/final_model --fusion moe --label_len 9 ...
```

### From Python

```python
from refun.predict import load_model, predict_one

model, tok, cfg = load_model("runs/moe_x64_O0/final_model")   # device auto
name = predict_one(
    model, tok,
    asm=open("func.asm").read(),
    code=open("func.c").read(),
    desc="Allocates a buffer and writes a build-id section to the output file.",
    tokens_per_encoder=cfg["tokens_per_encoder"],
)
print(name)
```

### Resuming or fine-tuning

`--resume` continues from the latest checkpoint in `--output_dir`. To fine-tune
one config's model on another, point `--output_dir` at a fresh directory and
keep `--fusion` and the token budget identical to the original run.

---

## 9. Ground-truth leakage — read before quoting a number

A function-name model is only interesting if the name is not already in its
input. On this corpus that is frequently false, via two distinct mechanisms.
Measured on the **x64_O0 test split** (n=794 sampled, names ≥4 chars):

| | rate | what it is |
|---|---|---|
| name in the function's **own signature** | **28.2%** | dynamically-linked and thunk functions keep their names in `.dynsym`, which survives `strip`. Ghidra recovers them, so the "stripped" body literally reads `void abort(void)`. **A copy task, not inference.** |
| name only in a **string literal** | 8.6% | assertion macros embed the enclosing function name: `FUN_0017c650("../../gold/object.h", 0x3c7, "do_output_section_offset")`. Genuine residual signal a human analyst would also use. |

Per-view leak rates: decompiled C **36.9%**, AST view **36.9%** (it quotes the
source), assembly **0.0%**, eval reasoning trace 22.3% (the teacher succeeding,
not a defect), training trace 3.0%.

These have **opposite implications**. The first is a dataset-construction defect
and inflates scores; the second is a property of the domain and is present in
every published binary corpus. To measure both across configs:

```bash
python -m refun.audit_leakage --configs all --split test --out leakage.json
```

and exclude the first from headline numbers with:

```bash
python -m refun.train --fusion moe --datasets x64_O0 --drop_selfnamed ...
```

`--drop_selfnamed` is off by default so that default runs stay comparable with
previously published numbers.

---

## 10. The four fusion strategies

All four encode the four views (512 tokens each) and feed a shared CodeT5
decoder. They differ in how the four encoder outputs become the decoder's
cross-attention memory — and, as shipped, in how many encoders there actually
are (see the note below).

**`concat`** — the four per-view sequences are concatenated along the time axis
into one 2048-position memory. No parameters are added; the decoder's
cross-attention does all the work. Cheapest, and the strongest "is fusion even
needed?" control.

**`cross_attention`** — a multi-head attention block (8 heads, residual + LayerNorm,
4× FFN) where the **assembly** stream is the query and all four streams
concatenated are the keys/values. Lets one view actively interrogate the others,
at the cost of privileging whichever view is chosen as query.

**`simple_gating`** — the four streams are stacked position-wise and a single
linear gate over their mean produces a softmax weight per view *per position*;
the output is the weighted sum. One small gate, and the weights are directly
interpretable as per-token view importance.

**`moe`** (proposed) — a mixture-of-experts layer over the fused sequence:
4 experts, **top-2** routing, outputs combined by the renormalised router
weights. A load-balancing term penalises squared deviation of mean routing mass
from uniform, keeping experts from collapsing onto one. Knobs: `--num_experts`,
`--num_selected_experts`, `--moe_loss_weight`.

Total loss: `L_sup + λ_kd·L_kd + λ_bal·L_bal`.

### Encoder count — a confound to control for

The four fusions were **not** trained with the same encoder capacity:

| fusion | encoders as trained |
|---|---|
| `moe` (proposed) | **4 independent** CodeT5 encoders (`deepcopy`), sharing only `embed_tokens` |
| `concat`, `cross_attention`, `simple_gating` | **1 shared** encoder applied to all four views |

In the shared configuration the views stay distinguishable through the `<ASM>`
/ `<DEC>` / `<SEXP>` / `<DESC>` marker token prepended to each, so it is a
legitimate design — but the proposed model carried roughly three extra
encoders' worth of parameters relative to its own baselines, so a gain measured
at the defaults confounds capacity with fusion strategy.

`train.py` reproduces the as-trained configuration by default. To hold encoder
capacity constant across fusions:

```bash
# All four fusions at equal encoder capacity
for f in concat cross_attention simple_gating moe; do
  python -m refun.train --fusion $f --encoder_mode independent \
      --datasets x64_O0 --output_dir runs/indep_$f
done
```

`nn.ModuleList([m.get_encoder() for _ in range(4)])` looks like it builds four
encoders and builds one: `get_encoder()` returns the same module object every
call. `build_encoders()` makes the choice explicit.

---

## 11. Hyperparameters (defaults, as used in the paper)

| Setting | Value |
|---|---|
| Base model | CodeT5-base — 4 encoders + 1 shared decoder |
| Tokens per encoder | 512 |
| Learning rate | 5e-4, encoders and decoder |
| Weight decay | 0.01 |
| Schedule | cosine, 2,000 warm-up steps |
| Batch × grad-accum | 16 × 8 (effective 128; `auto_find_batch_size` on OOM) |
| Epochs | up to 500, early stopping on val token-F1, patience 5 |
| Label smoothing | 0.1 |
| Precision | BF16 (`--no_bf16`, or `--use_fp16`) |
| MoE | 4 experts, top-2, load-balancing loss |
| λ_kd, λ_bal | 0.1, 0.1 |
| Seed | 42 (`random`, NumPy, Torch) |

---

## 12. Running many experiments

```bash
# Local, one or more GPUs
python scripts/run_experiments.py \
    --datasets x64_O0 x64_O1 \
    --architectures concat cross_attention simple_gating moe \
    --epochs 500 --output_root experiment_runs \
    --max_parallel 4 --cuda_devices 0,1,2,3        # --dry_run to preview

# SLURM: one job per (dataset × fusion)
python scripts/submit_slurm.py \
    --datasets x64_O0 --architectures moe --epochs 500 \
    --partition a100 --gpus 1 --cpus_per_task 16 \
    --conda_env <env> --repo_dir "$(pwd)"          # --dry_run writes .sh only

# LaTeX comparison table (MoE as reference row, baselines as % deltas)
python scripts/make_results_table.py --root experiment_runs --out fusion_table.tex
```

Generated SLURM scripts `cd` into `--repo_dir`; no absolute paths are baked in.

---

## 13. Evaluation

Names are scored at the **token level**: identifiers are split on camelCase and
underscores, and precision / recall / F1 computed over token overlap. Exact
match and CWordNet-F1 (synonym-cluster F1) are reported alongside.

Deduplication is SymGen-style and applied in memory at load time — by name, by
normalised assembly, and by normalised decompiled body — followed by a
cross-corpus pass that removes function names appearing in more than one config.
The per-epoch validation callback scores a random subset (`--eval_subset_cb`,
default 64) for speed; the full eval set is scored once at the end.

For UniFuN, `unifun_breakdown.json` reports per-config metrics plus a macro
average (every config weighted equally) and a row-weighted average (dominated
by the large O0 configs). A pooled score can improve while every hard config
degrades, so both are given.

---

## 14. Citing the data

The corpora are archived at **https://zenodo.org/records/15530083**. Cite that
record (not the Hub mirror) — it is versioned and has a DOI, so it stays
resolvable regardless of what happens to any account.

The author block is withheld while double-blind review is in progress; the
Zenodo record carries the canonical metadata.

