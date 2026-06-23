# ReFuN: Reasoning-Guided Mixture-of-Experts Function Name Inference for Stripped Binaries

> **Anonymous artifact for double-blind review.** This repository intentionally
> contains no author, institution, or account-identifying information. Please do
> not add any before the review process completes.

ReFuN infers meaningful function names from stripped binaries. Each function is
represented by four complementary views — **assembly code**, **decompiled source
code**, an **AST s-expression**, and a **reasoning trace** distilled from a
foundation LLM — encoded by four CodeT5 encoders and fused before a shared
decoder generates the function name. Four fusion strategies are provided; the
proposed model is the **mixture-of-experts (`moe`)** fusion, and the other three
(`concat`, `cross_attention`, `simple_gating`) are the ablation baselines.

---

## 1. Repository layout

```
refun-release/
├── refun/
│   ├── __init__.py
│   └── train.py            # Unified entry point for all four fusion strategies
├── scripts/
│   ├── run_experiments.py  # Run one or more (dataset × fusion) experiments locally
│   └── submit_slurm.py     # Generate + submit one SLURM job per (dataset × fusion)
├── configs/
│   └── .env.example        # Asset paths (segmentation model, word clusters)
├── requirements.txt
├── LICENSE
└── README.md
```

All four fusion strategies share one code path in `refun/train.py`, selected with
`--fusion`. This replaces the previous four near-duplicate training scripts.

---

## 2. Installation

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The model is built on `Salesforce/codet5-base`, downloaded automatically from the
HuggingFace Hub on first run.

---

## 3. Required assets

Two evaluation assets are **not** bundled (they are environment-specific) and must
be supplied locally. Point to them via environment variables or CLI flags:

| Asset | Purpose | Env var | CLI flag |
|-------|---------|---------|----------|
| SentencePiece segmentation model | Splits identifiers into subtokens for the token-level metric | `REFUN_SP_MODEL` | `--sp_model_path` |
| CodeWordNet word-cluster JSON | Synonym clustering for the CWordNet-F1 metric | `REFUN_WORD_CLUSTER` | `--word_cluster_path` |

```bash
cp configs/.env.example .env
# edit .env, then:
export $(grep -v '^#' .env | xargs)
# or place files at the defaults: ./assets/segmentation.model , ./assets/word_cluster.json
```

---

## 4. Dataset format

Datasets are loaded from the HuggingFace Hub by repo ID (pass with `--datasets`).
Each split must provide these columns:

| Column | Description |
|--------|-------------|
| `assembly_code` | Disassembled instructions of the stripped function |
| `decompiled_code_stripped` | Decompiled C source of the stripped function |
| `S-Expression_of_decompiled_code_stripped` | AST of the decompiled code, linearized as an s-expression |
| `original_function_name` | Ground-truth name (recovered from the unstripped binary) |
| *training description field* | Reasoning trace used for distillation (default column `reasoning_1`, set with `--train_desc_field`) |
| *eval description field* | Inference-time reasoning trace (default `model_generated_description_test`, set with `--eval_desc_field`) |

> **Note (double-blind):** dataset repo IDs are **not** hardcoded — supply your own
> with `--datasets`. Do not commit repo IDs that reveal an account name.

Reasoning/counterfactual traces are produced by prompting a foundation LLM with
the decompiled source (see the paper, §IV-B). The prompts are reproduced there.

---

## 5. Quick start — train a single model

```bash
# Proposed model: mixture-of-experts fusion
python -m refun.train \
  --fusion moe \
  --datasets <your_hf_dataset_repo_id> \
  --output_dir runs/moe_x64_o3 \
  --epochs 500 \
  --early_stopping_patience 5 \
  --sp_model_path ./assets/segmentation.model \
  --word_cluster_path ./assets/word_cluster.json
```

Swap `--fusion` for any of the ablation baselines:

```bash
python -m refun.train --fusion concat          --datasets <repo> --output_dir runs/concat
python -m refun.train --fusion cross_attention --datasets <repo> --output_dir runs/xattn
python -m refun.train --fusion simple_gating   --datasets <repo> --output_dir runs/gating
```

BF16 mixed precision is **on by default** (pass `--no_bf16` to disable, or
`--use_fp16` to force FP16). MoE-specific knobs: `--num_experts` (default 4),
`--num_selected_experts` (default 2), `--moe_loss_weight` (load-balancing weight).

Outputs land under `--output_dir`: `checkpoints/`, per-epoch eval dumps in
`eval_outputs/`, the best model (restored via early stopping) in `final_model/`,
and final-eval metrics in `inference_results/`.

---

## 6. Run multiple experiments

```bash
# All four fusions across one or more datasets, sequentially (1 GPU)
python scripts/run_experiments.py \
  --datasets <repo_A> <repo_B> \
  --architectures concat cross_attention simple_gating moe \
  --epochs 500 \
  --output_root experiment_runs

# Add --dry_run to print commands without launching.
# --max_parallel N and --cuda_devices 0,1,2,3 to fan out across GPUs.
```

---

## 7. SLURM cluster usage

```bash
# Generates one batch script per (dataset × fusion) under slurm_case_jobs/ and submits.
python scripts/submit_slurm.py \
  --datasets <repo_A> <repo_B> \
  --architectures moe \
  --epochs 500 \
  --partition a100 --gpus 1 --cpus_per_task 16 \
  --conda_env base \
  --repo_dir "$(pwd)"

# --dry_run writes the .sh files without calling sbatch.
```

The generated scripts `cd` into `--repo_dir` (defaults to the repo root) — no
machine-specific absolute paths are baked in.

---

## 8. Key hyperparameters (defaults, matching the paper)

| Setting | Value |
|---------|-------|
| Base model | CodeT5-base (4 encoders + 1 decoder) |
| Learning rate | 5e-4 (encoders and decoder) |
| Weight decay | 0.01 |
| Scheduler | cosine, 2,000 warm-up steps |
| Batch / grad accumulation | 4 × 32 (effective 128) |
| Max epochs | 500, **early stopping** on validation token-F1 (patience 5) |
| Label smoothing | 0.1 |
| MoE | 4 experts, top-2 routing, load-balancing loss |
| Loss | `L_sup + λ_kd·L_kd + λ_bal·L_bal` (λ_kd = λ_bal = 0.1) |

## 9. Evaluation metrics

Function-name quality is scored at the **token level**: identifiers are split
(camelCase / underscores) and precision/recall/F1 are computed over token overlap.
The code also reports exact match and CWordNet-F1 (synonym-cluster F1).

---

## 10. Reproducibility notes

- Seed is fixed (`--seed`, default 42) across `random`, NumPy, and Torch.
- Function deduplication (SymGen-style: by name, normalized assembly, and
  normalized decompiled body) is applied in-memory at load time; see `train.py`.
- The per-epoch validation callback scores a random subset (`--eval_subset_cb`,
  default 64) for speed; the full evaluation set is scored once at the end.
