#!/usr/bin/env python3
"""Generate and submit one SLURM batch job per (dataset x fusion) combination.

Datasets are HuggingFace dataset repo IDs supplied on the CLI (--datasets).
All four fusion architectures are supported. The generated batch script `cd`s
into the repo via --repo_dir (default: this script's parent-parent) and invokes
the experiment runner there.
"""
import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = SCRIPT_DIR.parent  # contains the `refun` package

ARCHITECTURES: List[str] = ["concat", "cross_attention", "simple_gating", "moe"]


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_job_id(output: str) -> str:
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    return match.group(1) if match else ""


def slugify_repo(repo: str) -> str:
    return repo.split("/")[-1]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_script(
    args: argparse.Namespace,
    repo_dir: Path,
    output_root: Path,
    slurm_log_dir: Path,
    dataset_repo: str,
    dataset_slug: str,
    architecture: str,
) -> str:
    job_name = f"refun_{dataset_slug}_{architecture}"
    if len(job_name) > 30:
        job_name = job_name[:30]

    command = [
        "python3",
        str(SCRIPT_DIR / "run_experiments.py"),
        "--epochs",
        str(args.epochs),
        "--output_root",
        str(output_root),
        "--train_desc_field",
        args.train_desc_field,
        "--eval_desc_field",
        args.eval_desc_field,
        "--eval_subset_cb",
        str(args.eval_subset_cb),
        "--cb_batch",
        str(args.cb_batch),
        "--early_stopping_patience",
        str(args.early_stopping_patience),
        "--max_parallel",
        "1",
        "--datasets",
        dataset_repo,
        "--architectures",
        architecture,
    ]
    if args.sp_model_path:
        command += ["--sp_model_path", args.sp_model_path]
    if args.word_cluster_path:
        command += ["--word_cluster_path", args.word_cluster_path]
    if args.use_fp16:
        command.append("--use_fp16")
    if not args.use_bf16:
        command.append("--no_bf16")
    if args.resume:
        command.append("--resume")

    out_file = slurm_log_dir / f"{dataset_slug}_{architecture}_%j.out"
    err_file = slurm_log_dir / f"{dataset_slug}_{architecture}_%j.err"
    command_line = " ".join(shell_quote(part) for part in command)

    return f"""#!/usr/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --job-name={job_name}
#SBATCH --partition={args.partition}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --output={out_file}
#SBATCH --error={err_file}

source ~/miniconda3/bin/activate
conda activate {args.conda_env}

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd {shell_quote(str(repo_dir))}
{command_line}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit one SLURM job per (dataset x fusion) combination."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="HuggingFace dataset repo IDs.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--run_id", type=str, default=run_id())
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=str(DEFAULT_REPO_DIR),
        help="Repository directory to cd into (must contain the `refun` package).",
    )
    parser.add_argument("--train_desc_field", type=str, default="reasoning_1")
    parser.add_argument("--eval_desc_field", type=str, default="model_generated_description_test")
    parser.add_argument("--eval_subset_cb", type=int, default=64)
    parser.add_argument("--cb_batch", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--sp_model_path", type=str, default="")
    parser.add_argument("--word_cluster_path", type=str, default="")
    parser.add_argument("--partition", type=str, default="a100")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus_per_task", type=int, default=16)
    parser.add_argument("--conda_env", type=str, default="base")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=ARCHITECTURES,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_bf16", dest="use_bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="use_bf16", action="store_false")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()

    output_root = Path(args.output_root) if args.output_root else SCRIPT_DIR / f"slurm_runs_{args.run_id}"
    if not output_root.is_absolute():
        output_root = SCRIPT_DIR / output_root

    slurm_log_dir = SCRIPT_DIR / "slurm_logs" / f"cases_{args.run_id}"
    job_script_dir = SCRIPT_DIR / "slurm_case_jobs" / f"cases_{args.run_id}"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    job_script_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    submitted_rows: List[dict] = []
    for dataset_repo in args.datasets:
        dataset_slug = slugify_repo(dataset_repo)
        for architecture in args.architectures:
            script_path = job_script_dir / f"{dataset_slug}_{architecture}.sh"
            script_text = build_script(
                args, repo_dir, output_root, slurm_log_dir, dataset_repo, dataset_slug, architecture
            )
            script_path.write_text(script_text)
            script_path.chmod(0o755)

            experiment_dir = output_root / dataset_slug / architecture
            row = {
                "dataset_slug": dataset_slug,
                "dataset_repo": dataset_repo,
                "architecture": architecture,
                "experiment_dir": str(experiment_dir),
                "job_script": str(script_path),
                "slurm_out_glob": str(slurm_log_dir / f"{dataset_slug}_{architecture}_*.out"),
                "slurm_err_glob": str(slurm_log_dir / f"{dataset_slug}_{architecture}_*.err"),
            }

            if args.dry_run:
                row["job_id"] = ""
                row["submit_status"] = "dry_run"
                print(f"[DryRun] {script_path}")
            else:
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    cwd=str(repo_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    check=False,
                )
                row["submit_output"] = result.stdout.strip()
                row["submit_returncode"] = result.returncode
                row["job_id"] = parse_job_id(result.stdout)
                row["submit_status"] = "submitted" if result.returncode == 0 and row["job_id"] else "failed"
                print(f"[{row['submit_status']}] {dataset_slug}/{architecture} {row['submit_output']}")

            submitted_rows.append(row)

    manifest = {
        "run_id": args.run_id,
        "repo_dir": str(repo_dir),
        "output_root": str(output_root),
        "slurm_log_dir": str(slurm_log_dir),
        "job_script_dir": str(job_script_dir),
        "jobs": submitted_rows,
    }
    write_json(output_root / "slurm_submission_manifest.json", manifest)
    write_csv(output_root / "slurm_submission_manifest.csv", submitted_rows)
    print(f"[Manifest] {output_root / 'slurm_submission_manifest.json'}")
    print(f"[Manifest] {output_root / 'slurm_submission_manifest.csv'}")

    failed = sum(1 for row in submitted_rows if row["submit_status"] == "failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
