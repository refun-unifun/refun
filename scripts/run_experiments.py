#!/usr/bin/env python3
"""Run one or more (dataset x fusion) training experiments.

Each experiment invokes the unified trainer via `python -m refun.train ...`.
All architectures (concat, cross_attention, simple_gating, moe) are supported.
Datasets are HuggingFace dataset repo IDs supplied on the CLI.
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, NamedTuple, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent  # contains the `refun` package

ARCHITECTURES: List[str] = ["concat", "cross_attention", "simple_gating", "moe"]


class ExperimentJob(NamedTuple):
    architecture: str
    dataset_repo: str
    dataset_slug: str
    output_dir: Path
    log_path: Path
    command: List[str]
    cuda_device: Optional[str]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def shell_join(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def resolve_output_root(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_DIR / path


def parse_cuda_devices(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def slugify_repo(repo: str) -> str:
    return repo.split("/")[-1]


def build_jobs(args: argparse.Namespace) -> List[ExperimentJob]:
    output_root = resolve_output_root(args.output_root)
    cuda_devices = parse_cuda_devices(args.cuda_devices)
    jobs: List[ExperimentJob] = []

    for dataset_repo in args.datasets:
        dataset_slug = slugify_repo(dataset_repo)
        for architecture in args.architectures:
            output_dir = output_root / dataset_slug / architecture
            log_path = output_dir / "logs" / "run.log"
            command = [
                args.python,
                "-m",
                "refun.train",
                "--fusion",
                architecture,
                "--epochs",
                str(args.epochs),
                "--output_dir",
                str(output_dir),
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
                "--datasets",
                dataset_repo,
            ]
            if args.sp_model_path:
                command += ["--sp_model_path", args.sp_model_path]
            if args.word_cluster_path:
                command += ["--word_cluster_path", args.word_cluster_path]
            if args.resume:
                command.append("--resume")
            if args.use_fp16:
                command.append("--use_fp16")
            # bf16 is on by default in the trainer; allow turning it off.
            if not args.use_bf16:
                command.append("--no_bf16")

            device_index = len(jobs) % len(cuda_devices) if cuda_devices else None
            jobs.append(ExperimentJob(
                architecture=architecture,
                dataset_repo=dataset_repo,
                dataset_slug=dataset_slug,
                output_dir=output_dir,
                log_path=log_path,
                command=command,
                cuda_device=cuda_devices[device_index] if device_index is not None else None,
            ))

    return jobs


def make_summary_row(job: ExperimentJob, status_payload: dict) -> dict:
    return {
        "dataset_slug": job.dataset_slug,
        "dataset_repo": job.dataset_repo,
        "architecture": job.architecture,
        "status": status_payload.get("status"),
        "return_code": status_payload.get("return_code"),
        "wall_time_s": status_payload.get("wall_time_s"),
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
    }


def run_job(job: ExperimentJob) -> dict:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = job.output_dir / "status.json"
    command_text = shell_join(job.command)
    started_at = now_text()
    start_time = time.perf_counter()

    status_payload = {
        "status": "running",
        "architecture": job.architecture,
        "dataset_slug": job.dataset_slug,
        "dataset_repo": job.dataset_repo,
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
        "command": job.command,
        "cuda_device": job.cuda_device,
        "started_at": started_at,
    }
    write_json(status_path, status_payload)
    write_json(job.output_dir / "experiment_config.json", status_payload)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if job.cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = job.cuda_device

    print(f"[Launch] {job.dataset_slug}/{job.architecture}")
    print(f"         log: {job.log_path}")

    with job.log_path.open("w") as log_file:
        log_file.write(f"[Runner] Started at {started_at}\n")
        log_file.write(f"[Runner] Command: {command_text}\n")
        if job.cuda_device is not None:
            log_file.write(f"[Runner] CUDA_VISIBLE_DEVICES={job.cuda_device}\n")
        log_file.flush()

        process = subprocess.Popen(
            job.command,
            cwd=str(REPO_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()

        finished_at = now_text()
        wall_time_s = time.perf_counter() - start_time
        log_file.write(f"\n[Runner] Finished at {finished_at}\n")
        log_file.write(f"[Runner] Return code: {return_code}\n")
        log_file.write(f"[Runner] Wall time seconds: {wall_time_s:.3f}\n")

    status_payload.update({
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "finished_at": finished_at,
        "wall_time_s": wall_time_s,
    })
    write_json(status_path, status_payload)

    print(f"[Done] {job.dataset_slug}/{job.architecture} -> {status_payload['status']}")
    return make_summary_row(job, status_payload)


def write_global_summary(output_root: Path, rows: List[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "experiment_summary.json"
    csv_path = output_root / "experiment_summary.csv"

    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Summary] Wrote {csv_path}")
    print(f"[Summary] Wrote {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run (dataset x fusion) training experiments via `python -m refun.train`."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="HuggingFace dataset repo IDs.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--output_root", type=str, default="experiment_runs")
    parser.add_argument("--train_desc_field", type=str, default="reasoning_1")
    parser.add_argument("--eval_desc_field", type=str, default="model_generated_description_test")
    parser.add_argument("--eval_subset_cb", type=int, default=64)
    parser.add_argument("--cb_batch", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--sp_model_path", type=str, default="")
    parser.add_argument("--word_cluster_path", type=str, default="")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=ARCHITECTURES,
    )
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--cuda_devices", type=str, default="", help="Comma-separated CUDA devices, e.g. 0,1,2,3")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_bf16", dest="use_bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="use_bf16", action="store_false")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max_parallel must be at least 1")

    output_root = resolve_output_root(args.output_root)
    jobs = build_jobs(args)
    print(f"[Runner] Prepared {len(jobs)} jobs under {output_root}")

    if args.dry_run:
        for job in jobs:
            device_text = f" CUDA_VISIBLE_DEVICES={job.cuda_device}" if job.cuda_device is not None else ""
            print(f"[DryRun]{device_text} {shell_join(job.command)}")
        return 0

    rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        future_map = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                status_payload = {
                    "status": "failed",
                    "return_code": None,
                    "wall_time_s": None,
                    "error": str(exc),
                }
                write_json(job.output_dir / "status.json", status_payload)
                rows.append(make_summary_row(job, status_payload))
                print(f"[Error] {job.dataset_slug}/{job.architecture}: {exc}")

    rows.sort(key=lambda row: (row["dataset_slug"], row["architecture"]))
    write_global_summary(output_root, rows)
    failed = sum(1 for row in rows if row.get("status") != "completed")
    print(f"[Runner] Completed {len(rows) - failed}/{len(rows)} jobs successfully.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
