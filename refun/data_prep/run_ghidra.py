"""Drive Ghidra headless over a tree of binaries.

Each binary is analysed twice -- as shipped and after strip -- because ReFuN
needs the stripped decompilation as input and the original name as the label.
Pairing is by entry-point address, which strip does not change.

    python -m refun.data_prep.run_ghidra \\
        --binaries ./binaries --out ./ghidra_json \\
        --ghidra $GHIDRA_INSTALL_DIR --jobs 8

Writes one `<binary>.{orig,stripped}.json` pair per input, consumed by
build_corpus.py. Ghidra is not vendored: point --ghidra at an install or set
$GHIDRA_INSTALL_DIR. Failures are recorded in failures.json, never dropped
silently.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

SCRIPT_NAME = "export_all_features.py"


def _headless(ghidra_dir: Path) -> Path:
    exe = ghidra_dir / "support" / "analyzeHeadless"
    if not exe.exists():
        raise FileNotFoundError(
            f"analyzeHeadless not found under {ghidra_dir}. Pass --ghidra "
            "<install dir> or set $GHIDRA_INSTALL_DIR."
        )
    return exe


def analyse_one(binary: Path, out_json: Path, ghidra_dir: Path,
                script_dir: Path, is_stripped: bool, timeout: int = 3600) -> dict:
    """Run one headless analysis, returning a status record."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="refun_ghidra_") as tmp:
        stage = Path(tmp) / "out"
        stage.mkdir()
        env = dict(os.environ)
        env["GHIDRA_OUTPUT_DIR"] = str(stage)
        env["IS_STRIPPED"] = "true" if is_stripped else "false"
        cmd = [
            str(_headless(ghidra_dir)), tmp, "proj",
            "-import", str(binary),
            "-scriptPath", str(script_dir),
            "-postScript", SCRIPT_NAME,
            "-analysisTimeoutPerFile", str(timeout),
            "-deleteProject",
        ]
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True,
                                  text=True, timeout=timeout + 300)
        except subprocess.TimeoutExpired:
            return {"binary": str(binary), "stripped": is_stripped,
                    "ok": False, "error": "timeout"}
        produced = stage / "all_features.json"
        if not produced.exists():
            return {"binary": str(binary), "stripped": is_stripped, "ok": False,
                    "error": "no output",
                    "stderr": (proc.stderr or "")[-2000:]}
        shutil.move(str(produced), str(out_json))
    return {"binary": str(binary), "stripped": is_stripped, "ok": True,
            "out": str(out_json)}


def strip_copy(binary: Path, dest: Path, strip_bin: str = "strip") -> Path:
    """Strip a copy of the binary, using a cross `strip` when given one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, dest)
    subprocess.run([strip_bin, "--strip-all", str(dest)],
                   check=True, capture_output=True)
    return dest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binaries", required=True, type=Path,
                    help="directory of unstripped binaries (searched recursively)")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--ghidra", type=Path,
                    default=os.environ.get("GHIDRA_INSTALL_DIR"),
                    help="Ghidra install dir ($GHIDRA_INSTALL_DIR)")
    ap.add_argument("--strip_bin", default="strip",
                    help="strip binary to use, e.g. aarch64-linux-gnu-strip")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.ghidra:
        ap.error("--ghidra or $GHIDRA_INSTALL_DIR is required")
    ghidra_dir = Path(args.ghidra)
    script_dir = Path(__file__).resolve().parent / "ghidra"

    binaries = [p for p in sorted(args.binaries.rglob("*")) if p.is_file()]
    if args.limit:
        binaries = binaries[: args.limit]
    if not binaries:
        print(f"no binaries under {args.binaries}", file=sys.stderr)
        return 1
    print(f"[ghidra] {len(binaries)} binaries, {args.jobs} workers")

    stripped_dir = args.out / "_stripped"
    jobs = []
    for b in binaries:
        stem = b.name
        jobs.append((b, args.out / f"{stem}.orig.json", False))
        sb = strip_copy(b, stripped_dir / stem, args.strip_bin)
        jobs.append((sb, args.out / f"{stem}.stripped.json", True))

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {
            pool.submit(analyse_one, b, o, ghidra_dir, script_dir, s, args.timeout): b
            for b, o, s in jobs
        }
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            flag = "ok " if rec["ok"] else "FAIL"
            print(f"  [{i}/{len(jobs)}] {flag} {Path(rec['binary']).name}", flush=True)

    failures = [r for r in results if not r["ok"]]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"[ghidra] {len(results) - len(failures)} ok, {len(failures)} failed "
          f"(see {args.out / 'failures.json'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
