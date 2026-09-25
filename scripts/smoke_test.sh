#!/usr/bin/env bash
# Prove every code path executes, with no GPU, no Hub access and no corpus.
#
# Builds a small synthetic dataset with the real column schema, then trains
# each of the four fusion strategies for one epoch. This exercises
# preprocessing, SymGen deduplication, all four fusion modules, the metric
# callbacks, early stopping and the inference dump.
#
#   bash scripts/smoke_test.sh              # all four fusions
#   bash scripts/smoke_test.sh moe          # just one
#   REFUN_SMOKE_N=128 bash scripts/smoke_test.sh
#
# Exit status is non-zero if any fusion fails.
set -uo pipefail

cd "$(dirname "$0")/.."

WORK="${REFUN_SMOKE_DIR:-${TMPDIR:-/tmp}/refun_smoke}"
FIXTURE="$WORK/fixture"
N="${REFUN_SMOKE_N:-64}"
FUSIONS=("$@")
if [ ${#FUSIONS[@]} -eq 0 ]; then
    FUSIONS=(concat cross_attention simple_gating moe)
fi

mkdir -p "$WORK"
echo "=== building fixture ($N rows) -> $FIXTURE"
rm -rf "$FIXTURE"
python tests/make_fixture.py --out "$FIXTURE" --n "$N" || {
    echo "FAILED: could not build fixture"; exit 1;
}

declare -a PASSED=() FAILED=()

for fusion in "${FUSIONS[@]}"; do
    out="$WORK/run_$fusion"
    log="$WORK/$fusion.log"
    rm -rf "$out"
    echo
    echo "=== $fusion ============================================="
    if python -u -m refun.train \
            --fusion "$fusion" \
            --datasets "$FIXTURE" \
            --output_dir "$out" \
            --epochs 1 \
            --eval_subset_cb 8 \
            --cb_batch 2 \
            --tokens_per_encoder "${REFUN_SMOKE_TOKENS:-128}" \
            > "$log" 2>&1; then
        echo "  PASS  (log: $log)"
        PASSED+=("$fusion")
    else
        echo "  FAIL  (exit $?) — last 25 lines:"
        tail -25 "$log" | sed 's/^/      /'
        FAILED+=("$fusion")
    fi
done

echo
echo "=========================================================="
echo "passed: ${#PASSED[@]}/${#FUSIONS[@]}  ${PASSED[*]:-}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "failed: ${FAILED[*]}"
    exit 1
fi
echo "all fusion strategies ran end-to-end."
