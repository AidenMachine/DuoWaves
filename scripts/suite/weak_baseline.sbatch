#!/bin/bash
#SBATCH --job-name=weak-baseline
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=sbatch_jobs/weak_baseline_%j.out
#SBATCH --error=sbatch_jobs/weak_baseline_%j.err

if [ -z "${SUITE_DIR:-}" ]; then
    echo "submit with --export=ALL,SUITE_DIR=...,WEAK_STORM=...,WEAK_SIZES=..." >&2
    exit 1
fi

source "$SUITE_DIR/common.sh"
suite_resolve_dirs
source "$HPC_INIT_SCRIPT"
export OMP_NUM_THREADS=1

set -e

WEAK_DIR="$(dirname "$RAW_DIR")/weak"
mkdir -p "$WEAK_DIR"

WORKTREE="$BUILD_DIR/weakseq_${SLURM_JOB_ID}"
trap 'suite_cleanup_worktree "$WORKTREE"' EXIT

suite_prepare_worktree "$SEQ_BRANCH" "$WORKTREE"
make -C "$WORKTREE" energy_storms_seq
BIN="$WORKTREE/energy_storms_seq"
test -x "$BIN"

echo "storm file: $WEAK_STORM"
echo "binary md5: $(suite_binary_fingerprint "$BIN")"

for size in $WEAK_SIZES; do
    out="$("$BIN" "$size" "$DUOWAVES_DIR/$WEAK_STORM" </dev/null)"
    result="$(suite_parse_result "$out")"
    if [ -z "$result" ]; then
        echo "size $size: no Result line, aborting" >&2
        exit 1
    fi
    printf '%s\n' "$result" > "$WEAK_DIR/baseline_${size}.result"
    echo "size $size: $(suite_parse_time "$out") s -> $result"
done

echo "--- BASELINES COMPLETE ---"
