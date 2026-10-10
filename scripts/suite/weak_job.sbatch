#!/bin/bash
#SBATCH --job-name=weak
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --time=00:30:00
#SBATCH --output=sbatch_jobs/weak_%j.out
#SBATCH --error=sbatch_jobs/weak_%j.err

if [ -z "${SUITE_DIR:-}" ]; then
    echo "submit through the ladder loop" >&2
    exit 1
fi

source "$SUITE_DIR/common.sh"
suite_resolve_dirs
source "$HPC_INIT_SCRIPT"

# Must come after the init script: it sets OMP_NUM_THREADS=32 for every job
# on this cluster, which would silently oversubscribe every rung.
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

set -e

WEAK_DIR="$(dirname "$RAW_DIR")/weak"
mkdir -p "$WEAK_DIR"
BASE="$WEAK_DIR/baseline_${LAYER_SIZE}.result"
CSV="$WEAK_DIR/weak_j${SLURM_JOB_ID}.csv"

WORKTREE="$BUILD_DIR/weak_${SLURM_JOB_ID}"
trap 'suite_cleanup_worktree "$WORKTREE"' EXIT

suite_prepare_worktree "$BRANCH" "$WORKTREE"
make -C "$WORKTREE" energy_storms_mpi_omp
BIN="$WORKTREE/energy_storms_mpi_omp"
test -x "$BIN"
MD5="$(suite_binary_fingerprint "$BIN")"

echo "=== WEAK SCALING RUNG ==="
echo "cores (p):      $P"
echo "nodes:          $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
echo "ranks total:    $MPI_TOTAL ($MPI_PPN per node)"
echo "threads/rank:   $OMP_THREADS"
echo "layer size:     $LAYER_SIZE"
echo "cells per core: $((LAYER_SIZE / P))"
echo "binary md5:     $MD5"
echo "========================="

check_result()
{
    if [ ! -f "$BASE" ]; then
        echo "MISSING_BASELINE"
        return 0
    fi
    awk -v a="$(cat "$BASE")" -v b="$1" 'BEGIN {
        na = split(a, A, " "); nb = split(b, B, " ");
        if (na != nb) { print "MISMATCH"; exit }
        for (i = 1; i <= na; i += 2)
            if (A[i] != B[i]) { print "FAIL_POSITION"; exit }
        for (i = 2; i <= na; i += 2) {
            d = A[i] - B[i]; if (d < 0) d = -d;
            m = A[i];        if (m < 0) m = -m;
            if (m > 0 && d / m > 1e-5) { print "FAIL_VALUE"; exit }
        }
        print "PASS"
    }'
}

for run in 1 2 3 4 5; do
    out="$(mpirun -np "$MPI_TOTAL" --report-bindings \
        --map-by "ppr:${MPI_PPN}:node:PE=${OMP_THREADS}" \
        --bind-to core \
        "$BIN" "$LAYER_SIZE" "$DUOWAVES_DIR/$WEAK_STORM" </dev/null 2>&2)"
    t="$(suite_parse_time "$out")"
    r="$(suite_parse_result "$out")"
    if [ -z "$t" ] || [ -z "$r" ]; then
        echo "run $run: time or result not found, abandoning"
        echo "$out"
        break
    fi
    v="$(check_result "$r")"
    echo "$P,$LAYER_SIZE,$MPI_PPN,$OMP_THREADS,$run,$t,$v,$MD5,\"$SLURM_JOB_NODELIST\"" >> "$CSV"
    echo "run $run: $t s, correctness $v"
done

echo "--- RUNG COMPLETE ---"
