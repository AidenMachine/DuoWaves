#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Launcher of the weak scaling campaign. The layer size is scaled in
# proportion to the core count, so every rung of the ladder keeps the same
# amount of work per core: an implementation that scales weakly takes the
# same time at every rung, and the curve is read against a flat line rather
# than against a falling one.
#
# Jobs are submitted one at a time and waited for, because the students QOS
# caps the number of jobs a user may have submitted at once and the whole
# ladder is only eight jobs. Rungs whose repetitions are already on disk are
# skipped, so an interrupted campaign resumes instead of repeating itself.
#
# Usage: run_weak_scaling.sh [--branch NAME] [--no-baseline] [--dry-run]
#

set -e
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
suite_resolve_dirs

BRANCH="Add_SIMD_Directives"
STORM="test_files/test_02_a30k_p20k_w1"
DRY_RUN=0
DO_BASELINE=1

# p | nodes | ranks per node | threads per rank | layer size
# 62500 cells per core throughout, and 4 threads per rank so that every rank
# owns exactly one NUMA domain of the EPYC 7301 nodes. The p=16 rung repeats
# the configuration 1n_4x4 over A_1M of the strong scaling campaign, so its
# time validates the whole ladder against a measurement already on disk.
LADDER="
4|1|1|4|250000
8|1|2|4|500000
16|1|4|4|1000000
32|1|8|4|2000000
64|2|8|4|4000000
128|4|8|4|8000000
"

# Sequential baselines are correctness references, not measurements. Only the
# sizes whose sequential run fits the 30 minute partition limit are computed:
# 4M and 8M cost about 3.5 and 7 minutes each and were dropped after the
# baseline job timed out. The two largest rungs record MISSING_BASELINE,
# which the payload handles and the report declares.
BASE_SMALL="250000 500000 1000000 2000000"
BASE_LARGE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --branch) BRANCH="$2"; shift ;;
        --no-baseline) DO_BASELINE=0 ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
    shift
done

#
# Submits one job and blocks until it leaves the queue, then verifies that it
# left it by completing. Serializing the campaign this way keeps at most one
# job submitted at a time, which is what the QOS allows, and makes a failure
# stop the ladder instead of letting the following rungs record measurements
# whose baseline never arrived.
#
submit_and_wait()
{
    local label="$1"
    shift
    local id state
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%q ' sbatch "$@"
        printf '\n'
        return 0
    fi
    until id="$(sbatch --parsable "$@")"; do
        echo "$(date +%H:%M:%S) $label: submission refused, retrying in 60s" >&2
        sleep 60
    done
    echo "$(date +%H:%M:%S) $label: job $id submitted, waiting"
    while squeue -j "$id" -h -o "%T" 2>/dev/null | grep -q .; do
        sleep 20
    done
    state="$(sacct -j "$id" --noheader -o State -X 2>/dev/null | tr -d ' ')"
    if [ "$state" != "COMPLETED" ]; then
        echo "$(date +%H:%M:%S) $label: job $id finished with state $state" >&2
        exit 1
    fi
    echo "$(date +%H:%M:%S) $label: job $id completed"
}

test -f "$DUOWAVES_DIR/$STORM" || { echo "missing storm file $STORM" >&2; exit 1; }

WEAK_DIR="$(dirname "$RAW_DIR")/weak"
mkdir -p "$WEAK_DIR"

#
# Counts how many runs are already recorded for the given P and layer size
# across every weak CSV on disk. grep prints zero and exits non zero when it
# matches nothing, so the exit status is swallowed rather than turned into a
# second zero on standard output, which used to make the caller compare a two
# line string as an integer.
#
weak_runs_on_disk()
{
    local p="$1" size="$2" n
    n="$(cat "$WEAK_DIR"/weak_*.csv 2>/dev/null | grep -c "^${p},${size},")" || true
    echo "${n:-0}"
}

#
# Filters a space separated list of layer sizes, keeping only those whose
# baseline result file is not on disk yet, so a relaunched campaign does not
# recompute references it already has.
#
filter_pending_baselines()
{
    local pending="" s
    for s in $1; do
        if [ ! -f "$WEAK_DIR/baseline_${s}.result" ]; then
            pending="${pending:+$pending }$s"
        else
            echo "$(date +%H:%M:%S) baseline for size $s already on disk, skipping" >&2
        fi
    done
    echo "$pending"
}

if [ "$DO_BASELINE" -eq 1 ]; then
    for raw_sizes in "$BASE_SMALL" "$BASE_LARGE"; do
        [ -z "$raw_sizes" ] && continue
        sizes="$(filter_pending_baselines "$raw_sizes")"
        [ -z "$sizes" ] && continue
        submit_and_wait "baseline ($sizes)" \
            --export=ALL,SUITE_DIR=$SUITE_DIR,WEAK_STORM=$STORM,WEAK_SIZES="$sizes" \
            "$SUITE_DIR/weak_baseline.sbatch"
    done
fi

N=0
SKIPPED=0
while IFS='|' read -r p n ppn thr size; do
    [ -z "$p" ] && continue
    done_runs="$(weak_runs_on_disk "$p" "$size")"
    if [ "$done_runs" -ge "$RUNS_PER_TEST" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo "$(date +%H:%M:%S) p=$p size=$size already has $done_runs runs, skipping"
        continue
    fi
    submit_and_wait "p=$p size=$size ${n}n x ${ppn} x ${thr}" \
        --nodes="$n" --ntasks-per-node="$ppn" --cpus-per-task="$thr" \
        --job-name="weak-p$p" \
        --export=ALL,SUITE_DIR=$SUITE_DIR,BRANCH=$BRANCH,WEAK_STORM=$STORM,LAYER_SIZE=$size,P=$p,MPI_TOTAL=$((n * ppn)),MPI_PPN=$ppn,OMP_THREADS=$thr \
        "$SUITE_DIR/weak_job.sbatch"
    N=$((N + 1))
done <<EOF
$LADDER
EOF

echo "$(date +%H:%M:%S) completed $N rung(s), skipped $SKIPPED on branch $BRANCH"
