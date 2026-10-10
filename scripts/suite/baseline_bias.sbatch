#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Diagnostic job answering one question the campaign cannot answer on its own:
# whether the sequential reference every speedup is divided by is slow for a
# reason that has nothing to do with parallelism.
#
# The suspicion comes from firstTry, whose kernel carries no omp pragma and no
# MPI call and differs from the sequential core only in allocating the two
# layer arrays itself. Measured as a hybrid branch it recorded 5.544 s on
# A_20k against a sequential baseline of 8.671 s, a factor of 1.56 for the
# same arithmetic on one active thread. Either the baseline is slow or the
# hybrid binaries are fast, and every speedup of the report carries that
# factor until it is explained.
#
# The job separates the candidate causes by measuring four variants of the
# same workload inside one allocation, so the node, its frequency and its
# neighbours are identical for all of them:
#
#   seq_gcc      the exact baseline recipe, energy_storms_seq built by gcc
#   seq_mpicc    the same sources built by mpicc, isolating the compiler
#   hyb_direct   the hybrid binary of the sequential-equivalent branch, run
#                without mpirun, isolating -fopenmp and the code shape
#   hyb_mpirun   the same binary under the launcher line the campaign uses,
#                isolating mpirun and its binding
#
# Reading the four medians in order tells which step introduces the factor:
# seq_gcc to seq_mpicc is the compiler driver, seq_mpicc to hyb_direct is the
# code and the OpenMP runtime, hyb_direct to hyb_mpirun is the launcher.
#
# Submit it through its launcher, which resolves the absolute paths of the
# suite and asks Slurm for the usual per-job log layout:
#
#   ./run_baseline_bias.sh
#
# The allocation is deliberately not exclusive, since the students partition
# does not always grant it and the job would queue indefinitely; the load of
# the node is reported instead, so a measurement taken on a busy node can be
# recognized and repeated.
#

#SBATCH --job-name=baseline-bias
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#
# TR-4, TR-5
# Resolves the suite directory without depending on the directory the job was
# submitted from. The launcher exports SUITE_DIR, which is the supported path;
# a job submitted by hand still finds common.sh by walking up from the
# submission directory. The first version of this payload derived the suite
# directory as SLURM_SUBMIT_DIR/scripts/suite and carried relative --output
# and --error directives, so submitting it from scripts/suite itself, the
# obvious place to submit it from, both looked for common.sh one level too
# deep and asked Slurm to write into a sbatch_jobs that was not there. It
# failed in one second, twice, before building anything.
#
if [ -z "${SUITE_DIR:-}" ]; then
    candidate="${SLURM_SUBMIT_DIR:-$PWD}"
    while [ "$candidate" != "/" ]; do
        if [ -f "$candidate/scripts/suite/common.sh" ]; then
            SUITE_DIR="$candidate/scripts/suite"
            break
        fi
        if [ -f "$candidate/common.sh" ]; then
            SUITE_DIR="$candidate"
            break
        fi
        candidate="$(dirname "$candidate")"
    done
fi
if [ ! -f "${SUITE_DIR:-}/common.sh" ]; then
    echo "cannot find common.sh: submit through run_baseline_bias.sh" >&2
    exit 1
fi

source "$SUITE_DIR/common.sh"
suite_resolve_dirs
source "$HPC_INIT_SCRIPT"
set -e

HYBRID_BRANCH="${HYBRID_BRANCH:-firstTry}"
TEST_SIZE="20000"
TEST_FILES="$DUOWAVES_DIR/test_files/test_02_a30k_p20k_w1"
REPETITIONS="${REPETITIONS:-5}"
SEQ_WORKTREE="$BUILD_DIR/bias_seq_${SLURM_JOB_ID}"
HYB_WORKTREE="$BUILD_DIR/bias_hyb_${SLURM_JOB_ID}"
REPORT="$RESULTS_DIR/analysis/baseline_bias.csv"
VARIANT_NAMES=""
VARIANT_MEDIANS=""

trap 'suite_cleanup_worktree "$SEQ_WORKTREE"; suite_cleanup_worktree "$HYB_WORKTREE"' EXIT

#
# ENV-META, MET-BASELINE
# Prints everything about the machine and the toolchain that could by itself
# explain a constant factor between two binaries running the same arithmetic:
# the identity of both compiler drivers, what mpicc actually wraps, the clock
# the node is running at, its governor and how busy it was when the job
# started.
#
print_environment()
{
    echo "=== JOB INFO ==="
    echo "Job ID:               $SLURM_JOB_ID"
    echo "Node:                 ${SLURM_JOB_NODELIST:-$(uname -n)}"
    echo "Sequential branch:    $SEQ_BRANCH"
    echo "Hybrid branch:        $HYBRID_BRANCH"
    echo "Workload:             size $TEST_SIZE, $(basename "$TEST_FILES")"
    echo "Repetitions:          $REPETITIONS"
    echo "================"
    echo ""
    echo "--- TOOLCHAIN ---"
    echo "gcc:    $(gcc --version | head -n 1)"
    echo "mpicc:  $(mpicc --version | head -n 1)"
    echo "mpicc wraps: $(mpicc -show 2>/dev/null || echo 'not reported')"
    echo ""
    echo "--- MACHINE ---"
    lscpu | grep -E "Model name|CPU\(s\)|MHz" || true
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'not readable')"
    echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"
}

#
# MET-BASELINE, CAM-PROVENANCE
# Builds one binary from a private worktree and prints the digest of what came
# out, after removing any previous build output so a rebuilt variant cannot
# accidentally reuse the object of the one measured before it. The remaining
# arguments are passed to make, which is how the same sources are built once
# by gcc and once by mpicc.
#
build_variant()
{
    local worktree="$1" target="$2"
    shift 2
    suite_purge_build_artifacts "$worktree"
    make -C "$worktree" "$target" "$@" >/dev/null
    test -x "$worktree/$target"
    echo "  built $target $* -> md5 $(suite_binary_fingerprint "$worktree/$target")"
}

#
# MET-BASELINE, MET-01
# Measures one variant: it executes the given command REPETITIONS times over
# the common workload, prints every wall clock time and the Result payload of
# the first execution, and keeps the median of the repetitions for the
# comparison table. The payload is printed so that four variants claiming to
# compute the same thing can be seen to actually compute it.
#
measure_variant()
{
    local name="$1"
    shift
    local times="" out t result median
    echo ""
    echo "--- VARIANT $name ---"
    for run in $(seq 1 "$REPETITIONS"); do
        out=$("$@" "$TEST_SIZE" "$TEST_FILES" 2>&1 </dev/null)
        t="$(suite_parse_time "$out")"
        if [ -z "$t" ]; then
            echo "  run $run: no time in the output, aborting this variant"
            echo "$out"
            return 0
        fi
        if [ "$run" -eq 1 ]; then
            result="$(suite_parse_result "$out")"
            echo "  result: $result"
        fi
        echo "  run $run: $t s"
        times="$times $t"
    done
    median="$(printf '%s\n' $times | sort -g | awk '{v[NR]=$1} END {print (NR%2) ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2}')"
    echo "  median: $median s"
    VARIANT_NAMES="$VARIANT_NAMES $name"
    VARIANT_MEDIANS="$VARIANT_MEDIANS $median"
    echo "$name,$median,$result" >> "$REPORT.tmp"
}

#
# MET-BASELINE, OUT-02
# Prints the comparison the job exists for: every variant next to its median
# and to its ratio against the sequential baseline recipe, which is the number
# the campaign divides by. A ratio of one says the variant is the baseline; the
# first ratio departing from one names the step that introduces the factor.
#
print_comparison()
{
    local reference name median index=0
    reference="$(printf '%s\n' $VARIANT_MEDIANS | head -n 1)"
    echo ""
    echo "--- COMPARISON (reference: the seq_gcc baseline recipe) ---"
    printf '  %-14s %14s %14s\n' "variant" "median [s]" "vs seq_gcc"
    set -- $VARIANT_MEDIANS
    for name in $VARIANT_NAMES; do
        index=$((index + 1))
        median="$(printf '%s\n' "$@" | sed -n "${index}p")"
        printf '  %-14s %14s %14s\n' "$name" "$median" \
            "$(awk -v a="$reference" -v b="$median" 'BEGIN {printf "%.3f", a/b}')"
    done
    echo ""
    echo "Reading: seq_gcc to seq_mpicc is the compiler driver, seq_mpicc to"
    echo "hyb_direct is the code and the OpenMP runtime, hyb_direct to"
    echo "hyb_mpirun is the launcher. The step whose ratio moves is the cause."
}

#
# MET-BASELINE, TR-3
# Entry point: takes both worktrees, builds the four variants and measures
# them one after the other inside the same allocation, so nothing but the
# variant itself changes between one measurement and the next.
#
main()
{
    mkdir -p "$RESULTS_DIR/analysis"
    rm -f "$REPORT.tmp"
    print_environment

    suite_prepare_worktree "$SEQ_BRANCH" "$SEQ_WORKTREE"
    suite_prepare_worktree "$HYBRID_BRANCH" "$HYB_WORKTREE"

    echo ""
    echo "--- BUILD ---"
    build_variant "$SEQ_WORKTREE" energy_storms_seq
    cp "$SEQ_WORKTREE/energy_storms_seq" "$SEQ_WORKTREE/energy_storms_seq_gcc"
    build_variant "$SEQ_WORKTREE" energy_storms_seq CC=mpicc
    cp "$SEQ_WORKTREE/energy_storms_seq" "$SEQ_WORKTREE/energy_storms_seq_mpicc"
    build_variant "$HYB_WORKTREE" energy_storms_mpi_omp

    export OMP_NUM_THREADS=1
    measure_variant seq_gcc "$SEQ_WORKTREE/energy_storms_seq_gcc"
    measure_variant seq_mpicc "$SEQ_WORKTREE/energy_storms_seq_mpicc"
    measure_variant hyb_direct "$HYB_WORKTREE/energy_storms_mpi_omp"

    export OMP_NUM_THREADS=2
    measure_variant hyb_mpirun mpirun -np 1 \
        --map-by "ppr:1:node:PE=2" --use-hwthread-cpus --bind-to hwthread \
        "$HYB_WORKTREE/energy_storms_mpi_omp"

    print_comparison

    echo "variant,median_s,result" > "$REPORT"
    cat "$REPORT.tmp" >> "$REPORT"
    rm -f "$REPORT.tmp"
    echo ""
    echo "wrote $REPORT"
    echo ""
    echo "--- JOB COMPLETE ---"
}

main
