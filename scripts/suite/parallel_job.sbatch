#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Generic Slurm payload of the parallel test suite. One job measures one
# hardware configuration of one section on one branch: it checks the branch
# out into a private worktree, builds the proper binary, executes the eleven
# common tests five times each, verifies every run against the sequential
# baseline and appends every measurement to its own raw CSV file. Already
# recorded runs are skipped, so resubmitting the same configuration resumes
# the work instead of repeating it. The job expects the launcher to provide
# SUITE_DIR, SECTION, BRANCH, CONFIG, MPI_TOTAL, MPI_PPN, OMP_THREADS and,
# for the CUDA section, BLOCK_SIZE, together with the matching resource
# options on the sbatch command line.
#
#SBATCH --job-name=suite-parallel
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --time=00:30:00

if [ -z "${SUITE_DIR:-}" ]; then
    echo "this payload must be submitted through run_parallel_tests.sh" >&2
    exit 1
fi

source "$SUITE_DIR/common.sh"
suite_resolve_dirs
if [ "${FORCE_RUN:-0}" = "1" ]; then
    # Keep fresh verification rows separate from the performance campaign.
    RAW_DIR="$RESULTS_DIR/verification/job_${SLURM_JOB_ID}"
    mkdir -p "$RAW_DIR"
    export RAW_DIR
fi
source "$HPC_INIT_SCRIPT"
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

set -e

WORKTREE="$BUILD_DIR/job_${SLURM_JOB_ID}"
trap 'suite_cleanup_worktree "$WORKTREE"' EXIT

#
# ENV-META
# Prints the identity of the job, of the measured configuration and of the
# exact commit under test, in the same style as the historical job logs, so
# every number in the spreadsheet can be traced back to its origin.
#
print_job_info()
{
    echo "=== JOB INFO ==="
    echo "Job ID:               $SLURM_JOB_ID"
    echo "Nodes:                $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
    echo "Section:              $SECTION"
    echo "Branch:               $BRANCH ($(git -C "$DUOWAVES_DIR" rev-parse --short "origin/$BRANCH"))"
    echo "Configuration:        $CONFIG"
    echo "MPI processes:        ${MPI_TOTAL:-1} (${MPI_PPN:-1} per node)"
    echo "OpenMP threads:       ${OMP_THREADS:-1}"
    echo "CUDA block size:      ${BLOCK_SIZE:-not applicable}"
    echo "Runs per test:        $RUNS_PER_TEST"
    echo "Fresh verification:   ${FORCE_RUN:-0}"
    echo "Measurement folder:   $RAW_DIR"
    echo "================"
    if [ "$SECTION" = "CUDA" ]; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    fi
}

#
# TM-CUDA
# Guarantees that the checked out CUDA sources accept a block size passed at
# compile time. The branches that predate the ifndef guard on main hardcode
# BLOCK_SIZE to 256, so on those branches the bare define is rewritten into
# its guarded form inside the private worktree only, leaving the repository
# untouched, and the rewrite is reported in the job log.
#
ensure_block_size_guard()
{
    local core="$WORKTREE/energy_storms_cuda_core.cu"
    if grep -q '#ifndef BLOCK_SIZE' "$core"; then
        return 0
    fi
    echo "NOTE: branch $BRANCH hardcodes BLOCK_SIZE, patching the worktree copy with an ifndef guard"
    sed -i 's/^#define BLOCK_SIZE 256$/#ifndef BLOCK_SIZE\n#define BLOCK_SIZE 256\n#endif/' "$core"
    grep -q '#ifndef BLOCK_SIZE' "$core"
}

#
# TM-MPI, TM-OMP, TM-HYB, TM-CUDA, CAM-PROVENANCE
# Builds the binary of the measured section inside the private worktree with
# the Makefile of the branch itself, so every column of the spreadsheet is
# produced by the build rules that branch was developed with. The CUDA build
# receives the block size of the configuration through CUDA_EXTRA_CFLAGS.
# The worktree reaches this point without any tracked build output, so the
# compilation really happens here, and the digest of what came out is
# published in the log and carried by every measurement of the job.
#
build_binary()
{
    echo "--- BUILD ---"
    if [ "$SECTION" = "CUDA" ]; then
        ensure_block_size_guard
        make -C "$WORKTREE" energy_storms_cuda CUDA_EXTRA_CFLAGS="-O3 -DBLOCK_SIZE=$BLOCK_SIZE"
        BINARY="$WORKTREE/energy_storms_cuda"
    else
        make -C "$WORKTREE" energy_storms_mpi_omp
        BINARY="$WORKTREE/energy_storms_mpi_omp"
    fi
    test -x "$BINARY"
    SUITE_BINARY_MD5="$(suite_binary_fingerprint "$BINARY")"
    export SUITE_BINARY_MD5
    echo "Binary md5:           $SUITE_BINARY_MD5"
}

#
# TM-MPI, TM-OMP, TM-HYB, TM-CUDA
# Executes one simulation of the given layer size over the given input files
# and prints its raw output. CPU sections are launched through mpirun with an
# explicit per-node mapping of MPI_PPN processes owning OMP_THREADS hardware
# threads each, which realizes the per-node layouts the sections prescribe;
# the CUDA section runs the binary directly inside its single-task
# allocation. Both forms take their standard input from /dev/null because
# mpirun forwards the standard input of the job to rank zero, and doing so
# would consume the test table the caller is iterating over, ending the walk
# after the first executed test.
#
# BUG: --use-hwthread-cpus makes PE= count hardware threads, not physical
# cores. On the 2-way SMT cluster nodes PE=4 maps a rank onto 2 physical
# cores, so every layout label overstates its core count by 2x. Combined
# with the OMP_THREADS bug in the launcher, oversubscription ranges from 8x
# to 64x depending on the layout and varies non-monotonically with the
# labelled core count.
#
execute_once()
{
    local size="$1"
    shift
    if [ "$SECTION" = "CUDA" ]; then
        "$BINARY" "$size" "$@" 2>&1 </dev/null
    else
        mpirun -np "$MPI_TOTAL" --report-bindings \
    --map-by "ppr:${MPI_PPN}:node:PE=${OMP_THREADS}" \
    --bind-to core \
    "$BINARY" "$size" "$@" </dev/null 2>&2
    fi
}

#
# MET-01, TM-RUNS, TM-CORRECTNESS, CAM-RESUME
# Measures one common test: it repeats the execution five times, extracts
# the wall clock time of every repetition, verifies the Result line of every
# repetition against the sequential baseline within the relative tolerance
# and records each repetition, Result payload included, in the raw CSV.
# Repetitions already present in the CSV are skipped so an interrupted job
# can be resubmitted, and the test is abandoned at the first failing
# execution because the remaining repetitions of a broken binary would fail
# identically.
#
run_test()
{
    local test_id="$1" size="$2"
    shift 2
    local files=() f wid key out t result verdict run
    for f in "$@"; do
        files+=("$DUOWAVES_DIR/$f")
    done
    wid="$(suite_workload_id "$size" "${files[@]}")"

    echo ""
    echo "--- TEST $test_id (size $size, ${#files[@]} storm file(s), workload $wid) ---"
    for run in $(seq 1 "$RUNS_PER_TEST"); do
        key="$SECTION|$BRANCH|$CONFIG|$test_id|$run"
        if suite_run_recorded "$key"; then
            echo "Run $run: already recorded, skipped"
            continue
        fi
        if ! out=$(execute_once "$size" "${files[@]}"); then
            echo "Run $run: execution failed, abandoning this test. Output:"
            echo "$out"
            return 0
        fi
        t="$(suite_parse_time "$out")"
        result="$(suite_parse_result "$out")"
        if [ -z "$t" ] || [ -z "$result" ]; then
            echo "Run $run: time or result not found, abandoning this test. Output:"
            echo "$out"
            return 0
        fi
        verdict="$(suite_check_correctness "$wid" "$result")"
        suite_record_run "$key" "$t" "$verdict" "$result"
        echo "Run $run: $t s, correctness $verdict"
    done
}

#
# TR-2, TM-SHEETS
# Entry point of the payload: prepares the worktree of the branch, builds the
# binary of the section and walks the whole common test table.
#
main()
{
    local line test_id size files
    print_job_info
    suite_prepare_worktree "$BRANCH" "$WORKTREE"
    build_binary
    while IFS='|' read -r test_id size files; do
        if suite_test_excluded "$test_id"; then
            echo ""
            echo "--- TEST $test_id excluded by request ---"
            continue
        fi
        run_test "$test_id" "$size" $files
    done < <(suite_tests)
    echo ""
    echo "--- JOB COMPLETE ---"
}

main
