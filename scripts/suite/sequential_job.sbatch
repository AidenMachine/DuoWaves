#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Slurm payload of the sequential test suite. One job measures one workload
# of the common test table with the reference sequential binary, whose
# sources are byte-identical on every branch and are therefore always built
# from the branch named in SEQ_BRANCH. The workload is executed
# SEQ_RUNS_PER_TEST times, mirroring the repeated measurement the parallel
# sections perform, so the Sequential sheet also carries a meaningful max,
# min and standard deviation instead of a single-point statistic. The Result
# line of the first execution is published as the correctness baseline of
# the workload, and the sequential program being deterministic, every
# further repetition is expected to reproduce it exactly. The job expects
# the launcher to provide SUITE_DIR and TEST_ID. The same script also runs
# outside Slurm, launched directly by the --local mode of the launcher:
# every Slurm-provided value then falls back to a local placeholder and the
# cluster init script is skipped when absent.
#
#SBATCH --job-name=suite-sequential
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

if [ -z "${SUITE_DIR:-}" ]; then
    echo "this payload must be submitted through run_sequential_tests.sh" >&2
    exit 1
fi

source "$SUITE_DIR/common.sh"
suite_resolve_dirs
if [ -f "$HPC_INIT_SCRIPT" ]; then
    source "$HPC_INIT_SCRIPT"
fi
set -e

JOB_TAG="${SLURM_JOB_ID:-local_$$}"
WORKTREE="$BUILD_DIR/job_${JOB_TAG}"
trap 'suite_cleanup_worktree "$WORKTREE"' EXIT

#
# ENV-META
# Prints the identity of the job, of the measured workload and of the exact
# commit the sequential binary is built from.
#
print_job_info()
{
    echo "=== JOB INFO ==="
    echo "Job ID:               ${SLURM_JOB_ID:-local}"
    echo "Nodes:                ${SLURM_JOB_NUM_NODES:-1} (${SLURM_JOB_NODELIST:-$(uname -n)})"
    echo "Section:              SEQ"
    echo "Branch:               $SEQ_BRANCH ($(git -C "$DUOWAVES_DIR" rev-parse --short "origin/$SEQ_BRANCH"))"
    echo "Test:                 $TEST_ID"
    echo "Repetitions:          $SEQ_RUNS_PER_TEST"
    echo "================"
}

#
# TM-SEQ, CAM-PROVENANCE
# Builds the reference sequential binary inside the private worktree of the
# job using the Makefile of the reference branch. The worktree arrives here
# stripped of any tracked build output, so the baselines the whole campaign
# is checked against come from sources compiled on the measuring machine,
# and the digest of the binary is published and carried by every row.
#
build_binary()
{
    echo "--- BUILD ---"
    make -C "$WORKTREE" energy_storms_seq
    BINARY="$WORKTREE/energy_storms_seq"
    test -x "$BINARY"
    SUITE_BINARY_MD5="$(suite_binary_fingerprint "$BINARY")"
    export SUITE_BINARY_MD5
    echo "Binary md5:           $SUITE_BINARY_MD5"
}

#
# TM-CORRECTNESS
# Publishes the given Result payload as the baseline of the workload if no
# baseline exists yet, writing through a temporary file and a non-clobbering
# move so concurrent jobs of the same workload cannot corrupt it, and records
# the mapping in a human readable manifest. If a baseline already exists the
# new payload is compared against it and any divergence is reported loudly,
# since the sequential program is deterministic and two honest runs of the
# same workload can never disagree.
#
publish_baseline()
{
    local wid="$1" result="$2" descr="$3"
    local baseline_file tmp
    baseline_file="$(suite_baseline_path "$wid")"
    if [ ! -f "$baseline_file" ]; then
        tmp="$baseline_file.tmp.$JOB_TAG"
        printf '%s\n' "$result" > "$tmp"
        mv -n "$tmp" "$baseline_file"
        rm -f "$tmp"
        echo "$wid,$descr" >> "$BASELINE_DIR/MANIFEST.csv"
        echo "Baseline published: $baseline_file"
    fi
    if [ "$(cat "$baseline_file")" != "$result" ]; then
        echo "WARNING: sequential result differs from the stored baseline of $wid" >&2
    fi
}

#
# MET-01, TM-SEQ, TM-CORRECTNESS, CAM-RESUME
# Measures the workload of this job: it repeats the sequential run
# SEQ_RUNS_PER_TEST times, publishes the Result line of the first execution
# as the baseline of the workload, verifies every repetition against that
# baseline and records each repetition's wall clock time in the raw CSV
# under the SEQ section. Repetitions already present in the CSV are skipped,
# so a resubmitted job does not pay again for a measurement the campaign
# already owns, and the workload is abandoned at the first failing execution
# because the remaining repetitions of a broken binary would fail
# identically.
#
run_workload()
{
    local test_id="$1" size="$2"
    shift 2
    local files=() f wid key out t result run verdict
    for f in "$@"; do
        files+=("$DUOWAVES_DIR/$f")
    done
    wid="$(suite_workload_id "$size" "${files[@]}")"

    echo ""
    echo "--- TEST $test_id (size $size, ${#files[@]} storm file(s), workload $wid) ---"
    for run in $(seq 1 "$SEQ_RUNS_PER_TEST"); do
        key="SEQ|$SEQ_BRANCH|seq|$test_id|$run"
        if suite_run_recorded "$key"; then
            echo "Run $run: already recorded, skipped"
            continue
        fi
        if ! out=$("$BINARY" "$size" "${files[@]}" 2>&1); then
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
        publish_baseline "$wid" "$result" "$test_id size=$size files=$*"
        verdict="$(suite_check_correctness "$wid" "$result")"
        suite_record_run "$key" "$t" "$verdict" "$result"
        echo "Run $run: $t s, correctness $verdict"
    done
}

#
# TR-3, TM-SEQ
# Entry point of the payload: prepares the worktree of the reference branch,
# builds the sequential binary and measures the single workload assigned to
# this job.
#
main()
{
    local line test_id size files found
    print_job_info
    suite_prepare_worktree "$SEQ_BRANCH" "$WORKTREE"
    build_binary
    found=0
    while IFS='|' read -r test_id size files; do
        if [ "$test_id" = "$TEST_ID" ]; then
            found=1
            run_workload "$test_id" "$size" $files
        fi
    done < <(suite_tests)
    if [ "$found" -eq 0 ]; then
        echo "unknown test id $TEST_ID" >&2
        exit 1
    fi
    echo ""
    echo "--- JOB COMPLETE ---"
}

main
