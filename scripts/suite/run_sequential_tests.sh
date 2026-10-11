#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Launcher of the sequential test suite, deliberately separate from the
# parallel launcher. It submits one sbatch job per distinct workload of the
# common test table, deduplicating the workloads that share layer size and
# input files. The sequential section exists to publish the correctness
# baselines the parallel suite compares against, and each workload is
# repeated SEQ_RUNS_PER_TEST times just like the parallel sections, so the
# Sequential sheet also carries a meaningful max, min and standard deviation
# instead of a single-point statistic. This launcher has to be run to
# completion before the parallel results can be verified.
#
# Usage: run_sequential_tests.sh [--test ID] [--exclude ID] [--dry-run] [--local]
#
# --test ID     restricts the submission to one test of the common table
# --exclude ID  leaves one test out, repeatable (e.g. --exclude A_1M)
# --dry-run     prints the commands without submitting or running anything
# --local       runs the payloads directly on this machine instead of sbatch
#

set -e
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
suite_resolve_dirs

DRY_RUN=0
LOCAL_RUN=0
ONLY_TEST=""
SUBMITTED=0
SEEN_WORKLOADS=""

#
# TR-3
# Parses the command line of the launcher, collecting the optional test
# filter, the repeatable exclusion list, the dry run switch and the local
# execution switch.
#
parse_arguments()
{
    while [ $# -gt 0 ]; do
        case "$1" in
            --test) ONLY_TEST="$2"; shift ;;
            --exclude) EXCLUDE_TESTS="${EXCLUDE_TESTS:+$EXCLUDE_TESTS:}$2"; shift ;;
            --dry-run) DRY_RUN=1 ;;
            --local) LOCAL_RUN=1 ;;
            *) echo "unknown option $1" >&2; exit 1 ;;
        esac
        shift
    done
}

#
# TM-SEQ
# Runs one sequential payload directly on the local machine, without Slurm,
# by invoking the very same payload script the cluster jobs execute and
# passing the workload assignment through the environment. In dry run mode
# the full command is printed instead of being executed.
#
run_local_job()
{
    local test_id="$1"
    local cmd=(env "SUITE_DIR=$SUITE_DIR" "TEST_ID=$test_id"
        bash "$SUITE_DIR/sequential_job.sbatch")
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
    SUBMITTED=$((SUBMITTED + 1))
}

#
# TR-1, TR-5
# Submits one sequential payload job through sbatch, combining the shared
# options of the suite with the export of the workload assignment. In dry
# run mode the full command is printed instead of being executed. In local
# mode the submission is replaced by a direct execution on this machine.
#
submit_job()
{
    local test_id="$1"
    if [ "$LOCAL_RUN" -eq 1 ]; then
        run_local_job "$test_id"
        return
    fi
    local cmd=(sbatch)
    local arg
    while IFS= read -r arg; do
        cmd+=("$arg")
    done < <(suite_sbatch_common_args)
    cmd+=("--job-name=ts-seq-$test_id")
    cmd+=("--export=ALL,SUITE_DIR=$SUITE_DIR,TEST_ID=$test_id")
    cmd+=("$SUITE_DIR/sequential_job.sbatch")
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        suite_submit_with_retry "seq/$test_id" "${cmd[@]}"
    fi
    SUBMITTED=$((SUBMITTED + 1))
}

#
# TR-3, TM-SEQ
# Entry point of the launcher: walks the common test table and submits one
# job per workload not measured under another test identifier yet. Since
# every distinct workload is covered by a single job regardless of the test
# identifiers that share it, the heavy sizes no longer need to be spread
# over several jobs to fit the time limit of the partition.
#
main()
{
    parse_arguments "$@"
    suite_check_input_files

    local test_id size files wid f abs_files
    while IFS='|' read -r test_id size files; do
        if [ -n "$ONLY_TEST" ] && [ "$test_id" != "$ONLY_TEST" ]; then
            continue
        fi
        if suite_test_excluded "$test_id"; then
            echo "skipping $test_id: excluded by request"
            continue
        fi
        abs_files=()
        for f in $files; do
            abs_files+=("$DUOWAVES_DIR/$f")
        done
        wid="$(suite_workload_id "$size" "${abs_files[@]}")"
        case " $SEEN_WORKLOADS " in
            *" $wid "*)
                echo "skipping $test_id: workload $wid already covered"
                continue
                ;;
        esac
        SEEN_WORKLOADS="$SEEN_WORKLOADS $wid"

        submit_job "$test_id"
    done < <(suite_tests)
    if [ "$LOCAL_RUN" -eq 1 ]; then
        echo "executed $SUBMITTED sequential payload(s) locally"
    else
        echo "submitted $SUBMITTED sequential job(s)"
    fi
}

main "$@"
