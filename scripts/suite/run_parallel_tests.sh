#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Launcher of the parallel test suite. Without options it submits every
# section over every branch selected for that section; with one or more of
# the section flags it submits only the requested sections. Every submission
# is one sbatch job measuring one (section, branch, configuration) triple,
# so the whole campaign is resumable at job granularity.
#
# Usage: run_parallel_tests.sh [--hybrid] [--cuda]
#                              [--branch NAME] [--exclude-branch NAME]
#                              [--exclude ID] [--force] [--dry-run]
#
# --branch NAME         restricts the submission to the given branch,
#                        repeatable (e.g. --branch main --branch immediate)
# --exclude-branch NAME leaves one branch out of the submission, repeatable
#                        (e.g. --exclude-branch firstTry)
# --exclude ID           leaves one common test out of every job, repeatable
#                        (e.g. --exclude A_1M)
# --force                submits also the configurations whose measurements
#                        are already complete, which are skipped by default
# --dry-run              prints the sbatch commands without submitting anything
#

set -e
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
suite_resolve_dirs

DRY_RUN=0
ONLY_BRANCHES=""
EXCLUDE_BRANCHES=""
SECTIONS=""
SUBMITTED=0
PLANNED=0
SKIPPED=0
FORCE=0
COMPLETE_CONFIGS=""

#
# TR-2
# Aborts when an option expecting a value was given without one, that is at
# the end of the command line or immediately before another flag. Without
# this check the trailing shift of the parser runs on an exhausted argument
# list, and its failure under set -e ends the launcher with status one and no
# message at all, which reads exactly like a campaign that had nothing to do.
#
require_value()
{
    case "${2:-}" in
        ""|-*) echo "option $1 requires a value" >&2; exit 1 ;;
    esac
}

#
# TR-2
# Adds a section to the selection unless it is already there. Repeating a
# section flag used to append the section twice, and every loop of the
# launcher walks the selection once per entry, so the whole campaign was
# submitted a second time: two jobs measuring one (section, branch,
# configuration) triple race over the same missing runs and record each of
# them twice, which is the one damage the resume mechanism cannot repair.
#
select_section()
{
    case " $SECTIONS " in
        *" $1 "*) return 0 ;;
    esac
    SECTIONS="$SECTIONS $1"
}

#
# TR-2
# Parses the command line of the launcher, collecting the sections to run,
# the optional branch include/exclude filters and the dry run switch. With no
# section flag every parallel section is selected, which realizes the
# requirement that the parallel tests run either all together or through a
# specific flag. --branch and --exclude-branch are both repeatable, so a
# campaign can be restricted to, or exclude, more than one branch, and --force
# turns off the skipping of the configurations already fully measured.
#
parse_arguments()
{
    while [ $# -gt 0 ]; do
        case "$1" in
            --hybrid) select_section HYB ;;
            --cuda)   select_section CUDA ;;
            --branch) require_value "$1" "${2:-}"; ONLY_BRANCHES="${ONLY_BRANCHES:+$ONLY_BRANCHES:}$2"; shift ;;
            --exclude-branch) require_value "$1" "${2:-}"; EXCLUDE_BRANCHES="${EXCLUDE_BRANCHES:+$EXCLUDE_BRANCHES:}$2"; shift ;;
            --exclude) require_value "$1" "${2:-}"; EXCLUDE_TESTS="${EXCLUDE_TESTS:+$EXCLUDE_TESTS:}$2"; shift ;;
            --force) FORCE=1 ;;
            --dry-run) DRY_RUN=1 ;;
            *) echo "unknown option $1" >&2; exit 1 ;;
        esac
        shift
    done
    if [ -z "$SECTIONS" ]; then
        SECTIONS="HYB CUDA"
    fi
}

#
# TR-2
# Prints, one per line, the branches carrying a column in the sections
# currently selected, which is the set the branch filters of the command line
# are meaningful against.
#
selected_section_branches()
{
    local section
    for section in $SECTIONS; do
        case "$section" in
            HYB)  printf '%s\n' $MPI_BRANCHES ;;
            CUDA) printf '%s\n' $CUDA_BRANCHES ;;
        esac
    done | sort -u
}

#
# TR-2
# Aborts when --branch or --exclude-branch names something no selected
# section measures, listing what it could have named. A name matching nothing
# is filtered out nowhere: it simply leaves the include list unsatisfiable,
# so a mistyped branch, or a branch belonging to the other section such as
# main under --hybrid, used to announce a campaign of zero jobs and exit
# successfully, indistinguishable from a campaign already complete.
#
check_branch_filters()
{
    local known name
    known="$(selected_section_branches)"
    for name in $(printf '%s' "$ONLY_BRANCHES:$EXCLUDE_BRANCHES" | tr ':' ' '); do
        if ! printf '%s\n' "$known" | grep -Fxq -- "$name"; then
            echo "unknown branch $name for the selected section(s):$(printf ' %s' $known)" >&2
            exit 1
        fi
    done
}

#
# TM-A, TM-B, TM-C
# Aborts when --exclude names a test outside the common table. An identifier
# matching nothing is silently ignored by every consumer of EXCLUDE_TESTS, so
# a typo left the campaign measuring the very test the invocation meant to
# leave out, and the launcher reported success either way.
#
check_excluded_tests()
{
    local known name
    known="$(suite_tests | cut -d'|' -f1)"
    for name in $(printf '%s' "${EXCLUDE_TESTS:-}" | tr ':' ' '); do
        if ! printf '%s\n' "$known" | grep -Fxq -- "$name"; then
            echo "unknown test $name, the common table holds:$(printf ' %s' $known)" >&2
            exit 1
        fi
    done
}

#
# TM-CORRECTNESS
# Warns when no sequential baseline has been published yet, because in that
# case every parallel run will be recorded with MISSING_BASELINE and the
# correctness block of the spreadsheet will stay empty until the sequential
# suite has been executed.
#
check_baselines()
{
    if ! ls "$BASELINE_DIR"/*.result >/dev/null 2>&1; then
        echo "WARNING: no sequential baselines found, run run_sequential_tests.sh first" >&2
    fi
}

#
# TR-1, TR-5, CAM-RESUME
# Submits one payload job through sbatch, combining the shared options of
# the suite with the resource options of the configuration and exporting the
# variables the payload needs. Configurations whose measurements are already
# complete are counted and left unsubmitted, so a campaign relaunched after a
# partial one only pays for what is missing. In dry run mode the full command
# is printed instead of being executed.
#
submit_job()
{
    local section="$1" branch="$2" config="$3" exports="$4"
    shift 4
    if config_complete "$section|$branch|$config"; then
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi
    local cmd=(sbatch)
    local arg
    while IFS= read -r arg; do
        cmd+=("$arg")
    done < <(suite_sbatch_common_args)
    cmd+=("--job-name=ts-$config" "$@")
    cmd+=("--export=ALL,SUITE_DIR=$SUITE_DIR,FORCE_RUN=$FORCE,EXCLUDE_TESTS=${EXCLUDE_TESTS:-},SECTION=$section,BRANCH=$branch,CONFIG=$config,$exports")
    cmd+=("$SUITE_DIR/parallel_job.sbatch")
    SUBMITTED=$((SUBMITTED + 1))
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        suite_submit_with_retry "[$SUBMITTED/$PLANNED] $branch/$config" "${cmd[@]}"
    fi
}

#
# TR-2
# Applies the optional branch filters of the command line: a branch is
# selected when it is not named by any --exclude-branch and, if at least one
# --branch was given, when it is also named by one of them.
#
branch_selected()
{
    local branch="$1"
    case ":${EXCLUDE_BRANCHES:-}:" in
        *":$branch:"*) return 1 ;;
    esac
    [ -z "$ONLY_BRANCHES" ] && return 0
    case ":$ONLY_BRANCHES:" in
        *":$branch:"*) return 0 ;;
        *) return 1 ;;
    esac
}

#
# CAM-RESUME
# Tells whether the given configuration already holds every measurement this
# invocation would ask of it, looking it up in the set collected once by
# suite_complete_configs. Under --force nothing is ever reported as complete,
# which is how a configuration gets measured again after its sources or the
# machine changed.
#
config_complete()
{
    if [ "$FORCE" -eq 1 ]; then
        return 1
    fi
    printf '%s\n' "$COMPLETE_CONFIGS" | grep -Fxq -- "$1"
}

#
# TR-2, CAM-RESUME
# Counts the jobs the current selection is going to submit, walking the same
# loops, the same branch filter and the same completeness check as the
# submission functions, so the position every submission carries is counted
# out of the work actually left to do. The count is what tells a launcher
# patiently queueing its first job apart from one that has stopped submitting
# altogether.
#
count_planned_jobs()
{
    local section branch nodes layout procs threads bs planned=0
    for section in $SECTIONS; do
        case "$section" in
            HYB)
                for branch in $MPI_BRANCHES; do
                    branch_selected "$branch" || continue
                    for nodes in $HYBRID_NODE_COUNTS; do
                        for layout in $HYBRID_LAYOUTS; do
                            procs="${layout%x*}"
                            threads="${layout#*x}"
                            config_complete "HYB|$branch|hyb_${nodes}n_${procs}x${threads}" && continue
                            planned=$((planned + 1))
                        done
                    done
                done
                ;;
            CUDA)
                for branch in $CUDA_BRANCHES; do
                    branch_selected "$branch" || continue
                    for bs in $CUDA_BLOCK_SIZES; do
                        config_complete "CUDA|$branch|cuda_bs${bs}" && continue
                        planned=$((planned + 1))
                    done
                done
                ;;
        esac
    done
    echo "$planned"
}

#
# TM-HYB
# Submits the hybrid section: every selected branch over one, two and four
# nodes, each with the seven prescribed per-node layouts of MPI processes
# times OpenMP threads.
#
# BUG: the export passes OMP_THREADS, which OpenMP ignores; the runtime
# reads OMP_NUM_THREADS, left at whatever init-hpc.sh set (32 on the
# cluster). Every rank spawns 32 threads regardless of the layout label.
# The launcher or the payload must export OMP_NUM_THREADS=$OMP_THREADS
# before calling mpirun.
#
submit_hybrid_section()
{
    local branch nodes layout procs threads
    for branch in $MPI_BRANCHES; do
        branch_selected "$branch" || continue
        for nodes in $HYBRID_NODE_COUNTS; do
            for layout in $HYBRID_LAYOUTS; do
                procs="${layout%x*}"
                threads="${layout#*x}"
                submit_job HYB "$branch" "hyb_${nodes}n_${procs}x${threads}" \
                    "MPI_TOTAL=$((nodes * procs)),MPI_PPN=$procs,OMP_THREADS=$threads" \
                    "--nodes=$nodes" "--ntasks-per-node=$procs" "--cpus-per-task=$threads"
            done
        done
    done
}

#
# TM-CUDA
# Submits the CUDA section: every selected branch rebuilt and measured at
# each of the six prescribed thread block sizes, on one GPU node.
#
submit_cuda_section()
{
    local branch bs
    for branch in $CUDA_BRANCHES; do
        branch_selected "$branch" || continue
        for bs in $CUDA_BLOCK_SIZES; do
            submit_job CUDA "$branch" "cuda_bs${bs}" \
                "MPI_TOTAL=1,MPI_PPN=1,OMP_THREADS=1,BLOCK_SIZE=$bs" \
                "--nodes=1" "--ntasks=1" "--cpus-per-task=1" "--gres=gpu:1"
        done
    done
}

#
# TR-2, TM-SHEETS, CAM-RESUME
# Entry point of the launcher: validates the environment, reads once which
# configurations are already fully measured and submits what is left of every
# selected section. The shared clone is assumed to already be up to date,
# since refreshing it from the remote is done manually before running the
# suite.
#
main()
{
    parse_arguments "$@"
    check_branch_filters
    check_excluded_tests
    suite_check_input_files
    check_baselines
    COMPLETE_CONFIGS="$(suite_complete_configs)"
    PLANNED="$(count_planned_jobs)"
    echo "$(date +%H:%M:%S) campaign of $PLANNED job(s) for sections:$( for s in $SECTIONS; do printf ' %s' "$s"; done )"

    local section
    for section in $SECTIONS; do
        case "$section" in
            HYB)  submit_hybrid_section ;;
            CUDA) submit_cuda_section ;;
        esac
    done
    if [ "$SUBMITTED" -eq 0 ] && [ "$SKIPPED" -eq 0 ]; then
        echo "$(date +%H:%M:%S) the selection matched no configuration at all, nothing was submitted" >&2
        exit 1
    fi
    echo "$(date +%H:%M:%S) submitted $SUBMITTED job(s), skipped $SKIPPED already complete"
}

main "$@"
