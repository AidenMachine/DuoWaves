#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Launcher of the sequential baseline bias check. It exists for the same
# reason the other launchers do: the payload must not have to guess where the
# suite lives, and the job logs must land in sbatch_jobs/JobID/JobID.out
# whatever directory the submission was made from.
#
# Usage: run_baseline_bias.sh [--branch NAME] [--repetitions N] [--dry-run]
#
# --branch NAME      the sequential-equivalent branch to measure as hybrid,
#                     firstTry by default
# --repetitions N    repetitions per variant, 5 by default
# --dry-run          prints the sbatch command without submitting it
#

set -e
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
suite_resolve_dirs

DRY_RUN=0
HYBRID_BRANCH="firstTry"
REPETITIONS="5"

#
# TR-2
# Parses the command line, which only carries the two knobs the check has and
# the dry run switch shared with the other launchers.
#
parse_arguments()
{
    while [ $# -gt 0 ]; do
        case "$1" in
            --branch) HYBRID_BRANCH="$2"; shift ;;
            --repetitions) REPETITIONS="$2"; shift ;;
            --dry-run) DRY_RUN=1 ;;
            *) echo "unknown option $1" >&2; exit 1 ;;
        esac
        shift
    done
}

#
# TR-1, TR-5, MET-BASELINE
# Submits the single diagnostic job, giving it the shared options of the suite
# so its log lands in the usual place, and exporting the absolute suite
# directory together with the two knobs, so the payload never has to derive
# anything from the directory the submission happened in.
#
main()
{
    parse_arguments "$@"
    if ! git -C "$DUOWAVES_DIR" rev-parse --verify "origin/$HYBRID_BRANCH" >/dev/null 2>&1; then
        echo "unknown branch origin/$HYBRID_BRANCH" >&2
        exit 1
    fi

    local cmd=(sbatch)
    local arg
    while IFS= read -r arg; do
        cmd+=("$arg")
    done < <(suite_sbatch_common_args)
    cmd+=("--job-name=baseline-bias")
    cmd+=("--export=ALL,SUITE_DIR=$SUITE_DIR,HYBRID_BRANCH=$HYBRID_BRANCH,REPETITIONS=$REPETITIONS")
    cmd+=("$SUITE_DIR/baseline_bias.sbatch")

    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        suite_submit_with_retry "baseline-bias/$HYBRID_BRANCH" "${cmd[@]}"
    fi
}

main "$@"
