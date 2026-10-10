#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Shared library of the multicore test suite. It is sourced both by the
# launchers running on the login node (or on the laptop for dry runs) and by
# the sbatch payloads running on the compute nodes, so everything in here has
# to work from any current working directory and must never depend on
# interactive shell features.
#

#
# TR-4
# Resolves every directory of the suite from the on-disk location of this
# file, assuming the committed layout where DuoWaves and multicore-relation
# are siblings inside the project folder. Exports the directories, sources
# suite.conf and creates the result folders, so any entry point can be
# launched from any working directory on any machine with the same layout.
#
suite_resolve_dirs()
{
    SUITE_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    RELATION_DIR="$(cd "$SUITE_DIR/../.." && pwd)"
    DUOWAVES_DIR="$(cd "$RELATION_DIR/../DuoWaves" && pwd)"
    RESULTS_DIR="$RELATION_DIR/results"
    RAW_DIR="$RESULTS_DIR/raw"
    BASELINE_DIR="$RESULTS_DIR/baselines"
    SBATCH_JOBS_DIR="$RELATION_DIR/sbatch_jobs"
    BUILD_DIR="$RELATION_DIR/build"
    export SUITE_DIR RELATION_DIR DUOWAVES_DIR RESULTS_DIR RAW_DIR BASELINE_DIR SBATCH_JOBS_DIR BUILD_DIR

    source "$SUITE_DIR/suite.conf"
    mkdir -p "$RAW_DIR" "$BASELINE_DIR" "$SBATCH_JOBS_DIR" "$BUILD_DIR"
}

#
# TM-A, TM-B, TM-C
# Prints the table of the eleven common tests, one per line, in the form
# "test_id|layer_size|relative input files". The A rows scale the layer size
# on a single test_02 wave, the B rows scale the number of cumulative storms
# at a fixed layer of 20000 positions using the six handout waves plus the
# four generated ones, and the C rows scale the particle count at a fixed
# layer of 20000 positions using the generated single-wave files. The order
# of the lines is the execution order of the payloads. The former A_100M row
# is deliberately absent: 20000 particles over 100 million positions is about
# 2e12 core updates, which no configuration of the campaign finishes inside
# the 30 minutes of the partition, so it only ever produced timed out jobs.
# The measurements it did produce before its removal stay in results/raw and
# keep their column in the spreadsheet.
#
suite_tests()
{
    local tf="test_files/test_02_a30k_p20k"
    local ct="custom-tests/test_02_a30k_p20k"
    echo "A_20k|20000|${tf}_w1"
    echo "A_100k|100000|${tf}_w1"
    echo "A_1M|1000000|${tf}_w1"
    echo "B_1|20000|${tf}_w1"
    echo "B_3|20000|${tf}_w1 ${tf}_w2 ${tf}_w3"
    echo "B_5|20000|${tf}_w1 ${tf}_w2 ${tf}_w3 ${tf}_w4 ${tf}_w5"
    echo "B_10|20000|${tf}_w1 ${tf}_w2 ${tf}_w3 ${tf}_w4 ${tf}_w5 ${tf}_w6 ${ct}_w7 ${ct}_w8 ${ct}_w9 ${ct}_w10"
    echo "C_3|20000|custom-tests/test_10_a20k_p3_w1"
    echo "C_100|20000|custom-tests/test_11_a20k_p100_w1"
    echo "C_1k|20000|custom-tests/test_12_a20k_p1k_w1"
    echo "C_5k|20000|custom-tests/test_13_a20k_p5k_w1"
}

#
# TM-A, TM-B, TM-C
# Verifies that every input file named by the common test table is present
# under DuoWaves, printing the missing ones and returning non zero when any
# is. The list is derived from the table itself rather than spot checked on
# two representatives, so a partially generated or partially copied
# custom-tests directory is caught by the launcher instead of being
# discovered inside the jobs, one wasted node allocation per configuration.
#
suite_check_input_files()
{
    local test_id size files f missing
    missing=""
    while IFS='|' read -r test_id size files; do
        for f in $files; do
            if [ ! -f "$DUOWAVES_DIR/$f" ]; then
                case " $missing " in
                    *" $f "*) ;;
                    *) missing="$missing $f" ;;
                esac
            fi
        done
    done < <(suite_tests)
    if [ -n "$missing" ]; then
        echo "missing input files:$missing" >&2
        echo "run scripts/suite/generate_inputs.py and commit DuoWaves/custom-tests" >&2
        return 1
    fi
}

#
# TM-A
# Tells whether the given test identifier appears in the colon separated
# EXCLUDE_TESTS list assembled by the --exclude options of the launchers,
# allowing campaigns to leave out tests that are known not to fit the time
# limit of the partition, such as A_1M on the least parallel layouts.
#
suite_test_excluded()
{
    case ":${EXCLUDE_TESTS:-}:" in
        *":$1:"*) return 0 ;;
        *) return 1 ;;
    esac
}

#
# TM-CORRECTNESS
# Derives the identifier of a workload from its layer size and the base names
# of its input files. Two tests with the same size and the same inputs, such
# as A_20k and B_1, map to the same identifier and therefore share one
# baseline file.
#
suite_workload_id()
{
    local size="$1"
    shift
    local names name
    names=""
    for name in "$@"; do
        names="$names $(basename "$name")"
    done
    echo "s${size}_$(echo "$size$names" | md5sum | cut -c1-10)"
}

#
# TM-CORRECTNESS
# Prints the path of the baseline file holding the sequential Result line of
# the given workload identifier.
#
suite_baseline_path()
{
    echo "$BASELINE_DIR/$1.result"
}

#
# CAM-RESUME
# Tells whether the run identified by the given key has already been recorded
# in any raw CSV file, allowing a resubmitted job to skip completed work and
# fit the remaining runs into the 30 minutes limit of the partition.
#
suite_run_recorded()
{
    grep -qsRF -- "$1," "$RAW_DIR"/ 2>/dev/null
}

#
# CAM-RESUME
# Prints the "section|branch|config" key of every configuration whose whole
# share of the campaign is already on disk, one per line. A configuration
# qualifies when the raw CSV files hold a measurement for each repetition of
# each test the current invocation is asking for, so the launchers can leave
# it out of the submission instead of spending a queue slot and a node
# allocation on a job that would only rediscover, run by run, that it has
# nothing left to measure.
#
# The whole answer is computed in one pass over the raw files, because the
# per-run suite_run_recorded lookup used inside the payloads greps the entire
# directory once per key and a campaign asks that question thousands of times.
# Only the tests of the current table are counted, and only repetitions inside
# the configured range, so measurements of retired tests such as the former
# A_100M row cannot make a configuration look finished, and the keys are
# deduplicated first so a repetition recorded twice cannot either.
#
suite_complete_configs()
{
    local wanted test_id
    wanted=""
    while IFS='|' read -r test_id _ _; do
        if suite_test_excluded "$test_id"; then
            continue
        fi
        wanted="${wanted:+$wanted,}$test_id"
    done < <(suite_tests)

    cat "$RAW_DIR"/*.csv 2>/dev/null | cut -d, -f1 | sort -u | awk -F'|' \
        -v tests="$wanted" -v runs="$RUNS_PER_TEST" '
        BEGIN {
            ntests = split(tests, list, ",")
            for (i = 1; i <= ntests; i++)
                wanted[list[i]] = 1
        }
        NF == 5 && ($4 in wanted) && $5 >= 1 && $5 <= runs {
            recorded[$1 "|" $2 "|" $3]++
        }
        END {
            for (key in recorded)
                if (recorded[key] == ntests * runs)
                    print key
        }'
}

#
# MET-01, OUT-01, CAM-PROVENANCE
# Appends one measurement to the raw CSV file owned by the current job. Every
# job writes only to its own file, named after its Slurm job id, so parallel
# jobs never contend on the same file and the .ods builder simply merges all
# of them. Row format:
# section|branch|config|test|run,time_s,correctness,job_id,nodes,timestamp,
# binary_md5,"result"
#
# The last two fields were added after a campaign whose branch columns turned
# out to measure a binary none of them had compiled. The digest names the
# executable the row came from, so two columns produced by the same executable
# are recognizable without rerunning anything, and the Result payload is kept
# verbatim, so a verdict can be re-examined, or recomputed under a different
# tolerance, from the file alone. The payload is quoted because it carries
# spaces; the fields before it keep their position, so the readers indexing
# the row by number are unaffected.
#
# The node list is quoted for a sharper reason. Slurm writes an allocation of
# several nodes as node[103,149], with a comma inside, and the rows of every
# multi node job were landing in the raw files with one field too many: the
# single node rows of the old campaign carry six fields, the four node ones
# up to eight. Nothing downstream noticed, because every reader indexes from
# the left and stops before the node list, but the digest and the payload sit
# after it, so leaving it unquoted would have shifted both on exactly the two
# and four node configurations of the rerun.
#
suite_record_run()
{
    local key="$1" time_s="$2" correctness="$3" result="$4"
    echo "$key,$time_s,$correctness,${SLURM_JOB_ID:-local},\"${SLURM_JOB_NODELIST:-localhost}\",$(date -Is),${SUITE_BINARY_MD5:-unknown},\"$result\"" \
        >> "$RAW_DIR/j${SLURM_JOB_ID:-local}.csv"
}

#
# CAM-PROVENANCE, ENV-META
# Prints the md5 digest of the binary under test. Every measurement carries
# it, so a column of the spreadsheet can be traced back to the exact
# executable that produced it and branches that unknowingly shared one
# executable cannot hide behind their labels.
#
suite_binary_fingerprint()
{
    md5sum "$1" | cut -d' ' -f1
}

#
# MET-01
# Extracts the wall clock time printed by the simulator from its captured
# standard output, using the same pattern as the historical batch scripts.
#
suite_parse_time()
{
    echo "$1" | grep -oP 'Time:\s*\K[0-9eE.+-]+' | tail -n 1
}

#
# TM-CORRECTNESS
# Extracts the payload of the Result line printed by the simulator, that is
# the alternating sequence of maximum positions and maximum values, one pair
# per storm file.
#
suite_parse_result()
{
    echo "$1" | grep -oP '^Result:\s*\K.*' | tail -n 1
}

#
# TM-CORRECTNESS
# Compares a Result payload against the sequential baseline payload of the
# same workload. Positions must match exactly and every maximum value must
# agree with the baseline within CORRECTNESS_REL_TOLERANCE, taken relative to
# the magnitude of the baseline value. Prints PASS, FAIL or MISSING_BASELINE.
#
# The comparison is relative because the layer is an array of float and the
# parallel sections reduce it in an order that depends on the number of
# ranks and threads. At the magnitude the campaign reaches, around 1.6e3, one
# ulp of float is already 1.22e-4, so the absolute 1e-5 this function used to
# apply demanded bit-identical output and reported as wrong every correct run
# of every configuration above two threads. Values below one are compared
# absolutely, since dividing by them would turn the rounding of a quantity
# close to zero into an arbitrarily large relative error.
#
suite_check_correctness()
{
    local wid="$1" result="$2"
    local baseline_file baseline
    baseline_file="$(suite_baseline_path "$wid")"
    if [ ! -f "$baseline_file" ]; then
        echo "MISSING_BASELINE"
        return
    fi
    baseline="$(cat "$baseline_file")"
    awk -v tol="$CORRECTNESS_REL_TOLERANCE" -v base="$baseline" -v got="$result" '
        BEGIN {
            nb = split(base, b, /[ \t]+/)
            ng = split(got, g, /[ \t]+/)
            if (nb != ng || nb == 0) { print "FAIL"; exit }
            for (i = 1; i <= nb; i += 2)
                if (b[i] != g[i]) { print "FAIL"; exit }
            for (i = 2; i <= nb; i += 2) {
                d = b[i] - g[i]
                if (d < 0) d = -d
                scale = b[i]
                if (scale < 0) scale = -scale
                if (scale < 1) scale = 1
                if (d > tol * scale) { print "FAIL"; exit }
            }
            print "PASS"
        }'
}

#
# TR-1, CAM-PROVENANCE
# Materializes the requested remote branch into a private detached git
# worktree for the current job, retrying a few times because concurrent jobs
# taking worktrees from the same shared clone briefly hold the repository
# lock. The build outputs the branch happens to track are purged from the
# fresh checkout before anything is built. Prints nothing and leaves the
# checkout at the given destination.
#
suite_prepare_worktree()
{
    local branch="$1" dest="$2" attempt
    for attempt in 1 2 3 4 5; do
        if git -C "$DUOWAVES_DIR" worktree add --detach "$dest" "origin/$branch" >/dev/null 2>&1; then
            suite_purge_build_artifacts "$dest"
            return 0
        fi
        sleep $(( (RANDOM % 5) + 2 ))
    done
    echo "unable to create worktree for origin/$branch at $dest" >&2
    return 1
}

#
# CAM-PROVENANCE
# Deletes from a freshly checked out worktree every build output the branch
# tracks, so the payload can only measure what it compiles itself.
#
# Several branches committed energy_storms_mpi_omp_core.o and the binaries
# linked from it. git gives a checked out file the mtime of the checkout, so
# the committed object is never older than its source and make declares it up
# to date: the job then links, or runs outright, an executable built on
# somebody's laptop months earlier instead of the branch under test. A whole
# campaign can be spent this way, three branch columns can end up holding one
# executable, and nothing in the logs says so. Removing the artifacts costs
# one compilation per job and makes that failure impossible.
#
suite_purge_build_artifacts()
{
    local dest="$1"
    rm -f "$dest"/*.o \
        "$dest/energy_storms_seq" \
        "$dest/energy_storms_mpi_omp" \
        "$dest/energy_storms_cuda"
}

#
# TR-1
# Removes the private worktree of the current job and prunes the bookkeeping
# of the shared clone, so aborted jobs do not accumulate stale checkouts.
#
suite_cleanup_worktree()
{
    local dest="$1"
    git -C "$DUOWAVES_DIR" worktree remove --force "$dest" >/dev/null 2>&1
    git -C "$DUOWAVES_DIR" worktree prune >/dev/null 2>&1
}

#
# CAM-THROTTLE
# Runs one sbatch submission, and whenever Slurm rejects it because the
# per-user submission cap of the qos is reached, waits and retries the same
# submission until a queue slot frees up. Any other sbatch failure is
# reported and aborts the launcher, since silently skipping a configuration
# would leave holes in the campaign. The first argument is a label for the
# messages, the remaining arguments are the full sbatch command.
#
# Every line carries a timestamp and the label of the configuration it talks
# about, and all of them are written to standard output, so the transcript of
# a launcher stays readable and correctly ordered whether it runs on a
# terminal or through a redirection. A blocked submission announces itself
# once and then only every SUITE_WAIT_NOTICE_RETRIES attempts, reporting the
# time waited so far, instead of repeating one identical line every retry
# interval: a campaign that waits an hour for the queue used to bury the few
# lines that report real progress under hundreds of identical ones.
#
suite_submit_with_retry()
{
    local label="$1" out attempts=0 waited=0
    shift
    while true; do
        if out=$("$@" 2>&1); then
            if [ "$attempts" -gt 0 ]; then
                echo "$(date +%H:%M:%S) $label: slot freed after ${waited}s"
            fi
            echo "$(date +%H:%M:%S) $label: ${out//$'\n'/ }"
            return 0
        fi
        case "$out" in
            *QOSMaxSubmitJobPerUserLimit*|*AssocMaxSubmitJobLimit*|*"job submit limit"*)
                attempts=$((attempts + 1))
                if [ "$attempts" -eq 1 ]; then
                    echo "$(date +%H:%M:%S) $label: queue full, waiting for a slot"
                elif [ $((attempts % SUITE_WAIT_NOTICE_RETRIES)) -eq 0 ]; then
                    echo "$(date +%H:%M:%S) $label: still waiting, ${waited}s so far"
                fi
                sleep "$SUBMIT_RETRY_SECONDS"
                waited=$((waited + SUBMIT_RETRY_SECONDS))
                ;;
            *)
                echo "$(date +%H:%M:%S) $label: submission failed, aborting the launcher"
                echo "$out"
                return 1
                ;;
        esac
    done
}

#
# TR-5
# Prints the sbatch options shared by every submission of the suite: the
# partition, the qos, the 30 minutes time limit and the per-job output files
# stored under sbatch_jobs in the same JobID/JobID.out and JobID/JobID.err
# format used by all the historical jobs of this repository.
#
suite_sbatch_common_args()
{
    echo "--partition=$SLURM_PARTITION"
    echo "--qos=$SLURM_QOS"
    echo "--time=$SLURM_TIME_LIMIT"
    echo "--chdir=$RELATION_DIR"
    echo "--output=$SBATCH_JOBS_DIR/%j/%j.out"
    echo "--error=$SBATCH_JOBS_DIR/%j/%j.err"
}
