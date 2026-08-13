#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Slurm batch script performing a problem-size scalability test of the CUDA
# version of the high-energy particle storms simulation. Every input file in
# test_files is benchmarked on its own, and every test suite made of several
# wave files is additionally benchmarked with all of its waves interleaved in
# a single execution. Each benchmark is repeated RUNS times and reported as
# the arithmetic mean and the standard deviation of the measured wall clock
# times.
#
#SBATCH --job-name=cuda-scal
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=sbatch_jobs/%j/%j.out
#SBATCH --error=sbatch_jobs/%j/%j.err

source /home/guest/init-hpc.sh
set -e
cd "$SLURM_SUBMIT_DIR"

RUNS=20
BINARY=./energy_storms_cuda
TEST_DIR=test_files
summary=()

#
# Prints the Slurm allocation details together with the GPU visible to the
# job, so that every recorded measurement can be traced back to the exact
# hardware that produced it.
#
print_job_info()
{
    echo "=== JOB INFO ==="
    echo "Job ID:               $SLURM_JOB_ID"
    echo "Nodes:                $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
    echo "Tasks:                $SLURM_NTASKS"
    echo "CPUs per task:        $SLURM_CPUS_PER_TASK"
    echo "GPUs requested:       $SLURM_JOB_GPUS"
    echo "Repetitions per test: $RUNS"
    echo "================"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
}

#
# Rebuilds the CUDA binary from a clean tree so that the timings always refer
# to the sources currently present in the working directory.
#
build_cuda()
{
    echo "--- COMPILAZIONE ---"
    make clean || true
    make energy_storms_cuda
}

#
# Converts the compact array size notation used in the input file names into
# the plain number of layer positions expected by the simulator, expanding the
# k and M multipliers (30k becomes 30000, 100M becomes 100000000).
#
expand_size()
{
    local raw="$1"
    case "$raw" in
        *k) echo $(( ${raw%k} * 1000 )) ;;
        *M) echo $(( ${raw%M} * 1000000 )) ;;
        *)  echo "$raw" ;;
    esac
}

#
# Extracts from an input file name the layer sizes it is meant to be executed
# with, reading the a<size> token embedded in the name. Names carrying a range
# such as a16-17 yield both endpoints, since the test file documentation asks
# for those cases to be run at each extreme of the interval.
#
sizes_for_file()
{
    local token part
    token=$(basename "$1" | grep -oP '_a\K[^_]+')
    if [[ "$token" == *-* ]]; then
        for part in ${token//-/ }; do
            expand_size "$part"
        done
    else
        expand_size "$token"
    fi
}

#
# Lists every input file of the test suite, in a stable alphabetical order and
# without the documentation file shipped alongside them.
#
collect_test_files()
{
    find "$TEST_DIR" -type f -name 'test_*' | sort
}

#
# Lists the identifiers of the test suites present in the input directory, so
# that the wave files belonging to the same suite can be grouped together.
#
collect_suites()
{
    collect_test_files | xargs -n 1 basename | grep -oP '^test_[0-9]+' | sort -u
}

#
# Executes the CUDA binary RUNS times on the given layer size and storm files,
# collecting the wall clock time reported by each execution and appending the
# resulting mean and standard deviation to the global summary. The job is
# aborted as soon as one execution fails or omits its time, since a partial
# series would make the aggregated statistics meaningless.
#
run_benchmark()
{
    local label="$1" size="$2"
    shift 2
    local files=("$@")
    local times=() out t i avg stddev

    echo ""
    echo "--- $label (size $size, ${#files[@]} storm file(s)) ---"
    for i in $(seq 1 "$RUNS"); do
        if ! out=$("$BINARY" "$size" "${files[@]}" 2>&1); then
            echo "Run $i: esecuzione fallita. Output:"
            echo "$out"
            exit 1
        fi
        t=$(echo "$out" | grep -oP 'Time:\s*\K[0-9eE.+-]+' | tail -n 1)
        if [ -z "$t" ]; then
            echo "Run $i: tempo non trovato. Output:"
            echo "$out"
            exit 1
        fi
        times+=("$t")
    done

    avg=$(printf '%s\n' "${times[@]}" | awk '{s+=$1} END {printf "%.6f", s/NR}')
    stddev=$(printf '%s\n' "${times[@]}" | awk -v avg="$avg" '{s+=($1-avg)^2} END {printf "%.6f", sqrt(s/NR)}')

    echo "Tempi:          ${times[*]}"
    echo "Tempo medio:    $avg s"
    echo "Std dev:        $stddev s"
    summary+=("$label;$size;${#files[@]};$avg;$stddev")
}

#
# Benchmarks every input file in isolation, repeating the measurement at each
# of the layer sizes the file name declares.
#
run_single_file_benchmarks()
{
    local file size
    echo ""
    echo "=== TEST SINGOLI ==="
    for file in $(collect_test_files); do
        for size in $(sizes_for_file "$file"); do
            run_benchmark "$(basename "$file")" "$size" "$file"
        done
    done
}

#
# Benchmarks each test suite owning more than one wave file by interleaving all
# of its waves in a single execution, reproducing the combined workload the
# suite was designed to represent. Suites made of a single file are skipped
# because the single file runs already cover them.
#
run_suite_benchmarks()
{
    local suite files size
    echo ""
    echo "=== TEST COMBINATI PER SUITE ==="
    for suite in $(collect_suites); do
        files=($(collect_test_files | grep "/${suite}_"))
        if [ "${#files[@]}" -lt 2 ]; then
            continue
        fi
        for size in $(sizes_for_file "${files[0]}"); do
            run_benchmark "$suite (combinato)" "$size" "${files[@]}"
        done
    done
}

#
# Prints the whole set of collected measurements as a single table ordered as
# they were executed, giving a compact view of how the execution time evolves
# with the size of the simulated layer.
#
print_summary()
{
    local entry
    echo ""
    echo "=== RIEPILOGO SCALABILITA ($RUNS run per test) ==="
    printf '%-28s %12s %6s %14s %14s\n' "TEST" "SIZE" "WAVES" "MEDIA (s)" "STD DEV (s)"
    for entry in "${summary[@]}"; do
        IFS=';' read -r label size waves avg stddev <<< "$entry"
        printf '%-28s %12s %6s %14s %14s\n' "$label" "$size" "$waves" "$avg" "$stddev"
    done
}

print_job_info
build_cuda
run_single_file_benchmarks
run_suite_benchmarks
print_summary
