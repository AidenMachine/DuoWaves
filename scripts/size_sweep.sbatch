#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Slurm batch script measuring how the CUDA version of the high-energy particle
# storms simulation scales with the size of the layer alone. Unlike
# scalability.sbatch, which compares different input files and therefore varies
# the layer size and the particle count at the same time, this sweep keeps one
# single storm file fixed and only changes the number of layer positions passed
# on the command line, so that the measured times isolate the effect of the
# problem size. Each size is repeated RUNS times and reported as the arithmetic
# mean and the standard deviation of the measured wall clock times, in the same
# textual format produced by scalability.sbatch so that plots/plot_scalability.py
# can parse this output without any change.
#
#SBATCH --job-name=cuda-sweep
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=sbatch_jobs/%j/%j.out
#SBATCH --error=sbatch_jobs/%j/%j.err

source /home/guest/init-hpc.sh
set -e
cd "$SLURM_SUBMIT_DIR"

RUNS=20
BINARY=./energy_storms_cuda
SWEEP_FILE=test_files/test_02_a30k_p20k_w1
SIZES="30000 60000 100000 300000 600000 1000000 3000000 6000000 10000000"
summary=()

#
# Prints the Slurm allocation details together with the GPU visible to the job
# and the parameters of the sweep, so that every recorded measurement can be
# traced back to the exact hardware and workload that produced it.
#
print_job_info()
{
    echo "=== JOB INFO ==="
    echo "Job ID:               $SLURM_JOB_ID"
    echo "Nodes:                $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
    echo "Tasks:                $SLURM_NTASKS"
    echo "CPUs per task:        $SLURM_CPUS_PER_TASK"
    echo "GPUs requested:       $SLURM_JOB_GPUS"
    echo "File di storm:        $SWEEP_FILE"
    echo "Particelle:           $(head -n 1 "$SWEEP_FILE")"
    echo "Dimensioni:           $SIZES"
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
# Verifies that the storm file to sweep exists and reports whether all of its
# particles fall inside the smallest layer of the sweep. A particle sitting
# beyond the end of the layer is still simulated correctly, since the impact
# only contributes an attenuated energy to every cell, but it makes the smaller
# sizes represent a slightly different physical setup, so the condition is
# worth stating explicitly next to the measurements.
#
check_sweep_file()
{
    local max_position smallest_size

    if [ ! -f "$SWEEP_FILE" ]; then
        echo "File di storm non trovato: $SWEEP_FILE"
        exit 1
    fi

    max_position=$(tail -n +2 "$SWEEP_FILE" | awk 'NR == 1 || $1 > m {m = $1} END {print m}')
    smallest_size=$(echo "$SIZES" | tr ' ' '\n' | sort -n | head -n 1)

    echo ""
    echo "--- CONTROLLO INPUT ---"
    echo "Posizione massima delle particelle: $max_position"
    echo "Dimensione minima del sweep:        $smallest_size"
    if [ "$max_position" -ge "$smallest_size" ]; then
        echo "NOTA: alle dimensioni piu' piccole alcune particelle cadono fuori dal layer."
    else
        echo "Tutte le particelle rientrano in ogni dimensione del sweep."
    fi
}

#
# Executes the CUDA binary RUNS times on the given layer size with the storm
# file of the sweep, collecting the wall clock time reported by each execution
# and appending the resulting mean and standard deviation to the global summary.
# The job is aborted as soon as one execution fails or omits its time, since a
# partial series would make the aggregated statistics meaningless.
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
    summary+=("$size;$avg;$stddev")
}

#
# Runs the sweep proper, benchmarking the same storm file at every layer size
# of the list. The label reported for each point is the name of the swept file,
# so that all the measurements form a single series when they are plotted.
#
run_size_sweep()
{
    local size label
    label=$(basename "$SWEEP_FILE")

    echo ""
    echo "=== SWEEP SULLA DIMENSIONE ==="
    for size in $SIZES; do
        run_benchmark "$label" "$size" "$SWEEP_FILE"
    done
}

#
# Prints the collected measurements as a single table ordered by layer size,
# adding the time per layer position so that a constant column immediately
# shows the region where the execution time grows linearly with the problem
# size and the region where the fixed overhead still dominates.
#
print_summary()
{
    local entry size avg stddev

    echo ""
    echo "=== RIEPILOGO SWEEP ($RUNS run per dimensione) ==="
    printf '%14s %14s %14s %16s\n' "SIZE" "MEDIA (s)" "STD DEV (s)" "ns/posizione"
    for entry in "${summary[@]}"; do
        IFS=';' read -r size avg stddev <<< "$entry"
        printf '%14s %14s %14s %16s\n' \
            "$size" "$avg" "$stddev" \
            "$(awk -v a="$avg" -v s="$size" 'BEGIN {printf "%.4f", a * 1000000000 / s}')"
    done
}

print_job_info
build_cuda
check_sweep_file
run_size_sweep
print_summary
