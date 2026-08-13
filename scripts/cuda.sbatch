#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Slurm batch script for the CUDA version of the high-energy particle storms
# simulation. Allocates a single GPU node, rebuilds energy_storms_cuda from
# scratch and benchmarks it over the test_02 workload, reporting mean and
# standard deviation of the wall clock times.
#
#SBATCH --job-name=cuda-juice
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

RUNS=10

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
# Executes the CUDA binary $RUNS times through the run_cuda Makefile target,
# extracting the wall clock time printed by each run. Aborts the job as soon
# as one execution fails to report a time, since a partial series would make
# the aggregated statistics meaningless.
#
run_benchmark()
{
    echo "--- ESECUZIONE ($RUNS run) ---"
    times=()
    for i in $(seq 1 $RUNS); do
        out=$(make -s run_cuda)
        t=$(echo "$out" | grep -oP 'Time:\s*\K[0-9eE.+-]+' | tail -n1)
        if [ -z "$t" ]; then
            echo "Run $i: tempo non trovato. Output:"
            echo "$out"
            exit 1
        fi
        echo "Run $i: $t s"
        times+=("$t")
    done
}

#
# Aggregates the collected wall clock times into their arithmetic mean and
# population standard deviation, mirroring the reporting format used by the
# MPI+OpenMP batch scripts so the two versions stay directly comparable.
#
print_statistics()
{
    avg=$(printf '%s\n' "${times[@]}" | awk '{s+=$1} END {printf "%.6f", s/NR}')
    stddev=$(printf '%s\n' "${times[@]}" | awk -v avg="$avg" '{s+=($1-avg)^2} END {printf "%.6f", sqrt(s/NR)}')

    echo "--- RISULTATI ---"
    echo "Tempo medio:    $avg s"
    echo "Std dev:        $stddev s"
}

print_job_info
build_cuda
run_benchmark
print_statistics
