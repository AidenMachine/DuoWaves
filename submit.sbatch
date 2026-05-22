#!/bin/bash
#SBATCH --job-name=mpi-juice
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --cpu-freq=Performance
#SBATCH --time=00:15:00
#SBATCH --output=sbatch_jobs/%j/%j.out
#SBATCH --error=sbatch_jobs/%j/%j.err

source /home/guest/init-hpc.sh
set -e
cd "$SLURM_SUBMIT_DIR"

export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "=== JOB INFO ==="
echo "Job ID:               $SLURM_JOB_ID"
echo "Nodes:                $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
echo "Tasks (MPI):          $SLURM_NTASKS"
echo "Tasks per node:       $SLURM_NTASKS_PER_NODE"
echo "CPUs per task (OMP):  $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS:      $OMP_NUM_THREADS"
echo "================"

echo "--- COMPILAZIONE ---"
make clean || true
make energy_storms_mpi_omp

RUNS=10
echo "--- ESECUZIONE ($RUNS run) ---"
times=()
for i in $(seq 1 $RUNS); do
    out=$(make -s run_mpi)
    t=$(echo "$out" | grep -oP 'Time:\s*\K[0-9eE.+-]+' | tail -n1)
    if [ -z "$t" ]; then
        echo "Run $i: tempo non trovato. Output:"
        echo "$out"
        exit 1
    fi
    echo "Run $i: $t s"
    times+=("$t")
done

avg=$(printf '%s\n' "${times[@]}" | awk '{s+=$1} END {printf "%.6f", s/NR}')
stddev=$(printf '%s\n' "${times[@]}" | awk -v avg="$avg" '{s+=($1-avg)^2} END {printf "%.6f", sqrt(s/NR)}')

echo "--- RISULTATI ---"
echo "Tempo medio:    $avg s"
echo "Std dev:        $stddev s"
