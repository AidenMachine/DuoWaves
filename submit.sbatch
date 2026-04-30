#!/bin/bash
#SBATCH --job-name=energy_storms
#SBATCH --partition=students
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=30
#SBATCH --time=00:29:00
#SBATCH --output=energy_storms_%j.out
#SBATCH --error=energy_storms_%j.err

set -e

# OpenMP: usa il numero di CPU allocate da SLURM per task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

cd "$SLURM_SUBMIT_DIR"

echo "=== JOB INFO ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
echo "MPI tasks:     $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo "================"

echo "--- COMPILAZIONE ---"
make energy_storms_mpi_omp

RUNS=10
TEST_FILES="test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6"

echo "--- ESECUZIONE ($RUNS run) ---"
times=()
for i in $(seq 1 $RUNS); do
    out=$(srun --mpi=pmix ./energy_storms_mpi_omp 20000 $TEST_FILES)
    t=$(echo "$out" | grep -oP 'Time:\s*\K[0-9eE.+-]+' | tail -n1)
    if [ -z "$t" ]; then
        echo "Run $i: tempo non trovato. Output completo:"
        echo "$out"
        exit 1
    fi
    echo "Run $i: $t s"
    times+=("$t")
done

avg=$(printf '%s\n' "${times[@]}" | awk '{s+=$1} END {printf "%.6f", s/NR}')
echo "--- TEMPO MEDIO SU $RUNS RUN: $avg s ---"
