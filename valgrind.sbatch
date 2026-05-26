#!/bin/bash
#SBATCH --job-name=mpi-valgrind
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --nodes=1               # 1 solo nodo per sicurezza Valgrind
#SBATCH --ntasks=4              # 4 processi totali
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --cpu-freq=Performance
#SBATCH --time=00:15:00
#SBATCH --output=sbatch_jobs/%j/%j.out
#SBATCH --error=sbatch_jobs/%j/%j.err

source /home/guest/init-hpc.sh
module load valgrind
set -e
cd "$SLURM_SUBMIT_DIR"

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

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

echo "--- ESECUZIONE VALGRIND (1 Nodo) ---"
# Utilizziamo rigorosamente il Makefile!
# Passiamo MPI_PROCS=$SLURM_NTASKS per sincronizzare il .mk con i 4 processi di Slurm
make -s run_valgrind MPI_PROCS=$SLURM_NTASKS

echo "--- RISULTATI ---"
echo "Esecuzione completata! Controlla i 4 file valgrind_log_rank_*.txt per il referto medico."
