#!/bin/bash
#SBATCH --job-name=mpi-pem
#SBATCH --output=sbatch_jobs/%j/out.txt
#SBATCH --error=sbatch_jobs/%j/err.txt
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --time=00:10:00

# (opzionale) carica i moduli necessari
module load openmpi

# lancia lo script bash
bash ./run_avg.sh
