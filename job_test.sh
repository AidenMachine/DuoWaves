#!/bin/bash
#SBATCH --job-name=DuoWaves_Run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --qos=students_limit
#SBATCH --output=risultati_%j.txt
#SBATCH --error=errori_%j.txt

cd ~/DuoWaves

make clean
make energy_storms_mpi_omp

export OMP_NUM_THREADS=4

echo "--- COMPILAZIONE FINITA. INIZIO ESECUZIONE ---"

# Ho messo -np 2 forzato per evitare scherzi di sistema
mpirun --bind-to none -np 8 ./energy_storms_mpi_omp 20000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6

echo "--- ESECUZIONE COMPLETATA CON SUCCESSO ---"
