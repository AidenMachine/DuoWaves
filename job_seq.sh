#!/bin/bash
#SBATCH --job-name=DuoWaves_Seq
#SBATCH --nodes=1                   # 1 solo nodo
#SBATCH --ntasks=1                  # 1 solo processo
#SBATCH --cpus-per-task=1           # 1 solo core (niente OpenMP)
#SBATCH --time=00:30:00             # Ho alzato a 30 min perché il sequenziale è più lento!
#SBATCH --qos=students_limit
#SBATCH --output=risultati_seq_%j.txt
#SBATCH --error=errori_seq_%j.txt

cd ~/DuoWaves

# 1. Pulisci e compila SOLO il programma sequenziale
make clean
make energy_storms_seq

echo "--- INIZIO ESECUZIONE SEQUENZIALE ---"

# 2. Lancia il programma nudo e crudo (NIENTE mpirun o srun)
./energy_storms_seq 20000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6

echo "--- ESECUZIONE COMPLETATA ---"
