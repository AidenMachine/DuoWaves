#!/bin/bash
#SBATCH --job-name=Avg_32C
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00            # ATTENZIONE: Ho alzato il tempo a 30 minuti! 10 run impiegano di più!
#SBATCH --qos=students_limit
#SBATCH --output=risultati_avg_%j.txt
#SBATCH --error=errori_avg_%j.txt

cd ~/DuoWaves

# 1. Compilazione pulita (ASSICURATI CHE CI SIA IL -O3 NEL MAKEFILE)
make clean
make energy_storms_mpi_omp

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUNS=10
echo "--- INIZIO $RUNS ESECUZIONI A 32 CORE ---"

times=()
for i in $(seq 1 $RUNS); do
    
    # Lanciamo direttamente mpirun per avere il controllo totale sui parametri e non rischiare crash
    out=$(mpirun --bind-to none -np $SLURM_NTASKS ./energy_storms_mpi_omp 20000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6)
    
    # Estraiamo il tempo dall'output
    t=$(echo "$out" | grep -oP 'Time:\s*\K[0-9.eE+-]+' | tail -n1)
    
    if [ -z "$t" ]; then
        echo "Run $i: tempo non trovato. Output:"
        echo "$out"
        exit 1
    fi
    echo "Run $i completato: $t secondi"
    times+=("$t")
done

# Calcolo della media
avg=$(printf '%s\n' "${times[@]}" | awk '{s+=$1} END {printf "%.6f", s/NR}')
echo "----------------------------------------"
echo "--- TEMPO MEDIO SU $RUNS RUN: $avg s ---"
echo "----------------------------------------"
