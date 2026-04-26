#!/bin/bash
# Esegue energy_storms_mpi_omp 10 volte tramite make e calcola il tempo medio.

set -e

RUNS=10
TARGET=${1:-run_mpi}

cd "$(dirname "$0")"

echo "--- COMPILAZIONE ---"
make energy_storms_mpi_omp

echo "--- ESECUZIONE ($RUNS run, target: $TARGET) ---"

times=()
for i in $(seq 1 $RUNS); do
    out=$(make -s $TARGET)
    t=$(echo "$out" | grep -oP 'Time:\s*\K[0-9.eE+-]+' | tail -n1)
    if [ -z "$t" ]; then
        echo "Run $i: tempo non trovato. Output:"
        echo "$out"
        exit 1
    fi
    echo "Run $i: $t s"
    times+=("$t")
done

avg=$(printf '%s\n' "${times[@]}" | awk '{s+=$1} END {printf "%.6f", s/NR}')
echo "--- TEMPO MEDIO SU $RUNS RUN: $avg s ---"
