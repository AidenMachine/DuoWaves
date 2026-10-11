#!/bin/bash
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/profiling/prof_%j.out
#SBATCH --error=logs/profiling/prof_%j.err

set -eo pipefail

REPO="$HOME/DuoWaves"
OUTDIR="$HOME/multicore-relation/results/profiling"
PROF_SRC="mpi_omp_core_profiling.c"
CORE_SRC="energy_storms_mpi_omp_core.c"
LAYOUTS="1x1 1x2 2x1 2x4 4x2 4x4 4x8 8x4"
RUNS=1

BRANCH="${PROF_BRANCH:?PROF_BRANCH not set}"
NODES="$SLURM_JOB_NUM_NODES"

: "${LD_LIBRARY_PATH:=}"; export LD_LIBRARY_PATH
source /home/guest/init-hpc.sh
set -u

mkdir -p "$OUTDIR"
CSV="$OUTDIR/prof_${BRANCH//\//_}_${NODES}n.csv"

# physical cores on this node (SMT-aware: the partition is heterogeneous)
PHYS=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
echo "physical cores/node: $PHYS   nodes: $NODES   branch: $BRANCH"

# --- disposable worktree; $REPO is never written to ---
mkdir -p "$HOME/.prof_worktrees"
WT="$HOME/.prof_worktrees/wt_${SLURM_JOB_ID}"
git -C "$REPO" worktree add --detach "$WT" "origin/$BRANCH"
trap 'cd /; git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"' EXIT
cd "$WT"

test -f "$PROF_SRC" || { echo "ABORT: $PROF_SRC missing on $BRANCH"; exit 1; }
cp "$PROF_SRC" "$CORE_SRC"
make clean >/dev/null 2>&1 || true
make energy_storms_mpi_omp
MD5=$(md5sum energy_storms_mpi_omp | cut -d' ' -f1)
echo "profiled binary md5: $MD5"

TF="$REPO/test_files/test_02_a30k_p20k"
CT="$REPO/custom-tests/test_02_a30k_p20k"
TESTS=(
  "A_20k|20000|${TF}_w1"
  "A_100k|100000|${TF}_w1"
  "A_1M|1000000|${TF}_w1"
  "B_10|20000|${TF}_w1 ${TF}_w2 ${TF}_w3 ${TF}_w4 ${TF}_w5 ${TF}_w6 ${CT}_w7 ${CT}_w8 ${CT}_w9 ${CT}_w10"
  "C_5k|20000|$REPO/custom-tests/test_13_a20k_p5k_w1"
)

field() { echo "$1" | grep -oP "$2" | tail -n 1; }

for layout in $LAYOUTS; do
    PPN="${layout%x*}"; THR="${layout#*x}"
    if [ $((PPN * THR)) -gt "$PHYS" ]; then
        echo "SKIP $layout: needs $((PPN*THR)) cores, node has $PHYS"; continue
    fi
    CONFIG="hyb_${NODES}n_${layout}"
    export OMP_NUM_THREADS="$THR"      # init-hpc.sh leaks 32
    TOTAL=$((PPN * NODES))

    for entry in "${TESTS[@]}"; do
        IFS='|' read -r tid size files <<<"$entry"
        for run in $(seq 1 "$RUNS"); do
            key="$BRANCH|$CONFIG|$tid|$run"
            grep -qsF "$key," "$CSV" && { echo "$key already recorded"; continue; }

            out=$(mpirun -np "$TOTAL" --map-by "ppr:${PPN}:node:PE=${THR}" \
                  --bind-to core ./energy_storms_mpi_omp "$size" $files </dev/null 2>&1) || {
                echo "$key FAILED"; echo "$out" | tail -20; continue; }

            wall=$(field "$out" 'Time:\s*\K[0-9eE.+-]+')
            bomb=$(field "$out" 'Fase di bombardamento:\s*\K[0-9.]+')
            exch=$(field "$out" 'Fase di scambio \(MPI\):\s*\K[0-9.]+')
            relx=$(field "$out" 'Fase di rilassamento:\s*\K[0-9.]+')
            redu=$(field "$out" 'Fase di reduction:\s*\K[0-9.]+')
            res=$(echo "$out" | grep -oP '^Result:\s*\K.*' | tail -n 1)

            if [ -z "$wall" ] || [ -z "$bomb" ]; then
                echo "$key PARSE FAILED"; echo "$out" | tail -20; continue; fi

            echo "$key,$wall,$bomb,$exch,$relx,$redu,$MD5,${SLURM_JOB_NODELIST:-?},$(date -Is),\"$res\"" >> "$CSV"
            echo "$key  wall=$wall  bomb=$bomb exch=$exch relax=$relx reduce=$redu"
        done
    done
done
echo "--- DONE -> $CSV ---"
