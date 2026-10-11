#!/bin/bash
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/profiling/cuda_prof_%j.out
#SBATCH --error=logs/profiling/cuda_prof_%j.err

set -eo pipefail
REPO="$HOME/DuoWaves"
OUTDIR="$HOME/multicore-relation/results/profiling"
TF="$REPO/test_files/test_02_a30k_p20k"
CT="$REPO/custom-tests"
BS=256
RUNS=1

TESTS=(
  "A_20k|20000|${TF}_w1"
  "A_100k|100000|${TF}_w1"
  "A_1M|1000000|${TF}_w1"
  "B_3|20000|${TF}_w1 ${TF}_w2 ${TF}_w3"
  "B_10|20000|${TF}_w1 ${TF}_w2 ${TF}_w3 ${TF}_w4 ${TF}_w5 ${TF}_w6 ${CT}/test_02_a30k_p20k_w7 ${CT}/test_02_a30k_p20k_w8 ${CT}/test_02_a30k_p20k_w9 ${CT}/test_02_a30k_p20k_w10"
  "C_1k|20000|${CT}/test_12_a20k_p1k_w1"
  "C_5k|20000|${CT}/test_13_a20k_p5k_w1"
)

: "${LD_LIBRARY_PATH:=}"; export LD_LIBRARY_PATH
source /home/guest/init-hpc.sh
set -u
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

for entry in "${TESTS[@]}"; do
  IFS='|' read -r tid size files <<<"$entry"
  for f in $files; do
    [ -f "$f" ] || { echo "ABORT: missing input $f"; exit 1; }
  done
done

mkdir -p "$OUTDIR" "$HOME/.prof_worktrees"
WT="$HOME/.prof_worktrees/cudaprof_${SLURM_JOB_ID}"
field() { echo "$1" | grep -oP "$2" | tail -n 1; }

for BRANCH in cuda-1D cuda-shared; do
  echo; echo "################ $BRANCH ################"
  CSV="$OUTDIR/prof_${BRANCH}_bs${BS}.csv"
  rm -rf "$WT"; git -C "$REPO" worktree prune
  git -C "$REPO" worktree add --detach "$WT" "origin/$BRANCH"
  cd "$WT"
  git log --oneline -1

  test -f cuda_core_profiler.cu || { echo "ABORT: cuda_core_profiler.cu missing on $BRANCH"; exit 1; }
  cp cuda_core_profiler.cu energy_storms_cuda_core.cu

  # same guard the suite applies, so -DBLOCK_SIZE takes effect
  grep -q '#ifndef BLOCK_SIZE' energy_storms_cuda_core.cu || \
    sed -i 's/^#define BLOCK_SIZE 256$/#ifndef BLOCK_SIZE\n#define BLOCK_SIZE 256\n#endif/' \
        energy_storms_cuda_core.cu

  make clean >/dev/null 2>&1 || true
  make energy_storms_cuda CUDA_EXTRA_CFLAGS="-O3 -DBLOCK_SIZE=$BS"
  MD5=$(md5sum energy_storms_cuda | cut -d' ' -f1)
  echo "profiled binary md5: $MD5"

  for entry in "${TESTS[@]}"; do
    IFS='|' read -r tid size files <<<"$entry"
    for run in $(seq 1 "$RUNS"); do
      key="$BRANCH|cuda_bs${BS}|$tid|$run"
      grep -qsF "$key," "$CSV" && { echo "$key already recorded"; continue; }

      out=$(./energy_storms_cuda "$size" $files 2>&1 </dev/null) || {
        echo "$key FAILED"; echo "$out" | tail -10; continue; }

      wall=$(field "$out" 'Time:\s*\K[0-9eE.+-]+')
      setup=$(field "$out" 'PROFILE:.*setup=\K[0-9.]+')
      bomb=$(field "$out" 'PROFILE:.*bomb=\K[0-9.]+')
      relax=$(field "$out" 'PROFILE:.*relax=\K[0-9.]+')
      redu=$(field "$out" 'PROFILE:.*reduce=\K[0-9.]+')
      d2h=$(field "$out" 'PROFILE:.*d2h=\K[0-9.]+')
      res=$(echo "$out" | grep -oP '^Result:\s*\K.*' | tail -n 1)

      if [ -z "$wall" ] || [ -z "$bomb" ]; then
        echo "$key PARSE FAILED"; echo "$out" | tail -10; continue; fi

      echo "$key,$wall,$setup,$bomb,$relax,$redu,$d2h,$MD5,${SLURM_JOB_NODELIST:-?},$(date -Is),\"$res\"" >> "$CSV"
      echo "$key  wall=$wall setup=$setup bomb=$bomb relax=$relax reduce=$redu d2h=$d2h"
    done
  done
  cd /; git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
done
echo; echo "=== DONE -> $OUTDIR ==="
