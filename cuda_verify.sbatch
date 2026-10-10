#!/bin/bash
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/profiling/cuda_verify_%j.out
#SBATCH --error=logs/profiling/cuda_verify_%j.err

set -eo pipefail
REPO="$HOME/DuoWaves"
TF="$REPO/test_files/test_02_a30k_p20k"
CT="$REPO/custom-tests"

TESTS=(
  "A_20k |20000  |${TF}_w1"
  "A_100k|100000 |${TF}_w1"
  "A_1M  |1000000|${TF}_w1"
  "B_3   |20000  |${TF}_w1 ${TF}_w2 ${TF}_w3"
  "B_10  |20000  |${TF}_w1 ${TF}_w2 ${TF}_w3 ${TF}_w4 ${TF}_w5 ${TF}_w6 ${CT}/test_02_a30k_p20k_w7 ${CT}/test_02_a30k_p20k_w8 ${CT}/test_02_a30k_p20k_w9 ${CT}/test_02_a30k_p20k_w10"
  "C_1k  |20000  |${CT}/test_12_a20k_p1k_w1"
  "C_5k  |20000  |${CT}/test_13_a20k_p5k_w1"
)

: "${LD_LIBRARY_PATH:=}"; export LD_LIBRARY_PATH
source /home/guest/init-hpc.sh
set -u
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# fail early if any input file is missing, instead of 70 identical errors
for entry in "${TESTS[@]}"; do
  IFS='|' read -r tid size files <<<"$entry"
  for f in $files; do
    [ -f "$f" ] || { echo "ABORT: missing input $f"; exit 1; }
  done
done

mkdir -p "$HOME/.prof_worktrees"
WT="$HOME/.prof_worktrees/cudaver_${SLURM_JOB_ID}"
FAILURES=0

for BRANCH in cuda-1D cuda-shared; do
  echo; echo "################ $BRANCH ################"
  rm -rf "$WT"; git -C "$REPO" worktree prune
  git -C "$REPO" worktree add --detach "$WT" "origin/$BRANCH"
  cd "$WT"
  git log --oneline -1
  grep -n "grid_copy =" energy_storms_cuda_core.cu

  grep -q '#ifndef BLOCK_SIZE' energy_storms_cuda_core.cu || \
    sed -i 's/^#define BLOCK_SIZE 256$/#ifndef BLOCK_SIZE\n#define BLOCK_SIZE 256\n#endif/' \
        energy_storms_cuda_core.cu

  unset REF; declare -A REF

  for BS in 256 32 64 128 512 1024; do
    make clean >/dev/null 2>&1 || true
    if ! make energy_storms_cuda CUDA_EXTRA_CFLAGS="-O3 -DBLOCK_SIZE=$BS" >/dev/null 2>&1; then
      echo "bs$BS  BUILD FAILED"; FAILURES=$((FAILURES+1)); continue
    fi
    md5=$(md5sum energy_storms_cuda | cut -d' ' -f1)
    echo "--- bs$BS  md5=$md5 ---"

    for entry in "${TESTS[@]}"; do
      IFS='|' read -r tid size files <<<"$entry"
      tid="${tid// /}"; size="${size// /}"
      if ! out=$(./energy_storms_cuda "$size" $files 2>&1 </dev/null); then
        echo "    $tid  RUN FAILED"; echo "$out" | tail -3
        FAILURES=$((FAILURES+1)); continue
      fi
      res=$(echo "$out" | grep '^Result:' | tail -1)
      if [ "$BS" = "256" ]; then
        REF["$tid"]="$res"
        echo "    $tid  REFERENCE   $res"
      elif [ "$res" = "${REF[$tid]:-}" ]; then
        echo "    $tid  MATCH"
      else
        echo "    $tid  MISMATCH"
        echo "        got: $res"
        echo "        ref: ${REF[$tid]:-<none>}"
        FAILURES=$((FAILURES+1))
      fi
    done
  done
  cd /; git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
done

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "=== VERDICT: ALL MATCH - fix confirmed across all tests and block sizes ==="
else
  echo "=== VERDICT: $FAILURES PROBLEM(S) - see MISMATCH/FAILED lines above ==="
fi
