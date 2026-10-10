#!/bin/bash
#SBATCH --job-name=safelen_check
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --output=safelen_check.out
#SBATCH --error=safelen_check.err

set -eo pipefail

REPO="$HOME/DuoWaves"
BRANCH="Add_SIMD_Directives"
SRC="energy_storms_mpi_omp_core.c"
SIZE=1000000
WAVE="test_files/test_02_a30k_p20k_w1"

: "${LD_LIBRARY_PATH:=}"
export LD_LIBRARY_PATH
source /home/guest/init-hpc.sh
set -u
export OMP_NUM_THREADS=4                      # init-hpc.sh leaks 32

WT="$(mktemp -d)/wt"
git -C "$REPO" worktree add --detach "$WT" "$BRANCH"
cd "$WT"
trap 'cd /; git -C "$REPO" worktree remove --force "$WT" || true' EXIT

run() {
    mpirun -np 4 --map-by ppr:4:node:PE=4 --bind-to core \
        ./energy_storms_mpi_omp "$SIZE" "$WAVE" </dev/null
}

echo "=== BUILD 1: unmodified ($BRANCH) ==="
make clean >/dev/null; make energy_storms_mpi_omp
md5sum energy_storms_mpi_omp
run > /tmp/before.$SLURM_JOB_ID.txt

echo
echo "=== EDIT: removing simd/safelen from the reduce loop ==="
n=$(grep -c 'omp for simd schedule(static) safelen(4)' "$SRC" || true)
echo "lines matched: $n"
if [ "$n" -ne 1 ]; then
    echo "ABORT: expected exactly 1 match, found $n. Inspect $SRC by hand."
    exit 1
fi
sed -i 's|omp for simd schedule(static) safelen(4)|omp for schedule(static)|' "$SRC"
grep -n 'omp for schedule(static)' "$SRC"

echo
echo "=== BUILD 2: fixed ==="
make clean >/dev/null; make energy_storms_mpi_omp
md5sum energy_storms_mpi_omp
run > /tmp/after.$SLURM_JOB_ID.txt

echo
echo "=== TIMES ==="
grep -h 'Time:' /tmp/before.$SLURM_JOB_ID.txt | sed 's/^/before: /'
grep -h 'Time:' /tmp/after.$SLURM_JOB_ID.txt  | sed 's/^/after:  /'

echo
echo "=== RESULT DIFF (empty means identical) ==="
if diff <(grep '^Result:' /tmp/before.$SLURM_JOB_ID.txt) \
        <(grep '^Result:' /tmp/after.$SLURM_JOB_ID.txt); then
    echo "VERDICT: IDENTICAL — the 550-row dataset stands."
else
    echo "VERDICT: DIFFERENT — stop and report this."
fi
