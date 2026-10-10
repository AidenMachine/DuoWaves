#!/bin/bash
cd ~/multicore-relation

for branch in Add_SIMD_Directives immediate; do
  for n in 1 2 4; do
    echo "=== $(date -Is) submitting $branch ${n}n ==="
    until jid=$(sbatch --parsable --nodes=$n \
                --export=ALL,PROF_BRANCH="$branch" prof_run.sbatch 2>/dev/null); do
      echo "slot busy, retrying in 30s"
      sleep 30
    done
    echo "job $jid submitted"
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do
      sleep 20
    done
    echo "job $jid finished"
  done
done
echo "=== ALL DONE $(date -Is) ==="
