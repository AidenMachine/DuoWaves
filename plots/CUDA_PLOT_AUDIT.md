# CUDA figure regeneration

The original figure 5 palette, fonts and axes are retained, split into two
standalone plots for a two-column report. Regenerate the PDF, PNG and source-row
audit with:

```sh
MPLCONFIGDIR=/tmp/duowaves-matplotlib python3 multicore-relation/plots/plot_figures.py --figure 5
python3 -m unittest discover -s multicore-relation/plots -p 'test_data.py' -v
```

Run from the workspace root. A figure-5-only rebuild leaves figures 1–4 and
`derived_numbers.json` untouched. Shared aggregation always uses medians;
existing figures 1–4 have not been regenerated or fully audited by this change.

Outputs: `fig5_cuda_time.pdf/.png`, `fig5_cuda_speedup.pdf/.png`, and
`fig5_cuda_audit.json`. The previous combined `fig5_cuda.pdf/.png` is kept as a
historical artifact and is no longer regenerated or used by the report.

## Eligibility and statistics

- Use active `results/raw/*.csv` only, never `results/raw-invalid/`.
- Exclude failed or unverified CUDA repetitions. Omit a configuration/workload
  cell entirely if any recorded repetition is rejected; a partially passing
  atomic configuration is not a stable performance result.
- Exclude historical `main` and `cuda-first` below 256 threads, even for PASS
  peaks: incomplete relaxation can leave the reported peak unchanged.
- For smaller blocks, admit `cuda-1D` and `cuda-shared` only when the job log
  identifies fixed commits `7d86f1e` or `6b05a7d` and matches the row's binary
  digest. New revisions need a provenance audit before extending this policy.
- Never mix different CUDA allocations or executable digests in one cell.
- Show medians and observed min–max ranges; no minimum fallback. All plotted
  CUDA cells in the current campaign have five valid repetitions.
- Exclude medians below the documented 0.02 s reporting floor from speedup
  selection. This is not a measured resolution of the timer.
- Select the smallest eligible median separately for each branch/workload;
  labels identify the winning block size. This is a configuration-selection
  envelope, not a fixed launch layout or multi-GPU scaling experiment.
- Use valid sequential cluster references, including `BASELINE` records.
  Keep allocations/builds separate and choose the allocation with the most
  repetitions; earliest numeric job id breaks ties. Never choose by runtime.
- Speedup error bars are the bounds `min(Tseq)/max(Tcuda)` through
  `max(Tseq)/min(Tcuda)`, not confidence intervals. They describe the samples
  of the selected configuration and do not quantify selection uncertainty or
  between-allocation uncertainty beyond those samples.

## Provenance and interpretation

`fig5_cuda_audit.json` records the exact input rows, job/node/build metadata,
repetition counts, selected configurations, baselines, timings, speedups and
exclusion reasons. Older rows lack executable digests and commit identifiers;
the figure labels their `main` results as historical, not evidence of a new
campaign on the final grid-corrected submission.

The time plot uses filled circles for historical `node111` measurements and open
squares for corrected `node122` reruns. Lines are visual guides across separate
allocations, not a controlled experiment isolating block size from node effects.

Original scripts and figure 5 outputs were copied to
`/tmp/duowaves-plots-before/` before regeneration. Project implementation,
measurements, branches and profiling captures were not modified.
