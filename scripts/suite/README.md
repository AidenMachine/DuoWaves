<!--
    Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International

    Documentation of the multicore test suite.
-->

# Multicore test suite

Implements the multicore test campaign under the technology requirements. Everything that
touches the cluster is an sbatch payload; the launchers only submit jobs and
can be invoked from any directory, both locally and on the cluster, as long as
`DuoWaves` and `multicore-relation` are siblings (locally under `progetto/`,
on the cluster under `~/progetto-multicore/`).

## Requirement codes used in the code comments

| Code | Source |
|------|--------|
| TR-1 .. TR-6 | `technology_requirements.md`, lines 1 to 6 |
| TM-RUNS | five distinct runs per test, including the sequential section |
| MET-01 | metrics per run: single time, max, min, standard deviation |
| MET-SPEEDUP | speedup of a parallel configuration over the sequential reference |
| MET-EFF | efficiency of a parallel configuration, its speedup per hardware unit |
| TM-CORRECTNESS | verification against the sequential version within 1e-5, relative |
| TM-SEQ | sequential suite providing the missing baselines |
| TM-SHEETS | one section equals one sheet of the final .ods |
| TM-HYB, TM-CUDA | the two hardware sections |
| TM-A, TM-B, TM-C | layer size, storm count and particle count scaling |
| OUT-01, OUT-02 | raw CSV records and spreadsheet blocks |
| CAM-RESUME | already recorded work is neither resubmitted nor repeated |
| CAM-THROTTLE | submissions wait and retry when the qos submit cap is hit |
| CAM-PROVENANCE | a measurement names the executable it came from, and that executable was compiled by the job that measured it |
| MET-BASELINE | the sequential reference is checked for build and environment bias before the speedups are believed |
| ENV-META | job logs carry the identity of what produced them |

## Workflow

1. `python3 generate_inputs.py` once, then commit `DuoWaves/custom-tests`
   and push, so the cluster clone receives the generated waves.
2. On the cluster: `./run_sequential_tests.sh` and wait for completion. This
   publishes the correctness baselines in `results/baselines/`. Each workload
   is executed `SEQ_RUNS_PER_TEST` (5) times, same as the parallel sections,
   so the Sequential sheet carries a real max, min and standard deviation
   instead of a single-point statistic. The first execution's Result line is
   published as the baseline and every further repetition is checked against
   it, since the sequential program is deterministic.
3. On the cluster: `./run_parallel_tests.sh` for everything, or a subset with
   `--hybrid`, `--cuda`, optionally a repeatable
   `--branch NAME` and/or `--exclude-branch NAME` to restrict or leave out
   one or more branches. `--dry-run` prints the sbatch commands of either
   launcher. Both launchers accept a repeatable `--exclude ID` that leaves
   one common test out, for example `--exclude A_1M` on a campaign restricted
   to the least parallel layouts.
   Every filter is validated before anything is submitted, because each of
   these mistakes used to produce a launcher that reported success and did
   what was not asked. A branch name no selected section measures, such as a
   typo or `main` under `--hybrid`, aborts with the list of the branches that
   section holds; an `--exclude` naming no test of the common table aborts
   with the table; an option given without a value aborts saying so, instead
   of dying wordlessly on `set -e`; a selection matching no configuration at
   all, such as `--branch X --exclude-branch X`, aborts rather than reporting
   an empty campaign, which is indistinguishable from a complete one.
   Repeating a section flag is collapsed: `--hybrid --hybrid` used to walk
   the selection twice and submit the whole section a second time, and two
   jobs measuring one triple record every run they both find missing twice.
   Relaunching the whole campaign after a partial one is the normal way to
   work, because resuming happens at two levels. The launcher reads the raw
   CSV files once and does not submit at all the configurations that already
   hold every repetition of every test the invocation asks for, reporting them
   as `skipped N already complete`; the jobs it does submit skip, run by run,
   the repetitions already recorded. So the queue slots go to the missing
   measurements only, and a configuration is resubmitted exactly while it is
   still incomplete. `--force` submits everything regardless, which is what to
   use after the sources or the machine changed and the old numbers are to be
   checked again. Forced jobs bypass historical run records and write fresh
   rows to `results/verification/job_<JobID>/j<JobID>.csv`, preserving the
   performance campaign in `results/raw/`. These verification rows are not
   included in the campaign spreadsheet. Excluded tests are not required, so `--exclude A_1M` also treats
   as complete a configuration that is only missing `A_1M`.
   The `students_limit` qos caps the number of jobs a user may have queued;
   when the cap is hit the launchers wait `SUBMIT_RETRY_SECONDS` and retry
   the same submission until a slot frees up, so a full campaign launcher
   can stay alive for hours: run it inside tmux or screen, or with
   `nohup ... &`, and keep the log. The transcript is built for that wait.
   Every line is timestamped, is written to standard output and names the
   configuration it refers to, prefixed by its position in the campaign, so
   `[37/192] immediate/hyb_2n_4x2: Submitted batch job 982713` says both what
   went in and how much is left. A submission blocked by the cap announces
   itself once and then reports the time waited so far every
   `SUITE_WAIT_NOTICE_RETRIES` attempts, five minutes apart with the shipped
   settings, rather than repeating one identical line every fifteen seconds.
   The sequential launcher also accepts `--local`, which executes the very
   same payload script directly on the current machine instead of going
   through sbatch, with no Slurm and no partition time limit: the
   measurements land in `results/raw/jlocal.csv` with job id `local` and the
   node name of the machine, and the baselines are published exactly as on
   the cluster. It is meant for testing the suite and producing baselines on
   the laptop; local timings must not be mixed into the cluster spreadsheet.
4. Copy `results/raw/` (and `sbatch_jobs/`) back to the laptop, for example
   by committing them or with rsync.
5. On the laptop: `./setup_venv.sh` once, then
   `../../.venv/bin/python build_ods.py` writes
   `results/test-results.ods`.
6. Anywhere, at any point of the campaign: `python3 hybrid_speedup.py` and
   `python3 hybrid_efficiency.py` evaluate the hybrid section against the
   sequential reference. They use the standard library only, so they need
   neither the virtual environment nor the cluster, and they never require
   the campaign to be complete: only the cells already recorded are
   evaluated and the gaps are listed explicitly.

## Checking the sequential reference itself

Every speedup the report publishes is a ratio against one sequential baseline,
so a baseline that is slow for a reason unrelated to parallelism inflates all
of them at once. There is a concrete reason to suspect it: `firstTry`, whose
kernel contains no `#pragma omp` and no `MPI_` call and differs from the
sequential core only in allocating its two layer arrays itself, recorded
5.544 s on `A_20k` against a sequential baseline of 8.671 s. That is a factor
of 1.56 for identical arithmetic on one active thread.

`./run_baseline_bias.sh` submits one job measuring four variants of one
workload inside a single allocation, so the node, its clock and its neighbours
are the same for all of them:

| variant | what it isolates |
|---------|------------------|
| `seq_gcc` | the baseline recipe exactly as the sequential suite builds it |
| `seq_mpicc` | the same sources built by `mpicc`, so the compiler driver alone |
| `hyb_direct` | the hybrid binary of a sequential-equivalent branch, without `mpirun` |
| `hyb_mpirun` | the same binary under the launcher line the campaign uses |

The step whose ratio departs from one names the cause. The job also prints
what `mpicc -show` wraps, both compiler versions, the clock of the node, its
governor and its load, since any of those can produce a constant factor on its
own. It writes `results/analysis/baseline_bias.csv` and takes a few minutes.
`--branch NAME` and `--repetitions N` override the defaults, `firstTry` and 5,
and `--dry-run` prints the sbatch command like the other launchers.

Like every other payload of the suite, `baseline_bias.sbatch` receives its
absolute `SUITE_DIR` and its log paths from the launcher and must not be
submitted with a bare `sbatch`. Its first version carried relative `--output`
directives and derived the suite directory from the submission directory,
which fails in one second when the submission happens from `scripts/suite`
itself. It now also walks up from the submission directory to find
`common.sh`, so a hand submission degrades into working rather than failing,
but the log then lands wherever Slurm's defaults put it instead of under
`sbatch_jobs/`.

On a laptop, where `gcc` and `mpicc` wrap the same compiler, the four variants
land within about one percent of each other, which is the shape of a null
result to compare the cluster run against.

## Speedup and efficiency of the hybrid section

`hybrid_speedup.py` reads the same `results/raw/` the spreadsheet is built
from and publishes, per `(branch, configuration, test)` cell:

* `speedup median` = median sequential time / median hybrid time, the figure
  representing the configuration;
* `speedup mean` = mean sequential time / mean hybrid time, kept because the
  earlier reports quoted it;
* `speedup best` = fastest sequential repetition / fastest hybrid repetition,
  the best case actually observed.

The median leads because the cluster is shared and a repetition occasionally
costs many times what its siblings cost. Of the cells measured with five
repetitions, better than a quarter span more than 1.5x between their fastest
and their slowest run, and the worst hold a single run tens of times above the
others: a cell of repetitions around 10.3 s carrying one of 310 s is reported
by its mean as a 70 s cell. Both figures are published side by side, so the
distance between them stays readable wherever contention disturbed the
measurement.

`hybrid_efficiency.py` divides both by the hardware the configuration
occupies. It calls into `hybrid_speedup.py` for the measurements and the
speedups, and stays a standalone entry point: run alone it produces the whole
efficiency report without the speedup one having been run first. The default
denominator is the number of hardware threads the launcher reserves
(`nodes x MPI processes per node x OpenMP threads per process`, exactly what
`--nodes`, `--ntasks-per-node` and `--cpus-per-task` allocate); `--unit core`
divides instead by the physical cores those threads sit on, using the two
hardware threads per core of the EPYC 7301 nodes and therefore assuming the
allocation is packed onto pairs of SMT siblings.

Both scripts share the same options: `--branch`, `--config` and `--test`
restrict the evaluation and are repeatable, `--no-csv` prints the report only,
`--csv PATH` redirects the file.

Two conventions are worth stating, because they decide the figures:

* The sequential reference is built from the **cluster** repetitions only.
  The `--local` runs of the sequential launcher were measured on the laptop,
  which is markedly faster per core than a cluster node, and pooling the two
  would inflate every speedup. `--include-local` opts back in, for example to
  read a provisional figure for a workload whose only sequential repetition so
  far is a local one.
* Repetitions whose result was wrong are excluded from every statistic, since
  a wrong result is fast for the wrong reason; the verdict of the cell is
  reported regardless, and `--include-failed` brings them back in.
* A cell whose median falls under `TIMING_RESOLUTION_FLOOR` (0.02 s) is marked
  `below floor` and can never win the best configuration table of either
  script. Under it the number recorded is the process built around the program
  rather than the program: repetitions of 0.00 and 0.01 s produce ratios that
  look like speedups and are not. The rows keep their figures, so nothing is
  hidden, but the summaries stop being decided by noise. The marker is worth
  reading before quoting anything from the CUDA sheet, where whole tests sit
  under the floor at every block size.

Tests sharing one workload share one reference: `B_1` is measured by the
sequential suite under `A_20k`, so the evaluation reproduces
`suite_workload_id` from `common.sh` and resolves the equivalence instead of
reporting `B_1` as missing. The test table itself is read from `common.sh`,
never duplicated.

## How results are stored

* Job logs: `sbatch_jobs/<JobID>/<JobID>.out` and `.err`, the same layout as
  every historical job of this repository (TR-5).
* Measurements: one CSV per job in `results/raw/`, row format
  `section|branch|config|test|run,time_s,correctness,job_id,nodes,timestamp,binary_md5,"result"`.
  The digest is the md5 of the executable the job built and measured
  (CAM-PROVENANCE) and the payload is the Result line verbatim, so a verdict
  can be recomputed under a different tolerance without running anything
  again. Rows written before these two fields existed have six fields; every
  reader indexes the row from the left and tolerates both lengths.
* Baselines: `results/baselines/<workload>.result` plus `MANIFEST.csv`.
* Spreadsheet: `results/test-results.ods` (TR-6), one sheet per section with
  the Median/Avg/Min/Max/STD blocks, the resolution block, the correctness
  block and the runs combined block of the reference example. Median leads;
  the resolution block marks the cells whose median is under the floor.
* Derived figures: `results/analysis/hybrid_speedup.csv` and
  `results/analysis/hybrid_efficiency.csv`, one row per evaluated cell, with
  empty speedup and efficiency fields wherever the campaign has not produced
  the numbers yet.

## Branch columns

Fixed in `suite.conf`, derived by hashing the section-relevant sources of
every remote branch and keeping, chronologically, only branches introducing
different sources. `main` carries the same MPI sources as `immediate`, so it
appears only in the CUDA sheet; the pre-CUDA branches all carry the untouched
handout CUDA stub, so the CUDA sheet starts at `cuda-first`.

`firstTry` was dropped from the hybrid list. Its kernel contains no `#pragma
omp` and no `MPI_` call, and differs from the sequential core only in
allocating the two layer arrays inside `core()` instead of receiving them as
parameters: it is the sequential algorithm compiled by `mpicc`, so it cannot
carry a hybrid column. Its measurements were deleted with the branch.

## To be stated in the report

The measurement uncertainty of this campaign is large enough that the report
has to declare it rather than let the reader assume the figures are exact.
It was quantified on the quarantined trio, the one set of measurements where a
byte-identical binary was run many times over: taking two jobs that measured
the same cell, the ratio of their medians is

| pair | median | p90 | worst |
|------|--------|-----|-------|
| same node, different job | 1.07x | 1.57x | 3.18x |
| different node | 1.13x | 1.70x | 4.20x |

Two consequences the report must carry:

* Five repetitions inside one job are not five independent samples. They run
  back to back in one allocation, so the median of a cell inherits the
  conditions of the job that produced it, and re-measuring the same cell in a
  new job moves that median by about 7 percent typically and occasionally by a
  factor of three. This was accepted deliberately rather than paid for with
  twice the submissions.
* Differences between branches smaller than roughly 1.5x are inside the noise
  and must not be presented as improvements. The `STD deviation`, `Min` and
  `Max` blocks of the `.ods` are what to quote next to any figure.

Node identity is explicitly *not* the cause: changing node adds about six
points to the median ratio above, and no node shows a systematic speed on the
data available. Pinning the campaign to a node set would buy nothing.

## Known limits, stated on purpose

* A single repetition that needs more than 30 minutes can never complete
  under the partition limit, no matter how the jobs are split. This is what
  retired `A_100M` from the common table: 20000 particles over 100 million
  positions is about 2e12 core updates, which neither the sequential section
  nor the least parallel hybrid layouts finish inside the limit, so the test
  only ever produced timed-out jobs and missing cells. The measurements it
  did produce before its removal, the complete CUDA block and one local
  sequential repetition, stay in `results/raw/` and keep their column in the
  spreadsheet, so nothing already paid for is thrown away.
* Raw CSV files produced while the sequential section was temporarily
  reduced to one run only carry that run 1. They are left untouched on disk;
  resubmitting `run_sequential_tests.sh` fills in the missing repetitions 2
  to 5 through the same CAM-RESUME mechanism the parallel sections rely on,
  and until then `build_ods.py` reports those cells as `PASS (1/5)`.
* Two concurrently running jobs measuring the same (section, branch,
  configuration) would duplicate work; the launchers never submit such
  pairs, so simply avoid submitting the same scope twice while jobs are
  still queued. Duplicated measurements are detected and reported by
  `build_ods.py` anyway.
* The CUDA branches that predate the `#ifndef BLOCK_SIZE` guard of `main`
  are patched inside the private per-job worktree only, and the patch is
  logged in the job output (ENV-META).
* Several branches track their build outputs, `energy_storms_mpi_omp_core.o`
  and the binaries linked from it, which are ignored only on `main`. A
  checkout gives them an mtime not older than their sources, so make used to
  declare them up to date and the job linked, or ran outright, an executable
  compiled elsewhere months earlier. The campaign of 2026-08-19 was lost this
  way: no job ever compiled the MPI kernel, and `Layers_OptimizationV2`,
  `AddParallelZone` and `Add_SIMD_Directives` all measured one object frozen
  at commit `cedf215`. `suite_prepare_worktree` now purges those artifacts
  from every fresh worktree and every row carries the digest of the binary,
  so the same failure cannot recur unnoticed. The branches themselves are
  left alone: rewriting them to untrack the objects would rewrite the
  team's history for a problem the suite can solve on its own side.
