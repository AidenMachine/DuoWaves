<!--
    Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International

    Why the measurements in this directory are kept but not used.
-->

# Quarantined measurements

These 126 job files hold 3465 real measurements taken on the cluster. They are
not wrong numbers: they are correct timings of the wrong code. They live here,
outside `results/raw/`, because `build_ods.py` merges `raw/*.csv` and resolves a
duplicated key by **keeping the first** row it meets, so leaving them next to
the reruns would make the reruns pointless.

## What happened

Several DuoWaves branches track their build outputs, `energy_storms_mpi_omp_core.o`
and the binaries linked from it; they are listed in `.gitignore` only on `main`.
A checkout gives a tracked file the mtime of the checkout, so the committed
object was never older than its source and make declared it up to date. No job
of the 2026-08-19 campaign ever compiled the MPI kernel, and 77 of them compiled
nothing at all and ran the binary committed in the branch.

`suite_prepare_worktree` now purges those artifacts from every fresh worktree
and every recorded row carries the md5 of the binary that produced it
(CAM-PROVENANCE), so this cannot recur unnoticed.

## What each subset actually measured

| Branch | rows | the object it ran | why it is not the branch |
|--------|------|-------------------|--------------------------|
| `Layers_OptimizationV2` | 1155 | `73d77463`, frozen at `cedf215` | the tip moves the omp directive to the third loop and removes `atomic` |
| `AddParallelZone` | 1155 | `73d77463`, frozen at `cedf215` | 283 lines of difference from the tip |
| `Add_SIMD_Directives` | 1155 | `73d77463`, frozen at `cedf215` | 299 lines of difference; its SIMD flags reached only the driver, never the kernel |

The 30 rows `Layers_OptimizationV2` carried under the retired `MPI` section were
deleted too: the suite has no MPI-only mode any more, the section is outside the
campaign grid `build_ods.py` accepts, and those rows came from the same frozen
object as the rest.

Two further subsets were quarantined and then deleted on purpose, since neither
had any salvage value and both branches are in the rerun anyway:
`Sqrt_Optimization`, object `c6f3e830` built from source `9d988ff8`, which is
the sqrt-table version the branch tip abandoned when it went back to `sqrtf()`;
and `immediate`, object `56e6d636` built from source `b204332e`, which still
passes `MPI_STATUS_IGNORE` where the tip passes a real `MPI_Request` handle to
`MPI_Waitall`. 337 rows of them were deleted first, then a further 2026 that
arrived with a later merge and carried the same two frozen objects, for 2363 in
total. All of it is recoverable from git history if it is ever wanted back.

## Why what is left is worth keeping

`Layers_OptimizationV2`, `AddParallelZone` and `Add_SIMD_Directives` all linked
the identical object `73d77463`, so everything in this directory is not three
columns but **one column measured fifteen times per cell**, of the code at
commit `cedf215`, "Eliminata la zona di lock()". Relabelled that way the data is
sound and is the most heavily sampled column of the whole campaign. Deciding
whether the report wants it is a separate question from this quarantine.

## Correctness verdicts in these files

Ignore them. They were produced under an absolute tolerance of 1e-5 applied to
`float` values of magnitude 1.6e3, whose ulp is 1.22e-4, so the check demanded
bit-identical output. Re-running the same binaries under the relative tolerance
now in `suite.conf` gives a worst relative error of 5.6e-6, with the positions
always exact.

## Branches deliberately left in results/raw

`firstTry` tracks no object and compiled from source in every job.
`Layers_Optimization` and `v1-da-runnare` track one, but it is a valid build of
their tip: the first differs from the branch source by comments only, the second
by an added `#include <string.h>` that produces byte-identical `.text`. The four
CUDA branches track nothing and recompiled the `.cu` in all 24 jobs.
