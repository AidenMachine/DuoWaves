<!--
    Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International

    Note on the batch scripts that predate the requirements catalogue.
-->

# Legacy batch scripts

These are the ad-hoc scripts written before the requirements catalogue existed. They are kept because they produced the 112 job outputs in
`sbatch_jobs/`, which is where the platform facts of section 1.3 were recovered
from, so deleting them would orphan that evidence.

They are **not** part of the harness and must not be used for report data. Each
of them violates several requirements: they aggregate to a mean and a standard
deviation inside bash instead of recording every repetition (MET-02, OUT-02),
they keep no warm-up distinction (MET-03), they capture no environment metadata
(ENV-02), they write no machine readable output (OUT-01), and they cannot be
resumed (CAM-03).

`size_sweep.sbatch` additionally flags its own defect in its header: it sweeps the
layer size while reusing a wave file whose particles were placed for a 30000 cell
layer, so it changes the physics together with the size. The generated inputs of
`DuoWaves/custom-tests` exist to remove exactly that caveat.

Use `scripts/suite/` instead, and see `scripts/suite/README.md`.
