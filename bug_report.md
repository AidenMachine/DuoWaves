# Bug Report: Heap Corruption in `energy_storms_mpi_omp`

**Date:** 2026-04-23  
**File:** `energy_storms_mpi_omp_core.c`  
**Severity:** Critical (program crash on every run)

## Symptoms

Running `make run_mpi` (10 MPI processes, 32 OpenMP threads) causes all processes to crash
with **signal 6 (SIGABRT)**. The glibc allocator detects heap corruption and aborts, producing
two types of error messages:

- `free(): invalid next size (normal)` — heap metadata of an adjacent block has been overwritten
- `double free or corruption (!prev)` — similar corruption on the previous block's metadata

No process produces any useful output before crashing.

## Root Cause

### Primary: shared loop variable `k` in OpenMP parallel region

In `energy_storms_mpi_omp_core.c`, the variable `k` is declared at function scope (line 54)
and is therefore **shared by default** across all OpenMP threads. At lines 120-139, the outer
loop over particles (`j`) is parallelized, but the inner loop over the local layer uses `k`
as its control variable:

```c
#pragma omp parallel for          /* only j is made private */
for( j=0; j<storms[i].size; j++ ) {
    ...
    for( k = 1; k <= local_size; k++ ) {   /* k is shared: DATA RACE */
        update( layer, layer_size, k, position, energy, offset + k - 1);
    }
}
```

Since all threads read and write `k` concurrently:

1. `k` takes unpredictable values due to the race condition.
2. `k` can exceed the valid index range (`local_size + 1`), going out of bounds.
3. `update()` writes to `layer[k]`, and an out-of-bounds write corrupts the heap
   metadata placed by `malloc` right after the `layer` buffer.
4. When `free(layer)` or `free(layer_copy)` is called at end of `core()`, glibc
   detects the corrupted metadata and calls `abort()`.

### Secondary: wrong compiler flag in `student.mk`

Line 12 of `student.mk` reads:

```makefile
MPI_OMP_EXTRA_CFLAGS =-o3
```

The lowercase `-o3` is interpreted by the compiler as "write output to a file named `3`",
not as optimization level 3. The correct flag is `-O3` (uppercase O). This does not cause
the crash (the Makefile's later `-o $@` overrides the output destination), but it means
the code is compiled without any optimization.

## Suggested Fix

Make `k` private in the parallel for directive at line 120:

```c
#pragma omp parallel for private(k)
for( j=0; j<storms[i].size; j++ ) {
    ...
}
```

Alternatively, declare `k` inside the inner loop body so that each thread gets its own copy:

```c
for( int k = 1; k <= local_size; k++ ) {
    ...
}
```

And in `student.mk`, change `-o3` to `-O3`.
