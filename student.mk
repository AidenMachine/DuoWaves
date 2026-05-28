# --- Execution Parameters ---
OMP_NUM_THREADS ?= 4
MPI_PROCS ?= 8

# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) --bind-to numa -map-by numa
#MPIRUN_FLAGS = -np $(MPI_PROCS) --map-by l3cache --bind-to core --report-bindings

# --- Compiler Flags ---
MPI_OMP_EXTRA_CFLAGS = -O3 -march=znver1 -mtune=znver1 -mavx2 -mprefer-vector-width=256 -fno-math-errno -ffinite-math-only -fno-signaling-nans -fno-rounding-math -fcx-limited-range -fno-signed-zeros -fno-trapping-math -fprefetch-loop-arrays -funroll-loops -mfma
#( serve per Valgrind )
#MPI_OMP_EXTRA_LIBS =

# Flags for CUDA code
# Uncomment and add extra flags if you need them
#CUDA_EXTRA_CFLAGS =
#CUDA_EXTRA_LIBS =
