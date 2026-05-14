# --- Execution Parameters ---
OMP_NUM_THREADS=8
MPI_PROCS=16

# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) --bind-to numa -map-by numa

# --- Compiler Flags ---
MPI_OMP_EXTRA_CFLAGS = -O3
#MPI_OMP_EXTRA_LIBS =

# Flags for CUDA code
# Uncomment and add extra flags if you need them
#CUDA_EXTRA_CFLAGS =
#CUDA_EXTRA_LIBS =
