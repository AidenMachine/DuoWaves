# --- Execution Parameters ---
export OMP_NUM_THREADS=2
MPI_PROCS=32

# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) \
               --bind-to none \
	       --oversubscribe #assegna piu processi allo stesso nodo quando runnato su piu nodi

# --- Compiler Flags ---
# Flags for MPI+OpenMP code
# Uncomment and add extra flags if you need them
MPI_OMP_EXTRA_CFLAGS =-O3
#MPI_OMP_EXTRA_LIBS =

# Flags for CUDA code
# Uncomment and add extra flags if you need them
#CUDA_EXTRA_CFLAGS =
#CUDA_EXTRA_LIBS =
