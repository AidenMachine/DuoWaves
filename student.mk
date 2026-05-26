# --- Execution Parameters ---
OMP_NUM_THREADS=8
MPI_PROCS=16

# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) --bind-to numa -map-by numa

# --- Compiler Flags ---
MPI_OMP_EXTRA_CFLAGS = -O3 #-g	# Cancellare poi -g ( serve per Valgrind )
#MPI_OMP_EXTRA_LIBS =

# Flags for CUDA code
# Uncomment and add extra flags if you need them
#CUDA_EXTRA_CFLAGS =
#CUDA_EXTRA_LIBS =

# --- Valgrind Execution ---
run_valgrind:
	mpirun $(MPIRUN_FLAGS) valgrind --leak-check=full --track-origins=yes --log-file="valgrind_log_rank_%p.txt" ./energy_storms_mpi_omp 20000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2
