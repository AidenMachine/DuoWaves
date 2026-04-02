#
# Simplified simulation of high-energy particle storms
#
# Parallel computing (Degree in Computer Engineering)
# 2017/2018
#
# EduHPC 2018: Peachy assignment
#
# (c) 2018 Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
# Grupo Trasgo, Universidad de Valladolid (Spain)
#
# This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-sa/4.0/
#
#
# The current Parallel Computing course includes contests using:
# OpenMP, MPI, and CUDA.
#
include student.mk
# Compilers
CC=gcc
MPICC=mpicc
CUDACC=nvcc -arch=sm_75
OMPFLAG=-fopenmp

# Flags for optimization and libs
FLAGS=-O3
LIBS=-lm

# Targets to build
OBJS=energy_storms_seq energy_storms_mpi_omp energy_storms_cuda energy_storms_core.o energy_storms_mpi_omp_core.o energy_storms_cuda_core.o

all: $(OBJS)

# Rules. By default show help
help:
		@echo
		@echo "Simplified simulation of high-energy particle storms"
		@echo
		@echo "Multicore Programming 2025/2026 Exam"
		@echo
		@echo "make all Build all versions (Sequential, OpenMP, MPI, CUDA)"
		@echo "make debug	   Build all version with demo output for small arrays (size<=35)"
		@echo "make clean	   Remove targets"
		@echo
		@echo "make energy_storms_seq		   Build only the sequential version"
		@echo "make energy_storms_mpi_omp	   Build only the MPI+OpenMP version"
		@echo "make energy_storms_cuda		  Build only the CUDA version"

energy_storms_core.o: energy_storms_core.c energy_storms.h
		$(CC) $(DEBUG) -c $< $(LIBS) -o $@

energy_storms_seq: energy_storms_seq.c energy_storms.h energy_storms_core.o
		$(CC) $(DEBUG) $< energy_storms_core.o $(LIBS) -o $@

energy_storms_mpi_omp_core.o: energy_storms_mpi_omp_core.c energy_storms.h
		$(MPICC) $(DEBUG) $(OMPFLAG) $(MPI_OMP_EXTRA_CFLAGS) -c $< $(LIBS) $(MPI_OMP_EXTRA_LIBS) -o $@

energy_storms_mpi_omp: energy_storms_mpi_omp.c energy_storms.h energy_storms_mpi_omp_core.o
		$(MPICC) $(DEBUG) $(OMPFLAG) $(MPI_OMP_EXTRA_CFLAGS) $< energy_storms_mpi_omp_core.o $(LIBS) $(MPI_OMP_EXTRA_LIBS) -o $@
	
energy_storms_cuda_core.o: energy_storms_cuda_core.cu energy_storms.h
		$(CUDACC) $(DEBUG) $(CUDA_EXTRA_CFLAGS) -c $< $(LIBS) $(CUDA_EXTRA_LIBS) -o $@

energy_storms_cuda: energy_storms_cuda.cpp energy_storms.h energy_storms_cuda_core.o
		$(CUDACC) $(DEBUG) $(CUDA_EXTRA_CFLAGS) $< energy_storms_cuda_core.o $(LIBS) $(CUDA_EXTRA_LIBS) -o $@

# Remove the target files
clean:
		rm -rf $(OBJS)

# Compile in debug mode
debug:
		make DEBUG=-DDEBUG

run_mpi:
		mpirun $(MPIRUN_FLAGS) ./energy_storms_mpi_omp 20000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6

run_cuda:
		srun -N 1 -n 1 ./energy_storms_cuda 20000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6