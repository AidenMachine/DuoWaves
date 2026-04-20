/*
 * Simplified simulation of high-energy particle storms
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2017/2018
 *
 * Version: 2.0
 *
 * MPI+OpenMP code.
 *
 * Multicore Programming 2025/2026 project.
 *
 * Based on the EduHPC 2018 Peachy Assignment by
 * Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include "energy_storms.h"

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions);

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i;

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    Storm* storms = (Storm*) malloc(sizeof(Storm)*num_storms);

    /* 1.2. Read storms information */
    for( i=2; i<argc; i++ ) 
        storms[i-2] = read_storm_file( argv[i] );

    /* 1.3. Intialize maximum levels to zero */
    float maximum[ num_storms ];
    int positions[ num_storms ];
    for (i=0; i<num_storms; i++) {
        maximum[i] = 0.0f;
        positions[i] = 0;
    }

    /* 2. Begin time measurement */
    MPI_Barrier(MPI_COMM_WORLD);
    double ttotal = cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */
    core(layer_size, num_storms, storms, maximum, positions);
    /* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
    MPI_Barrier(MPI_COMM_WORLD);
    ttotal = cp_Wtime() - ttotal;

    if ( rank == 0 ) {
	    /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
	    #ifdef DEBUG
	    debug_print( layer_size, layer, positions, maximum, num_storms );
	    #endif

	    /* 7. Results output, used by the Tablon online judge software */
	    printf("\n");
	    /* 7.1. Total computation time */
	    printf("Time: %lf\n", ttotal );
	    /* 7.2. Print the maximum levels */
	    printf("Result:");
	    for (i=0; i<num_storms; i++)
		printf(" %d %f", positions[i], maximum[i] );
	    printf("\n");
    }

    /* 8. Free resources */    
    for( i=0; i<argc-2; i++ )
        free( storms[i].posval );
    free(storms);
    /* 9. Program ended successfully */
	MPI_Finalize();
    return 0;
}

