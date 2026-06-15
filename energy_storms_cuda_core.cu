#include <__clang_cuda_runtime_wrapper.h>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"

#define BLOCK_SIZE 256

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
static void update( float *layer, int layer_size, int k, int pos, float energy ) {
    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = pos - k;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[k] = layer[k] + energy_k;
}

__global__
void set_zeros_cuda(int layer_size, float *layer, float *layer_copy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > layer_size) return;

    layer[i] = 0.0f;
    layer_copy[i] = 0.0f;
}

__global__
void phase1() {}


/* Allocating the memory to the VRAM and transferring all the arrays */
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int num_blocks = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *layer, *layer_copy, *maximum_cuda;
    int *positions_cuda;
    Storm *storms_cuda;
    if (int alloc_layer = cudaMalloc(&layer, sizeof(float) * layer_size) != 0) {
       fprintf(stderr, "[cudaMalloc]: error when allocating memory for layer, returned %d", alloc_layer);
       exit(EXIT_FAILURE);
    }

    if (int alloc_layer_copy = cudaMalloc(&layer_copy, sizeof(float) * layer_size) != 0) {
       fprintf(stderr, "[cudaMalloc]: error when allocating memory for layer_copy, returned %d", alloc_layer_copy);
       exit(EXIT_FAILURE);
    }

    if (int maximum_error = cudaMemcpy(maximum_cuda, maximum , sizeof(maximum), cudaMemcpyHostToDevice) != 0) {
       fprintf(stderr, "[cudaMalloc]: error when copying to device for maximum, returned %d", maximum_error);
       exit(EXIT_FAILURE);
    }

    if (int positions_error = cudaMemcpy(positions_cuda, positions, sizeof(positions), cudaMemcpyHostToDevice) != 0) {
       fprintf(stderr, "[cudaMalloc]: error when copying to device for positions, returned %d", positions_error);
       exit(EXIT_FAILURE);
    }

    if (int storms_error = cudaMemcpy(storms_cuda, storms, sizeof(storms), cudaMemcpyHostToDevice) != 0) {
       fprintf(stderr, "[cudaMalloc]: error when copying to device for storms, returned %d", storms_error);
       exit(EXIT_FAILURE);
    }

    set_zeros_cuda<<<num_blocks, BLOCK_SIZE>>>(layer_size, layer, layer_copy);

    
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {

        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        for( j=0; j<storms[i].size; j++ ) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Get impact position */
            int position = storms[i].posval[j*2];

            /* For each cell in the layer */
            for( k=0; k<layer_size; k++ ) {
                /* Update the energy value for the cell */
                update( layer, layer_size, k, position, energy );
            }
        }

        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        for( k=0; k<layer_size; k++ ) 
            layer_copy[k] = layer[k];

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        for( k=1; k<layer_size-1; k++ )
            layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;

        /* 4.3. Locate the maximum value in the layer, and its position */
        for( k=1; k<layer_size-1; k++ ) {
            /* Check it only if it is a local maximum */
            if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
                if ( layer[k] > maximum[i] ) {
                    maximum[i] = layer[k];
                    positions[i] = k;
                }
            }
        }
    }
}
