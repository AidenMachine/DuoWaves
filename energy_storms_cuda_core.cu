#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <cuda.h>

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

__global__ void bomb(Storm *d_storms, float *d_layer, int layer_size, int num_storms) {
    int i_layer = blockIdx.x * blockDim.x + threadIdx.x;
    int i_particle = blockIdx.y * blockDim.y + threadIdx.y;
    int i_storm = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i_storm < num_storms && i_particle < d_storms[i_storm].size && i_layer < layer_size) {}
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k; // TBR
    float *d_layer, *d_layer_copy, *d_storms;
    size_t size = layer_size * sizeof(float);
    
    cudaError_t m1_err = cudaMalloc((void**)&d_layer, layer_size);
    cudaError_t m2_err = cudaMalloc((void**)&d_layer_copy, layer_size);
    cudaError_t memc_err = cudaMemcpy(d_storms, storms, num_storms * sizeof(Storm *), cudaMemcpyHostToDevice);
    cudaError_t mems1_err = cudaMemset(&d_layer, 0.0f, layer_size);
    cudaError_t mems2_err = cudaMemset(&d_layer_copy, 0.0f, layer_size);

    if (m1_err != cudaSuccess || m2_err != cudaSuccess ||
            memc_err != cudaSuccess || 
            mems1_err != cudaSuccess || mems2_err != cudaSuccess) {
        printf("Error in device variables initialization:\n");                
        printf("layer:%s, %s\n", cudaGetErrorName(m1_err), cudaGetErrorString(m1_err));                
        printf("layer_copy:%s, %s\n", cudaGetErrorName(m2_err), cudaGetErrorString(m2_err));
        printf("memcpy:%s, %s\n", cudaGetErrorName(memc_err), cudaGetErrorString(memc_err));
        printf("memset layer:%s, %s\n", cudaGetErrorName(mems1_err), cudaGetErrorString(mems1_err));
        printf("memset layer_copy:%s, %s\n", cudaGetErrorName(mems2_err), cudaGetErrorString(mems2_err));
    }
    
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
                update( d_layer, layer_size, k, position, energy );
            }
        }

        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        for( k=0; k<layer_size; k++ ) 
            d_layer_copy[k] = d_layer[k];

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        for( k=1; k<layer_size-1; k++ )
            d_layer[k] = ( d_layer_copy[k-1] + d_layer_copy[k] + d_layer_copy[k+1] ) / 3;

        /* 4.3. Locate the maximum value in the layer, and its position */
        for( k=1; k<layer_size-1; k++ ) {
            /* Check it only if it is a local maximum */
            if ( d_layer[k] > d_layer[k-1] && d_layer[k] > d_layer[k+1] ) {
                if ( d_layer[k] > maximum[i] ) {
                    maximum[i] = d_layer[k];
                    positions[i] = k;
                }
            }
        }
    }
}
