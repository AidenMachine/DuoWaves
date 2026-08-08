#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <cuda.h>

/* Kernel per il bombardamento, al momento errato perché pensato per la vecchia versione*/
__global__ void bomb(float *d_flattened, float *d_layer, int* d_offsets, int layer_size, int total_size, int num_storms) {
    int i_layer = blockIdx.x * blockDim.x + threadIdx.x;
    int i_particle = blockIdx.y * blockDim.y + threadIdx.y;
    int i_storm = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i_storm < num_storms && i_particle < (d_offsets[i_storm + 1] - d_offsets[i_storm]) && i_layer < layer_size) {
        float energy = (float)d_flattened[d_offsets[i_storm] + i_particle*2+1] * 1000;
        float position = (float)d_flattened[d_offsets[i_storm] + i_particle*2];

        int distance = position - i_layer;
        if ( distance < 0 ) distance = - distance;

        distance = distance + 1;
        float atenuacion = sqrtf( (float)distance );

        float energy_k = energy / layer_size / atenuacion;

        if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
            d_layer[i_layer] = d_layer[i_layer] + energy_k;
    }
}

/* Kernel che copia i valori da layer a layer_copy */
__global__ void copy(float *layer, float *layer_copy, int layer_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < layer_size) {
        layer_copy[i] = layer[i];
    }
}

/* Kernel per la fase di rilassamento */
__global__ void relax(float * layer, float * layer_copy, int layer_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < layer_size -1 && i > 0) {
       layer[i] = ( layer_copy[i-1] + layer_copy[i] + layer_copy[i+1] ) / 3;
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    /* Prima parte: il problema principale da affrontare è che storms sia un array di strutture con
     * puntatori ad altri array, e  quindi allocare tutto nel device diventerebbe lungo. Per tale 
     * motivo, prima di allocare nel device si effettua una operazione di "flattening" dell'arrray 
     * di storms in un gigantesco array che contenga i valori di tutte le storms uno dietro l'altro,
     * con un array di offset che per ogni storm dice esattamente dove si trova l'inizio della storm */
    int *h_offsets = (int *)malloc((num_storms + 1) * sizeof(int)); 
    float* h_storms; 

    h_offsets[0] = 0;

    /* Popolazione dell'array di offset nell'host. ATTENZIONE: per ogni struct di una Storm si ha
     * l'array posval che ha 2 valori per ogni particella, ma size descrive il numero di particelle.
     * Da qui la necessaria moltiplicazione per 2. */
    for (int i = 0; i < num_storms; i++) {
        h_offsets[i+1] = h_offsets[i] + (storms[i].size * 2);
    }

    int total_size = h_offsets[num_storms];
    float *h_flattened = (float *)malloc(total_size * sizeof(float));

    /* Popolazione dell'array flattened nell'host tramite memcpy dal posval corrispondente. Stesse
     * attenzioni di cui sopra quando si tratta di definire la dimensione. */
    for (int i = 0; i < num_storms; i++) {
        memcpy(h_flattened + h_offsets[i], storms[i].posval, storms[i].size * 2);
    }


    /* Trasferimento nel device: creazione e popolazione di tutte le strutture che serviranno nel
     * device. VALUTARE SPOSTAMENTO IN UNA FUNZIONE A PARTE */
    float *d_layer, *d_layer_copy , *d_flattened;
    int * d_offsets;
    size_t size = layer_size * sizeof(float);

    cudaError_t offsets_err = cudaMalloc((void **)&d_offsets, (num_storms + 1) * sizeof(int));
    cudaError_t flattened_err = cudaMalloc((void **)&d_flattened, total_size * sizeof(float));
    cudaError_t layer_err = cudaMalloc((void**)&d_layer, layer_size);
    cudaError_t layercopy_err = cudaMalloc((void**)&d_layer_copy, layer_size);

    cudaError_t offsetcpy_err = cudaMemcpy(d_offsets, h_offsets, num_storms + 1, cudaMemcpyHostToDevice);
    cudaError_t flatcpy_err = cudaMemcpy(d_flattened, h_flattened, total_size, cudaMemcpyHostToDevice);

    cudaError_t mems1_err = cudaMemset(&d_layer, 0.0f, layer_size);
    cudaError_t mems2_err = cudaMemset(&d_layer_copy, 0.0f, layer_size);

    if (offsets_err != cudaSuccess || flattened_err != cudaSuccess || 
            layer_err != cudaSuccess || layercopy_err != cudaSuccess ||
            offsetcpy_err != cudaSuccess || flatcpy_err != cudaSuccess ||
            mems1_err != cudaSuccess || mems2_err != cudaSuccess) {
        printf("Error in device variables initialization:\n");                
        printf("layer:%s, %s\n", cudaGetErrorName(layer_err), cudaGetErrorString(layer_err));
        printf("layer_copy:%s, %s\n", cudaGetErrorName(layercopy_err), cudaGetErrorString(layercopy_err));
        printf("offsets:%s, %s\n", cudaGetErrorName(offsets_err), cudaGetErrorString(offsets_err));
        printf("flattened:%s, %s\n", cudaGetErrorName(flattened_err), cudaGetErrorString(flattened_err));

        printf("memcpy offset:%s, %s\n", cudaGetErrorName(offsetcpy_err), cudaGetErrorString(offsetcpy_err));
        printf("memcpy flattened:%s, %s\n", cudaGetErrorName(flatcpy_err), cudaGetErrorString(flatcpy_err));
        printf("memset layer:%s, %s\n", cudaGetErrorName(mems1_err), cudaGetErrorString(mems1_err));
        printf("memset layer_copy:%s, %s\n", cudaGetErrorName(mems2_err), cudaGetErrorString(mems2_err));

        exit(1);
    }

    int max_storm = 0;
    for (int i = 0; i + 1 < num_storms + 1; i++) {
        int gap = h_offsets[i + 1] - h_offsets[i];
        if (gap > max_storm) max_storm = gap;
    }

    /* 256 threads per block */
    dim3 block(32, 4, 2);
    dim3 grid_bomb((layer_size + 31)/32, (max_storm + 4)/4, (num_storms + 1)/2);

    bomb<<<grid_bomb, block>>>(d_flattened, d_layer, d_offsets, layer_size, total_size, num_storms);
    cudaDeviceSynchronize();

    int grid_copy = (layer_size + 256 - 1)/256;

    copy<<<grid_copy, block>>>(d_layer, d_layer_copy, layer_size);
    cudaDeviceSynchronize();
    
    relax<<<grid_copy, block>>>(d_layer, d_layer_copy, layer_size);
    cudaDeviceSynchronize();

}
