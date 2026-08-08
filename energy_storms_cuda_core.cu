#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <cuda.h>

/* Kernel per il bombardamento, al momento errato perché pensato per la vecchia versione*/
__global__ void bomb(Storm *d_storms, float *d_layer, int layer_size, int num_storms) {
    int i_layer = blockIdx.x * blockDim.x + threadIdx.x;
    int i_particle = blockIdx.y * blockDim.y + threadIdx.y;
    int i_storm = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i_storm < num_storms && i_particle < d_storms[i_storm].size && i_layer < layer_size) {
        float energy = (float)d_storms[i_storm].posval[i_particle*2+1] * 1000;
        float position = (float)d_storms[i_storm].posval[i_particle*2];

        int distance = position - i_layer;
        if ( distance < 0 ) distance = - distance;

        distance = distance + 1;
        float atenuacion = sqrtf( (float)distance );

        float energy_k = energy / layer_size / atenuacion;

        if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
            d_layer[i_layer] = d_layer[i_layer] + energy_k;
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
    float *d_layer, *d_layer_copy, *d_offsets, *d_flattened;
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

    
    /*SERIAL storm simulation, for reference purpose only*/
    //for( i=0; i<num_storms; i++) {
   // 
    //    /* For each particle */
     //   for( j=0; j<storms[i].size; j++ ) {
      //      /* Get impact energy (expressed in thousandths) */
       //     float energy = (float)storms[i].posval[j*2+1] * 1000;
        //    /* Get impact position */
         //   int position = storms[i].posval[j*2];
//
 //           /* For each cell in the layer */
  //          for( k=0; k<layer_size; k++ ) {
   //             /* Update the energy value for the cell */
    //            update( d_layer, layer_size, k, position, energy );
     //       }
      //  }
//
 //       /* 4.2. Energy relaxation between storms */
  //      /* 4.2.1. Copy values to the ancillary array */
   //     for( k=0; k<layer_size; k++ ) 
    //        d_layer_copy[k] = d_layer[k];
//
 //       /* 4.2.2. Update layer using the ancillary values.
  //                Skip updating the first and last positions */
   //     for( k=1; k<layer_size-1; k++ )
    //        d_layer[k] = ( d_layer_copy[k-1] + d_layer_copy[k] + d_layer_copy[k+1] ) / 3;
//
 //       /* 4.3. Locate the maximum value in the layer, and its position */
  //      for( k=1; k<layer_size-1; k++ ) {
   //         /* Check it only if it is a local maximum */
    //        if ( d_layer[k] > d_layer[k-1] && d_layer[k] > d_layer[k+1] ) {
     //           if ( d_layer[k] > maximum[i] ) {
      //              maximum[i] = d_layer[k];
       //             positions[i] = k;
        //        }
         //   }
        //}
    //}
}
