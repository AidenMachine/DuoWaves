#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <cuda.h>

#define BLOCK_SIZE 256

/* Kernel per il bombardamento. Ora 1D */
__global__ void bomb(int i_storm, int *d_flattened, float *d_layer, unsigned int* d_offsets, int layer_size, float calc_threshold) {
    unsigned int i_layer = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* Variabili in memoria condivisa. */
    __shared__ int s_pos[BLOCK_SIZE];
    __shared__ float s_energy[BLOCK_SIZE];

    unsigned int base = d_offsets[i_storm];
    int num_particles = (d_offsets[i_storm + 1] - base) / 2;
    
    bool active = i_layer < layer_size;
    
    /* Dichiarazione della variabile per ridurre gli accessi alla memoria */
    float save_cell_energy = 0.0f;
    if(active) {
    	save_cell_energy = d_layer[i_layer];
    }

    for (int tile = 0; tile < num_particles; tile += BLOCK_SIZE) {
        int tile_count = num_particles - tile;
        if (tile_count > BLOCK_SIZE) tile_count = BLOCK_SIZE;

        /* Sezione in cui si carica il blocco corrispondente al thread in questione in shared memory. */
        __syncthreads();

        if (threadIdx.x < tile_count) {
            s_pos[threadIdx.x] = d_flattened[base + (tile + threadIdx.x)*2];
            s_energy[threadIdx.x] = ((float)d_flattened[base + (tile + threadIdx.x)*2 + 1] * 1000) / layer_size;
        }

        __syncthreads();
        
        /* Calcolo del contributo di ogni particella del tile */
        if (active) {
            for (int j = 0; j < tile_count; j++) {
                float energy = s_energy[j];
                int position = s_pos[j];

                int distance = position - i_layer;
                if ( distance < 0 ) distance = - distance;

                distance = distance + 1;
                float atenuacion = sqrtf( (float)distance );

                float energy_k = energy / atenuacion;

                /* Cambiata la threshold con quella pre-calcolata, come nella versione MPI/OMP. */
                if ( energy_k >= calc_threshold || energy_k <= -calc_threshold ) {
                    save_cell_energy += energy_k;
                }
            }
        }
    }
    /* Solo fuori dal for andiamo a salvare i dati in VRAM. Meno accessi == piu veloce */
    if(active) d_layer[i_layer] = save_cell_energy;
}

/* Kernel per la fase di rilassamento */
__global__ void relax(float * layer, float * layer_copy, int layer_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0 || i == layer_size - 1) {
        layer[i] = layer_copy[i];
    } else if (i < layer_size - 1) {
        layer[i] = (layer_copy[i-1] + layer_copy[i] + layer_copy[i+1]) / 3.0f;
    }
}
/* Kernel per il calcolo del massimo, è nei fatti una riduzione. */
__global__ void reduce_argmax(float* in, int* in_idx, float* out, int* out_idx, int n, bool first_pass) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sdata[BLOCK_SIZE];
    __shared__ int   sidx[BLOCK_SIZE];

    sdata[tid] = -INFINITY;
      sidx[tid]  = -1;

      if (i < n) {
          if (!first_pass) {
              sdata[tid] = in[i];
              sidx[tid]  = in_idx[i];
          } else if (i > 0 && i < n - 1 && in[i] > in[i-1] && in[i] > in[i+1]) {
              sdata[tid] = in[i];
              sidx[tid]  = (int)i;
          }
      }
      __syncthreads();

      for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
          if (tid < s) {
              bool better = sdata[tid + s] > sdata[tid];
              bool tie_lower = sdata[tid + s] == sdata[tid] && sidx[tid + s] >= 0 &&
                               (sidx[tid] < 0 || sidx[tid + s] < sidx[tid]);
              if (better || tie_lower) {
                  sdata[tid] = sdata[tid + s];
                  sidx[tid]  = sidx[tid + s];
              }
          }
          __syncthreads();
      }

      if (tid == 0) {
          out[blockIdx.x]     = sdata[0];
          out_idx[blockIdx.x] = sidx[0];
      }
}

/* SOLO PER DEBUG: stampa solamente i primi 35 elementi di un array
 * in memoria del device, portandolo in memoria host */
void DEBUG_print_device_array(float *d_layer, int layer_size) {
    float *h_layer = (float *) malloc(layer_size * sizeof(float));
    cudaMemcpy(h_layer, d_layer, layer_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < layer_size; i++) {
        if (i > 35)
            break;
        printf("%f ", h_layer[i]);
    }
    printf("\n");
    free(h_layer);
}

/* Stessa cosa di sopra ma per un array in memoria host */
void DEBUG_print_host_array(int *h_layer, int layer_size) {
    for (int i = 0; i < layer_size; i++) {
        if (i > 35)
            break;
        printf("%d ", h_layer[i]);
    }
    printf("\n");
}

/* Versione con interi */
void DEBUG_print_host_array(unsigned int *h_layer, int layer_size) {
    for (int i = 0; i < layer_size; i++) {
        if (i > 35)
            break;
        printf("%d ", h_layer[i]);
    }
    printf("\n");
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    /* Per ottimizzare un pò i calcoli, calcolo la "THRESHOLD" da controllare in "bomb" subito ed una sola volta */
    float calc_threshold = (float) THRESHOLD / layer_size;
    printf("calc_threshold: %f\n", calc_threshold);

    unsigned int *h_offsets = (unsigned int *)malloc((num_storms + 1) * sizeof(unsigned int));

    h_offsets[0] = 0;

    /* Popolazione dell'array di offset nell'host. ATTENZIONE: per ogni struct di una Storm si ha
     * l'array posval che ha 2 valori per ogni particella, ma size descrive il numero di particelle.
     * Da qui la necessaria moltiplicazione per 2. */
    for (int i = 0; i < num_storms; i++) {
        h_offsets[i+1] = h_offsets[i] + (storms[i].size * 2);                   /* Calcolo inizio tempesta successiva */
    }

    /* Numero celle di memoria massime per contenere tutte le particelle di ogni tempesta*/
    int total_size = (h_offsets[num_storms]);

    /* Allocazione dell'array "flattato" nella dimensione appena calcolata. Conterrà tutti i dati ( posizione ed energia ) di tutte le tempeste */
    int *h_flattened = (int *)malloc(total_size * sizeof(int));

    /* Popolazione dell'array flattened nell'host tramite memcpy dal posval corrispondente. Stesse
     * attenzioni di cui sopra quando si tratta di definire la dimensione. */
    for (int i = 0; i < num_storms; i++) {
        memcpy(h_flattened + h_offsets[i], storms[i].posval, storms[i].size * 2 * sizeof(int));
    }


    /* Trasferimento nel device: creazione e popolazione di tutte le strutture che serviranno nel
     * device. */
    float *d_layer, *d_layer_copy;
    unsigned int * d_offsets;
    int *d_flattened;

    cudaError_t offsets_err = cudaMalloc((void **)&d_offsets, (num_storms + 1) * sizeof(unsigned int));
    cudaError_t flattened_err = cudaMalloc((void **)&d_flattened, total_size * sizeof(int));
    cudaError_t layer_err = cudaMalloc((void**)&d_layer, layer_size * sizeof(float));
    cudaError_t layercopy_err = cudaMalloc((void**)&d_layer_copy, layer_size * sizeof(float));

    cudaError_t offsetcpy_err = cudaMemcpy(d_offsets, h_offsets, (num_storms + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaError_t flatcpy_err = cudaMemcpy(d_flattened, h_flattened, total_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaError_t mems1_err = cudaMemset(d_layer, 0, layer_size * sizeof(float));
    cudaError_t mems2_err = cudaMemset(d_layer_copy, 0, layer_size * sizeof(float));

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
        /* Calcola la grandezza di ogni tempesta. */
        int gap = h_offsets[i + 1] - h_offsets[i];
        if (gap > max_storm) max_storm = gap;
    }

    free(h_offsets);
    free(h_flattened);

    int grid = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int max_reduction_blocks = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Allocazione dei DUE buffer di supporto FUORI dal loop for (una sola volta)
     * Ne utilizziamo due perché se ne utilizzassi uno singolo, andrei a scrivere e leggere contemporaneamente i dati dallo stesso buffer ( ricorda i thread che lavorano in parallelo )
     * Usandone due di buffer invece faccio cosi:
     * 1. Leggo dal layer
     * 2. Scrivo i risultati su A
     * 3. Ora leggo i dati da A
     * 4. Scrivo i risultati su B
     * 5. Leggo dati da B
     * 6. Scrivo su A...
     *
     * una sorta di effetto "ping pong"
     */
    float *d_max_output[2];
    int *d_pos_output[2];

    /* Alloco subito ( ed una volta sola ) lo spazio necessario. Invece che usare spazi diversi ogni volta, si va ad utilizzare sempre lo stesso
     * cosi da risparmiare qualche ciclo di clock.
     */
    cudaMalloc((void**)&d_max_output[0], max_reduction_blocks * sizeof(float));
    cudaMalloc((void**)&d_max_output[1], max_reduction_blocks * sizeof(float));
    cudaMalloc((void**)&d_pos_output[0], max_reduction_blocks * sizeof(int));
    cudaMalloc((void**)&d_pos_output[1], max_reduction_blocks * sizeof(int));
    /* ----------------------------------------------------------------------------------------------------------------- */

    for (int i = 0; i < num_storms; i++) {
        bomb<<<grid, BLOCK_SIZE>>>(i, d_flattened, d_layer, d_offsets, layer_size, calc_threshold);

        /* Pointer-swap, come in MPI/OMP. */
        float *temp = d_layer_copy;
        d_layer_copy = d_layer;
        d_layer = temp;

        /* Applicazione del rilassamento. Principalmente molto simile ad MPI/OMP */
        relax<<<grid, BLOCK_SIZE>>>(d_layer, d_layer_copy, layer_size);
    
        float *in = d_layer;
        int   *in_idx = nullptr;
        int n = layer_size;
        bool first = true;

        /* Alterna tra 0 ed 1 */
        int pingpong = 0;

        /* Riduce progressivamente la dimensione dei dati su cui lavorare. Da "n" fino ad 1. */
        while (n > 1) {
            /* Calcolo progressivo dei blocchi che servono per lavorare su "n" elementi */
            int nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

             reduce_argmax<<<nblocks, BLOCK_SIZE>>>(in, in_idx, d_max_output[pingpong], d_pos_output[pingpong], n, first);

            in = d_max_output[pingpong];
            in_idx = d_pos_output[pingpong];

            /* GESTIONE DELLA MEMORIA TEMPORANEA
             * Se non siamo al primo passaggio, 'in' e 'in_idx' puntano a buffer intermedi creati
             * nelle iterazioni precedenti del while, che ora non servono più e vanno liberati per evitare memory leak. */

            pingpong = 1 - pingpong;
            n = nblocks;
            first = false;
        }

        /* TRASFERIMENTO DEI RISULTATI FINALI ALL'HOST (CPU)
         * Terminata la riduzione (n == 1), 'in[0]' contiene il massimo globale della tempesta attuale
         * e 'in_idx[0]' contiene l'indice della cella in cui si trova tale massimo.
         */
        float h_max = 0.0f;
        int h_pos = -1;
        
        cudaMemcpy(&h_max, &in[0], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pos, &in_idx[0], sizeof(int), cudaMemcpyDeviceToHost);

        if (h_pos >= 0 && h_max > maximum[i]) {
            maximum[i] = h_max;
            positions[i] = h_pos;
    }
    }

    cudaFree(d_max_output[0]);
    cudaFree(d_max_output[1]);
    cudaFree(d_pos_output[0]);
    cudaFree(d_pos_output[1]);
    cudaFree(d_flattened);
    cudaFree(d_offsets);
    cudaFree(d_layer);
    cudaFree(d_layer_copy);
}
