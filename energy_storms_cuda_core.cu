#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <cuda.h>

#define BLOCK_SIZE 256

/* Kernel per il bombardamento */
__global__ void bomb(int i_storm, float *d_flattened, float *d_layer, unsigned int* d_offsets, int layer_size, int total_size, int num_storms, float calc_threshold) {
    unsigned int i_layer = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i_particle = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_storm < num_storms && i_particle < ((d_offsets[i_storm + 1] - d_offsets[i_storm]) / 2) && i_layer < layer_size) {
        float energy = d_flattened[d_offsets[i_storm] + i_particle*2+1] * 1000;
        float position = d_flattened[d_offsets[i_storm] + i_particle*2];

        int distance = position - i_layer;
        if ( distance < 0 ) distance = - distance;

        distance = distance + 1;
        float atenuacion = sqrtf( (float)distance );

        float energy_k = energy / layer_size / atenuacion;

        /* Cambiata la threshold con quella pre-calcolata. Almeno cosi facendo i calcoli non verranno ripetuti inutilmente */
        if ( energy_k >= calc_threshold || energy_k <= -calc_threshold )
            /* In teoria dovremmo aggiungere "AtomicAdd"
             * atomicAdd(&d_layer[i_layer], energy_k);
             * In questo modo andiamo ad eliminare la race_condition, rallentando però l'esecuzione. */
            d_layer[i_layer] = d_layer[i_layer] + energy_k;
    }
}

/* Kernel che copia i valori da layer a layer_copy
 * DA ELIMINARE QUANDO VUOI, BROTHER
 */
__global__ void copy(float *layer, float *layer_copy, int layer_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < layer_size) {
        layer_copy[i] = layer[i];
    }

}

/* Kernel per la fase di rilassamento */
__global__ void relax(float * layer, float * layer_copy, int layer_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Non dovremmo fare tipo cosi ?
    * if (i == 0 || i == layer_size - 1) {
        layer[i] = layer_copy[i]; // Mantiene il valore aggiornato dal bombardamento
    } else if (i < layer_size - 1) {
        layer[i] = (layer_copy[i-1] + layer_copy[i] + layer_copy[i+1]) / 3.0f;
    }
    */

    if (i < layer_size -1 && i > 0) {
       layer[i] = ( layer_copy[i-1] + layer_copy[i] + layer_copy[i+1] ) / 3.0f;            /* Ho cambiato il valore da 3 a 3.0f come in MPI/OMP */
    }
}

/* Kernel per il calcolo del massimo, sfrutta la tecnica di reduce */
__global__ void reduce_argmax(float* in, int* in_idx, float* out, int* out_idx, int n, bool first_pass) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sdata[BLOCK_SIZE];
    __shared__ int   sidx[BLOCK_SIZE];

    if (i < n) {
        sdata[tid] = in[i];
        sidx[tid]  = first_pass ? i : in_idx[i];
    } else {
        sdata[tid] = 0.0f;
        sidx[tid]  = -1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            sidx[tid]  = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x]     = sdata[0];
        out_idx[blockIdx.x] = sidx[0];
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    /* Per ottimizzare un pò i calcoli, calcolo la "THRESHOLD" da controllare in "bomb" subito ed una sola volta */
    float calc_threshold = (float) THRESHOLD / layer_size;

    /* Prima parte: il problema principale da affrontare è che storms sia un array di strutture con
     * puntatori ad altri array, e  quindi allocare tutto nel device diventerebbe lungo. Per tale 
     * motivo, prima di allocare nel device si effettua una operazione di "flattening" dell'arrray 
     * di storms in un gigantesco array che contenga i valori di tutte le storms uno dietro l'altro,
     * con un array di offset che per ogni storm dice esattamente dove si trova l'inizio della storm */

    /* Spiegazione per me ( Anto )
     * Essnzialmente, l'offset viene calcolato con una "tempesta extra" per capire dove finisce l'ultima tempesta ( ovvero la [num_storms - 1] )
     * Con il "+1" abbiamo che:
     * h_offset[0] = 0 ( T1 inizia a 0 )
     * h_offset[1] = 4 ( finisce T1 ed inizia T2 )
     * h_offset[2] = 10 ( finisce T2 ed inizia T3 )
     * h_offset[3] = 12 ( finisce T3 e basta )
     *
     * Senza il "+1" abbiamo invece che:
     * h_offset[0] = 0 ( T1 inizia a 0 )
     * h_offset[1] = 4 ( finisce T1 ed inizia T2 )
     * h_offset[2] = 10 ( finisce T2 ed inizia T3 ma cosi facendo non sappiamo dove finisce T3! )
     */
    unsigned int *h_offsets = (unsigned int *)malloc((num_storms + 1) * sizeof(int));                 // Contiene gli indici di partenza di ogni tempesta

    /* Qui inizia la prima tempesta */
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
    float *h_flattened = (float *)malloc(total_size * sizeof(float));

    /* Popolazione dell'array flattened nell'host tramite memcpy dal posval corrispondente. Stesse
     * attenzioni di cui sopra quando si tratta di definire la dimensione. */
    for (int i = 0; i < num_storms; i++) {
        /* Letteralmente
         * 1. Dove incollare i dati
         * 2. Da dove copiare i dati
         * 3. Quanti byte incollare
         */
        memcpy(h_flattened + h_offsets[i], storms[i].posval, storms[i].size * 2 * sizeof(unsigned int));
    }


    /* Trasferimento nel device: creazione e popolazione di tutte le strutture che serviranno nel
     * device. VALUTARE SPOSTAMENTO IN UNA FUNZIONE A PARTE */
    float *d_layer, *d_layer_copy , *d_flattened;
    unsigned int * d_offsets;

    cudaError_t offsets_err = cudaMalloc((void **)&d_offsets, (num_storms + 1) * sizeof(unsigned int));
    cudaError_t flattened_err = cudaMalloc((void **)&d_flattened, total_size * sizeof(float));
    cudaError_t layer_err = cudaMalloc((void**)&d_layer, layer_size * sizeof(float));
    cudaError_t layercopy_err = cudaMalloc((void**)&d_layer_copy, layer_size * sizeof(float));

    cudaError_t offsetcpy_err = cudaMemcpy(d_offsets, h_offsets, num_storms + 1 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaError_t flatcpy_err = cudaMemcpy(d_flattened, h_flattened, total_size * sizeof(float), cudaMemcpyHostToDevice);

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

    printf("allocato tutto in device\n");

    int max_storm = 0;
    for (int i = 0; i + 1 < num_storms + 1; i++) {
        /* Calcola la grandezza di ogni tempesta. Nello specifico vede quanti elementi ha ogni tempesta */
        int gap = h_offsets[i + 1] - h_offsets[i];
        if (gap > max_storm) max_storm = gap;                           /* Se il valore è superiore al massimo registrato fino ad ora allora lo cambia */
    }

    /* Liberazione della memoria degli array sull'host */
    free(h_offsets);
    free(h_flattened);

    /* Non utilizziamo "ceil()" per limitare conversioni etc... in teoria cosi facendo dovremo salvare qualche ciclo di clock
     * Inoltre, la X viene calcolata in questo modo perché cosi facendo possiamo ottenere incrementare il risultato. Questo serve perché
     * C tronca il risultato. Senza quel "+31" essenzialmente andremo a creare un valore inferiore a quello che ci serve per il block
     *
     * Il principio è praticamente lo stesso per la Y
     * La Z invece la manteniamo ad 1, cosi da avere una "griglia 2D ( a due dimensioni )
     */
    dim3 grid_bomb((layer_size + 31)/32, (max_storm + 4)/4, 1);
    int grid_copy = (layer_size + 256 - 1)/256;                     /* Essendo solo la copia di un array il lavoro verrà fatto in una struttura ad una dimensione */

    for (int i = 0; i < num_storms; i++) {
        bomb<<<grid_bomb, BLOCK_SIZE>>>(i, d_flattened, d_layer, d_offsets, layer_size, total_size, num_storms, calc_threshold);
        cudaDeviceSynchronize();

        /* PER LUCA ----------------------------------------------------------------------------------------------------------- */
        /* Tutta questa parte qui sotto la possiamo direttamente eliminare. Il pointer-swap si può fare anche in questo caso.
         * Non solo in MPI/OMP */
        copy<<<grid_copy, BLOCK_SIZE>>>(d_layer, d_layer_copy, layer_size);
        cudaDeviceSynchronize();
        /* -------------------------------------------------------------------------------------------------------------------- */

        /* Scommenta come puoi ------------------------------------------------------------------------------------------------ */
        /**
        float *temp = layer_copy;
        layer_copy = layer;
        layer = temp;
        * ---------------------------------------------------------------------------------------------------------------------- */
        
        relax<<<grid_copy, BLOCK_SIZE>>>(d_layer, d_layer_copy, layer_size);
        cudaDeviceSynchronize();
    
        float *in = d_layer;
        int   *in_idx = nullptr;
        int n = layer_size;
        bool first = true;

        while (n > 1) {
            int nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            float *out;
            int   *out_idx;
            cudaMalloc(&out, nblocks * sizeof(float));
            cudaMalloc(&out_idx, nblocks * sizeof(int));

            reduce_argmax<<<nblocks, BLOCK_SIZE>>>(in, in_idx, out, out_idx, n, first);

            if (!first) { cudaFree(in); cudaFree(in_idx); }
            in = out; in_idx = out_idx; n = nblocks; first = false;
        }


        cudaMemcpy((void *) &maximum[i], (void *) &in[0], sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy((void *) &positions[i], (void *) &in_idx[0], sizeof(float),cudaMemcpyDeviceToHost);

        cudaFree(in);
        cudaFree(in_idx);
    }

    cudaFree(d_flattened);
    cudaFree(d_offsets);
    cudaFree(d_layer);
    cudaFree(d_layer_copy);
}
