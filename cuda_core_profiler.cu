#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <cuda.h>

#define BLOCK_SIZE 256

/* ============================================================================
 * VERSIONE CON PROFILING - branch cuda-shared
 *
 * PER I COMMENTI SUL CODICE GUARDA IL FILE ORIGINALE energy_storms_cuda_core.cu
 *
 * Il timing usa cudaEvent_t e NON un clock host: i lanci dei kernel sono
 * asincroni, quindi un timer host attorno a "bomb<<<>>>" misurerebbe solo la
 * chiamata di lancio e non l'esecuzione sul device. Gli eventi vengono
 * registrati NELLO STREAM, quindi misurano il tempo reale sulla GPU.
 *
 * ATTENZIONE: nessuna riga stampata da questo file deve contenere la stringa
 * "Time:". Il ramo CUDA di execute_once() usa 2>&1 e suite_parse_time() prende
 * l'ULTIMO match di "Time:", quindi una riga del profiler sostituirebbe
 * silenziosamente il tempo reale del programma in ogni riga del CSV.
 *
 * NOTA IMPORTANTE SULLA FASE "reduce" DI QUESTO BRANCH:
 * a differenza di cuda-1D, questo branch NON pre-alloca i due buffer di
 * ping-pong: chiama cudaMalloc e cudaFree DENTRO il ciclo while, ad ogni
 * passata e per ogni tempesta. Il tempo di allocazione fa quindi legittimamente
 * parte del costo della riduzione ed e' incluso in t_reduce. Confrontando
 * t_reduce con quello di cuda-1D si misura esattamente quanto costa questa
 * differenza.
 * ========================================================================== */

#define PROF_TIC()        cudaEventRecord(ev_start)
#define PROF_TOC(acc)     do {                                   \
        cudaEventRecord(ev_stop);                                \
        cudaEventSynchronize(ev_stop);                           \
        float _ms = 0.0f;                                        \
        cudaEventElapsedTime(&_ms, ev_start, ev_stop);           \
        (acc) += (double)_ms / 1000.0;                           \
    } while (0)


/* Kernel per il bombardamento. 1D con tiling in shared memory */
__global__ void bomb(int i_storm, int *d_flattened, float *d_layer, unsigned int* d_offsets, int layer_size, float calc_threshold) {
    unsigned int i_layer = blockIdx.x * blockDim.x + threadIdx.x;

    /* Variabili in memoria condivisa. */
    __shared__ int s_pos[BLOCK_SIZE];
    __shared__ float s_energy[BLOCK_SIZE];

    unsigned int base = d_offsets[i_storm];
    int num_particles = (d_offsets[i_storm + 1] - base) / 2;

    bool active = i_layer < layer_size;

    for (int tile = 0; tile < num_particles; tile += BLOCK_SIZE) {
        int n = num_particles - tile;
        if (n > BLOCK_SIZE) n = BLOCK_SIZE;

        __syncthreads();

        if (threadIdx.x < n) {
            s_pos[threadIdx.x] = d_flattened[base + (tile + threadIdx.x)*2];
            s_energy[threadIdx.x] = ((float)d_flattened[base + (tile + threadIdx.x)*2 + 1] * 1000) / layer_size;
        }

        __syncthreads();

        if (active) {
            for (int j = 0; j < n; j++) {
                float energy = s_energy[j];
                int position = s_pos[j];

                int distance = position - i_layer;
                if ( distance < 0 ) distance = - distance;

                distance = distance + 1;
                float atenuacion = sqrtf( (float)distance );

                float energy_k = energy / atenuacion;

                if ( energy_k >= calc_threshold || energy_k <= -calc_threshold ) {
                    d_layer[i_layer] = d_layer[i_layer] + energy_k;
                }
            }
        }
    }
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

/* Kernel per il calcolo del massimo, sfrutta la tecnica di reduce */
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


void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    /* ---- VARIABILI PER IL PROFILING ---------------------------------------
     * t_setup  -> flattening host + cudaMalloc + cudaMemcpy H2D + cudaMemset
     * t_bomb   -> kernel di bombardamento con tiling shared (accumulato)
     * t_relax  -> kernel di rilassamento (accumulato)
     * t_reduce -> ciclo while della riduzione, cudaMalloc/cudaFree INCLUSI
     * t_d2h    -> le due cudaMemcpy scalari device->host (accumulato)
     * --------------------------------------------------------------------- */
    double t_setup = 0.0, t_bomb = 0.0, t_relax = 0.0, t_reduce = 0.0, t_d2h = 0.0;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    /* ===== FASE 1: SETUP =================================================== */
    PROF_TIC();

    float calc_threshold = (float) THRESHOLD / layer_size;

    unsigned int *h_offsets = (unsigned int *)malloc((num_storms + 1) * sizeof(unsigned int));

    h_offsets[0] = 0;
    for (int i = 0; i < num_storms; i++) {
        h_offsets[i+1] = h_offsets[i] + (storms[i].size * 2);
    }

    int total_size = (h_offsets[num_storms]);
    int *h_flattened = (int *)malloc(total_size * sizeof(int));

    for (int i = 0; i < num_storms; i++) {
        memcpy(h_flattened + h_offsets[i], storms[i].posval, storms[i].size * 2 * sizeof(int));
    }

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
        int gap = h_offsets[i + 1] - h_offsets[i];
        if (gap > max_storm) max_storm = gap;
    }

    free(h_offsets);
    free(h_flattened);

    int grid_bomb = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_copy = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    PROF_TOC(t_setup);
    /* ====================================================================== */


    for (int i = 0; i < num_storms; i++) {

        /* ===== FASE 2: BOMBARDAMENTO (tiling shared memory) =============== */
        PROF_TIC();
        bomb<<<grid_bomb, BLOCK_SIZE>>>(i, d_flattened, d_layer, d_offsets, layer_size, calc_threshold);
        PROF_TOC(t_bomb);
        /* ================================================================== */

        /* Pointer-swap: operazione host, costo trascurabile, non misurata */
        float *temp = d_layer_copy;
        d_layer_copy = d_layer;
        d_layer = temp;

        /* ===== FASE 3: RILASSAMENTO ======================================= */
        PROF_TIC();
        relax<<<grid_copy, BLOCK_SIZE>>>(d_layer, d_layer_copy, layer_size);
        PROF_TOC(t_relax);
        /* ================================================================== */

        float *in = d_layer;
        int   *in_idx = nullptr;
        int n = layer_size;
        bool first = true;

        /* ===== FASE 4: RIDUZIONE (cudaMalloc/cudaFree INCLUSI) ============ */
        PROF_TIC();
        while (n > 1) {
            int nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            float *out;
            int   *out_idx;

            cudaMalloc(&out, nblocks * sizeof(float));
            cudaMalloc(&out_idx, nblocks * sizeof(int));

            reduce_argmax<<<nblocks, BLOCK_SIZE>>>(in, in_idx, out, out_idx, n, first);

            if (!first) {
                cudaFree(in);
                cudaFree(in_idx);
            }

            in = out;
            in_idx = out_idx;

            n = nblocks;
            first = false;
        }
        PROF_TOC(t_reduce);
        /* ================================================================== */

        float h_max = 0.0f;
        int h_pos = -1;

        /* ===== FASE 5: TRASFERIMENTO RISULTATI DEVICE -> HOST ============= */
        PROF_TIC();
        cudaMemcpy(&h_max, &in[0], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pos, &in_idx[0], sizeof(int), cudaMemcpyDeviceToHost);
        PROF_TOC(t_d2h);
        /* ================================================================== */

        if (h_pos >= 0 && h_max > maximum[i]) {
            maximum[i] = h_max;
            positions[i] = h_pos;
        }

        /* Liberazione dell'ultimo buffer della riduzione: fa parte del costo
         * della gestione memoria della riduzione, quindi si accumula in
         * t_reduce e non in una fase a se stante. */
        PROF_TIC();
        cudaFree(in);
        cudaFree(in_idx);
        PROF_TOC(t_reduce);
    }

    cudaFree(d_flattened);
    cudaFree(d_offsets);
    cudaFree(d_layer);
    cudaFree(d_layer_copy);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    /* ---- STAMPA DEI RISULTATI DEL PROFILING -------------------------------
     * Riga leggibile dalla macchina, su stdout.
     * NESSUNA riga contiene la stringa "Time:" (vedi nota in testa al file).
     * --------------------------------------------------------------------- */
    double t_sum = t_setup + t_bomb + t_relax + t_reduce + t_d2h;

    printf("PROFILE: setup=%.6f bomb=%.6f relax=%.6f reduce=%.6f d2h=%.6f sum=%.6f\n",
           t_setup, t_bomb, t_relax, t_reduce, t_d2h, t_sum);

    fprintf(stderr, "\n=== RISULTATI PROFILING CUDA (Layer: %d, Storms: %d, Block: %d) ===\n",
            layer_size, num_storms, BLOCK_SIZE);
    fprintf(stderr, "Setup (malloc + H2D):        %8.4f s (%5.2f%%)\n", t_setup,  t_sum > 0 ? (t_setup /t_sum)*100.0 : 0.0);
    fprintf(stderr, "Bombardamento (shared):      %8.4f s (%5.2f%%)\n", t_bomb,   t_sum > 0 ? (t_bomb  /t_sum)*100.0 : 0.0);
    fprintf(stderr, "Rilassamento (kernel):       %8.4f s (%5.2f%%)\n", t_relax,  t_sum > 0 ? (t_relax /t_sum)*100.0 : 0.0);
    fprintf(stderr, "Riduzione (+malloc/free):    %8.4f s (%5.2f%%)\n", t_reduce, t_sum > 0 ? (t_reduce/t_sum)*100.0 : 0.0);
    fprintf(stderr, "Trasferimento D2H:           %8.4f s (%5.2f%%)\n", t_d2h,    t_sum > 0 ? (t_d2h   /t_sum)*100.0 : 0.0);
    fprintf(stderr, "Somma delle fasi:            %8.4f s\n", t_sum);
    fprintf(stderr, "==================================================================\n\n");
}
