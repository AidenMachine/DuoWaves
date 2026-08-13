#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <cuda.h>

#define BLOCK_SIZE 256

/* Kernel per il bombardamento. Ora 1D */
__global__ void bomb(int i_storm, int *d_flattened, float *d_layer, unsigned int* d_offsets, int layer_size, float calc_threshold) {
    unsigned int i_layer = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_layer < layer_size) {
        for (int j = 0; 
                j < (d_offsets[i_storm +1] - d_offsets[i_storm]) / 2;
                    j++) {
            float energy = (float)d_flattened[d_offsets[i_storm] + j*2+1] * 1000;
            int position = d_flattened[d_offsets[i_storm] + j*2];

            int distance = position - i_layer;
            if ( distance < 0 ) distance = - distance;

            distance = distance + 1;
            float atenuacion = sqrtf( (float)distance );

            float energy_k = energy / layer_size / atenuacion;

            /* Cambiata la threshold con quella pre-calcolata. Almeno cosi facendo i calcoli non verranno ripetuti inutilmente */
            if ( energy_k >= calc_threshold || energy_k <= -calc_threshold ) {
                d_layer[i_layer] = d_layer[i_layer] + energy_k;
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

/* Versione con interi perché non so come cazzo se fa sennò */
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
    unsigned int *h_offsets = (unsigned int *)malloc((num_storms + 1) * sizeof(unsigned int));                 // Contiene gli indici di partenza di ogni tempesta

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
    int *h_flattened = (int *)malloc(total_size * sizeof(int));

    /* Popolazione dell'array flattened nell'host tramite memcpy dal posval corrispondente. Stesse
     * attenzioni di cui sopra quando si tratta di definire la dimensione. */
    for (int i = 0; i < num_storms; i++) {
        /* Letteralmente
         * 1. Dove incollare i dati
         * 2. Da dove copiare i dati
         * 3. Quanti byte incollare
         */
        memcpy(h_flattened + h_offsets[i], storms[i].posval, storms[i].size * 2 * sizeof(int));
    }


    /* Trasferimento nel device: creazione e popolazione di tutte le strutture che serviranno nel
     * device. VALUTARE SPOSTAMENTO IN UNA FUNZIONE A PARTE */
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
    int grid_bomb = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_copy = (layer_size + 256 - 1)/256;                     /* Essendo solo la copia di un array il lavoro verrà fatto in una struttura ad una dimensione */

    for (int i = 0; i < num_storms; i++) {
        /* Ora non utilizziamo più un kernel 2D. Passando con un kernel 1D possiamo calcolare in modo corretto i valori esatti dei bombardamenti
         * ed, inoltre, non vi è più race-condition.
         * Invece che passare una variabile "dim3" ci basta passsare BLOCK_SIZE. Essendo un singolo valore, i valori di Y e Z saranno uguali ad 1.
         */
        bomb<<<grid_bomb, BLOCK_SIZE>>>(i, d_flattened, d_layer, d_offsets, layer_size, calc_threshold);

        /* Pointer-swap
         * Utilizzando questo andiamo a risparmiare tempo. Invece che perdere tempo con la copia, andiamo semplicemente a
         * swappare gli indirizzi dei puntatori. Lo stesso ragionamento applicato in MPI/OMP.
         */
        float *temp = d_layer_copy;
        d_layer_copy = d_layer;
        d_layer = temp;

        /* Applicazione del rilassamento. Principalmente molto simile ad MPI/OMP */
        relax<<<grid_copy, BLOCK_SIZE>>>(d_layer, d_layer_copy, layer_size);
    
        float *in = d_layer;                /* Puntatore "d'ingresso" ( che porta all'inizio ) della reduction */
        int   *in_idx = nullptr;
        int n = layer_size;                 /* Numero di elementi su cui fare la reduction */
        bool first = true;                  /* Flag per indicare al kernel che si tratta del primo passaggio della riduzione */

        /* Riduce progressivamente la dimensione dei dati su cui lavorare. Da "n" fino ad 1. */
        while (n > 1) {
            int nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;                            /* Calcolo progressivo dei blocchi che servono per lavorare su "n" elementi */
            float *out;                                                                 /* ( Questo ed il successivo ) sono puntatori temporanei per contenere i risultati intermedi di questa specifica passata */
            int   *out_idx;

            /* Allocazione dinamica in VRAM dei buffer di output per i massimi/indici parziali (uno per ogni blocco) */
            cudaMalloc(&out, nblocks * sizeof(float));
            cudaMalloc(&out_idx, nblocks * sizeof(int));

            reduce_argmax<<<nblocks, BLOCK_SIZE>>>(in, in_idx, out, out_idx, n, first);

            /* GESTIONE DELLA MEMORIA TEMPORANEA
             * Se non siamo al primo passaggio, 'in' e 'in_idx' puntano a buffer intermedi creati
             * nelle iterazioni precedenti del while, che ora non servono più e vanno liberati per evitare memory leak.
             * NOTA: Al primo passaggio (first == true) 'in' punta a d_layer, quindi NON deve essere liberato qui!
             */
            if (!first) {
                cudaFree(in); 
                cudaFree(in_idx); 
            }

            in = out;
            in_idx = out_idx;
            n = nblocks;
            first = false;
        }

        /* TRASFERIMENTO DEI RISULTATI FINALI ALL'HOST (CPU)
         * Terminata la riduzione (n == 1), 'in[0]' contiene il massimo globale della tempesta attuale
         * e 'in_idx[0]' contiene l'indice della cella in cui si trova tale massimo.
         */
        float h_max = 0.0f;
        int h_pos = -1;

        /* Copia dei due valori scalari dalla VRAM (GPU) alla memoria RAM (CPU) */
        cudaMemcpy(&h_max, &in[0], sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pos, &in_idx[0], sizeof(int),cudaMemcpyDeviceToHost);

        if (h_pos >= 0 && h_max > maximum[i]) {
            maximum[i] = h_max;
            positions[i] = h_pos;
    }

        cudaFree(in);
        cudaFree(in_idx);
    }

    cudaFree(d_flattened);
    cudaFree(d_offsets);
    cudaFree(d_layer);
    cudaFree(d_layer_copy);
}

/* APPUNTO SUL CAMBIAMENTO DEL KERNEL -------------------------------------------------------------------------------------------------------------------
 * Con il Kernel-2D si hanno due assi di threads:
 * Asse X -> il vecchio "i_layer"                   IL LAYER SU CUI LAVORAVA
 * Asse Y -> i_particle                             LA PARTICELLA SU CUI LAVORAVA
 * La GPU lanciava più threads per la stessa cella. Esempio...
 * Threads(5, 0) calcolava l'effetto della Particella 0 su layer[5]
 * Threads(5, 1) calcolava l'effetto della Particella 1 su layer[5]
 * Threads(5, 2) calcolava l'effetto della Particella 2 su layer[5]
 * Il problema nasceva quando ogni thread andava a scrivere in VRAM il valore.
 * Dato che non vi è nessun ordine e che ogni thread poteva finire prima dell'altro, succedeva che
 * magari nel primo test finiva il thread 2. Nel successivo lo 0... e così via.
 * Di conseguenza, dato che sulla macchina i float sommati in ordine diverso possono portare a risultati diversi, questo portava a risultati errati.
 *
 * Con il Kernel-1D invece le cose cambiano perché, semplicemente, ogni thread prende una SINGOLA CELLA DEL LAYER e cicla tutte le particelle.
 * In questo modo non vi è race-condition e non serve utilizzare AtomicAdd() perché quel singolo thread sarà l'unico ad accedere
 * a quell'area di memoria/registro. Inoltre non vi sarà alcuna "Data race" quindi l'ordine d'acccesso in memoria sarà sempre uguale e, quindi,
 * il risultato del programma sarà sempre deterministico.
 * ------------------------------------------------------------------------------------------------------------------------------------------------------ */

/* Possibile ottimizzazione:
 * Invece che chiamare CudaMalloc() nel while per cosi tante volte, non sarebbe male chiamarlo una singola volta.ù
 * Limitando le chiamate, in teoria, andiamo a risparmiare qualche ms.
 * In teoria credo si faccia prima del ciclo for finale.
 */
