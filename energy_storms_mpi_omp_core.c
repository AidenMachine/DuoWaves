#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>
#include<omp.h>
#include <string.h>

#include"energy_storms.h"

typedef struct{
    float local_maximum;
    int local_positions;
}LocalData;

// Ricorda di inserire nel "verbale" il fatto di aver aggiunto due parametri alla funzione
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k, size, rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Per eliminazione lock(), vedi a fine codice ( nella zona parallela con dentro anche un "for" ) ------------------
    int n_threads = omp_get_max_threads();
    LocalData *threads_results = malloc( n_threads * sizeof( LocalData ) );
    // -----------------------------------------------------------------------------------------------------------------

    // Calcolo chunk_locale
    int local_chunk = layer_size / size;            // Il chunk è "per tutti"
    int resto = layer_size % size;                  // Remainder come parola mi fa cagare quindi uso "resto"

    int local_size;             // Quante celle devo calcolare
    int offset;                 // Da quale indice ( della var. globale ) devo partire ( sul chunk locale )

    if ( rank < resto ) {
        local_size = local_chunk + 1;       // I primi rank ( processi ) prendono una cella extra
        offset = rank * local_size;        // Anche quelli prima di me hanno una cella in piu? -> scalo l'offset
    }else{
        local_size = local_chunk;           // Non prendo la cella extra
        offset = rank * local_chunk + resto;        // Scalo dai precedenti con
    }

    /* Ogni processo avrà il suo pezzo di layer sul quale farà i propri calcoli.
     * Utilizziamo sempre due copie per la fase futura di "rilassamento"
     * Aggiungiamo inoltre 2 CELLE EXTRA per layer ( perché per la fase di rilassamento ci
     * serve sapere che valore ha la cella precedente ( che però risiede nel layer di un altro
     * processo. APPLICARE RAGIONAMENTO DOPO BOMBARDAMENTO
     *
     * Va fatto ( local_size + 2 ) e non lasciando il 2 fuori dalla parentesi perché altrimenti
     * si andrebbe incontro alla creazione di "n" celle di memoria ( da 4 byte ) piu 2 celle da 2
     * Tutto questo perché l'operatore * con "sizeof" ha la precedenza
     */
    float *layer = malloc( sizeof( float ) * (local_size + 2) );
    float *layer_copy = malloc( sizeof( float ) * (local_size + 2) );
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error_rank_%d: Allocating the layer memory\n", rank);
        exit( EXIT_FAILURE );
    }

    /* TODO Tramite parallelizzazione sembra che questa rimanga l'opzione migliore...
     * peccato... tutte quelle ricerche sono state inutili. Almeno "chiamo" i threads una volta sola, cosi mi risparmio il tempo di
     * "richiamarli" in futuro ( vedi branch precedenti per capire di cosa sto parlando )
     */
    #pragma omp parallel for private(k)
       for ( k = 0; k < local_size + 2; k++ ) {
           layer[k] = 0.0f;
           layer_copy[k] = 0.0f;
       }

    // ---------------------------------------------------------------------
    
    /* Le storms non possono essere parallelizzate perché vale come "fattore temporale".
     * C'è la storm_0 che avviene PRIMA della storm_1. Se le parallelizzassimo allora i core e threads
     * andrebbero a lavorare su dati sbagliati. Il che non va bene perché uscirebbero risultati errati
     * ( magari ogni volta diversi )
     *
     * RICORDA: Ogni processo lavora sulle stesse particelle MA suddividendo l'effetto in layer diversi!ss
     */
    /* INTERA TEMPESTA */
    for( i=0; i<num_storms; i++) {

        /* SINGOLA PARTICELLA */
        #pragma omp parallel for private(j,k)
        for( j=0; j<storms[i].size; j++ ) {

            /* Energia dell'impatto, calcolata x 1000 */
            float energy = (float)storms[i].posval[j*2+1] * 1000;

            /* Calcola la posizione dell'impatto sul layer */
            int position = storms[i].posval[j*2];

            // DISPERSIONE ENERGIA DELLA SINGOLA PARTICELLA
            for( k = 1; k <= local_size; k++ ) {
                /* Si parte da k = 1 per non contaminare le "celle fantasma" ma come k globale
                 * consideriamo offset + k - 1 perché altrimenti si andrebbe a contare la cella
                 * successiva
                 *
                 * k = 1 -> Dove dovrei partire con l'array normale
                 * offset + k - 1 -> Parto dall'indice del pezzo che ha il processo
                 */

                // PARTE NUOVA ----------------------------------------------------------------------------------------
                // Evito i commenti perché tanto stanno sopra ( calcoli equivalenti )
                // Faccio cosi perché, in teoria, non vi è chiamata alla funzione, dunque, nessuna creazione e copia di parametri etc...
                int global_k = offset + k - 1;
                int distance = position - global_k;
                if ( distance < 0 ) distance = - distance;
                distance = distance + 1;
                float atenuacion = sqrtf((float)distance);//sqrt_table[distance];               // Speriamo sia piu fast ( spoiler, non lo era )

                float energy_k = energy / layer_size / atenuacion;

                if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size ) {
                    #pragma omp atomic
                    layer[k] += energy_k;
                }
                // ----------------------------------------------------------------------------------------------------
            }
        }

        /*
         * Manca solo questa parte Zi
         * Tocca riempire layer[0] e layer[local_size + 1] con i valori dei vicini.
         * Ricorda che, giustamente, il primo e l'ultimo layer NON devono, rispettivamente,
         * mandarsi nulla. Inoltre, usiamo Sendrecv() e non i singoli Send() e Recv().
         * ( dobbiamo usare layer_copy ( credo solo in questo caso ).
         * Il resto lo lascio a te ( anche perché non so alto ahah ).
         * Ma confido in te, Brother.
         */
        int left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int right_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;


        // 1. Tutti mandano il proprio bordo destro al vicino di destra,
        // e ricevono il bordo sinistro dal vicino di sinistra (Tag 0)
        MPI_Sendrecv(&layer[local_size], 1, MPI_FLOAT, right_neighbor, 0,
                     &layer[0], 1, MPI_FLOAT, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 2. Tutti mandano il proprio bordo sinistro al vicino di sinistra,
        // e ricevono il bordo destro dal vicino di destra (Tag 1)
        MPI_Sendrecv(&layer[1], 1, MPI_FLOAT, left_neighbor, 1,
                     &layer[local_size + 1], 1, MPI_FLOAT, right_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* METODO VECCHIO
        MPI_Sendrecv(&layer[1], 1, MPI_FLOAT, left_neighbor, 0,
                     &layer[0], 1, MPI_FLOAT, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&layer[local_size], 1, MPI_FLOAT, right_neighbor, 0,
                     &layer[local_size + 1], 1, MPI_FLOAT, right_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        */

        // Pointer-swap ( risparmia un bel pò di tempo utile )
        float *temp = layer_copy;
        layer_copy = layer;
        layer = temp;

        #pragma omp parallel for private(k)
        for (k = 1; k <= local_size; k++) {
            if (rank == 0 && k == 1) continue;
            if (rank == size - 1 && k == local_size) continue;
            layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
        }


        /* --- MODIFICA: SECONDO SCAMBIO MPI (Per la ricerca del massimo) --- */
        /* Serve a far sapere ai vicini che la tua energia  cambiata dopo la media */
        MPI_Sendrecv(&layer[local_size], 1, MPI_FLOAT, right_neighbor, 0,
                     &layer[0], 1, MPI_FLOAT, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&layer[1], 1, MPI_FLOAT, left_neighbor, 1,
                     &layer[local_size + 1], 1, MPI_FLOAT, right_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* --- MODIFICA: SECONDO SCAMBIO MPI (Per la ricerca del massimo) --- */
        /* Serve a far sapere ai vicini che la tua energia è cambiata dopo la media
        MPI_Sendrecv(&layer[1], 1, MPI_FLOAT, left_neighbor, 0,
                     &layer[0], 1, MPI_FLOAT, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&layer[local_size], 1, MPI_FLOAT, right_neighbor, 0,
                     &layer[local_size + 1], 1, MPI_FLOAT, right_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        */

        // PARTE NUOVA ( per eliminare i locks ) -----------------------------------------------------------------------
        for ( k = 0; k < n_threads; k++ ) {
            threads_results[k].local_maximum = -1.0f;
            threads_results[k].local_positions = -1;
        }

        int start = 1, end = local_size;
        if ( rank == 0 ) start++;
        if ( rank == size - 1 ) end--;

        #pragma omp parallel
        {
            // Ogni thread calcola il singolo
            int my_id = omp_get_thread_num();
            LocalData my_data = { -1.0,-1 };

            // Il calcolo è praticamente lo stesso, solo che si va a scrivere i dati in un array condiviso, evitando il delay delle attese
            #pragma omp for private(k)
            for( k = start; k <= end; k++ ) {
                if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {

                    if ( layer[k] > my_data.local_maximum ) {
                        my_data.local_maximum = layer[k];
                        my_data.local_positions = offset + k - 1;
                    }
                }
            }
            // A fine "zona parallela" si va a scrivere tutto sull'array
            threads_results[my_id] = my_data;
        }

        // Calcolo del massimo dell'array ( evito di usare OMP perché tanto il tempo di settarlo che perdo solo tempo )
        LocalData global_max = { -1.0,-1 };
        for ( k = 0; k < n_threads; k++ ) {
            if ( threads_results[k].local_maximum > global_max.local_maximum ) {
                global_max = threads_results[k];
            }
        }


        LocalData data_recv;        // Sarà la var. dove il processo MPI riceverà global_max
        MPI_Allreduce(&global_max, &data_recv, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        maximum[i] = data_recv.local_maximum;
        positions[i] = data_recv.local_positions;
        // ------------------------------------------------------------------------------------------------------------
    }
    free(threads_results);
    free(layer);
    free(layer_copy);
}
