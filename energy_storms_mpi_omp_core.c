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

    // Invece che farla calcolare volta per volta nell'if, è possibile calcolarla una singola volta e risparmiare molti cicli
    float local_threshold = THRESHOLD / layer_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Le variabili le ho dovute spostare qui perché altrimenti avrebbero mandato in crash il programma ( le avevo create in una zona parallela "single" )
    int left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

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
    //float *layer = malloc( sizeof( float ) * (local_size + 2) );
    //float *layer_copy = malloc( sizeof( float ) * (local_size + 2) );

    // PROVA CON POSIX_MEMALIGN PER EVITARE FALSE_SHARING
    float *layer = NULL, *layer_copy = NULL;
    int err_layer = posix_memalign((void**) &layer, 64, sizeof( float ) * (local_size + 2));
    int err_layer_copy = posix_memalign((void**) &layer_copy, 64, sizeof( float ) * (local_size + 2));

    if ( err_layer != 0 || err_layer_copy != 0 ) {
        fprintf(stderr,"Error_rank_%d: Posix-Memalign error\n", rank);
        exit( EXIT_FAILURE );
    }

    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error_rank_%d: Allocating the layer memory\n", rank);
        exit( EXIT_FAILURE );
    }

    /* TODO Tramite parallelizzazione sembra che questa rimanga l'opzione migliore...
     * peccato... tutte quelle ricerche sono state inutili. Almeno "chiamo" i threads una volta sola, cosi mi risparmio il tempo di
     * "richiamarli" in futuro ( vedi branch precedenti per capire di cosa sto parlando )
     */

    #pragma omp parallel private(i,j,k)
    {
        #pragma omp for
        for ( k = 0; k < local_size + 2; k++ ) {
            layer[k] = 0.0f;
            layer_copy[k] = 0.0f;
        }

        /* INTERA TEMPESTA. Qui ogni thread fa la stessa cosa aimé... che spreco Zi ------------------------------------*/
        for( i=0; i<num_storms; i++) {

            #pragma omp for
            for (k = 1; k <= local_size; k++) {
                int global_k = offset + k - 1;

                // L'idea è la stessa di prima ma con una roba diversa ovvero creiamo si la var. per salvare l'energia volta per vola
                // ( essendo nel for e con -O3, in teoria, dovrebbe essere una var. salvata tra i registri del core )
                // Andiamo a salvare il valore del layer[k] nel registro ( valore corretto ) -> andiamo ad aggiornare il registro volta per volta ( cosi non ci perdiamo nessun val. post virgola )
                // SOLO ALLA FINE del ciclo mettiamo tutto in RAM. In teoria i valori dovrebbero uscire sempre corretti e saltiamo tanti passaggi in RAM
                float save_cell_energy = layer[k];

                for (j=0; j<storms[i].size; j++) {
                    /* Energia dell'impatto, calcolata x 1000 */
                    float energy = (float)storms[i].posval[j*2+1] * 1000;

                    /* Calcola la posizione dell'impatto sul layer */
                    int position = storms[i].posval[j*2];

                    int distance = position - global_k;
                    if ( distance < 0 ) distance = - distance;
                    distance = distance + 1;
                    float atenuacion = sqrtf((float)distance);

                    float energy_k = energy / layer_size / atenuacion;

                    // Qui dovremmo risparmiare un pò di tempo
                    if ( energy_k >= local_threshold || energy_k <= -local_threshold ) {
                        save_cell_energy += energy_k;
                        //layer[k] += energy_k;
                    }
                }
                layer[k] = save_cell_energy;
            }
            // ---------------------------------------------------------------------------------------------------------

            // Solo un thread fa questa parte ( anche perché altrimenti volerebbero pacchetti a destra e sinistra )     <- #pragma omp single
            // Dato che però è il "thread master" che deve entrare per primo, cambio in #pragma omp master ( ma devo aggiungere la barriera manualmente perché non la ha )
            #pragma omp master
            {
                MPI_Isend(&layer[local_size], 1, MPI_FLOAT, right_neighbor, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Irecv(&layer[0], 1, MPI_FLOAT, left_neighbor, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Isend(&layer[1], 1, MPI_FLOAT, left_neighbor, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Irecv(&layer[local_size + 1], 1, MPI_FLOAT, right_neighbor, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Pointer-swap ( risparmia un bel pò di tempo utile )
                float *temp = layer_copy;
                layer_copy = layer;
                layer = temp;
            }
            #pragma omp barrier

            #pragma omp for
            for (k = 1; k <= local_size; k++) {
                // Con il pointer-swap i valori ai bordi rimangono del vecchio layer. Bisogna dunque copiarli per ottenere i valori corretti
                if (rank == 0 && k == 1) {
                    layer[k] = layer_copy[k];
                    continue;
                }
                if (rank == size - 1 && k == local_size) {
                    layer[k] = layer_copy[k];
                    continue;
                }
                layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
            }

            #pragma omp master
            {
                MPI_Isend(&layer[local_size], 1, MPI_FLOAT, right_neighbor, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Irecv(&layer[0], 1, MPI_FLOAT, left_neighbor, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Isend(&layer[1], 1, MPI_FLOAT, left_neighbor, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Irecv(&layer[local_size + 1], 1, MPI_FLOAT, right_neighbor, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Per eliminare i locks  ----------------------------------------------------------------------------------
                // Proviamo a dividere lo stesso il lavoro
                for ( k = 0; k < n_threads; k++ ) {
                    threads_results[k].local_maximum = -1.0f;
                    threads_results[k].local_positions = -1;
                }
            }
            #pragma omp barrier

            int start = 1, end = local_size;
            if ( rank == 0 ) start++;
            if ( rank == size - 1 ) end--;

            // Ogni thread calcola il singolo
            int my_id = omp_get_thread_num();
            LocalData my_data = { -1.0,-1 };

            // Il calcolo è praticamente lo stesso, solo che si va a scrivere i dati in un array condiviso, evitando il delay delle attese
            #pragma omp for
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

            // Per provare a fixare la non-associatività dei float
            #pragma omp barrier

            #pragma omp master
            {
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
            }
            #pragma omp barrier
        }
    }
    free(threads_results);
    free(layer);
    free(layer_copy);
}
