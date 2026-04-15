#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>
#include<omp.h>
#include"energy_storms.h"

typedef struct{
    float local_maximum;
    int local_positions;
}LocalData;

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
static void update( float *layer, int layer_size, int k, int pos, float energy, int global_k) {
    /* Il calcolo rimane lo stesso, solo che si utilizza l'indice k globale, altrimenti
     * si andrebbe in Segmentation_fault */

    /* Anche se i due processi non comunicano, le particelle vengono elaborate tutte allo stesso modo,
     * dunque attraverso il calcolo di "distance", "sqrt(...)" etc... si ottiene quanta energia viene
     * dispersa ( se è minore di THRESHOLD/layer_size allora non viene salvata )
     */
    int distance = pos - global_k;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold
     * TODO Probabilmente è l'unico punto dove inserirò ATOMIC ( per ora )
     */
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size ) {
        #pragma omp atomic
        layer[k] = layer[k] + energy_k;
    }
    /*
     * Inserire #PRAGMA_OMP_ATOMIC qui sopra non è stato utile se il "parallel for" divide il layer della singola particella,
     * per il semplice fatto che, tecnicamente, nessuna cella verrà mai visitata contemporaneamente.
     * Diventa utile ora, passando il "parallel for" da quello "del layer" a quello delle particelle.
     */
}


// Ricorda di inserire nel "verbale" il fatto di aver aggiunto due parametri alla funzione
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions, const int rank, const int size) {
    int i, j, k;
    // PARTE NUOVA --------------------------------------------------------
    float local_maximum = -1.0f;
    int local_positions = -1;

    omp_lock_t writelock;
    omp_init_lock(&writelock);

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
    float *layer = ( float * )malloc( sizeof( float ) * (local_size + 2) );
    float *layer_copy = ( float * )malloc( sizeof( float ) * (local_size + 2) );
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error_rank_%d: Allocating the layer memory\n", rank);
        exit( EXIT_FAILURE );
    }

    /* Due for staccati in questo modo funzionano bene in seriale. Ma in parallelo non possiamo
     * saperlo. Forse converrebbe unificare tutto (?). Perché facendo "pragma" due volte vado incontro
     * al "set" dei threads per due volte ( e non so se questo comporta rallentamenti )
     * TODO Proviamo a fine parallelizzazione! */
    #pragma omp parallel for
        for ( k = 0; k < local_size + 2; k++ ) layer[k] = 0.0f;

    #pragma omp parallel for
        for ( k = 0; k < local_size + 2; k++ ) layer_copy[k] = 0.0f;


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
        #pragma omp parallel for
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
                update( layer, layer_size, k, position, energy , offset + k - 1);
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

        /* LUCA: parallelizzato senza troppi cazzi, é solo una copia */
        # pragma omp parallel for
        for( k=0; k<local_size; k++ ) 
            layer_copy[k] = layer[k];

        /* LUCA: qui invece i cazzoni rischiamo di prenderli se non si fa attenzione; il primo e l'ultimo
         * GLOBALI non devono essere presi in considerazione, e bisogna mandare/ricevere tra processi
         * la prima e ultima cella; OCCHIO (cit. Lele Adani)*/
        int prev_rank = rank-1; int succ_rank = rank+1;                                     // TODO Letteralmente i "layer affianco a quello del processo"
        float my_first, my_last;                                                            // TODO Contiene i valori da mandare ai vicini
        float prev_last, succ_first;                                                        // TODO Riceve i valori dai vicini ( l'ultimo del processo precedente e il primo del processo successivo )

        /* Fase 1: si fanno le SendRecv. Ovviamente, il primo rank non riceve/manda da/a quello precedente  
         * l'ultimo rank non riceve/manda da/a quello dopo */
        if (!(prev_rank==-1)) // TODO Ma perche hai usato "!"? Non sarebbe meglio direttamente fare " != " ?
            // TODO Non hai inserito il valore in my_first
            MPI_Sendrecv(&my_first, 1, MPI_FLOAT, prev_rank, 0,
                       &prev_last, 1, MPI_FLOAT, prev_rank, 0,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (!(succ_rank==size)) // TODO Idem qui Zi
            // TODO Non hai inserito il valore in my_last
            MPI_Sendrecv(&my_last, 1, MPI_FLOAT, succ_rank, local_size-1,
                       &succ_first, 1, MPI_FLOAT, succ_rank, local_size-1,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for( k=0; k<local_size; k++ ) {
            /* Fase 2: se sono la prima o l'ultima cella globali, salta */
            if ((prev_rank == -1 && k==0) || (succ_rank == size && k == local_size-1)) continue;
            
            /* Fase 3: computa l'effettivo valore. Si fa adoperando prev (o il valore prima nell'array
*              o il valore ricevuto dal rank precedente) e succ (o il valore dopo nell'array o il valore
*              ricevuto dal rank successivo) */
            float prev = k == 0 ? prev_last : layer_copy[k-1];
            float succ = k == local_size ? succ_first : layer_copy[k-1];              // TODO Avevi scritto "prev" di nuovo

            layer[k] = ( prev + layer_copy[k] + succ ) / 3;

            // TODO Penso che per l'OpenMP ( per quanto riguarda il for ) si possa fare senza incappare a problemi
            // TODO Al limite lo vediamo domani
          }       

        int start = 1, end = local_size;
        if ( rank == 0 ) start++;
        if ( rank == size - 1 ) end--;

        #pragma omp parallel for
        for( k = start; k <= end; k++ ) {
            if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {

                omp_set_lock(&writelock);
                if ( layer[k] > local_maximum ) {
                    local_maximum = layer[k];
                    local_positions = offset + k - 1;
                }
                omp_unset_lock(&writelock);
            }
        }

        LocalData data;
        data.local_maximum = local_maximum;
        data.local_positions = local_positions;

        MPI_Allreduce(MPI_IN_PLACE, &data, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        maximum[i] = data.local_maximum;
        positions[i] = data.local_positions;
    }
    omp_destroy_lock(&writelock);
    free(layer);
    free(layer_copy);
}
