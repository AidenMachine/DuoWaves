#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<mpi.h>
#include<omp.h>
#include "energy_storms.h"

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


// Ricorda di inserire nel "verbale" il fatto di aver aggiunto due parametri alla funzioni
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions, const int rank, const int size) {
    int i, j, k;
    // PARTE NUOVA --------------------------------------------------------

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
     */
    for( i=0; i<num_storms; i++) {

        /* TODO Parallelizzare con OpenMP */
        for( j=0; j<storms[i].size; j++ ) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Get impact position */
            int position = storms[i].posval[j*2];

            /* TODO Da vedere DA DOVE deve partire e FINO A DOVE deve arrivare */
            for( k=0; k<layer_size; k++ ) {
                /* Update the energy value for the cell */
                update( layer, layer_size, k, position, energy );
            }
        }

        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        for( k=0; k<layer_size; k++ ) 
            layer_copy[k] = layer[k];

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        for( k=1; k<layer_size-1; k++ )
            layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;

        /* 4.3. Locate the maximum value in the layer, and its position */
        for( k=1; k<layer_size-1; k++ ) {
            /* Check it only if it is a local maximum */
            if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
                if ( layer[k] > maximum[i] ) {
                    maximum[i] = layer[k];
                    positions[i] = k;
                }
            }
        }
    }
}