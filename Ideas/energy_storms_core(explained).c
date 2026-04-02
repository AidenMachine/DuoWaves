#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"

/* * FUNZIONE DI SUPPORTO: Calcola e aggiunge l'energia di una singola particella 
 * a una specifica cella 'k' dell'array. Questa funzione viene chiamata migliaia di volte.
 */
 
 // PARAMETRI SPIEGATI DA Zi'Ndo:
 /*
    - Il vettore float
    - La grandezza del vettore
    - La cella corrente
    - Il punto di impatto della particella
    - L'energia irradiata
 */
static void update( float *layer, int layer_size, int k, int pos, float energy ) {
    
    /* Calcola la distanza tra il punto di impatto della particella (pos) e la cella corrente (k).
       Usa un semplice valore assoluto per non avere distanze negative. */
    int distance = pos - k;
    if ( distance < 0 ) distance = - distance;

    /* Aggiunge 1 alla distanza. Questo è fondamentale per evitare una divisione per zero 
       quando la particella colpisce esattamente la cella k (distanza 0 diventa 1). */
    distance = distance + 1;

    /* Calcola l'attenuazione dell'energia. Invece della classica legge del quadrato della distanza,
       i professori hanno usato la radice quadrata. Perché? Per far spalmare l'energia molto più
       lontano e costringere il processore a fare calcoli più pesanti. */
    float atenuacion = sqrtf( (float)distance );

    /* L'energia che arriva alla cella k è inversamente proporzionale all'attenuazione 
       e alla grandezza totale della superficie. */
    float energy_k = energy / layer_size / atenuacion;

    /* Se l'energia che arriverebbe alla cella è troppo piccola (minore della soglia), la ignora. 
       Se è sufficientemente grande, la somma al valore già presente nella cella. 
       -> ALERT CONCORRENZA: In OpenMP, se due thread chiamano update() per particelle diverse 
       ma sulla STESSA cella 'k', c'è un rischio di sovrascrittura qui! */
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[k] = layer[k] + energy_k;
}


/*
 * IL CUORE DELLA SIMULAZIONE: Questa è la funzione in cui orchestrerai MPI e OpenMP.
 */
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k;
    
    /* ALLOCAZIONE MEMORIA: Crea la superficie (layer) e un array di supporto (layer_copy).
       -> ALERT MPI: Attualmente questo alloca tutta la superficie. In MPI, il rank dovrà 
       calcolare la grandezza della SUA fetta e allocare array molto più piccoli, 
       aggiungendo lo spazio per le "celle fantasma" ai bordi. 
       Conviene suddividere il lavoro totale dell'allocazione in parti. So gia che sarà una rogna suddividere a mano.
       Magari creiamo una funzione che calcola la parte restante della suddivisione e la suddivide a sua volta.
       ESEMPIO: 4 processi, 101 layer_size
       101/4 = 25 con resto di 1
       ERGO: 25 per ogni processo con la differenza che il primo prenderà ne prenderà 26
       ( FOSSE STATO RESTO 3 ALLORA DISTRIBUIVA UNO CIASCUNO PER OGNI PROCESSO )
       P.S. -> spero di essere chiaro quando scrivo perchè ho timore di far cagare HAHAHA -----> X */
    float *layer = (float *)malloc( sizeof(float) * layer_size );
    float *layer_copy = (float *)malloc( sizeof(float) * layer_size );
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    
    /* Inizializza tutte le celle a 0.0 (superficie pulita prima della tempesta). */
    // Sembra uno spreco utilizzare due for diversi... ed invece no. Letteralmente, se andassimo a far scrivere in un singolo for due info diverse su
    // possibili punti della memoria, andremo incontro a dei "jump" che richiederebbeero quindi più tempo. Quindi un singolo for magari no...
    
    /* X -----> Riprendendo il discorso di prima... anche qui vale la stessa cosa. I for andranno eliminati ed inseriti, a loro volta, dei
                "for locali". In questo modo credo si possa ottimizzare di molto il tutto. */
    for( k=0; k<layer_size; k++ ) layer[k] = 0.0f;
    for( k=0; k<layer_size; k++ ) layer_copy[k] = 0.0f;
    
    /* CICLO PRINCIPALE: Ripete le 3 fasi per ogni ondata (tempesta) di particelle passata dal main.
       Qui invece è ancora da vedere come suddivide il tutto ( principalmente serve una strategia per divide il lavoro tra i nodi MPI */
    for( i=0; i<num_storms; i++) {

        /* ----------------------------------------------------------------------------------
           FASE 1: BOMBARDAMENTO
           ---------------------------------------------------------------------------------- */
        /* Prende una ad una tutte le particelle di questa ondata 'i'. */
        for( j=0; j<storms[i].size; j++ ) {
            
            /* Legge l'energia e la converte (si trovava negli indici dispari dell'array posval) */
            // Aggiunta: Letteralmente l'array è fatto cosi -> [ pos0, val0, pos1, val1 ... ] cosi è più ottimizzato
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Legge la posizione di impatto (si trovava negli indici pari dell'array posval) */
            int position = storms[i].posval[j*2];

            /* La particella appena estratta trasferisce la sua energia a TUTTE le celle della superficie.
               -> ALERT MPI: Tutti i processi leggono le stesse particelle. Il trucco sarà fare in 
               modo che il ciclo for qui sotto aggiorni solo le celle da 'mio_inizio' a 'mia_fine', 
               non tutte le celle da 0 a layer_size! */
            for( k=0; k<layer_size; k++ ) {
                update( layer, layer_size, k, position, energy );
            }
        }

        /* ----------------------------------------------------------------------------------
           FASE 2: RILASSAMENTO
           ---------------------------------------------------------------------------------- */
        /* PASSO 2.1: Copia lo stato del livello su layer_copy. 
           Questo impedisce che i calcoli successivi si inquinino a vicenda. */
        for( k=0; k<layer_size; k++ ) 
            layer_copy[k] = layer[k];

        /* PASSO 2.2: Spalma l'energia facendo la media di 3 celle ( manco la nutella ). 
           Nota i limiti del for: k parte da 1 e arriva a layer_size-1. I bordi estremi 
           della superficie non partecipano al rilassamento e restano immobili.
           -> ALERT MPI: Per calcolare layer[mio_inizio], ti servirà layer_copy[mio_inizio - 1], 
           che si trova nella memoria del processo vicino! Qui devi scambiare le celle fantasma. */
        for( k=1; k<layer_size-1; k++ )
            layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;

        /* ----------------------------------------------------------------------------------
           FASE 3: RICERCA DEL RISCHIO MASSIMO
           ---------------------------------------------------------------------------------- */
        /* Controlla tutte le celle (tranne i bordi) per trovare il punto più critico dell'ondata. */
        for( k=1; k<layer_size-1; k++ ) {
            
            /* Una cella viene presa in considerazione solo se è un "picco locale", 
               ovvero se la sua energia è strettamente maggiore dei suoi vicini a destra e sinistra. */
            if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
                
                /* Se questo picco locale è il più alto trovato finora per questa ondata, lo salva. 
                   -> ALERT MPI: Quando ogni processo avrà finito questo ciclo, avrà il SUO 
                   massimo locale. Dovrai usare una MPI_Allreduce per confrontarli tutti e 
                   salvare il vincitore assoluto nei due array globali maximum[i] e positions[i]. */
                // Magari qui si inserisce un lock (?)
                if ( layer[k] > maximum[i] ) {
                    maximum[i] = layer[k];
                    positions[i] = k;
                }
            }
        }
    }
}
