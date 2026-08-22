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

static void update( float *layer, int layer_size, int k, int pos, float energy, int global_k) {
    int distance = pos - global_k;
    if ( distance < 0 ) distance = - distance;
    distance = distance + 1;
    float atenuacion = sqrtf((float)distance);
    float energy_k = energy / layer_size / atenuacion;

    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size ) {
        #pragma omp atomic
        layer[k] = layer[k] + energy_k;
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k;

    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    omp_lock_t writelock;
    omp_init_lock(&writelock);

    int local_chunk = layer_size / size;
    int resto = layer_size % size;

    int local_size;
    int offset;

    if ( rank < resto ) {
        local_size = local_chunk + 1;
        offset = rank * local_size;
    } else {
        local_size = local_chunk;
        offset = rank * local_chunk + resto;
    }

    float *layer = (float *)calloc((local_size + 2), sizeof(float));
    float *layer_copy = (float *)calloc((local_size + 2), sizeof(float));
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error_rank_%d: Allocating the layer memory\n", rank);
        exit( EXIT_FAILURE );
    }

    /* VARIABILI PER IL PROFILING
     * - t_bomb -> fase di bombardamento
     * - t_exchange -> fase di scambio dati tra celle fantasma
     * - t_relax -> fase di rilassamento
     * - t_reduce -> ricerca del massimo locale
     */
    double t_bomb = 0.0, t_exchange = 0.0, t_relax = 0.0, t_reduce = 0.0;
    double t0, t1;

    /* INTERA TEMPESTA */
    for( i=0; i<num_storms; i++) {

        /* CALCOLO TEMPO FASE DI BOMBARDAMENTO ------------------------------------------------------------------------- */
        t0 = MPI_Wtime();
        #pragma omp parallel for private(j,k)
        for( j=0; j<storms[i].size; j++ ) {

            float energy = (float)storms[i].posval[j*2+1] * 1000;
            int position = storms[i].posval[j*2];

            for( k = 1; k <= local_size; k++ ) {
                update( layer, layer_size, k, position, energy , offset + k - 1);
            }
        }
        t1 = MPI_Wtime();
        t_bomb += t1 - t0;
        /* ------------------------------------------------------------------------------------------------------------- */

        int left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int right_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        /* CALCOLO TEMPO FASE DI "SCAMBIO_1" --------------------------------------------------------------------------- */
        t0 = MPI_Wtime();
        MPI_Sendrecv(&layer[local_size], 1, MPI_FLOAT, right_neighbor, 0,
                     &layer[0], 1, MPI_FLOAT, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&layer[1], 1, MPI_FLOAT, left_neighbor, 1,
                     &layer[local_size + 1], 1, MPI_FLOAT, right_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        t1 = MPI_Wtime();
        t_exchange += t1 - t0;
        /* ------------------------------------------------------------------------------------------------------------- */

        /* CALCOLO TEMPO FASE DI RILASSAMENTO -------------------------------------------------------------------------- */
        t0 = MPI_Wtime();
        float *temp = layer_copy;
        layer_copy = layer;
        layer = temp;

        #pragma omp parallel for private(k)
        for (k = 1; k <= local_size; k++) {
            if (rank == 0 && k == 1) continue;
            if (rank == size - 1 && k == local_size) continue;
            layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
        }
        t1 = MPI_Wtime();
        t_relax += t1 - t0;
        /* ------------------------------------------------------------------------------------------------------------- */

        /* CALCOLO TEMPO FASE DI "SCAMBIO_2" --------------------------------------------------------------------------- */
        t0 = MPI_Wtime();
        MPI_Sendrecv(&layer[local_size], 1, MPI_FLOAT, right_neighbor, 0,
                     &layer[0], 1, MPI_FLOAT, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&layer[1], 1, MPI_FLOAT, left_neighbor, 1,
                     &layer[local_size + 1], 1, MPI_FLOAT, right_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        t1 = MPI_Wtime();
        t_exchange += t1 - t0;
        /* ------------------------------------------------------------------------------------------------------------- */

        /* CALCOLO TEMPO FASE DI REDUCTION E CALCOLO MASSIMO LOCALE ---------------------------------------------------- */
        t0 = MPI_Wtime();
        float local_maximum = -1.0f;
        int local_positions = -1;

        int start = 1, end = local_size;
        if ( rank == 0 ) start++;
        if ( rank == size - 1 ) end--;

        #pragma omp parallel for private(k)
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

        LocalData data_send, data_recv;
        data_send.local_maximum = local_maximum;
        data_send.local_positions = local_positions;

        MPI_Allreduce(&data_send, &data_recv, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        maximum[i] = data_recv.local_maximum;
        positions[i] = data_recv.local_positions;
        t1 = MPI_Wtime();
        t_reduce += t1 - t0;
        /* ------------------------------------------------------------------------------------------------------------- */
    }

    /* AGGREGAZIONE DATI SUL RANK 0 RACCOLTA DATI */
    double max_bomb, max_exchange, max_relax, max_reduce;
    MPI_Reduce(&t_bomb, &max_bomb, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_exchange, &max_exchange, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_relax, &max_relax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_reduce, &max_reduce, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* STAMPA DATI */
    if (rank == 0) {
        double total = max_bomb + max_exchange + max_relax + max_reduce;
        fprintf(stderr, "\n=== RISULTATI PROFILING (Layer: %d, Storms: %d, Nodes: %d) ===\n", layer_size, num_storms, size);
        fprintf(stderr, "Fase di bombardamento:       %8.4f s (%5.2f%%)\n", max_bomb, (max_bomb / total) * 100.0);
        fprintf(stderr, "Fase di scambio (MPI):       %8.4f s (%5.2f%%)\n", max_exchange, (max_exchange / total) * 100.0);
        fprintf(stderr, "Fase di rilassamento:        %8.4f s (%5.2f%%)\n", max_relax, (max_relax / total) * 100.0);
        fprintf(stderr, "Fase di reduction:           %8.4f s (%5.2f%%)\n", max_reduce, (max_reduce / total) * 100.0);
        fprintf(stderr, "Tempo totale ( circa ):      %8.4f s\n", total);
        fprintf(stderr, "==============================================================\n\n");
    }

    omp_destroy_lock(&writelock);
    free(layer);
    free(layer_copy);
}