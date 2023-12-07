#include <mpi.h>
#include <cstdio>
#include <cstdlib>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr) {
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
    }

    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    *a_mat_ptr = (int*)malloc((*n_ptr) * (*m_ptr) * sizeof(int));
    *b_mat_ptr = (int*)malloc((*m_ptr) * (*l_ptr) * sizeof(int));

    if (world_rank == 0) {
        for(int i = 0; i < (*n_ptr) * (*m_ptr); i++) {
            scanf("%d", &((*a_mat_ptr)[i]));
        }
        for(int i = 0; i < (*m_ptr); i++) {
            for(int j = 0 ; j < *l_ptr ; j++) {
                scanf("%d", &((*b_mat_ptr)[i + j * (*m_ptr)]));
            }
        }
    }

    MPI_Bcast(*a_mat_ptr, (*n_ptr) * (*m_ptr), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat) {
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int local_n = n / world_size;
    int *c_mat = (int*)malloc(n * l * sizeof(int));

    int start = world_rank * local_n;
    int end = (world_rank == world_size - 1) ? n : start + local_n;
    
    for (int i = start; i < end; i++) {
        for (int j = 0; j < l; j++) {
            c_mat[i * l + j] = 0;
            for (int k = 0; k < m; k++) {
                c_mat[i * l + j] += a_mat[i * m + k] * b_mat[j * m + k];
            }
        }
    }

    int* gathered_c_mat = (int*)malloc(n * l * sizeof(int));

    MPI_Reduce(c_mat, gathered_c_mat, n * l, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                printf("%d ", gathered_c_mat[i * l + j]);
            }
            printf("\n");
        }
        free(gathered_c_mat);
    }

    free(c_mat);
}

void destruct_matrices(int *a_mat, int *b_mat) {
    free(a_mat);
    free(b_mat);
}
