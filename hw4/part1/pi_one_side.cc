#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    long long local_toss_num = tosses / world_size;
    long long local_number_in_circle = 0;   

    unsigned int seed = time(0) * world_rank;

    for (long long i = 0; i < local_toss_num; i++) {
        double x = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double y = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        
        if (x * x + y * y <= 1) {
            local_number_in_circle++;
        }
    }

    long long *value;
    if (world_rank == 0)
    {
        // Master
        MPI_Alloc_mem(world_size * sizeof(long long), MPI_INFO_NULL, &value);
        *value = 0;
        MPI_Win_create(value, world_size * sizeof(long long), sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&local_number_in_circle, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // handle PI result
        local_number_in_circle += *value;
        pi_result = 4.0 * (local_number_in_circle / (double)tosses);        

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}