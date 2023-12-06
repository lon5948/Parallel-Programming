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

    // init MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    long long local_toss_num = tosses / world_size;
    long long local_number_in_circle = 0, global_number_in_circle = 0;    

    unsigned int seed = time(0) * world_rank;

    for (long long i = 0; i < local_toss_num; i++) {
        double x = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double y = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        
        if (x * x + y * y <= 1) {
            local_number_in_circle++;
        }
    }

    if (world_rank > 0)
    {
        // handle workers
        MPI_Send(&local_number_in_circle, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // master
        global_number_in_circle = local_number_in_circle;
        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&local_number_in_circle, 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_number_in_circle += local_number_in_circle;
        }
    }

    if (world_rank == 0)
    {
        // process PI result
        pi_result = 4.0 * (global_number_in_circle / (double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
