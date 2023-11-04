#include <iostream>
#include <random>
#include <pthread.h>

using namespace std;

pthread_mutex_t mutexT;
pthread_t* threads;
long long* total_number_in_circle;

void* toss(void* arg) {
    long long* toss_num_per_thread = (long long*)arg;
    long long number_in_circle = 0;

    unsigned int seed = 47;

    for (long long i = 0; i < *toss_num_per_thread; i++) {
        double x = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double y = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) {
            number_in_circle++;
        }
    }

    pthread_mutex_lock(&mutexT);
    *total_number_in_circle += number_in_circle;
    pthread_mutex_unlock(&mutexT);
    return NULL;
}

int main(int argc, char** argv) {
    int thread_num = atoi(argv[1]);
    long long number_of_tosses = atoll(argv[2]);
    long long tosses_num_per_thread = number_of_tosses / thread_num;

    threads = new pthread_t[thread_num];
    total_number_in_circle = new long long;
    *total_number_in_circle = 0;

    pthread_mutex_init(&mutexT, NULL);

    for (int i = 0; i < thread_num; i++) {
        pthread_create(&threads[i], NULL, toss, &tosses_num_per_thread);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(threads[i], NULL);
    }
    double pi_estimate = 4 * (*total_number_in_circle) / (double)number_of_tosses;
    cout << pi_estimate << endl;

    pthread_mutex_destroy(&mutexT);
    delete[] threads;
    delete total_number_in_circle;

    return 0;
}