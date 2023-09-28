#include <iostream>

using namespace std;

int main() {
    long long number_in_circle = 0, number_of_tosses = 1e9;
    for (long long toss = 0; toss < number_of_tosses; toss++) {
        double x = (double)rand() / RAND_MAX * 2 - 1;
        double y = (double)rand() / RAND_MAX * 2 - 1;
        if (x * x + y * y <= 1)
            number_in_circle++;
    }
    double pi_estimate = 4 * number_in_circle / (double)number_of_tosses;
    cout << pi_estimate << endl;
}