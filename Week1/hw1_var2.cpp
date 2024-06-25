#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;

// Tridiagonal matrix method
// Thomas algorithm
// Метод прогонки
// Variant 2
// 1D Poisson equation
// Boundary conditions
// P(x=0) = 1
// P(x=1) = 0

const double dx = 0.1;
const double start_x = 0, end_x = 1;
const int N = (end_x - start_x) / dx + 1;
const double A = 1.0 / (dx*dx);
const double B = -2.0 / (dx*dx);
const double C = 1.0 / (dx*dx);

int main(){
    double x[N], f[N], F[N], P[N], D[N], alpha[N], beta[N];
    
    auto start = high_resolution_clock::now();
    for(int i = 0; i < N; ++i){
        x[i] = start_x + i*dx;
        f[i] = sin(x[i]);
        F[i] = sin(x[i]) - x[i]*(sin(1) + 1) + 1;
        P[i] = 0;
        alpha[i] = 0;
        beta[i] = 0;
        D[i] = -f[i];
    }

    alpha[1] = 0;
    beta[1] = 1;
    for(int i = 1; i < N-1; ++i){
        alpha[i+1] = -A/(B + C*alpha[i]);
        beta[i+1] = (D[i] - C*beta[i])/(B + C*alpha[i]);
    }

    P[N-1] = 0;
    for(int i = N-2; i >= 0; i--){
        P[i] = alpha[i+1]*P[i+1] + beta[i+1];
    }
    
    double maxdiff = 0;
    for(int i = 0; i < N; ++i){
        if(maxdiff < fabs(F[i] - P[i])){
            maxdiff = fabs(F[i] - P[i]);
        }
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);
    printf("Maximum difference: %.6f seconds\n", maxdiff);

    FILE *File;
    File = freopen("HW1_U_cpp.txt", "w", stdout);
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf("%.6f", P[i]);
        }else{
            printf("%.6f\t", P[i]);
        }
    }
    fclose(File);

    return 0;
}
