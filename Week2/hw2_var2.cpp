#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;


// 1D Heat equation
// Thomas algorithm
// Variant 2

// Initial condition
// U(t=0, x) = 1 - x^3

// Boundary conditions
// Ux(t, x=0) = 0
// U(t, x=1) = 0

const double pi = 3.14159265359;
const double dx = 0.01;
const double dt = 0.01;
const double start_x = 0, end_x = 1;
const int N = (end_x - start_x) / dx + 1;

const double eps = 1e-6;
const int stop_iteration = 3e5;

const double A = -1.0 / (dx*dx);
const double B = 1.0 / dt + 2.0 / (dx*dx);
const double C = -1.0 / (dx*dx);

double x[N], F[N], D[N], alpha[N], beta[N];
double U_old[N], U_new[N];

void Analytical_solution(double t, int m=100){
    for(int n = 1; n <= m; ++n){
        for(int i = 0; i < N; ++i){
            F[i] += -96*(pow(-1, n) + 2/((2*n - 1)*pi))
                / pow((2*n - 1)*pi, 3)
                * exp(-pow((2*n - 1)*pi/2, 2)*t)
                * cos((2*n - 1)*pi/2*x[i]);
        }
    }    
}

int main(){
    
    for(int i = 0; i < N; ++i){
        x[i] = start_x + i*dx;
    }

    // Initialize arrays
    for(int i = 0; i < N; ++i){
        F[i] = 0;
        U_old[i] = 0;
        U_new[i] = 0;
        alpha[i] = 0;
        beta[i] = 0;
    }

    // initial condition
    // U(x, t=0) = 1 - x^3
    for(int i = 0; i < N; ++i){
        U_old[i] = 1 - pow(x[i], 3);
    }

    // Finding D
    for(int i = 0; i < N; ++i){
        D[i] = U_old[i] / dt;
    }

    auto start = high_resolution_clock::now();

    int iteration = 0;
    double maximum = 0;
    do{
        for(int i = 0; i < N; ++i){
            D[i] = U_old[i] / dt;
        }

        alpha[1] = 1;
        beta[1] = 0;
        for(int i = 1; i < N-1; ++i){
            alpha[i+1] = -A / (B + C*alpha[i]);
            beta[i+1] = (D[i] - C*beta[i]) / (B + C*alpha[i]);
        }

        U_new[N-1] = 0;
        for(int i = N-2; i >= 0; i--){
            U_new[i] = alpha[i+1]*U_new[i+1] + beta[i+1];
        }

        maximum = 0;
        for(int i = 0; i < N; ++i){
            if(maximum < fabs(U_new[i] - U_old[i])){
                maximum = fabs(U_new[i] - U_old[i]);
            }
        }

        for(int i = 0; i < N; ++i){
            U_old[i] = U_new[i];
        }

        iteration++;
    }while(maximum > eps and iteration < stop_iteration);

    // Finding analytical solution
    Analytical_solution(iteration * dt);

    maximum = 0;
    for(int i = 0; i < N; ++i){
        if(maximum < fabs(F[i] - U_new[i])){
            maximum = fabs(F[i] - U_new[i]);
        }
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);
    printf("Maximum difference: %.9f\n", maximum);
    printf("Number of iterations: %d\n", iteration);

    FILE *File;
    File = freopen("HW2_X_cpp.txt", "w", stdout);
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf("%.9f", x[i]);
        }else{
            printf("%.9f\t", x[i]);
        }
    }
    fclose(File);

    File = freopen("HW2_U_cpp.txt", "w", stdout);
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf("%.6e", U_new[i]);
        }else{
            printf("%.6e\t", U_new[i]);
        }
    }
    fclose(File);

    File = freopen("HW2_F_cpp.txt", "w", stdout);
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf("%.6e", F[i]);
        }else{
            printf("%.6e\t", F[i]);
        }
    }
    fclose(File);

    return 0;
}
