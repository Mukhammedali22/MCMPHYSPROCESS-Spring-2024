#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;


// 1D Heat conductivity equation
// Homework 3, Problem 2

// Initial condition
// U(t=0, x) = 1 - x^3

// Boundary conditions
// Ux(t, x=0) = 0
// U(t, x=1) = 0

const double start_x = 0, end_x = 1;
const double dx = 0.1, dt = 0.001;
const double pi = 3.14159265359;
const double eps = 1e-6;
const double a2 = 1;

const int N = (end_x - start_x) / dx + 1;
const int stop_iteration = 3e5;

double x[N], F_S[N], F_T[N], U_old[N], U_S[N], U_T[N];

void FillArray(double arr[N], double value=0){
    for(int i = 0; i < N; ++i){
        arr[i] = value;
    }
}

double FindMaxDiff(double U_old[N], double U_new[N]){
    double maximum = 0;
    for(int i = 0; i < N; ++i){
        if(maximum < fabs(U_new[i] - U_old[i])){
            maximum = fabs(U_new[i] - U_old[i]);
        }
    }
    return maximum;
}

void Analytical_solution(double F[N], double t, int m=100){
    for(int n = 1; n <= m; ++n){
        for(int i = 0; i < N; ++i){
            F[i] += -96*(pow(-1, n) + 2/((2*n - 1)*pi))
                / pow((2*n - 1)*pi, 3)
                * exp(-pow((2*n - 1)*pi/2, 2)*t)
                * cos((2*n - 1)*pi/2*x[i]);
        }
    }    
}

void Simple_Iterative_Method(double U_old[N], double U_new[N], int N, double dt, 
                                double dx, double a2, double eps=1e-6, int stop_iteration=3e4){
    for(int i = 1; i < N-1; ++i){
        U_new[i] = U_old[i] + a2 * dt / (dx*dx)
            * (U_old[i+1] - 2*U_old[i] + U_old[i-1]);
    }
}

void set_boundary_U(double U[N]){
    U[0] = U[1];
    U[N-1] = 0;
}

void Thomas_algorithm(double U_old[N], double U_new[N], int N, double dt, 
                        double dx, double a2, double eps=1e-6, int stop_iteration=3e4){

    double alpha[N], beta[N], D[N];

    double A = -a2 / (dx*dx);
    double B = 1 / dt + 2*a2 / (dx*dx);
    double C = -a2 / (dx*dx);
    
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
    for(int i = N-2; i >= 0; --i){
        U_new[i] = alpha[i+1]*U_new[i+1] + beta[i+1];
    }
}

int main(){
    for(int i = 0; i < N; ++i){
        x[i] = start_x + i*dx;
    }

    // Initialize arrays
    FillArray(U_old, 0);
    FillArray(U_S, 0);
    FillArray(U_T, 0);
    FillArray(F_S, 0);
    FillArray(F_T, 0);

    // Simple Iterative method
    // initial condition
    // U(x, t=0) = 1 - x^3
    for(int i = 0; i < N; ++i){
        U_old[i] = 1 - pow(x[i], 3);
    }

    auto start_S = high_resolution_clock::now();

    int iter_S = 0;
    double maximum = 0;
    do{
        Simple_Iterative_Method(U_old, U_S, N, dt, dx, a2);
        set_boundary_U(U_S);

        maximum = FindMaxDiff(U_old, U_S);

        for(int i = 0; i < N; ++i){
            U_old[i] = U_S[i];
        }

        iter_S++;
    }while(maximum > eps and iter_S < stop_iteration);

    Analytical_solution(F_S, dt*iter_S);

    maximum = FindMaxDiff(F_S, U_S);

    auto stop_S = high_resolution_clock::now();
    auto duration_S = duration_cast<microseconds>(stop_S - start_S);
    
    printf("calculating time: %.6f seconds\n", duration_S.count() / 1e6);
    printf("Maximum error (Simple): %e\n", maximum);
    printf("Number of iterations: %d\n", iter_S);

    // Thomas algorithm
    // initial condition
    // U(x, t=0) = 1 - x^3
    for(int i = 0; i < N; ++i){
        U_old[i] = 1 - pow(x[i], 3);
        U_T[i] = 0;
    }

    set_boundary_U(U_old);
    set_boundary_U(U_T);

    auto start = high_resolution_clock::now();

    int iter_T = 0;
    maximum = 0;
    do{
        Thomas_algorithm(U_old, U_T, N, dt, dx, a2);
        
        maximum = FindMaxDiff(U_old, U_T);

        for(int i = 0; i < N; ++i){
            U_old[i] = U_T[i];
        }

        iter_T++;
    }while(maximum > eps and iter_T < stop_iteration);

    Analytical_solution(F_T, dt*iter_T);

    maximum = FindMaxDiff(F_T, U_T);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);
    printf("Maximum error (Thomas): %e\n", maximum);
    printf("Number of iterations: %d\n", iter_T);

    FILE *File;
    File = freopen("Results\\HW3_X_cpp.txt", "w", stdout);
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf("%.9f", x[i]);
        }else{
            printf("%.9f\t", x[i]);
        }
    }
    fclose(File);

    File = freopen("Results\\HW3_U_Simple_cpp.txt", "w", stdout);
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf("%.9f", U_S[i]);
        }else{
            printf("%.9f\t", U_S[i]);
        }
    }
    fclose(File);

    File = freopen("Results\\HW3_F_Simple_cpp.txt", "w", stdout);
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf("%.9f", F_S[i]);
        }else{
            printf("%.9f\t", F_S[i]);
        }
    }
    fclose(File);

    File = freopen("Results\\HW3_U_Thomas_cpp.txt", "w", stdout);
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf("%.9f", U_T[i]);
        }else{
            printf("%.9f\t", U_T[i]);
        }
    }
    fclose(File);

    File = freopen("Results\\HW3_F_Thomas_cpp.txt", "w", stdout);
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf("%.9f", F_T[i]);
        }else{
            printf("%.9f\t", F_T[i]);
        }
    }
    fclose(File);

    return 0;
}
