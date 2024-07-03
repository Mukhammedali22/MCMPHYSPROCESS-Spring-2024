#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
using namespace std::chrono;


// 1D Poisson equation
// Five-diagonal matrix method
// Problem 2

// Boundary conditions
// P(x=0) = 1
// P(x=1) = 0

// f(x) = sin(x)

const double start_x = 0, end_x = 1;
const double dx = 0.1;
const double a1 = 0, b1 = 0, g1 = 1;
const int N = (end_x - start_x) / dx + 1;


double Analytical_solution(double x){
    return sin(x) - (1 + sin(1))*x + 1;
}

void fill_array(double arr[N], double value=0){
    for(int i = 0; i < N; ++i){
        arr[i] = value;
    }
}

double abs_max(double A[N], double B[N]){
    double maximum = 0;
    for(int i = 0; i < N; ++i){
        if(maximum < fabs(A[i] - B[i])){
            maximum = fabs(A[i] - B[i]);
        }
    }
    return maximum;
}

void savetxt(std::string path, double arr[N], 
                std::string fmt="%.6f", char delimeter='\t'){
    const char * c = path.c_str();
    FILE *File;
    File = freopen(c, "w", stdout);
    std::string fmtwd = fmt + "%c";
    for(int i = 0; i < N; ++i){
        if(i == N-1){
            printf(fmt.c_str(), arr[i]);
        }else{
            printf(fmtwd.c_str(), arr[i], delimeter);
        }
    }
    fclose(File);
}

int main(){
    double x[N], F[N];
    double A[N], B[N], C[N], D[N], E[N], H[N];
    double alpha[N], beta[N], gamma[N];
    double P_old[N], P_new[N];

    auto start = high_resolution_clock::now();
    
    for(int i = 0; i < N; ++i){
        x[i] = start_x + i*dx;
    }

    // Initialize arrays
    fill_array(P_old);
    fill_array(P_new);
    fill_array(F);
    fill_array(A, -1 / (12*dx*dx));
    fill_array(B, 16 / (12*dx*dx));
    fill_array(C, -30 / (12*dx*dx));
    fill_array(D, 16 / (12*dx*dx));
    fill_array(E, -1 / (12*dx*dx));
    fill_array(H);
    fill_array(alpha);
    fill_array(beta);
    fill_array(gamma);

    for(int i = 0; i < N; ++i){
        F[i] = Analytical_solution(x[i]);
        H[i] = -F[i];
    }

    alpha[1] = 0;
    beta[1] = 0;
    gamma[1] = 1;

    alpha[2] = -(B[1] + D[1] * beta[1]) / (C[1] + D[1] * alpha[1]);
    beta[2] = -A[1] / (C[1] + D[1] * alpha[1]);
    gamma[2] = (H[1] - D[1] * gamma[1]) / (C[1] + D[1] * alpha[1]);

    for(int i = 2; i < N-1; ++i){
        alpha[i+1] = -(B[i] + D[i]*beta[i] + E[i]*alpha[i-1]*beta[i])
            / (C[i] + D[i]*alpha[i] + E[i]*alpha[i-1]*alpha[i] + E[i]*beta[i-1]);
        beta[i+1] = -A[i]
            / (C[i] + D[i]*alpha[i] + E[i]*alpha[i-1]*alpha[i] + E[i]*beta[i-1]);
        gamma[i+1] = (H[i] - D[i]*gamma[i] - E[i]*alpha[i-1]*gamma[i] - E[i]*gamma[i-1])
            / (C[i] + D[i]*alpha[i] + E[i]*alpha[i-1]*alpha[i] + E[i]*beta[i-1]);
    }

    P_new[N-1] = 0;
    P_new[N-2] = alpha[N-1]*P_new[N-1] + gamma[N-1];
    for(int i = N-3; i >= 0; --i){
        P_new[i] = alpha[i+1]*P_new[i+1] 
            + beta[i+1]*P_new[i+2] + gamma[i+1];
    }

    double maximum = abs_max(F, P_new);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);
    printf("Maximum difference: %.9f\n", maximum);
    printf("dx = %f", dx);

    savetxt("Results\\HW4_X_cpp.txt", x, "%.6f", '\t');
    savetxt("Results\\HW4_U_cpp.txt", P_new, "%.6f", '\t');
    savetxt("Results\\HW4_F_cpp.txt", F, "%.6f", '\t');
    
    return 0;
}
