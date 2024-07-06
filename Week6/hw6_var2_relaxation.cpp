#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;


// 2D Laplace equation
// Over Relaxation method
// Problem 2

// Boundary conditions
// P(x=0, 0<y<0.7) = 0
// P(x=0, 0.7<y<1) = 1
// P(x=1, 0<y<0.3) = 1
// P(x=1, 0.3<y<1) = 0
// P(x, y=0) = 0
// P(x, y=1) = 0

const double dx = 0.01;
const double dy = 0.01;

const double eps = 1e-6;
const int stop_iteration = 1e5;

const double start_x = 0, end_x = 1;
const double start_y = 0, end_y = 1;
const int N = (end_x - start_x) / dx + 1;
const int M = (end_y - start_y) / dy + 1;

const int M1 = 0.3 * M;
const int M2 = 0.7 * M;

// Fills array with some value
void fill_array(double arr[M][N], double value=0){
    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            arr[j][i] = value;
        }
    }
}

// Returns maximum absolute difference of two arrays
double max_abs_diff(double A[M][N], double B[M][N]){
    double maximum = 0;
    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            if(maximum < fabs(A[j][i] - B[j][i])){
                maximum = fabs(A[j][i] - B[j][i]);
            }
        }
    }
    return maximum;
}

// copies values of array A to the array B
void copy_array(double A[M][N], double B[M][N]){
    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            B[j][i] = A[j][i];
        }
    }
}

void savetxt(std::string path, double arr[M][N], 
                std::string fmt="%f", char delimeter=' '){
    const char * c = path.c_str();
    FILE *File;
    File = freopen(c, "w", stdout);
    std::string fmtwd = fmt + "%c";
    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            if(i == N-1){
                printf(fmt.c_str(), arr[j][i]);
            }else{
                printf(fmtwd.c_str(), arr[j][i], delimeter);
            }
        }
    }
    fclose(File);
}

// Boundary conditions
// P(x=0, 0<y<0.7) = 0
// P(x=0, 0.7<y<1) = 1
// P(x=1, 0<y<0.3) = 1
// P(x=1, 0.3<y<1) = 0
// P(x, y=0) = 0
// P(x, y=1) = 0
void set_boundary_P(double P[M][N]){
    // P(x=0, 0<y<0.7) = 0
    for(int j = 0; j < M2; ++j){
        P[j][0] = 0;
    }
    // P(x=0, 0.7<y<1) = 1
    for(int j = M2; j < M; ++j){
        P[j][0] = 1;
    }
    // P(x=1, 0<y<0.3) = 1
    for(int j = 0; j < M1; ++j){
        P[j][N-1] = 1;
    }
    // P(x=1, 0.3<y<1) = 0
    for(int j = M1; j < M; ++j){
        P[j][N-1] = 0;
    }
    // P(x, y=0) = 0
    // P(x, y=1) = 0
    for(int i = 0; i < N; ++i){
        P[0][i] = 0;
        P[M-1][i] = 0;
    }
}

void Over_Relaxation_method(double P_old[M][N], double P_new[M][N], int N, int M, 
                    double dx, double dy, double w=1.9, double eps=1e-6, int stop_iteration=3e4){
    
    set_boundary_P(P_old);
    set_boundary_P(P_new);

    int iteration = 0;
    double maximum = 0;
    do{
        for(int j = 1; j < M-1; ++j){
            for(int i = 1; i < N-1; ++i){
                P_new[j][i] = w*(
                    dy*dy*(P_old[j][i+1] + P_new[j][i-1])
                    + dx*dx*(P_old[j+1][i] + P_new[j-1][i])
                    ) / (2*(dx*dx + dy*dy))
                    + (1 - w)*P_old[j][i];
            }
        }

        set_boundary_P(P_old);
        set_boundary_P(P_new);
        maximum = max_abs_diff(P_new, P_old);
        copy_array(P_new, P_old);

        iteration += 1;
    }while(maximum > eps and iteration < stop_iteration);

    std::cout << "Over Relaxation method\n";
    std::cout << "Number of iterations = " << iteration << std::endl;
    std::cout << "N = " << N << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "dy = " << dy << std::endl;
    std::cout << "w = " << w << std::endl;
    std::cout << "eps = " << eps << std::endl;
    std::cout << "Maximum = " << maximum << std::endl;
}

int main(){
    double P_old[M][N], P_new[M][N];
    double x[N], y[M], X[M][N], Y[M][N];

    auto start = high_resolution_clock::now();

    for(int i = 0; i < N; ++i){
        x[i] = start_x + i*dx;
    }

    for(int j = 0; j < M; ++j){
        y[j] = start_y + j*dy;
    }

    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            X[j][i] = x[i];
            Y[j][i] = y[j];
        }
    }

    fill_array(P_old, 0);
    fill_array(P_new, 0);

    Over_Relaxation_method(P_old, P_new, N, M, dx, dy, 1.94);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);

    savetxt("Results\\HW6_X_cpp.txt", X, "%.6f", '\t');
    savetxt("Results\\HW6_Y_cpp.txt", Y, "%.6f", '\t');
    savetxt("Results\\HW6_P3_cpp.txt", P_new, "%.6f", '\t');

    std::cout << "Results are recorded" << std::endl;
    return 0;
}
