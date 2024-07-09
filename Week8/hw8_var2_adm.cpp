#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;


// 2D Heat equation
// Alternating direction method
// Problem 2

// Initial condition
// U(t=0, x, y) = 0

// Boundary conditions
// U(x=0, 0<y<0.7) = 0
// U(x=0, 0.7<y<1) = 1
// U(x=1, 0<y<0.3) = 1
// U(x=1, 0.3<y<1) = 0
// U(x, y=0) = 0
// U(x, y=1) = 0

const double start_x = 0, end_x = 1;
const double start_y = 0, end_y = 1;

const int N = 101;
const int M = 101;

const double dx = (end_x - start_x) / (N - 1);
const double dy = (end_y - start_y) / (M - 1);

const int M1 = 0.3 * M;
const int M2 = 0.7 * M;

const double dt = 0.01;
const double a2 = 1;
const double eps = 1e-6;
const int stop_iteration = 1e5;

double U_old[M][N], U_new[M][N];
double X[M][N], Y[M][N];
double A[M][N], B[M][N], C[M][N], D[M][N];
double alpha[M][N], beta[M][N];

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

void meshgrid(double X[M][N], double Y[M][N]){
    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            X[j][i] = start_x + i*dx;
            Y[j][i] = start_y + j*dy;
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
            if(i != N-1){
                printf(fmt.c_str(), arr[j][i]);
                if(i != N-2){
                    printf("%c", delimeter);
                }
            }else{
                printf("\n");
            }
        }
    }
    fclose(File);
}

void Alternating_direction_method(double U_old[M][N], double U_new[M][N], int N, int M, double dx, double dy, 
                            double dt, double a2=1, double eps=1e-6, int stop_iteration =3e4){
    int iteration = 0;
    double maximum = 0;
    do{
        // Finding U^(n+1/2)
        for(int j = 0; j < M; ++j){
            for(int i = 0; i < N; ++i){
                A[j][i] = -a2 / (dx*dx);
                B[j][i] = 1 / dt + 2*a2 / (dx*dx);
                C[j][i] = -a2 / (dx*dx);
            }
        }
        
        for(int j = 1; j < M-1; ++j){
            for(int i = 1; i < N-1; ++i){
                D[j][i] = U_old[j][i] / dt
                    + a2*(U_old[j+1][i] - 2*U_old[j][i] + U_old[j-1][i])
                        / (dy*dy);
            }
        }

        // Thomas algorithm for x
        // U(t, x=0, 0<y<0.7) = 0
        for(int j = 0; j < M2; ++j){
            alpha[j][1] = 0;
            beta[j][1] = 0;
        }
        // U(t, x=0, 0.7<y<1) = 1
        for(int j = M2; j < M; ++j){
            alpha[j][1] = 0;
            beta[j][1] = 1;
        }

        for(int j = 1; j < M-1; ++j){
            for(int i = 1; i < N-1; ++i){
                alpha[j][i+1] = -A[j][i]
                    / (B[j][i] + C[j][i]*alpha[j][i]);
                beta[j][i+1] = (D[j][i] - C[j][i]*beta[j][i])
                    / (B[j][i] + C[j][i]*alpha[j][i]);
            }
        }
        
        for(int j = 1; j < M-1; ++j){
            // U^(n+1/2)
            // U(t, x=1, 0<y<0.3) = 1
            if(j < M1){
                U_new[j][N-1] = 1;
            }
            // U(t, x=1, 0.3<y<1) = 0
            else if(j >= M1 and j < M){
                U_new[j][N-1] = 0;
            }
            for(int i = N-2; i >= 0; --i){
                U_new[j][i] = alpha[j][i+1]*U_new[j][i+1] 
                                + beta[j][i+1];
            }
        }

        // Finding U^(n+1)
        for(int j = 0; j < M; ++j){
            for(int i = 0; i < N; ++i){
                A[j][i] = - a2 / (dy*dy);
                B[j][i] = 1 / dt + 2*a2 / (dy*dy);
                C[j][i] = -a2 / (dy*dy);
            }
        }
        
        for(int j = 1; j < M-1; ++j){
            for(int i = 1; i < N-1; ++i){
                D[j][i] = U_new[j][i] / dt 
                    + a2*(U_new[j][i+1] - 2*U_new[j][i] + U_new[j][i-1])
                        / (dx*dx);
            }
        }

        // Thomas algorithm for y
        // U(t, x, y=0) = 0
        for(int i = 0; i < N; ++i){
            alpha[1][i] = 0;
            beta[1][i] = 0;
        }

        for(int i = 1; i < N-1; ++i){
            for(int j = 1; j < M-1; ++j){
                alpha[j+1][i] = -A[j][i]
                    / (B[j][i] + C[j][i]*alpha[j][i]);
                beta[j+1][i] = (D[j][i] - C[j][i]*beta[j][i])
                    / (B[j][i] + C[j][i]*alpha[j][i]);
            }
        }

        for(int i = 1; i < N-1; ++i){
            // U^(n+1)
            // U(t, x, y=1) = 0
            U_new[M-1][i] = 0;
            for(int j = M-2; j >= 0; --j){
                U_new[j][i] = alpha[j+1][i]*U_new[j+1][i] 
                                + beta[j+1][i];
            }
        }

        maximum = max_abs_diff(U_new, U_old);
        copy_array(U_new, U_old);

        iteration++;
    }while(maximum > eps and iteration < stop_iteration);

    std::cout << "Alternating direction method\n";
    std::cout << "Number of iterations = " << iteration << std::endl;
    std::cout << "N = " << N << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "dy = " << dy << std::endl;
    std::cout << "dt = " << dt << std::endl;
    std::cout << "a2 = " << a2 << std::endl;
    std::cout << "eps = " << eps << std::endl;
    std::cout << "Maximum = " << maximum << std::endl;
}

int main(){
    // Filling grid
    meshgrid(X, Y);

    // Initialize arrays
    fill_array(alpha);
    fill_array(beta);
    fill_array(A);
    fill_array(B);
    fill_array(C);
    fill_array(D);

    // Initial condition
    fill_array(U_old, 0);
    fill_array(U_new, 0);

    auto start = high_resolution_clock::now();

    Alternating_direction_method(U_old, U_new, N, M, dx, dy, dt, a2);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);

    savetxt("Results\\HW8_X_cpp.txt", X, "%.6f", '\t');
    savetxt("Results\\HW8_Y_cpp.txt", Y, "%.6f", '\t');
    savetxt("Results\\HW8_U_cpp.txt", U_new, "%.6f", '\t');

    std::cout << "Results are recorded" << std::endl;
    return 0;
}
