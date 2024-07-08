#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;


// 3D Heat equation
// Alternating direction method (ADM)
// Problem 2

// Initial condition
// U(t=0, x, y, z) = 0

// Boundary conditions
// U(t, x=0, 0.3<y<0.6, 0.7<z<1) = 1
// U(t, 0.4<x<0.7, 0<y<0.3, z=0) = 1
// Otherwise is 0
// Or

// U(t, x=0, 0.3<y<0.6, 0.7<z<1) = 1
// U(t, x=0, y, z) = 0
// U(t, x=1, y, z) = 0
// U(t, x, y=0, z) = 0
// U(t, x, y=1, z) = 0
// U(t, x, y, z=0) = 0
// U(t, 0.4<x<0.7, 0<y<0.3, z=0) = 1
// U(t, x, y, z=1) = 0

const double start_x = 0, end_x = 1;
const double start_y = 0, end_y = 1;
const double start_z = 0, end_z = 1;

const int N = 41;
const int M = 41;
const int P = 41;

const double dx = (end_x - start_x) / (N - 1);
const double dy = (end_y - start_y) / (M - 1);
const double dz = (end_z - start_z) / (P - 1);

const int N1 = (1.0 / 3) * N;
const int N2 = (2.0 / 3) * N;
const int M1 = (1.0 / 3) * M;
const int M2 = (2.0 / 3) * M;
const int P1 = (2.0 / 3) * P;

const double dt = 0.0001;
const double a2 = 1;
const double eps = 1e-6;
const int stop_iteration = 3e4;

double U_old[P][M][N], U_new[P][M][N];
double x[N], y[M], z[P];
double A[P][M][N], B[P][M][N], C[P][M][N], D[P][M][N];
double alpha[P][M][N], beta[P][M][N];

// Fills array with some value
void fill_array(double arr[P][M][N], double value=0){
    for(int k = 0; k < P; ++k){
        for(int j = 0; j < M; ++j){
            for(int i = 0; i < N; ++i){
                arr[k][j][i] = value;
            }
        }
    }
}

// Returns maximum absolute difference of two arrays
double max_abs_diff(double A[P][M][N], double B[P][M][N]){
    double maximum = 0;
    for(int k = 0; k < P; ++k){
        for(int j = 0; j < M; ++j){
            for(int i = 0; i < N; ++i){
                if(maximum < fabs(A[k][j][i] - B[k][j][i])){
                    maximum = fabs(A[k][j][i] - B[k][j][i]);
                }
            }
        }
    }
    return maximum;
}

// copies values of array A to the array B
void copy_array(double A[P][M][N], double B[P][M][N]){
    for(int k = 0; k < P; ++k){
        for(int j = 0; j < M; ++j){
            for(int i = 0; i < N; ++i){
                B[k][j][i] = A[k][j][i];
            }
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

void Alternating_direction_method(double U_old[P][M][N], double U_new[P][M][N], int N, int M, int P, double dx, double dy, 
                                double dz, double dt, double a2=1, double eps=1e-6, int stop_iteration=3e4){
    
    int iteration = 0;
    double maximum = 0;
    do{
        // Finding U^(n+1/2)
        for(int k = 0; k < P; ++k){
            for(int j = 0; j < M; ++j){
                for(int i = 0; i < N; ++i){
                    A[k][j][i] = -a2 / (dx*dx);
                    B[k][j][i] = 1 / dt + 2*a2 / (dx*dx);
                    C[k][j][i] = -a2 / (dx*dx);
                }
            }
        }
        
        for(int k = 1; k < P-1; ++k){
            for(int j = 1; j < M-1; ++j){
                for(int i = 1; i < N-1; ++i){
                    D[k][j][i] = U_old[k][j][i] / dt
                        + a2*(
                        (U_old[k][j+1][i] - 2*U_old[k][j][i] + U_old[k][j-1][i])
                            / (dy*dy)
                        + (U_old[k+1][j][i] - 2*U_old[k][j][i] + U_old[k-1][j][i])
                            / (dz*dz));                
                }
            }
        }

        // Thomas algorithm for x
        // U(t, x=0, y, z) = 0
        for(int k = 0; k < P; ++k){
            for(int j = 0; j < M; ++j){
                alpha[k][j][1] = 0;
                beta[k][j][1] = 0;
            }
        }
        //  U(t, x=0, 1/3<y<2/3, 2/3<z<1) = 1
        for(int k = P1; k < P; ++k){
            for(int j = M1; j < M2; ++j){
                alpha[k][j][1] = 0;
                beta[k][j][1] = 1;
            }
        }

        for(int k = 1; k < P-1; ++k){
            for(int j = 1; j < M-1; ++j){
                for(int i = 1; i < N-1; ++i){
                    alpha[k][j][i+1] = -A[k][j][i]
                        / (B[k][j][i] + C[k][j][i]*alpha[k][j][i]);
                    beta[k][j][i+1] = (D[k][j][i] - C[k][j][i]*beta[k][j][i])
                        / (B[k][j][i] + C[k][j][i]*alpha[k][j][i]);
                }
            }
        }

        // U^(n+1/3)
        // U(t, x=1, y, z) = 0
        for(int k = 1; k < P-1; ++k){
            for(int j = 1; j < M-1; ++j){
                U_new[k][j][N-1] = 0;
                for(int i = N-2; i >= 0; --i){
                    U_new[k][j][i] = alpha[k][j][i+1]*U_new[k][j][i+1] 
                                        + beta[k][j][i+1];
                }
            }
        }

        // Finding U^(n+2/3)
        for(int k = 0; k < P; ++k){
            for(int j = 0; j < M; ++j){
                for(int i = 0; i < N; ++i){
                    A[k][j][i] = -a2 / (dy*dy);
                    B[k][j][i] = 1 / dt + 2*a2 / (dy*dy);
                    C[k][j][i] = -a2 / (dy*dy);
                }
            }
        }

        // U_new is U^(n+1/3) now
        for(int k = 1; k < P-1; ++k){
            for(int j = 1; j < M-1; ++j){
                for(int i = 1; i < N-1; ++i){
                    D[k][j][i] = U_new[k][j][i] / dt
                        + a2*(
                        (U_new[k][j][i+1] - 2*U_new[k][j][i] + U_new[k][j][i-1])
                            / (dx*dx)
                        + (U_new[k+1][j][i] - 2*U_new[k][j][i] + U_new[k-1][j][i])
                            / (dz*dz));
                }
            }
        }

        // Thomas algorithm for y
        // U(t, x, y=0, z) = 0
        for(int k = 0; k < P; ++k){
            for(int i = 0; i < M; ++i){
                alpha[k][1][i] = 0;
                beta[k][1][i] = 0;
            }
        }

        for(int k = 1; k < P-1; ++k){
            for(int i = 1; i < N-1; ++i){
                for(int j = 1; j < M-1; ++j){
                    alpha[k][j+1][i] = -A[k][j][i]
                        / (B[k][j][i] + C[k][j][i]*alpha[k][j][i]);
                    beta[k][j+1][i] = (D[k][j][i] - C[k][j][i]*beta[k][j][i])
                        / (B[k][j][i] + C[k][j][i]*alpha[k][j][i]);
                }
            }
        }

        // U^(n+2/3)
        // U(t, x, y=1, z) = 0
        for(int k = 1; k < P-1; ++k){
            for(int i = 1; i < N-1; ++i){
                U_new[k][M-1][i] = 0;
                for(int j = M-2; j >= 0; --j){
                    U_new[k][j][i] = alpha[k][j+1][i]*U_new[k][j+1][i] 
                                        + beta[k][j+1][i];
                }
            }
        }

        // Finding U^(n+1)
        for(int k = 0; k < P; ++k){
            for(int j = 0; j < M; ++j){
                for(int i = 0; i < N; ++i){
                    A[k][j][i] = -a2 / (dz*dz);
                    B[k][j][i] = 1 / dt + 2*a2 / (dz*dz);
                    C[k][j][i] = -a2 / (dz*dz);
                }
            }
        }

        // U_new is U^(n+2/3) now
        for(int k = 1; k < P-1; ++k){
            for(int j = 1; j < M-1; ++j){
                for(int i = 1; i < N-1; ++i){
                    D[k][j][i] = U_new[k][j][i] / dt
                        + a2*(
                        (U_new[k][j][i+1] - 2*U_new[k][j][i] + U_new[k][j][i-1])
                            / (dx*dx)
                        + (U_new[k][j+1][i] - 2*U_new[k][j][i] + U_new[k][j-1][i])
                            / (dy*dy));
                }
            }
        }

        // Thomas algorithm for z
        // from bottom to the right
        // U(t, x, y, z=0) = 0
        for(int j = 0; j < M; ++j){
            for(int i = 0; i < N; ++i){
                alpha[1][j][i] = 0;
                beta[1][j][i] = 0;
            }
        }
        // U(t, 1/3<x<2/3, 0<y<1/3, z=0) = 1
        for(int j = 0; j < M1; ++j){
            for(int i = N1; i < N2; ++i){
                alpha[1][j][i] = 0;
                beta[1][j][i] = 1;
            }
        }

        for(int j = 1; j < M-1; ++j){
            for(int i = 1; i < N-1; ++i){
                for(int k = 1; k < P-1; ++k){
                    alpha[k+1][j][i] = -A[k][j][i]
                        / (B[k][j][i] + C[k][j][i]*alpha[k][j][i]);
                    beta[k+1][j][i] = (D[k][j][i] - C[k][j][i]*beta[k][j][i])
                        / (B[k][j][i] + C[k][j][i]*alpha[k][j][i]);
                }
            }
        }

        // U(t, x, y, z=1) = 0
        for(int j = 1; j < M-1; ++j){
            for(int i = 1; i < N-1; ++i){
                U_new[P-1][j][i] = 0;
                for(int k = P-2; k >= 0; --k){
                    U_new[k][j][i] = alpha[k+1][j][i]*U_new[k+1][j][i] 
                                        + beta[k+1][j][i];
                }
            }
        }

        maximum = max_abs_diff(U_new, U_old);
        copy_array(U_new, U_old);

        // std::cout << iteration << "\t" << maximum << "\n";

        iteration++;
    }while(maximum > eps and iteration < stop_iteration);

    std::cout << "Alternating direction method\n";
    std::cout << "Number of iterations = " << iteration << std::endl;
    std::cout << "N = " << N << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "P = " << P << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "dy = " << dy << std::endl;
    std::cout << "dz = " << dz << std::endl;
    std::cout << "dt = " << dt << std::endl;
    std::cout << "a2 = " << a2 << std::endl;
    std::cout << "eps = " << eps << std::endl;
    std::cout << "Maximum = " << maximum << std::endl;
}

int main(){
    // Initialize arrays
    fill_array(alpha);
    fill_array(beta);
    fill_array(A);
    fill_array(B);
    fill_array(C);
    fill_array(D);

    for(int i = 0; i < N; ++i){
        x[i] = start_x + i*dx;
    }

    for(int j = 0; j < M; ++j){
        y[j] = start_y + j*dy;
    }

    for(int k = 0; k < P; ++k){
        z[k] = start_z + k*dz;
    }

    // Initial condition
    fill_array(U_old, 0);
    fill_array(U_new, 0);

    auto start = high_resolution_clock::now();

    Alternating_direction_method(U_old, U_new, N, M, P, dx, dy, dz, dt, a2);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);

    freopen("Results\\HW9_cpp.dat", "w", stdout);
    std::cout << "VARIABLES = \"X\", \"Y\", \"Z\", \"U\"" << std::endl;
    std::cout << "ZONE I = " << N << ", J = " << M << ", K = " << P << std::endl;

    for(int k = 0; k < P; ++k){
        for(int j = 0; j < M; ++j){
            for(int i = 0; i < N; ++i){
                std::cout << x[i] << "\t" << y[j] << "\t" << z[k] 
                    << "\t" << U_new[k][j][i] << std::endl;
            }
        }
    }

    return 0;
}
