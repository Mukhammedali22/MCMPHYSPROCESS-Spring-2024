#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;


// 2D Burger's equation
// Fractional step method (FSM)
// Problem 1

// Initial condition
// U(t=0, x, y) = 0
// V(t=0, x, y) = 0

// Boundary conditions
// U(t, x=0, 0<y<0.4) = 0
// Ux(t, x=0, 0.4<y<0.7) = 0
// U(t, x=0, 0.7<y<1) = 0
// U(t, x=1, y) = 0
// U(t, 0<x<0.7, y=0) = 0
// Uy(t, 0.7<x<1, y=0) = 0
// U(t, 0<x<0.7, y=1) = 0
// U(t, 0.7<x<1, y=1) = 0

// V(t, x=0, 0<y<0.4) = 0
// Vx(t, x=0, 0.4<y<0.7) = 0
// V(t, x=0, 0.7<y<1) = 0
// V(t, x=1, y) = 0
// V(t, 0<x<0.7, y=0) = 0
// Vy(t, 0.7<x<1, y=0) = 0
// V(t, 0<x<0.7, y=1) = 0
// V(t, 0.7<x<1, y=1) = -1

const double start_x = 0, end_x = 1;
const double start_y = 0, end_y = 1;

const int N = 101;
const int M = 101;

const double dx = (end_x - start_x) / (N - 1);
const double dy = (end_y - start_y) / (M - 1);

const int N1 = 0.7 * N;
const int M1 = 0.4 * M;
const int M2 = 0.7 * M;

const double dt = 0.01;
const double Re = 40;
const double nu = 1 / Re;
const double eps = 1e-6;
const int stop_iteration = 3e4;

double U_old[M][N], U_new[M][N];
double V_old[M][N], V_new[M][N];
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

void FSM_Burger(double S_new[M][N], double S_old[M][N], double U_old[M][N], double V_old[M][N]){
    
    // U[y, x] or U[j, i]
    const int N1 = 0.7 * N;
    const int M1 = 0.4 * M;
    const int M2 = 0.7 * M;

    // Finding U^(n+1/2)
    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            A[j][i] = -nu / (2*dx*dx) + U_old[j][i] / (2*dx);
            B[j][i] = 1 / dt + nu / (dx*dx) - U_old[j][i] / (2*dx);
            C[j][i] = -nu / (2*dx*dx);
        }
    }
    
    for(int j = 1; j < M-1; ++j){
        for(int i = 1; i < N-1; ++i){
            D[j][i] = S_old[j][i] / dt + 0.5*(
                nu*(S_old[j][i+1] - 2*S_old[j][i] + S_old[j][i-1])
                    / (dx*dx)
                - U_old[j][i] * (S_old[j][i+1] - S_old[j][i]) 
                    / dx)
                + nu*(S_old[j+1][i] - 2*S_old[j][i] + S_old[j-1][i])
                    / (dy*dy)
                - V_old[j][i] * (S_old[j+1][i] - S_old[j][i]) 
                    / dy;
        }
    }

    // Thomas algorithm for x
    // U(t, x=0, 0<y<0.4) = 0
    for(int j = 0; j < M1; ++j){
        alpha[j][1] = 0;
        beta[j][1] = 0;
    }
    // Ux(t, x=0, 0.4<y<0.7) = 0
    for(int j = M1; j < M2; ++j){
        alpha[j][1] = 1;
        beta[j][1] = 0;
    }
    // U(t, x=0, 0.7<y<1) = 0
    for(int j = M2; j < M; ++j){
        alpha[j][1] = 0;
        beta[j][1] = 0;
    }

    for(int j = 1; j < M-1; ++j){
        for(int i = 1; i < N-1; ++i){
            alpha[j][i+1] = -A[j][i]
                / (B[j][i] + C[j][i]*alpha[j][i]);
            beta[j][i+1] = (D[j][i] - C[j][i]*beta[j][i])
                / (B[j][i] + C[j][i]*alpha[j][i]);
        }
    }
    
    // U^(n+1/2)
    // U(t, x=1, y) = 0
    for(int j = 1; j < M-1; ++j){
        S_new[j][N-1] = S_old[j][N-1];
        for(int i = N-2; i >= 0; --i){
            S_new[j][i] = alpha[j][i+1]*S_new[j][i+1] 
                            + beta[j][i+1];
        }
    }

    // Finding U^(n+1)
    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            A[j][i] = -nu / (2*dy*dy) + V_old[j][i] / (2*dy);
            B[j][i] = 1 / dt + nu / (dy*dy) - V_old[j][i] / (2*dy);
            C[j][i] = -nu / (2*dy*dy);
        }
    }
    
    for(int j = 1; j < M-1; ++j){
        for(int i = 1; i < N-1; ++i){
            D[j][i] = S_new[j][i] / dt - 0.5*(
                nu*(S_old[j+1][i] - 2*S_old[j][i] + S_old[j-1][i])
                    / (dy*dy)
                - V_old[j][i] * (S_old[j+1][i] - S_old[j][i])
                    / dy);
        }
    }

    // Thomas algorithm for y
    // U(t, 0<x<0.7, y=0) = 0
    for(int i = 0; i < N1; ++i){
        alpha[1][i] = 0;
        beta[1][i] = 0;
    }
    // Uy(t, 0.7<x<1, y=0) = 0
    for(int i = N1; i < N; ++i){
        alpha[1][i] = 1;
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

    // U^(n+1)
    // U(t, 0<x<0.7, y=1) = 0
    // U(t, 0.7<x<1, y=1) = 0
    for(int i = 1; i < N-1; ++i){
        S_new[M-1][i] = S_old[M-1][i];
        for(int j = M-2; j >= 0; --j){
            S_new[j][i] = alpha[j+1][i]*S_new[j+1][i] 
                            + beta[j+1][i];
        }
    }
}

// Boundary conditions
// U(t, x=0, 0<y<0.4) = 0
// Ux(t, x=0, 0.4<y<0.7) = 0
// U(t, x=0, 0.7<y<1) = 0
// U(t, x=1, y) = 0
// U(t, 0<x<0.7, y=0) = 0
// Uy(t, 0.7<x<1, y=0) = 0
// U(t, 0<x<0.7, y=1) = 0
// U(t, 0.7<x<1, y=1) = 0
void set_boundary_U(double U[M][N]){
    for(int j = 0; j < M1; ++j){
        U[j][0] = 0;
    }
    for(int j = M1; j < M2; ++j){
        U[j][0] = U[j][1];
    }
    for(int j = M2; j < M; ++j){
        U[j][0] = 0;
    }
    for(int j = 0; j < M; ++j){
        U[j][N-1] = 0;
    }
    for(int i = 0; i < N1; ++i){
        U[0][i] = 0;
        U[M-1][i] = 0;
    }
    for(int i = N1; i < N; ++i){
        U[0][i] = U[1][i];
        U[M-1][i] = 0;
    }
}

// Boundary conditions
// V(t, x=0, 0<y<0.4) = 0
// Vx(t, x=0, 0.4<y<0.7) = 0
// V(t, x=0, 0.7<y<1) = 0
// V(t, x=1, y) = 0
// V(t, 0<x<0.7, y=0) = 0
// Vy(t, 0.7<x<1, y=0) = 0
// V(t, 0<x<0.7, y=1) = 0
// V(t, 0.7<x<1, y=1) = -1
void set_boundary_V(double V[M][N]){
    for(int j = 0; j < M1; ++j){
        V[j][0] = 0;
    }
    for(int j = M1; j < M2; ++j){
        V[j][0] = V[j][1];
    }
    for(int j = M2; j < M; ++j){
        V[j][0] = 0;
    }
    for(int j = 0; j < M; ++j){
        V[j][N-1] = 0;
    }
    for(int i = 0; i < N1; ++i){
        V[0][i] = 0;
        V[M-1][i] = 0;
    }
    for(int i = N1; i < N; ++i){
        V[0][i] = V[1][i];
        V[M-1][i] = -1;
    }
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
    fill_array(V_old, 0);
    fill_array(V_new, 0);

    // Boundary conditions
    set_boundary_U(U_old);
    set_boundary_V(V_old);    

    auto start = high_resolution_clock::now();

    int iteration = 0;
    double maximum = 0;
    do{
        FSM_Burger(U_new, U_old, U_old, V_old);
        FSM_Burger(V_new, V_old, U_old, V_old);

        double maximum_U = max_abs_diff(U_new, U_old);
        double maximum_V = max_abs_diff(V_new, V_old);
        maximum = std::fmax(maximum_U, maximum_V);
        
        copy_array(U_new, U_old);
        copy_array(V_new, V_old);
        // std::cout << iteration << "\t" << "\t" << maximum << "\n";

        iteration++;
    }while(maximum > eps and iteration < stop_iteration);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);

    std::cout << "Fractional step method\n";
    std::cout << "Number of iterations = " << iteration << std::endl;
    std::cout << "N = " << N << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "dy = " << dy << std::endl;
    std::cout << "dt = " << dt << std::endl;
    std::cout << "Re = " << Re << std::endl;
    std::cout << "nu = " << nu << std::endl;
    std::cout << "eps = " << eps << std::endl;
    std::cout << "Maximum = " << maximum << std::endl;

    savetxt("Results\\HW11_X_cpp.txt", X, "%.6f", '\t');
    savetxt("Results\\HW11_Y_cpp.txt", Y, "%.6f", '\t');
    savetxt("Results\\HW11_U_cpp.txt", U_new, "%.6f", '\t');
    savetxt("Results\\HW11_V_cpp.txt", V_new, "%.6f", '\t');
    
    std::cout << "Results are recorded" << std::endl;

    freopen("Results\\HW11_cpp.dat", "w", stdout);
    std::cout << "VARIABLES = \"X\", \"Y\", \"U\", \"V\"" << std::endl;
    std::cout << "ZONE I = " << N << ", J = " << M << std::endl;

    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            std::cout << X[i] << "\t" << Y[j] << "\t" << U_new[j][i] 
                << "\t" << V_new[j][i] << std::endl;
        }
    }

    return 0;
}
