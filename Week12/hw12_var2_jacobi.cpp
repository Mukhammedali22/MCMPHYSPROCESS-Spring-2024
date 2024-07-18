#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;


// 2D Navier-Stokes equation
// Projection method
// Explicit + Jacobi
// Problem 1

// Initial condition
// U(t=0, x, y) = 0
// V(t=0, x, y) = 0

// Boundary conditions
// U(t, x=0, y) = 0
// U(t, x=1, y) = 0
// Uy(t, 0<x<0.2, y=0) = 0
// U(t, 0.2<x<0.8, y=0) = 0
// Uy(t, 0.8<x<1, y=0) = 0
// U(t, 0<x<0.3, y=1) = 0
// U(t, 0.3<x<0.7, y=1) = 0
// U(t, 0.7<x<1, y=1) = 0

// V(t, x=0, y) = 0
// V(t, x=1, y) = 0
// Vy(t, 0<x<0.2, y=0) = 0
// V(t, 0.2<x<0.8, y=0) = 0
// Vy(t, 0.8<x<1, y=0) = 0
// V(t, 0<x<0.3, y=1) = 0
// V(t, 0.3<x<0.7, y=1) = -1
// V(t, 0.7<x<1, y=1) = 0

// Px(t, x=0, y) = 0
// Px(t, x=1, y) = 0
// P(t, 0<x<0.2, y=0) = 0
// Py(t, 0.2<x<0.8, y=0) = 0
// P(t, 0.8<x<1, y=0) = 0
// Py(t, 0<x<0.3, y=1) = 0
// P(t, 0.3<x<0.7, y=1) = 1
// Py(t, 0.7<x<1, y=1) = 0

const double start_x = 0, end_x = 1;
const double start_y = 0, end_y = 1;

const int N = 101;
const int M = 101;

const double dx = (end_x - start_x) / (N - 1);
const double dy = (end_y - start_y) / (M - 1);

const int N1 = 0.2 * N;
const int N2 = 0.3 * N;
const int N3 = 0.7 * M;
const int N4 = 0.8 * M;

const double dt = 0.0001;
const double Re = 100;
const double nu = 1 / Re;
const double rho = 1;
const double eps = 1e-6;
const int stop_iteration = 3e4;
const double eps_P = 1e-6;
const int stop_iteration_P = 1e5;

double U_old[M][N], U_new[M][N];
double V_old[M][N], V_new[M][N];
double P_old[M][N], P_new[M][N];
double X[M][N], Y[M][N];

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

void Explicit_Burger(double S_new[M][N], double S_old[M][N], double U_old[M][N], double V_old[M][N], 
                        int N, int M, double dx, double dy, double dt, double nu){
    for(int j = 1; j < M-1; ++j){
        for(int i = 1; i < N-1; ++i){
            // central scheme
            S_new[j][i] = S_old[j][i] + dt*(
                - U_old[j][i]*(S_old[j][i+1] - S_old[j][i-1]) 
                                / (2*dx)
                - V_old[j][i]*(S_old[j+1][i] - S_old[j-1][i]) 
                                / (2*dy)
                + nu*(
                    (S_old[j][i+1] - 2*S_old[j][i] + S_old[j][i-1]) 
                        / (dx*dx) 
                    + (S_old[j+1][i] - 2*S_old[j][i] + S_old[j-1][i])
                        / (dy*dy))
            );
        }
    }
}

void Jacobi_Poisson(double P_new[M][N], double P_old[M][N], double U_new[M][N], double V_new[M][N], 
                      int N, int M, double dx, double dy, double dt, double rho){
    for(int j = 1; j < M-1; ++j){
        for(int i = 1; i < N-1; ++i){
            // central scheme
            P_new[j][i] = (
                dy*dy*(P_old[j][i+1] + P_old[j][i-1])
                + dx*dx*(P_old[j+1][i] + P_old[j-1][i])) 
                    / (2*(dx*dx + dy*dy))
                - dx*dx*dy*dy*rho
                    / (2*dt*(dx*dx + dy*dy))
                * ((U_new[j][i+1] - U_new[j][i-1]) 
                        / (2*dx) 
                    + (V_new[j+1][i] - V_new[j-1][i]) 
                        / (2*dy));
        }
    }           
}

// Boundary conditions
// U(t, x=0, y) = 0
// U(t, x=1, y) = 0
// Uy(t, 0<x<0.2, y=0) = 0
// U(t, 0.2<x<0.8, y=0) = 0
// Uy(t, 0.8<x<1, y=0) = 0
// U(t, 0<x<0.3, y=1) = 0
// U(t, 0.3<x<0.7, y=1) = 0
// U(t, 0.7<x<1, y=1) = 0
void set_boundary_U(double U[M][N]){
    for(int j = 0; j < M; ++j){
        U[j][0] = 0;
        U[j][N-1] = 0;
    }
    for(int i = 0; i < N1; ++i){
        U[0][i] = U[1][i];
    }
    for(int i = N1; i < N4; ++i){
        U[0][i] = 0;
    }
    for(int i = N4; i < N; ++i){
        U[0][i] = U[1][i];
    }
    for(int i = 0; i < N2; ++i){
        U[M-1][i] = 0;
    }
    for(int i = N2; i < N3; ++i){
        U[M-1][i] = 0;
    }
    for(int i = N3; i < N; ++i){
        U[M-1][i] = 0;
    }
}

// Boundary conditions
// V(t, x=0, y) = 0
// V(t, x=1, y) = 0
// Vy(t, 0<x<0.2, y=0) = 0
// V(t, 0.2<x<0.8, y=0) = 0
// Vy(t, 0.8<x<1, y=0) = 0
// V(t, 0<x<0.3, y=1) = 0
// V(t, 0.3<x<0.7, y=1) = -1
// V(t, 0.7<x<1, y=1) = 0
void set_boundary_V(double V[M][N]){
    for(int j = 0; j < M; ++j){
        V[j][0] = 0;
        V[j][N-1] = 0;
    }
    for(int i = 0; i < N1; ++i){
        V[0][i] = V[1][i];
    }
    for(int i = N1; i < N4; ++i){
        V[0][i] = 0;
    }
    for(int i = N4; i < N; ++i){
        V[0][i] = V[1][i];
    }
    for(int i = 0; i < N2; ++i){
        V[M-1][i] = 0;
    }
    for(int i = N2; i < N3; ++i){
        V[M-1][i] = -1;
    }
    for(int i = N3; i < N; ++i){
        V[M-1][i] = 0;
    }
}

// Px(t, x=0, y) = 0
// Px(t, x=1, y) = 0
// P(t, 0<x<0.2, y=0) = 0
// Py(t, 0.2<x<0.8, y=0) = 0
// P(t, 0.8<x<1, y=0) = 0
// Py(t, 0<x<0.3, y=1) = 0
// P(t, 0.3<x<0.7, y=1) = 1
// Py(t, 0.7<x<1, y=1) = 0
void set_boundary_P(double P[M][N]){
    for(int j = 0; j < M; ++j){
        P[j][0] = P[j][1];
        P[j][N-1] = P[j][N-2];
    }
    for(int i = 0; i < N1; ++i){
        P[0][i] = 0;
    }
    for(int i = N1; i < N4; ++i){
        P[0][i] = P[1][i];
    }
    for(int i = N4; i < N; ++i){
        P[0][i] = 0;
    }
    for(int i = 0; i < N2; ++i){
        P[M-1][i] = P[M-2][i];
    }
    for(int i = N2; i < N3; ++i){
        P[M-1][i] = 1;
    }
    for(int i = N3; i < N; ++i){
        P[M-1][i] = P[M-2][i];
    }
}


int main(){
    // Filling grid
    meshgrid(X, Y);

    // Initial condition
    fill_array(U_old, 0);
    fill_array(U_new, 0);
    fill_array(V_old, 0);
    fill_array(V_new, 0);
    fill_array(P_old, 0);
    fill_array(P_new, 0);

    auto start = high_resolution_clock::now();

    // Boundary conditions
    set_boundary_U(U_old);
    set_boundary_V(V_old);

    int iteration = 0;
    double maximum = 0;
    int iteration_P = 0;
    double maximum_P = 0;
    do{
        Explicit_Burger(U_new, U_old, U_old, V_old, N, M, dx, dy, dt, nu);
        Explicit_Burger(V_new, V_old, U_old, V_old, N, M, dx, dy, dt, nu);
        set_boundary_U(U_new);
        set_boundary_V(V_new);

        set_boundary_P(P_old);
        set_boundary_P(P_new);

        iteration_P = 0;
        maximum_P = 0;
        do{
            Jacobi_Poisson(P_new, P_old, U_new, V_new, N, M, dx, dy, dt, rho);
            
            set_boundary_P(P_new);            
            maximum_P = max_abs_diff(P_new, P_old);
            copy_array(P_new, P_old);
            iteration_P++;
        }while(maximum_P > eps_P and iteration_P < stop_iteration_P);

        // Correct U, V
        for(int j = 1; j < M-1; ++j){
            for(int i = 1; i < N-1; ++i){
                U_new[j][i] = U_new[j][i] - dt / rho * (P_new[j][i+1] - P_new[j][i-1]) / (2*dx);
                V_new[j][i] = V_new[j][i] - dt / rho * (P_new[j+1][i] - P_new[j-1][i]) / (2*dy);
            }
        }
        
        set_boundary_U(U_new);
        set_boundary_V(V_new);

        double maximum_U = max_abs_diff(U_new, U_old);
        double maximum_V = max_abs_diff(V_new, V_old);
        maximum = std::fmax(maximum_U, maximum_V);
        
        copy_array(U_new, U_old);
        copy_array(V_new, V_old);
        
        if(iteration % 2 == 0){
            std::cout << iteration << "\t" << maximum << "\t" 
            << iteration_P << "\t" << maximum_P << "\n";
        }

        iteration++;
    }while(maximum > eps and iteration < stop_iteration);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);

    std::cout << "Explicit method\n";
    std::cout << "Number of iterations = " << iteration << std::endl;
    std::cout << "N = " << N << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "dy = " << dy << std::endl;
    std::cout << "dt = " << dt << std::endl;
    std::cout << "Re = " << Re << std::endl;
    std::cout << "rho = " << rho << std::endl;
    std::cout << "nu = " << nu << std::endl;
    std::cout << "eps = " << eps << std::endl;
    std::cout << "Maximum = " << maximum << std::endl;

    savetxt("Results\\HW12_X_cpp.txt", X, "%.6f", '\t');
    savetxt("Results\\HW12_Y_cpp.txt", Y, "%.6f", '\t');
    savetxt("Results\\HW12_U_cpp.txt", U_new, "%.6f", '\t');
    savetxt("Results\\HW12_V_cpp.txt", V_new, "%.6f", '\t');
    savetxt("Results\\HW12_P_cpp.txt", P_new, "%.6f", '\t');
    
    std::cout << "Results are recorded" << std::endl;

    freopen("Results\\HW12_cpp.dat", "w", stdout);
    std::cout << "VARIABLES = \"X\", \"Y\", \"U\", \"V\", \"P\"" << std::endl;
    std::cout << "ZONE I = " << N << ", J = " << M << std::endl;

    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            std::cout << X[i] << "\t" << Y[j] << "\t" << U_new[j][i] 
                << "\t" << V_new[j][i] << "\t" << P_new[j][i] << std::endl;
        }
    }

    return 0;
}
