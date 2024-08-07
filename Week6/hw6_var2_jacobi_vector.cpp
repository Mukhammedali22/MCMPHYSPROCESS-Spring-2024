#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std::chrono;


// 2D Laplace equation
// Jacobi method
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

std::vector<std::vector<double>> create_2D_vector(unsigned height, unsigned width, double value=0){
   return std::vector<std::vector<double>> (height, std::vector<double>(width, value));
}

// Passing vector by value
void print_2D_vector(std::vector<std::vector<double>> U, std::string sep=" ", std::string end="\n"){
    int M = U.size();
    int N = U[0].size();

    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            if(i != N-1){
                std::cout << U[j][i];
                if(i != N-2){
                    std::cout << sep;
                }
            }else{
                std::cout << end;
            }
        }
    }
}

// Passing vector by reference
void fill_2D_vector(std::vector<std::vector<double>>& U, double value=0){
    std::fill(U.begin(), U.end(), std::vector<double> (U[0].size(), value));
}

// Passing vector by reference
void meshgrid(std::vector<std::vector<double>>& X, 
                std::vector<std::vector<double>>& Y){
    
    for(int j = 0; j < X.size(); ++j){
        for(int i = 0; i < X[j].size(); ++i){
            X[j][i] = start_x + i*dx;
            Y[j][i] = start_y + j*dy;
        }
    }
}

// Returns maximum absolute difference of two vectors
double max_abs_diff(std::vector<std::vector<double>>& A, 
                    std::vector<std::vector<double>>& B){
    
    double maximum = 0;
    for(int j = 0; j < A.size(); ++j){
        for(int i = 0; i < A[j].size(); ++i){
            if(maximum < fabs(A[j][i] - B[j][i])){
                maximum = fabs(A[j][i] - B[j][i]);
            }
        }
    }
    return maximum;
}

void savetxt(std::string path, std::vector<std::vector<double>> arr, 
                std::string fmt="%f", std::string sep=" ", std::string end="\n"){
    
    int M = arr.size();
    int N = arr[0].size();
    
    const char * c = path.c_str();
    FILE *File;
    File = freopen(c, "w", stdout);
    for(int j = 0; j < M; ++j){
        for(int i = 0; i < N; ++i){
            if(i != N-1){
                printf(fmt.c_str(), arr[j][i]);
                if(i != N-2){
                    std::cout << sep;
                }
            }else{
                std::cout << end;
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
void set_boundary_P(std::vector<std::vector<double>>& P){
    for(int j = 0; j < M2; ++j){
        P[j][0] = 0;
    }
    for(int j = M2; j < M; ++j){
        P[j][0] = 1;
    }
    for(int j = 0; j < M1; ++j){
        P[j][N-1] = 1;
    }
    for(int j = M1; j < M; ++j){
        P[j][N-1] = 0;
    }
    for(int i = 0; i < N; ++i){
        P[0][i] = 0;
        P[M-1][i] = 0;
    }
}

std::vector<std::vector<double>> Jacobi_method(std::vector<std::vector<double>> P, int N, int M, double dx, double dy, 
                                                double eps=1e-6, int stop_iteration=3e4){
    "Jacobi method for solving 2D Laplace equation";
    std::vector<std::vector<double>> P_old = P;
    std::vector<std::vector<double>> P_new = create_2D_vector(M, N);

    set_boundary_P(P_old);

    int iteration = 0;
    double maximum = 0;
    do{
        for(int j = 1; j < M-1; ++j){
            for(int i = 1; i < N-1; ++i){
                P_new[j][i] = (dy*dy*(
                    P_old[j][i+1] + P_old[j][i-1])
                     + dx*dx*(
                    P_old[j+1][i] + P_old[j-1][i])
                    ) / (2*(dx*dx + dy*dy));
            }
        }

        set_boundary_P(P_new);

        maximum = max_abs_diff(P_new, P_old);
        P_old = P_new;
        iteration++;
    }while(maximum > eps and iteration < stop_iteration);

    std::cout << "Jacobi method\n";
    std::cout << "Number of iterations = " << iteration << std::endl;
    std::cout << "N = " << N << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "dy = " << dy << std::endl;
    std::cout << "eps = " << eps << std::endl;
    std::cout << "Maximum = " << maximum << std::endl;

    return P_new;
}

int main(){
    // Declaration and initialization of 2D vector with 0
    std::vector<std::vector<double>> P_old = create_2D_vector(M, N, 0);
    std::vector<std::vector<double>> P_new = create_2D_vector(M, N, 0);
    std::vector<std::vector<double>> X = create_2D_vector(M, N, 0);
    std::vector<std::vector<double>> Y = create_2D_vector(M, N, 0);

    meshgrid(X, Y);
    // P_new = P_old; // copies values
    auto start = high_resolution_clock::now();

    P_new = Jacobi_method(P_old, N, M, dx, dy);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);

    // Vectors 3-4 times slower than raw arrays :(
    savetxt("Results\\HW6_X_cpp.txt", X, "%.6f", "\t");
    savetxt("Results\\HW6_Y_cpp.txt", Y, "%.6f", "\t");
    savetxt("Results\\HW6_P1_cpp.txt", P_new, "%.6f", "\t");

    std::cout << "Results are recorded" << std::endl;
    return 0;
}
