#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;


// 1D Transport equation
// The first scheme against the flow method
// Problem 2

// Initial condition
// U(t=0, x) = cos(pi*x/2)

// Boundary conditions
// U(t, x=0) = 1
// U(t, x=1) = 0

// c = -1
// If c is negative use forward scheme else backward scheme


const double pi = 3.14159265359;
const double c = -1;
const double dx = 0.01;
const double dt = dx / fabs(c);

const double eps = 1e-6;
const int stop_iteration = 3e4;

const double start_x = 0, end_x = 3;
const int N = (end_x - start_x) / dx + 1;

void set_boundary_U(double U[N]){
    U[0] = 1;
    U[N-1] = 0;

}

// Fills array with some value
void fill_array(double arr[N], double value=0){
    for(int i = 0; i < N; ++i){
        arr[i] = value;
    }
}

// Returns maximum absolute difference of two arrays
double max_abs_diff(double A[N], double B[N]){
    double maximum = 0;
    for(int i = 0; i < N; ++i){
        if(maximum < fabs(A[i] - B[i])){
            maximum = fabs(A[i] - B[i]);
        }
    }
    return maximum;
}

// copies values of array A to the array B
void copy_array(double A[N], double B[N]){
    for(int i = 0; i < N; ++i){
        B[i] = A[i];
    }
}

void savetxt(std::string path, double arr[N], 
                std::string fmt="%f", char delimeter=' '){
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
    double x[N], U_old[N], U_new[N];
    
    for(int i = 0; i < N; ++i){
        x[i] = start_x + i*dx;
    }

    // Initialize arrays
    fill_array(U_old, 0);
    fill_array(U_new, 0);

    auto start = high_resolution_clock::now();

    // initial condition
    // U(x, t=0) = cos(pi*x/2)
    for(int i = 0; i < N; ++i){
        U_old[i] = cos(pi*x[i]/2);
    }

    // Boundary conditions
    set_boundary_U(U_old);

    int iteration = 0;
    double maximum = 0;
    do{
        if(c > 0){
            // Use backward scheme
            for(int i = 1; i < N-1; ++i){
                U_new[i] = U_old[i] - c * dt / dx * (U_old[i] - U_old[i-1]);
            }
        }else{
            // Use forward scheme
            for(int i = 1; i < N-1; ++i){
                U_new[i] = U_old[i] - c * dt / dx * (U_old[i+1] - U_old[i]);
            }
        }

        set_boundary_U(U_new);

        maximum = max_abs_diff(U_new, U_old);
        printf("Iteration = %d\tmaximum = %.6f\n", iteration, maximum);
        copy_array(U_new, U_old);

        iteration++;
    }while(maximum > eps and iteration < stop_iteration);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    printf("calculating time: %.6f seconds\n", duration.count() / 1e6);
    printf("Maximum difference: %.9f\n", maximum);
    printf("Time t when U reached steady state: %.3f\n", iteration*dt);
    printf("Number of iterations: %d\n", iteration);
    printf("Parameters:\nc=%f\ndx=%f\ndt=%f\n", c, dx, dt);

    savetxt("Results\\HW5_X_cpp.txt", x, "%.6f", '\t');
    savetxt("Results\\HW5_U_cpp.txt", U_new, "%.6f", '\t');
    
    return 0;
}
