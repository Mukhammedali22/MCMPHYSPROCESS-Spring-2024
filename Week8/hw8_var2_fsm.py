import numpy as np
from numpy import (pi, exp, sin, cos)
import matplotlib.pyplot as plt
from time import perf_counter
from functools import wraps
from numba import jit
import os


# 2D Heat equation
# Fractional step method (FSM)
# Problem 2

# Initial condition
# U(t=0, x, y) = 0

# Boundary conditions
# U(x=0, 0<y<0.7) = 0
# U(x=0, 0.7<y<1) = 1
# U(x=1, 0<y<0.3) = 1
# U(x=1, 0.3<y<1) = 0
# U(x, y=0) = 0
# U(x, y=1) = 0

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print(f"{func.__doc__}")
        print(f"Calculation time: {(end_time - start_time):.3f} seconds\n")
        return result
    
    return timeit_wrapper

@timeit
def Fractional_step_method(U:np.ndarray, N, M, dx, dy, dt, 
                           a2=1, eps=1e-6, stop_iteration=3e4):
    """Fractional step method for solving 2D Heat equation"""
    U_old = U.copy()
    U_new = np.zeros_like(U)

    M1 = int(0.3 * M)
    M2 = int(0.7 * M)

    A = np.zeros((M, N))
    B = np.zeros((M, N))
    C = np.zeros((M, N))
    D = np.zeros((M, N))

    alpha = np.zeros((M, N))
    beta = np.zeros((M, N))

    iteration = 0
    maximum = 1
    while maximum > eps and iteration < stop_iteration:
        # Finding U^(n+1/2)
        A[0:M, 0:N] = -a2 / (2*dx**2)
        B[0:M, 0:N] = 1 / dt + a2 / dx**2
        C[0:M, 0:N] = -a2 / (2*dx**2)
        
        D[1:M-1, 1:N-1] = U_old[1:M-1, 1:N-1] / dt \
            + a2*(U_old[1:M-1, 2:N] - 2*U_old[1:M-1, 1:N-1] + U_old[1:M-1, 0:N-2]) \
                / (2*dx**2) \
            + a2*(U_old[2:M, 1:N-1] - 2*U_old[1:M-1, 1:N-1] + U_old[0:M-2, 1:N-1]) \
                / dy**2
        
        # Thomas algorithm for x
        # U(t, x=0, 0<y<0.7) = 0
        alpha[0:M2, 1] = 0
        beta[0:M2, 1] = 0
        # U(t, x=0, 0.7<y<1) = 1
        alpha[M2:M, 1] = 0
        beta[M2:M, 1] = 1
        
        for i in range(1, N-1):
            alpha[1:M-1, i+1] = -A[1:M-1, i] \
                / (B[1:M-1, i] + C[1:M-1, i]*alpha[1:M-1, i])
            beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:M-1, i]*beta[1:M-1, i]) \
                / (B[1:M-1, i] + C[1:M-1, i]*alpha[1:M-1, i])
            
        # U^(n+1/2)
        # U(t, x=1, 0<y<0.3) = 1
        U_new[0:M1, N-1] = 1
        # U(t, x=1, 0.3<y<1) = 0
        U_new[M1:M, N-1] = 0
        for i in range(N-2, -1, -1):
            U_new[1:M-1, i] = alpha[1:M-1, i+1]*U_new[1:M-1, i+1] + beta[1:M-1, i+1]   
        
        # Finding U^(n+1)
        A[0:M, 0:N] = -a2 / (2*dy**2)
        B[0:M, 0:N] = 1 / dt + a2 / dy**2
        C[0:M, 0:N] = -a2 / (2*dy**2)
        
        D[1:M-1, 1:N-1] = U_new[1:M-1, 1:N-1] / dt \
            - a2*(U_old[2:M, 1:N-1] - 2*U_old[1:M-1, 1:N-1] + U_old[0:M-2, 1:N-1]) \
                / (2*dy**2)
            
        # Thomas algorithm for y
        # U(t, x, y=0) = 0
        alpha[1, 0:N] = 0
        beta[1, 0:N] = 0
        
        for j in range(1, M-1):
            alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            
        # U^(n+1)
        # U(t, x, y=1) = 0
        U_new[M-1, 0:N] = 0
        for j in range(M-2, -1, -1):
            U_new[j, 1:N-1] = alpha[j+1, 1:N-1]*U_new[j+1, 1:N-1] + beta[j+1, 1:N-1]

        maximum = np.max(np.abs(U_new - U_old))        
        # print("Iteration", iteration, "\t", "maximum", maximum)
        U_old = U_new.copy()
        iteration += 1

    print("Number of iterations:", iteration)
    print("Maximum absolute difference:", maximum)

    return U_new

@timeit
@jit(nopython=True)
def Numba_Fractional_step_method(U:np.ndarray, N, M, dx, dy, dt, 
                           a2=1, eps=1e-6, stop_iteration=3e4):
    """Boosted Fractional step method for solving 2D Heat equation"""
    U_old = U.copy()
    U_new = np.zeros_like(U)

    M1 = int(0.3 * M)
    M2 = int(0.7 * M)

    A = np.zeros((M, N))
    B = np.zeros((M, N))
    C = np.zeros((M, N))
    D = np.zeros((M, N))

    alpha = np.zeros((M, N))
    beta = np.zeros((M, N))

    iteration = 0
    maximum = 1
    while maximum > eps and iteration < stop_iteration:
        # Finding U^(n+1/2)
        A[0:M, 0:N] = -a2 / (2*dx**2)
        B[0:M, 0:N] = 1 / dt + a2 / dx**2
        C[0:M, 0:N] = -a2 / (2*dx**2)
        
        D[1:M-1, 1:N-1] = U_old[1:M-1, 1:N-1] / dt \
            + a2*(U_old[1:M-1, 2:N] - 2*U_old[1:M-1, 1:N-1] + U_old[1:M-1, 0:N-2]) \
                / (2*dx**2) \
            + a2*(U_old[2:M, 1:N-1] - 2*U_old[1:M-1, 1:N-1] + U_old[0:M-2, 1:N-1]) \
                / dy**2
        
        # Thomas algorithm for x
        # U(t, x=0, 0<y<0.7) = 0
        alpha[0:M2, 1] = 0
        beta[0:M2, 1] = 0
        # U(t, x=0, 0.7<y<1) = 1
        alpha[M2:M, 1] = 0
        beta[M2:M, 1] = 1
        
        for i in range(1, N-1):
            alpha[1:M-1, i+1] = -A[1:M-1, i] \
                / (B[1:M-1, i] + C[1:M-1, i]*alpha[1:M-1, i])
            beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:M-1, i]*beta[1:M-1, i]) \
                / (B[1:M-1, i] + C[1:M-1, i]*alpha[1:M-1, i])
            
        # U^(n+1/2)
        # U(t, x=1, 0<y<0.3) = 1
        U_new[0:M1, N-1] = 1
        # U(t, x=1, 0.3<y<1) = 0
        U_new[M1:M, N-1] = 0
        for i in range(N-2, -1, -1):
            U_new[1:M-1, i] = alpha[1:M-1, i+1]*U_new[1:M-1, i+1] + beta[1:M-1, i+1]   
        
        # Finding U^(n+1)
        A[0:M, 0:N] = -a2 / (2*dy**2)
        B[0:M, 0:N] = 1 / dt + a2 / dy**2
        C[0:M, 0:N] = -a2 / (2*dy**2)
        
        D[1:M-1, 1:N-1] = U_new[1:M-1, 1:N-1] / dt \
            - a2*(U_old[2:M, 1:N-1] - 2*U_old[1:M-1, 1:N-1] + U_old[0:M-2, 1:N-1]) \
                / (2*dy**2)
            
        # Thomas algorithm for y
        # U(t, x, y=0) = 0
        alpha[1, 0:N] = 0
        beta[1, 0:N] = 0
        
        for j in range(1, M-1):
            alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            
        # U^(n+1)
        # U(t, x, y=1) = 0
        U_new[M-1, 0:N] = 0
        for j in range(M-2, -1, -1):
            U_new[j, 1:N-1] = alpha[j+1, 1:N-1]*U_new[j+1, 1:N-1] + beta[j+1, 1:N-1]

        maximum = np.max(np.abs(U_new - U_old))        
        # print("Iteration", iteration, "\t", "maximum", maximum)
        U_old = U_new.copy()
        iteration += 1

    print("Number of iterations:", iteration)
    print("Maximum absolute difference:", maximum)

    return U_new
 
def plot_result(X, Y, P, name="Numerical method"):
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contourf(X, Y, P)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


start_x, end_x = (0, 1)
start_y, end_y = (0, 1)

dt = 0.01
a2 = 1

N = 101
M = 101
dx = (end_x - start_x) / (N - 1)
dy = (end_y - start_y) / (M - 1)

M1 = int(0.3 * M)
M2 = int(0.7 * M)

x = start_x + np.arange(start=0, stop=N) * dx
y = start_y + np.arange(start=0, stop=M) * dy
X, Y = np.meshgrid(x, y)

U_old = np.zeros((M, N))
U_new = np.zeros((M, N))

U_1 = Fractional_step_method(U_old, N, M, dx, dy, dt, a2=a2)

U_1N = Numba_Fractional_step_method(U_old, N, M, dx, dy, dt, a2=a2)
U_1N = Numba_Fractional_step_method(U_old, N, M, dx, dy, dt, a2=a2)
U_1N = Numba_Fractional_step_method(U_old, N, M, dx, dy, dt, a2=a2)
U_1N = Numba_Fractional_step_method(U_old, N, M, dx, dy, dt, a2=a2)
U_1N = Numba_Fractional_step_method(U_old, N, M, dx, dy, dt, a2=a2)

path = "Results"
if not os.path.exists(path):
    os.makedirs(path)

np.savetxt(f"{path}\\HW8_X_py.txt", X, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW8_Y_py.txt", Y, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW8_U_py.txt", U_1, fmt="%.6f", delimiter="\t")

plot_result(X, Y, U_1, name="Fractional step method")
plot_result(X, Y, U_1N, name="Fractional step method")
