import numpy as np
from numpy import (pi, exp, sin, cos)
import matplotlib.pyplot as plt
from time import perf_counter
from functools import wraps
from numba import jit
import os


# 3D Heat equation
# Alternating direction method (ADM)
# Problem 2

# Initial condition
# U(t=0, x, y, z) = 0

# Boundary conditions
# U(t, x=0, y, z) = 0
# U(t, x=0, 1/3<y<2/3, 2/3<z<1) = 1
# U(t, x=1, y, z) = 0
# U(t, x, y=0, z) = 0
# U(t, x, y=1, z) = 0
# U(t, x, y, z=0) = 0
# U(t, 1/3<x<2/3, 0<y<1/3, z=0) = 1
# U(t, x, y, z=1) = 0

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
def Alternating_direction_method(U:np.ndarray, N, M, P, dx, dy, dz, dt, 
                           a2=1, eps=1e-6, stop_iteration=3e4):
    """Alternating direction method for solving 3D Heat equation"""
    # U[z, y, x] or U[k, j, i]
    U_old = U.copy()
    U_new = np.zeros_like(U)

    N1 = int(1/3 * N)
    N2 = int(2/3 * N)
    M1 = int(1/3 * N)
    M2 = int(2/3 * N)
    P1 = int(2/3 * N)

    A = np.zeros((P, M, N))
    B = np.zeros((P, M, N))
    C = np.zeros((P, M, N))
    D = np.zeros((P, M, N))

    alpha = np.zeros((P, M, N))
    beta = np.zeros((P, M, N))

    iteration = 0
    maximum = 1
    while maximum > eps and iteration < stop_iteration:
        # Finding U^(n+1/3)
        A[0:P, 0:M, 0:N] = -a2 / dx**2
        B[0:P, 0:M, 0:N] = 1 / dt + 2 * a2 / dx**2
        C[0:P, 0:M, 0:N] = -a2 / dx**2
        
        D[1:P-1, 1:M-1, 1:N-1] = U_old[1:P-1, 1:M-1, 1:N-1] / dt \
            + a2*(
            (U_old[1:P-1, 2:M, 1:N-1] - 2*U_old[1:P-1, 1:M-1, 1:N-1] \
                + U_old[1:P-1, 0:M-2, 1:N-1]) \
                    / dy**2 \
            + (U_old[2:P, 1:M-1, 1:N-1] - 2*U_old[1:P-1, 1:M-1, 1:N-1] \
                + U_old[0:P-2, 1:M-1, 1:N-1]) \
                    / dz**2)
            
        # Thomas algorithm for x
        # from back to the front
        # U(t, x=0, y, z) = 0
        alpha[0:P, 0:M, 1] = 0
        beta[0:P, 0:M, 1] = 0
        # U(t, x=0, 1/3<y<2/3, 2/3<z<1) = 1
        alpha[P1:P, M1:M2, 1] = 0
        beta[P1:P, M1:M2, 1] = 1

        # alpha[0:P, 0:M, 1] = 0
        # beta[0:P, 0:M, 1] = U_old[0:P, 0:M, 0]
        
        for i in range(1, N-1):
            alpha[1:P-1, 1:M-1, i+1] = -A[1:P-1, 1:M-1, i] \
                / (B[1:P-1, 1:M-1, i] + C[1:P-1, 1:M-1, i]*alpha[1:P-1, 1:M-1, i])
            
            beta[1:P-1, 1:M-1, i+1] = (D[1:P-1, 1:M-1, i] \
                    - C[1:P-1, 1:M-1, i]*beta[1:P-1, 1:M-1, i]) \
                / (B[1:P-1, 1:M-1, i] + C[1:P-1, 1:M-1, i]*alpha[1:P-1, 1:M-1, i])
            
        # U^(n+1/3)
        # U(t, x=1, y, z) = 0
        U_new[0:P, 0:M, N-1] = 0
        for i in range(N-2, -1, -1):
            U_new[1:P-1, 1:M-1, i] = alpha[1:P-1, 1:M-1, i+1]*U_new[1:P-1, 1:M-1, i+1] \
                + beta[1:P-1, 1:M-1, i+1]
        
        # Finding U^(n+2/3)
        A[0:P, 0:M, 0:N] = -a2 / dy**2
        B[0:P, 0:M, 0:N] = 1 / dt + 2 * a2 / dy**2
        C[0:P, 0:M, 0:N] = -a2 / dy**2
        
        # U_new is U^(n+1/3) now
        D[1:P-1, 1:M-1, 1:N-1] = U_new[1:P-1, 1:M-1, 1:N-1] / dt \
            + a2*(
            (U_new[1:P-1, 1:M-1, 2:N] - 2*U_new[1:P-1, 1:M-1, 1:N-1] \
                + U_new[1:P-1, 1:M-1, 0:N-2]) \
                    / dx**2 \
            + (U_new[2:P, 1:M-1, 1:N-1] - 2*U_new[1:P-1, 1:M-1, 1:N-1] \
                + U_new[0:P-2, 1:M-1, 1:N-1]) \
                    / dz**2)
                    
        # Thomas algorithm for y
        # from left to the right
        # U(t, x, y=0, z) = 0
        alpha[0:P, 1, 0:N] = 0
        beta[0:P, 1, 0:N] = 0    
        
        for j in range(1, M-1):
            alpha[1:P-1, j+1, 1:N-1] = -A[1:P-1, j, 1:N-1] \
                / (B[1:P-1, j, 1:N-1] + C[1:P-1, j, 1:N-1]*alpha[1:P-1, j, 1:N-1])
            
            beta[1:P-1, j+1, 1:N-1] = (D[1:P-1, j, 1:N-1] \
                    - C[1:P-1, j, 1:N-1]*beta[1:P-1, j, 1:N-1]) \
                / (B[1:P-1, j, 1:N-1] + C[1:P-1, j, 1:N-1]*alpha[1:P-1, j, 1:N-1])

        # U^(n+2/3)
        # U(t, x, y=1, z) = 0
        U_new[0:P, M-1, 0:N] = 0
        for j in range(M-2, -1, -1):
            U_new[1:P-1, j, 1:N-1] = alpha[1:P-1, j+1, 1:N-1]*U_new[1:P-1, j+1, 1:N-1] \
                + beta[1:P-1, j+1, 1:N-1]
        
        # Finding U^(n+1)
        A[0:P, 0:M, 0:N] = -a2 / dz**2
        B[0:P, 0:M, 0:N] = 1 / dt + 2 * a2 / dz**2
        C[0:P, 0:M, 0:N] = -a2 / dz**2
        
        # U_new is U^(n+2/3) now
        D[1:P-1, 1:M-1, 1:N-1] = U_new[1:P-1, 1:M-1, 1:N-1] / dt \
            + a2*(
            (U_new[1:P-1, 1:M-1, 2:N] - 2*U_new[1:P-1, 1:M-1, 1:N-1] \
                + U_new[1:P-1, 1:M-1, 0:N-2]) \
                    / dx**2 \
            + (U_new[1:P-1, 2:M, 1:N-1] - 2*U_new[1:P-1, 1:M-1, 1:N-1] \
                + U_new[1:P-1, 0:M-2, 1:N-1]) \
                    / dy**2)
    
        # Thomas algorithm for z
        # from bottom to the right
        # U(t, x, y, z=0) = 0
        alpha[1, 0:M, 0:N] = 0
        beta[1, 0:M, 0:N] = 0
        # U(t, 1/3<x<2/3, 0<y<1/3, z=0) = 1
        alpha[1, 0:M1, N1:N2] = 0
        beta[1, 0:M1, N1:N2] = 1
        
        for k in range(1, P-1):
            alpha[k+1, 1:M-1, 1:N-1] = -A[k, 1:M-1, 1:N-1] \
                / (B[k, 1:M-1, 1:N-1] + C[k, 1:M-1, 1:N-1]*alpha[k, 1:M-1, 1:N-1])
            beta[k+1, 1:M-1, 1:N-1] = (D[k, 1:M-1, 1:N-1] \
                    - C[k, 1:M-1, 1:N-1]*beta[k, 1:M-1, 1:N-1]) \
                / (B[k, 1:M-1, 1:N-1] + C[k, 1:M-1, 1:N-1]*alpha[k, 1:M-1, 1:N-1])

        # U^(n+1)
        # U(t, x, y, z=1) = 0
        U_new[P-1, 0:M, 0:N] = 0
        for k in range(P-2, -1, -1):
            U_new[k, 1:M-1, 1:N-1] = alpha[k+1, 1:M-1, 1:N-1]*U_new[k+1, 1:M-1, 1:N-1] \
                + beta[k+1, 1:M-1, 1:N-1]

        maximum = np.max(np.abs(U_new - U_old))        
        # print("Iteration", iteration, "\t", "maximum", maximum)
        U_old = U_new.copy()
        iteration += 1

    print("Number of iterations:", iteration)
    print("Maximum absolute difference:", maximum)

    return U_new

@timeit
@jit(nopython=True)
def Numba_Alternating_direction_method(U:np.ndarray, N, M, P, dx, dy, dz, dt, 
                                    a2=1, eps=1e-6, stop_iteration=3e4):
    """Boosted Alternating direction method for solving 3D Heat equation"""
    # U[z, y, x] or U[k, j, i]
    U_old = U.copy()
    U_new = np.zeros_like(U)

    N1 = int(1/3 * N)
    N2 = int(2/3 * N)
    M1 = int(1/3 * N)
    M2 = int(2/3 * N)
    P1 = int(2/3 * N)

    A = np.zeros((P, M, N))
    B = np.zeros((P, M, N))
    C = np.zeros((P, M, N))
    D = np.zeros((P, M, N))

    alpha = np.zeros((P, M, N))
    beta = np.zeros((P, M, N))

    iteration = 0
    maximum = 1
    while maximum > eps and iteration < stop_iteration:
        # Finding U^(n+1/3)
        A[0:P, 0:M, 0:N] = -a2 / dx**2
        B[0:P, 0:M, 0:N] = 1 / dt + 2 * a2 / dx**2
        C[0:P, 0:M, 0:N] = -a2 / dx**2
        
        D[1:P-1, 1:M-1, 1:N-1] = U_old[1:P-1, 1:M-1, 1:N-1] / dt \
            + a2*(
            (U_old[1:P-1, 2:M, 1:N-1] - 2*U_old[1:P-1, 1:M-1, 1:N-1] \
                + U_old[1:P-1, 0:M-2, 1:N-1]) \
                    / dy**2 \
            + (U_old[2:P, 1:M-1, 1:N-1] - 2*U_old[1:P-1, 1:M-1, 1:N-1] \
                + U_old[0:P-2, 1:M-1, 1:N-1]) \
                    / dz**2)
            
        # Thomas algorithm for x
        # from back to the front
        # U(t, x=0, y, z) = 0
        alpha[0:P, 0:M, 1] = 0
        beta[0:P, 0:M, 1] = 0
        # U(t, x=0, 1/3<y<2/3, 2/3<z<1) = 1
        alpha[P1:P, M1:M2, 1] = 0
        beta[P1:P, M1:M2, 1] = 1

        # alpha[0:P, 0:M, 1] = 0
        # beta[0:P, 0:M, 1] = U_old[0:P, 0:M, 0]
        
        for i in range(1, N-1):
            alpha[1:P-1, 1:M-1, i+1] = -A[1:P-1, 1:M-1, i] \
                / (B[1:P-1, 1:M-1, i] + C[1:P-1, 1:M-1, i]*alpha[1:P-1, 1:M-1, i])
            
            beta[1:P-1, 1:M-1, i+1] = (D[1:P-1, 1:M-1, i] \
                    - C[1:P-1, 1:M-1, i]*beta[1:P-1, 1:M-1, i]) \
                / (B[1:P-1, 1:M-1, i] + C[1:P-1, 1:M-1, i]*alpha[1:P-1, 1:M-1, i])
            
        # U^(n+1/3)
        # U(t, x=1, y, z) = 0
        U_new[0:P, 0:M, N-1] = 0
        for i in range(N-2, -1, -1):
            U_new[1:P-1, 1:M-1, i] = alpha[1:P-1, 1:M-1, i+1]*U_new[1:P-1, 1:M-1, i+1] \
                + beta[1:P-1, 1:M-1, i+1]
        
        # Finding U^(n+2/3)
        A[0:P, 0:M, 0:N] = -a2 / dy**2
        B[0:P, 0:M, 0:N] = 1 / dt + 2 * a2 / dy**2
        C[0:P, 0:M, 0:N] = -a2 / dy**2
        
        # U_new is U^(n+1/3) now
        D[1:P-1, 1:M-1, 1:N-1] = U_new[1:P-1, 1:M-1, 1:N-1] / dt \
            + a2*(
            (U_new[1:P-1, 1:M-1, 2:N] - 2*U_new[1:P-1, 1:M-1, 1:N-1] \
                + U_new[1:P-1, 1:M-1, 0:N-2]) \
                    / dx**2 \
            + (U_new[2:P, 1:M-1, 1:N-1] - 2*U_new[1:P-1, 1:M-1, 1:N-1] \
                + U_new[0:P-2, 1:M-1, 1:N-1]) \
                    / dz**2)
                    
        # Thomas algorithm for y
        # from left to the right
        # U(t, x, y=0, z) = 0
        alpha[0:P, 1, 0:N] = 0
        beta[0:P, 1, 0:N] = 0    
        
        for j in range(1, M-1):
            alpha[1:P-1, j+1, 1:N-1] = -A[1:P-1, j, 1:N-1] \
                / (B[1:P-1, j, 1:N-1] + C[1:P-1, j, 1:N-1]*alpha[1:P-1, j, 1:N-1])
            
            beta[1:P-1, j+1, 1:N-1] = (D[1:P-1, j, 1:N-1] \
                    - C[1:P-1, j, 1:N-1]*beta[1:P-1, j, 1:N-1]) \
                / (B[1:P-1, j, 1:N-1] + C[1:P-1, j, 1:N-1]*alpha[1:P-1, j, 1:N-1])

        # U^(n+2/3)
        # U(t, x, y=1, z) = 0
        U_new[0:P, M-1, 0:N] = 0
        for j in range(M-2, -1, -1):
            U_new[1:P-1, j, 1:N-1] = alpha[1:P-1, j+1, 1:N-1]*U_new[1:P-1, j+1, 1:N-1] \
                + beta[1:P-1, j+1, 1:N-1]
        
        # Finding U^(n+1)
        A[0:P, 0:M, 0:N] = -a2 / dz**2
        B[0:P, 0:M, 0:N] = 1 / dt + 2 * a2 / dz**2
        C[0:P, 0:M, 0:N] = -a2 / dz**2
        
        # U_new is U^(n+2/3) now
        D[1:P-1, 1:M-1, 1:N-1] = U_new[1:P-1, 1:M-1, 1:N-1] / dt \
            + a2*(
            (U_new[1:P-1, 1:M-1, 2:N] - 2*U_new[1:P-1, 1:M-1, 1:N-1] \
                + U_new[1:P-1, 1:M-1, 0:N-2]) \
                    / dx**2 \
            + (U_new[1:P-1, 2:M, 1:N-1] - 2*U_new[1:P-1, 1:M-1, 1:N-1] \
                + U_new[1:P-1, 0:M-2, 1:N-1]) \
                    / dy**2)
    
        # Thomas algorithm for z
        # from bottom to the right
        # U(t, x, y, z=0) = 0
        alpha[1, 0:M, 0:N] = 0
        beta[1, 0:M, 0:N] = 0
        # U(t, 1/3<x<2/3, 0<y<1/3, z=0) = 1
        alpha[1, 0:M1, N1:N2] = 0
        beta[1, 0:M1, N1:N2] = 1
        
        for k in range(1, P-1):
            alpha[k+1, 1:M-1, 1:N-1] = -A[k, 1:M-1, 1:N-1] \
                / (B[k, 1:M-1, 1:N-1] + C[k, 1:M-1, 1:N-1]*alpha[k, 1:M-1, 1:N-1])
            beta[k+1, 1:M-1, 1:N-1] = (D[k, 1:M-1, 1:N-1] \
                    - C[k, 1:M-1, 1:N-1]*beta[k, 1:M-1, 1:N-1]) \
                / (B[k, 1:M-1, 1:N-1] + C[k, 1:M-1, 1:N-1]*alpha[k, 1:M-1, 1:N-1])

        # U^(n+1)
        # U(t, x, y, z=1) = 0
        U_new[P-1, 0:M, 0:N] = 0
        for k in range(P-2, -1, -1):
            U_new[k, 1:M-1, 1:N-1] = alpha[k+1, 1:M-1, 1:N-1]*U_new[k+1, 1:M-1, 1:N-1] \
                + beta[k+1, 1:M-1, 1:N-1]

        maximum = np.max(np.abs(U_new - U_old))        
        # print("Iteration", iteration, "\t", "maximum", maximum)
        U_old = U_new.copy()
        iteration += 1

    print("Number of iterations:", iteration)
    print("Maximum absolute difference:", maximum)

    return U_new


start_x, end_x = (0, 1)
start_y, end_y = (0, 1)
start_z, end_z = (0, 1)

dt = 0.0001
a2 = 1

N = 41
M = 41
P = 41
dx = (end_x - start_x) / (N - 1)
dy = (end_y - start_y) / (M - 1)
dz = (end_z - start_z) / (P - 1)

N1 = int(1/3 * N)
N2 = int(2/3 * N)
M1 = int(1/3 * N)
M2 = int(2/3 * N)
P1 = int(2/3 * N)

x = start_x + np.arange(start=0, stop=N) * dx
y = start_y + np.arange(start=0, stop=M) * dy
z = start_z + np.arange(start=0, stop=P) * dz

# U[z, y, x] or U[k, j, i]
U_old = np.zeros((P, M, N))
U_new = np.zeros((P, M, N))

# Initial condition
# U(t=0, x, y, z) = 0
U_old[0:P, 0:M, 0:N] = 0

U_1 = Alternating_direction_method(U_old, N, M, P, dx, dy, dz, dt, a2=a2)

U_1N = Numba_Alternating_direction_method(U_old, N, M, P, dx, dy, dz, dt, a2=a2)
U_1N = Numba_Alternating_direction_method(U_old, N, M, P, dx, dy, dz, dt, a2=a2)
U_1N = Numba_Alternating_direction_method(U_old, N, M, P, dx, dy, dz, dt, a2=a2)

print(f"{N = }\n{M = }\n{P = }")
print(f"{dx = }\n{dy = }\n{dz = }\n{dt = }")
print(f"{a2 = }")

path = "Results"
if not os.path.exists(path):
    os.makedirs(path)

with open(f"{path}\\HW9_py.dat", "w") as file:
    file.write(f"VARIABLES = \"X\", \"Y\", \"Z\", \"U\"\n")
    file.write(f"ZONE I = {N}, J = {M}, K = {P}\n")
    
    for k in range(N):
        for j in range(N):
            for i in range(N):
                file.write(f"{x[i]}\t{y[j]}\t{z[k]}\t{U_1[k, j, i]}\n")
    
    print("Results are recorded")
