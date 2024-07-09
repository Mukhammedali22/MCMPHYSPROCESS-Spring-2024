import numpy as np
from numpy import (pi, exp, sin, cos)
import matplotlib.pyplot as plt
from time import perf_counter
from functools import wraps
from numba import jit
import os


# 2D Burger's equation
# Fractional step method (FSM)
# Problem 1

# Initial condition
# U(t=0, x, y) = 0
# V(t=0, x, y) = 0

# Boundary conditions
# U(t, x=0, 0<y<0.4) = 0
# Ux(t, x=0, 0.4<y<0.7) = 0
# U(t, x=0, 0.7<y<1) = 0
# U(t, x=1, y) = 0
# U(t, 0<x<0.7, y=0) = 0
# Uy(t, 0.7<x<1, y=0) = 0
# U(t, 0<x<0.7, y=1) = 0
# U(t, 0.7<x<1, y=1) = 0

# V(t, x=0, 0<y<0.4) = 0
# Vx(t, x=0, 0.4<y<0.7) = 0
# V(t, x=0, 0.7<y<1) = 0
# V(t, x=1, y) = 0
# V(t, 0<x<0.7, y=0) = 0
# Vy(t, 0.7<x<1, y=0) = 0
# V(t, 0<x<0.7, y=1) = 0
# V(t, 0.7<x<1, y=1) = -1

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
def Fractional_step_method(U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, 
                           nu=1, eps=1e-6, stop_iteration=3e4):
    """Fractional step method for solving 2D Heat equation"""
    # U[y, x] or U[j, i]
    U_old = U.copy()
    U_new = np.zeros_like(U)

    V_old = V.copy()
    V_new = np.zeros_like(V)

    N1 = int(0.7 * N)
    M1 = int(0.4 * M)
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
        A[0:M, 0:N] = -nu / (2*dx**2) + U_old[0:M, 0:N] / (2*dx)
        B[0:M, 0:N] = 1 / dt + nu / (dx**2) - U_old[0:M, 0:N] / (2*dx)
        C[0:M, 0:N] = -nu / (2*dx**2)
        
        D[1:M-1, 1:N-1] = U_old[1:M-1, 1:N-1] / dt + 0.5*(
            nu*(U_old[1:M-1, 2:N] - 2*U_old[1:M-1, 1:N-1] + U_old[1:M-1, 0:N-2]) \
                / (dx**2) \
            - U_old[1:M-1, 1:N-1] * (U_old[1:M-1, 2:N] - U_old[1:M-1, 1:N-1]) \
                / dx) \
            + nu*(U_old[2:M, 1:N-1] - 2*U_old[1:M-1, 1:N-1] + U_old[0:M-2, 1:N-1]) \
                / (dy**2) \
            - V_old[1:M-1, 1:N-1] * (U_old[2:M, 1:N-1] - U_old[1:M-1, 1:N-1]) \
                / dy
        
        # Thomas algorithm for x
        # U(t, x=0, 0<y<0.4) = 0
        alpha[0:M1, 1] = 0
        beta[0:M1, 1] = 0
        # Ux(t, x=0, 0.4<y<0.7) = 0
        alpha[M1:M2, 1] = 1
        beta[M1:M2, 1] = 0
        # U(t, x=0, 0.7<y<1) = 0
        alpha[M2:M, 1] = 0
        beta[M2:M, 1] = 0
        
        for i in range(1, N-1):
            alpha[1:M-1, i+1] = -A[1:N-1, i] \
                / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
            beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:N-1, i]*beta[1:M-1, i]) \
                / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
            
        # U^(n+1/2)
        # U(t, x=1, y) = 0
        U_new[0:M, N-1] = U_old[0:M, N-1]
        for i in range(N-2, -1, -1):
            U_new[1:M-1, i] = alpha[1:M-1, i+1]*U_new[1:M-1, i+1] + beta[1:M-1, i+1]   
        
        # Finding U^(n+1)
        A[0:M, 0:N] = -nu / (2*dy**2) + V_old[0:M, 0:N] / (2*dy)
        B[0:M, 0:N] = 1 / dt + nu / dy**2 - V_old[0:M, 0:N] / (2*dy)
        C[0:M, 0:N] = -nu / (2*dy**2)
        
        D[1:M-1, 1:N-1] = U_new[1:M-1, 1:N-1] / dt - 0.5*(
            nu*(U_old[2:M, 1:N-1] - 2*U_old[1:M-1, 1:N-1] + U_old[0:M-2, 1:N-1]) \
                / (dy**2) \
            - V_old[1:M-1, 1:N-1] * (U_old[2:M, 1:N-1] - U_old[1:M-1, 1:N-1]) \
                / dy)
            
        # Thomas algorithm for y
        # U(t, 0<x<0.7, y=0) = 0
        alpha[1, 0:N1] = 0
        beta[1, 0:N1] = 0
        # Uy(t, 0.7<x<1, y=0) = 0
        alpha[1, N1:N] = 1
        beta[1, N1:N] = 0

        for j in range(1, M-1):
            alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            
        # U^(n+1)
        # U(t, 0<x<0.7, y=1) = 0
        # U(t, 0.7<x<1, y=1) = 0
        U_new[M-1, 0:N] = U_old[M-1, 0:N]
        for j in range(M-2, -1, -1):
            U_new[j, 1:N-1] = alpha[j+1, 1:N-1]*U_new[j+1, 1:N-1] + beta[j+1, 1:N-1]

        # -------------------------------------------------------------------- #
        
        # Finding V^(n+1/2)
        A[0:M, 0:N] = -nu / (2*dx**2) + U_old[0:M, 0:N] / (2*dx)
        B[0:M, 0:N] = 1 / dt + nu / (dx**2) - U_old[0:M, 0:N] / (2*dx)
        C[0:M, 0:N] = -nu / (2*dx**2)
        
        D[1:M-1, 1:N-1] = V_old[1:M-1, 1:N-1] / dt + 0.5*(
            nu*(V_old[1:M-1, 2:N] - 2*V_old[1:M-1, 1:N-1] + V_old[1:M-1, 0:N-2]) \
                / (dx**2) \
            - U_old[1:M-1, 1:N-1] * (V_old[1:M-1, 2:N] - V_old[1:M-1, 1:N-1]) \
                / dx) \
            + nu*(V_old[2:M, 1:N-1] - 2*V_old[1:M-1, 1:N-1] + V_old[0:M-2, 1:N-1]) \
                / (dy**2) \
            - V_old[1:M-1, 1:N-1] * (V_old[2:M, 1:N-1] - V_old[1:M-1, 1:N-1]) \
                / dy

        # Thomas algorithm for x
        # U(t, x=0, 0<y<0.4) = 0
        alpha[0:M1, 1] = 0
        beta[0:M1, 1] = 0
        # Ux(t, x=0, 0.4<y<0.7) = 0
        alpha[M1:M2, 1] = 1
        beta[M1:M2, 1] = 0
        # U(t, x=0, 0.7<y<1) = 0
        alpha[M2:M, 1] = 0
        beta[M2:M, 1] = 0
            
        for i in range(1, N-1):
            alpha[1:M-1, i+1] = -A[1:N-1, i] \
                / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
            beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:N-1, i]*beta[1:M-1, i]) \
                / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
            
        # V^(n+1/2)
        # V(t, x=1, y) = 0
        V_new[0:M, N-1] = V_old[0:M, N-1]
        for i in range(N-2, -1, -1):
            V_new[1:M-1, i] = alpha[1:M-1, i+1]*V_new[1:M-1, i+1] \
                                + beta[1:M-1, i+1]   

        # Finding V^(n+1)
        A[0:M, 0:N] = -nu / (2*dy**2) + V_old[0:M, 0:N] / (2*dy)
        B[0:M, 0:N] = 1 / dt + nu / (dy**2) - V_old[0:M, 0:N] / (2*dy)
        C[0:M, 0:N] = -nu / (2*dy**2)
        
        # V_new is V^(n+1/2) now
        D[1:M-1, 1:N-1] = V_new[1:M-1, 1:N-1] / dt - 0.5*(
            nu*(V_old[2:M, 1:N-1] - 2*V_old[1:M-1, 1:N-1] + V_old[0:M-2, 1:N-1]) \
                / (dy**2) \
            - V_old[1:M-1, 1:N-1] * (V_old[2:M, 1:N-1] - V_old[1:M-1, 1:N-1]) \
                / dy)

        # Thomas algorithm for y
        # U(t, 0<x<0.7, y=0) = 0
        alpha[1, 0:N1] = 0
        beta[1, 0:N1] = 0
        # Uy(t, 0.7<x<1, y=0) = 0
        alpha[1, N1:N] = 1
        beta[1, N1:N] = 0
        
        for j in range(1, M-1):
            alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            
        # V^(n+1)
        # U(t, 0<x<0.7, y=1) = 0
        # U(t, 0.7<x<1, y=1) = -1
        V_new[M-1, 0:N] = V_old[M-1, 0:N]
        for j in range(M-2, -1, -1):
            V_new[j, 1:N-1] = alpha[j+1, 1:N-1]*V_new[j+1, 1:N-1] \
                                + beta[j+1, 1:N-1] 
        
        maximum_U = np.max(np.abs(U_new - U_old))
        maximum_V = np.max(np.abs(V_new - V_old))

        maximum = max(maximum_U, maximum_V)
        # print("Iteration", iteration, "\t", "maximum", maximum)
        U_old = U_new.copy()
        V_old = V_new.copy()
        iteration += 1

    print("Number of iterations:", iteration)
    print("Maximum absolute difference:", maximum)

    return U_new, V_new

@timeit
@jit(nopython=True)
def Numba_Fractional_step_method(U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, 
                           nu=1, eps=1e-6, stop_iteration=3e4):
    """Boosted Fractional step method for solving 2D Heat equation"""
        # U[y, x] or U[j, i]
    U_old = U.copy()
    U_new = np.zeros_like(U)

    V_old = V.copy()
    V_new = np.zeros_like(V)

    N1 = int(0.7 * N)
    M1 = int(0.4 * M)
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
        A[0:M, 0:N] = -nu / (2*dx**2) + U_old[0:M, 0:N] / (2*dx)
        B[0:M, 0:N] = 1 / dt + nu / (dx**2) - U_old[0:M, 0:N] / (2*dx)
        C[0:M, 0:N] = -nu / (2*dx**2)
        
        D[1:M-1, 1:N-1] = U_old[1:M-1, 1:N-1] / dt + 0.5*(
            nu*(U_old[1:M-1, 2:N] - 2*U_old[1:M-1, 1:N-1] + U_old[1:M-1, 0:N-2]) \
                / (dx**2) \
            - U_old[1:M-1, 1:N-1] * (U_old[1:M-1, 2:N] - U_old[1:M-1, 1:N-1]) \
                / dx) \
            + nu*(U_old[2:M, 1:N-1] - 2*U_old[1:M-1, 1:N-1] + U_old[0:M-2, 1:N-1]) \
                / (dy**2) \
            - V_old[1:M-1, 1:N-1] * (U_old[2:M, 1:N-1] - U_old[1:M-1, 1:N-1]) \
                / dy
        
        # Thomas algorithm for x
        # U(t, x=0, 0<y<0.4) = 0
        alpha[0:M1, 1] = 0
        beta[0:M1, 1] = 0
        # Ux(t, x=0, 0.4<y<0.7) = 0
        alpha[M1:M2, 1] = 1
        beta[M1:M2, 1] = 0
        # U(t, x=0, 0.7<y<1) = 0
        alpha[M2:M, 1] = 0
        beta[M2:M, 1] = 0
        
        for i in range(1, N-1):
            alpha[1:M-1, i+1] = -A[1:N-1, i] \
                / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
            beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:N-1, i]*beta[1:M-1, i]) \
                / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
            
        # U^(n+1/2)
        # U(t, x=1, y) = 0
        U_new[0:M, N-1] = U_old[0:M, N-1]
        for i in range(N-2, -1, -1):
            U_new[1:M-1, i] = alpha[1:M-1, i+1]*U_new[1:M-1, i+1] + beta[1:M-1, i+1]   
        
        # Finding U^(n+1)
        A[0:M, 0:N] = -nu / (2*dy**2) + V_old[0:M, 0:N] / (2*dy)
        B[0:M, 0:N] = 1 / dt + nu / dy**2 - V_old[0:M, 0:N] / (2*dy)
        C[0:M, 0:N] = -nu / (2*dy**2)
        
        D[1:M-1, 1:N-1] = U_new[1:M-1, 1:N-1] / dt - 0.5*(
            nu*(U_old[2:M, 1:N-1] - 2*U_old[1:M-1, 1:N-1] + U_old[0:M-2, 1:N-1]) \
                / (dy**2) \
            - V_old[1:M-1, 1:N-1] * (U_old[2:M, 1:N-1] - U_old[1:M-1, 1:N-1]) \
                / dy)
            
        # Thomas algorithm for y
        # U(t, 0<x<0.7, y=0) = 0
        alpha[1, 0:N1] = 0
        beta[1, 0:N1] = 0
        # Uy(t, 0.7<x<1, y=0) = 0
        alpha[1, N1:N] = 1
        beta[1, N1:N] = 0

        for j in range(1, M-1):
            alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            
        # U^(n+1)
        # U(t, 0<x<0.7, y=1) = 0
        # U(t, 0.7<x<1, y=1) = 0
        U_new[M-1, 0:N] = U_old[M-1, 0:N]
        for j in range(M-2, -1, -1):
            U_new[j, 1:N-1] = alpha[j+1, 1:N-1]*U_new[j+1, 1:N-1] + beta[j+1, 1:N-1]

        # -------------------------------------------------------------------- #
        
        # Finding V^(n+1/2)
        A[0:M, 0:N] = -nu / (2*dx**2) + U_old[0:M, 0:N] / (2*dx)
        B[0:M, 0:N] = 1 / dt + nu / (dx**2) - U_old[0:M, 0:N] / (2*dx)
        C[0:M, 0:N] = -nu / (2*dx**2)
        
        D[1:M-1, 1:N-1] = V_old[1:M-1, 1:N-1] / dt + 0.5*(
            nu*(V_old[1:M-1, 2:N] - 2*V_old[1:M-1, 1:N-1] + V_old[1:M-1, 0:N-2]) \
                / (dx**2) \
            - U_old[1:M-1, 1:N-1] * (V_old[1:M-1, 2:N] - V_old[1:M-1, 1:N-1]) \
                / dx) \
            + nu*(V_old[2:M, 1:N-1] - 2*V_old[1:M-1, 1:N-1] + V_old[0:M-2, 1:N-1]) \
                / (dy**2) \
            - V_old[1:M-1, 1:N-1] * (V_old[2:M, 1:N-1] - V_old[1:M-1, 1:N-1]) \
                / dy

        # Thomas algorithm for x
        # U(t, x=0, 0<y<0.4) = 0
        alpha[0:M1, 1] = 0
        beta[0:M1, 1] = 0
        # Ux(t, x=0, 0.4<y<0.7) = 0
        alpha[M1:M2, 1] = 1
        beta[M1:M2, 1] = 0
        # U(t, x=0, 0.7<y<1) = 0
        alpha[M2:M, 1] = 0
        beta[M2:M, 1] = 0
            
        for i in range(1, N-1):
            alpha[1:M-1, i+1] = -A[1:N-1, i] \
                / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
            beta[1:M-1, i+1] = (D[1:M-1, i] - C[1:N-1, i]*beta[1:M-1, i]) \
                / (B[1:M-1, i] + C[1:N-1, i]*alpha[1:N-1, i])
            
        # V^(n+1/2)
        # V(t, x=1, y) = 0
        V_new[0:M, N-1] = V_old[0:M, N-1]
        for i in range(N-2, -1, -1):
            V_new[1:M-1, i] = alpha[1:M-1, i+1]*V_new[1:M-1, i+1] \
                                + beta[1:M-1, i+1]   

        # Finding V^(n+1)
        A[0:M, 0:N] = -nu / (2*dy**2) + V_old[0:M, 0:N] / (2*dy)
        B[0:M, 0:N] = 1 / dt + nu / (dy**2) - V_old[0:M, 0:N] / (2*dy)
        C[0:M, 0:N] = -nu / (2*dy**2)
        
        # V_new is V^(n+1/2) now
        D[1:M-1, 1:N-1] = V_new[1:M-1, 1:N-1] / dt - 0.5*(
            nu*(V_old[2:M, 1:N-1] - 2*V_old[1:M-1, 1:N-1] + V_old[0:M-2, 1:N-1]) \
                / (dy**2) \
            - V_old[1:M-1, 1:N-1] * (V_old[2:M, 1:N-1] - V_old[1:M-1, 1:N-1]) \
                / dy)

        # Thomas algorithm for y
        # U(t, 0<x<0.7, y=0) = 0
        alpha[1, 0:N1] = 0
        beta[1, 0:N1] = 0
        # Uy(t, 0.7<x<1, y=0) = 0
        alpha[1, N1:N] = 1
        beta[1, N1:N] = 0
        
        for j in range(1, M-1):
            alpha[j+1, 1:N-1] = -A[j, 1:N-1] \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            beta[j+1, 1:N-1] = (D[j, 1:N-1] - C[j, 1:N-1]*beta[j, 1:N-1]) \
                / (B[j, 1:N-1] + C[j, 1:N-1]*alpha[j, 1:N-1])
            
        # V^(n+1)
        # U(t, 0<x<0.7, y=1) = 0
        # U(t, 0.7<x<1, y=1) = -1
        V_new[M-1, 0:N] = V_old[M-1, 0:N]
        for j in range(M-2, -1, -1):
            V_new[j, 1:N-1] = alpha[j+1, 1:N-1]*V_new[j+1, 1:N-1] \
                                + beta[j+1, 1:N-1] 
        
        maximum_U = np.max(np.abs(U_new - U_old))
        maximum_V = np.max(np.abs(V_new - V_old))

        maximum = max(maximum_U, maximum_V)
        # print("Iteration", iteration, "\t", "maximum", maximum)
        U_old = U_new.copy()
        V_old = V_new.copy()
        iteration += 1

    print("Number of iterations:", iteration)
    print("Maximum absolute difference:", maximum)

    return U_new

def set_boundary_U(U:np.ndarray):
    U[0:M1, 0] = 0
    U[M1:M2, 0] = U[M1:M2, 1]
    U[M2:M, 0] = 0
    U[0:M, N-1] = 0
    U[0, 0:N1] = 0
    U[0, N1:N] = U[1, N1:N]
    U[M-1, 0:N1] = 0
    U[M-1, N1:N] = 0

def set_boundary_V(V:np.ndarray):
    V[0:M1, 0] = 0
    V[M1:M2, 0] = V[M1:M2, 1]
    V[M2:M, 0] = 0
    V[0:M, N-1] = 0
    V[0, 0:N1] = 0
    V[0, N1:N] = V[1, N1:N]
    V[M-1, 0:N1] = 0
    V[M-1, N1:N] = -1

def plot_result(X, Y, P, lvl=7, name="Numerical method"):
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contourf(X, Y, P, levels=lvl)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


start_x, end_x = (0, 1)
start_y, end_y = (0, 1)

dt = 0.01
Re = 40

N = 101
M = 101
dx = (end_x - start_x) / (N - 1)
dy = (end_y - start_y) / (M - 1)

N1 = int(0.7 * N)
M1 = int(0.4 * M)
M2 = int(0.7 * M)

x = start_x + np.arange(start=0, stop=N) * dx
y = start_y + np.arange(start=0, stop=M) * dy
X, Y = np.meshgrid(x, y)

U_old = np.zeros((M, N))
U_new = np.zeros((M, N))

V_old = np.zeros((M, N))
V_new = np.zeros((M, N))

set_boundary_U(U=U_old)
set_boundary_V(V=V_old)

U_1, V_1 = Fractional_step_method(U_old, V_old, N, M, dx, dy, dt, nu=1/Re)

U_1N = Numba_Fractional_step_method(U_old, V_old, N, M, dx, dy, dt, nu=1/Re)
U_1N = Numba_Fractional_step_method(U_old, V_old, N, M, dx, dy, dt, nu=1/Re)
U_1N = Numba_Fractional_step_method(U_old, V_old, N, M, dx, dy, dt, nu=1/Re)

path = "Results"
if not os.path.exists(path):
    os.makedirs(path)

np.savetxt(f"{path}\\HW11_X_py.txt", X, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW11_Y_py.txt", Y, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW11_U_py.txt", U_1, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW11_V_py.txt", V_1, fmt="%.6f", delimiter="\t")

# Saving data to the Tecplot program
with open(f"{path}\\HW11_py.dat", "w") as file:
    file.write(f"VARIABLES = \"X\", \"Y\", \"U\", \"V\"\n")
    file.write(f"ZONE I = {N}, J = {M}\n")
    
    for j in range(M):
        for i in range(N):
            file.write(f"{x[i]}\t{y[j]}\t{U_1[j, i]}\t{V_1[j, i]}\n")

    print("Results are recorded")

plot_result(X, Y, (U_1**2 + V_1**2)**0.5, lvl=20, name="Fractional step method")
plot_result(X, Y, U_1, lvl=20, name="Fractional step method")
plot_result(X, Y, V_1, lvl=20, name="Fractional step method")
