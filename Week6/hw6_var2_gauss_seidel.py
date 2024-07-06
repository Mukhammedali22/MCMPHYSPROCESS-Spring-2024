import numpy as np
from numpy import (pi, exp, sin, cos)
import matplotlib.pyplot as plt
from time import perf_counter
from functools import wraps
from numba import jit
import os


# 2D Laplace equation
# Gauss-Seidel method
# Problem 2

# Boundary conditions
# P(x=0, 0<y<0.7) = 0
# P(x=0, 0.7<y<1) = 1
# P(x=1, 0<y<0.3) = 1
# P(x=1, 0.3<y<1) = 0
# P(x, y=0) = 0
# P(x, y=1) = 0

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
def Gauss_Seidel_method(P0:np.ndarray, P1:np.ndarray, N, M, 
                        dx, dy, eps=1e-6, stop_iteration=3e4):
    """Gauss-Seidel method for solving 2D Laplace equation"""
    P_old = P0.copy()
    P_new = P1.copy()

    set_boundary_P(P=P_old)
    set_boundary_P(P=P_new)
    
    iteration = 0
    maximum = 1
    while maximum > eps and iteration < stop_iteration:
        for j in range(1, M-1):
            for i in range(1, N-1):
                P_new[j, i] = (
                    dy**2*(P_old[j, i+1] + P_new[j, i-1]) \
                    + dx**2*(P_old[j+1, i] + P_new[j-1, i])
                    ) / (2*(dx**2 + dy**2))
                
        set_boundary_P(P=P_old)
        set_boundary_P(P=P_new)

        maximum = np.max(np.abs(P_new - P_old))
        # print(f"{iteration = }\t{maximum = }")
        P_old = P_new.copy()
        iteration += 1

    print(f"Number of iterations: {iteration}")
    print(f"Maximum absolute difference: {maximum}")
    
    return P_new


@timeit
@jit(nopython=True)
def Numba_Gauss_Seidel_method(P0:np.ndarray, P1:np.ndarray, N, M, 
                        dx, dy, eps=1e-6, stop_iteration=3e4):
    """Boosted Gauss-Seidel method for solving 2D Laplace equation"""
    P_old = P0.copy()
    P_new = P1.copy()
    
    M1 = int(0.3 * M)
    M2 = int(0.7 * M)

    P_old[0:M2, 0] = 0
    P_old[M2:M, 0] = 1
    P_old[0:M1, N-1] = 1
    P_old[M1:M, N-1] = 0
    P_old[0, 0:N] = 0
    P_old[M-1, 0:N] = 0

    P_new[0:M2, 0] = 0
    P_new[M2:M, 0] = 1
    P_new[0:M1, N-1] = 1
    P_new[M1:M, N-1] = 0
    P_new[0, 0:N] = 0
    P_new[M-1, 0:N] = 0

    iteration = 0
    maximum = 1
    while maximum > eps and iteration < stop_iteration:
        for j in range(1, M-1):
            for i in range(1, N-1):
                P_new[j, i] = (
                    dy**2*(P_old[j, i+1] + P_new[j, i-1]) \
                    + dx**2*(P_old[j+1, i] + P_new[j-1, i])
                    ) / (2*(dx**2 + dy**2))
                
        P_old[0:M2, 0] = 0
        P_old[M2:M, 0] = 1
        P_old[0:M1, N-1] = 1
        P_old[M1:M, N-1] = 0
        P_old[0, 0:N] = 0
        P_old[M-1, 0:N] = 0

        P_new[0:M2, 0] = 0
        P_new[M2:M, 0] = 1
        P_new[0:M1, N-1] = 1
        P_new[M1:M, N-1] = 0
        P_new[0, 0:N] = 0
        P_new[M-1, 0:N] = 0

        maximum = np.max(np.abs(P_new - P_old))
        # print("Iteration", iteration, "\t", "maximum", maximum)
        P_old = P_new.copy()
        iteration += 1

    print("Number of iterations:", iteration)
    print("Maximum absolute difference:", maximum)
    
    return P_new


def set_boundary_P(P:np.ndarray):
    P[0:M2, 0] = 0
    P[M2:M, 0] = 1
    P[0:M1, N-1] = 1
    P[M1:M, N-1] = 0
    P[0, 0:N] = 0
    P[M-1, 0:N] = 0    

def plot_result(X, Y, P, name="Numerical method"):
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contourf(X, Y, P)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


dx = 0.01
dy = 0.01

start_x, end_x = (0, 1)
start_y, end_y = (0, 1)

N = int((end_x - start_x) / dx) + 1
M = int((end_y - start_y) / dy) + 1

M1 = int(0.3 * M)
M2 = int(0.7 * M)

x = start_x + np.arange(start=0, stop=N) * dx
y = start_y + np.arange(start=0, stop=M) * dy
X, Y = np.meshgrid(x, y)

P_old = np.zeros((M, N))
P_new = np.zeros((M, N))

P_2 = Gauss_Seidel_method(P_old, P_new, N, M, dx, dy)
# First run
P_2N = Numba_Gauss_Seidel_method(P_old, P_new, N, M, dx, dy)
# Other runs
P_2N = Numba_Gauss_Seidel_method(P_old, P_new, N, M, dx, dy)
P_2N = Numba_Gauss_Seidel_method(P_old, P_new, N, M, dx, dy)
P_2N = Numba_Gauss_Seidel_method(P_old, P_new, N, M, dx, dy)
P_2N = Numba_Gauss_Seidel_method(P_old, P_new, N, M, dx, dy)

print(np.max(np.abs(P_2 - P_2N)))

path = "Results"
if not os.path.exists(path):
    os.makedirs(path)

np.savetxt(f"{path}\\HW6_X_py.txt", X, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW6_Y_py.txt", Y, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW6_P2_py.txt", P_2N, fmt="%.6f", delimiter="\t")

plot_result(X, Y, P_2N, name="Gauss-Seidel method")
