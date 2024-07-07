import numpy as np
from numpy import (pi, exp, sin, cos)
from numpy.linalg import inv
import matplotlib.pyplot as plt
from time import perf_counter
from functools import wraps
from numba import jit
import os


# 2D Laplace equation
# Tridiagonal matrix method
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
        print(f"Calculation time: {(end_time - start_time):.3f} seconds")
        return result
    
    return timeit_wrapper

@timeit
def Tridiagonal_Matrix_method(P:np.ndarray, N, dx, dy):
    """Tridiagonal matrix method for solving 2D Laplace equation"""
    # P[y, x] or P[j, i]
    # 2D Thomas algotihm by x
    P_new = P.copy()

    A = np.zeros((N-2, N-2))
    B = np.zeros((N-2, N-2))
    C = np.zeros((N-2, N-2))
    D = np.zeros((N-2, N))

    np.fill_diagonal(A, 1 / dx**2)
    np.fill_diagonal(C, 1 / dx**2)
    np.fill_diagonal(B, -2 / dx**2 - 2 / dy**2)
    np.fill_diagonal(B[0:, 1:], 1 / dy**2)
    np.fill_diagonal(B[1:, 0:], 1 / dy**2)

    # Vector of matrices
    alpha = np.zeros((N, N-2, N-2))
    # Vector of vectors
    beta = np.zeros((N-2, N))

    np.fill_diagonal(alpha[1], 0)
    beta[0:M2, 1] = 0
    beta[M2:N, 1] = 1

    for i in range(1, N-1):
        denom = inv(B + C @ alpha[i])
        alpha[i+1] = -denom @ A
        beta[:, i+1] = denom @ (D[:, i] - C @ beta[:, i])

    P_new[0:M1, N-1] = 1
    P_new[M1:N, N-1] = 0
    for i in range(N-2, -1, -1):
        P_new[1:N-1, i] = alpha[i+1] @ P_new[1:N-1, i+1] + beta[:, i+1]

    return P_new

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

# N and M must be the same because we will use matrix operations
N = 101
M = N
dx = (end_x - start_x) / (N - 1)
dy = (end_y - start_y) / (M - 1)

M1 = int(0.3 * M)
M2 = int(0.7 * M)

x = start_x + np.arange(start=0, stop=N) * dx
y = start_y + np.arange(start=0, stop=M) * dy
X, Y = np.meshgrid(x, y)

P_old = np.zeros((M, N))
P_new = np.zeros((M, N))

P_new = Tridiagonal_Matrix_method(P_old, N, dx, dy)

path = "Results"
if not os.path.exists(path):
    os.makedirs(path)

np.savetxt(f"{path}\\HW7_X_py.txt", X, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW7_Y_py.txt", Y, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW7_P_py.txt", P_new, fmt="%.6f", delimiter="\t")

plot_result(X, Y, P_new, name="Tridiagonal matrix method")
