import numpy as np
from numpy import (pi, exp, sin, cos)
import matplotlib.pyplot as plt
from time import perf_counter
import os


# 1D Poisson equation
# Problem 2

# Boundary conditions
# P(x=0) = 1
# P(x=1) = 0

# f(x) = sin(x)

def Analytical_solution(x):
    return sin(x) - (1 + sin(1))*x + 1

def Five_diagonal_matrix_method(P, f, dx, a1, b1, g1):
    N = len(P)
    P_new = np.zeros(N)
    alpha = np.zeros(N)
    beta = np.zeros(N)
    gamma = np.zeros(N)
    
    A = np.zeros(N)
    B = np.zeros(N)
    C = np.zeros(N)
    D = np.zeros(N)
    E = np.zeros(N)
    H = np.zeros(N)
    
    A[:] = -1 / (12 * dx**2)
    B[:] = 16 / (12 * dx**2)
    C[:] = -30 / (12 * dx**2)
    D[:] = 16 / (12 * dx**2)
    E[:] = -1 / (12 * dx**2)
    H[:] = -f
    
    alpha[1] = a1
    beta[1] = b1
    gamma[1] = g1

    alpha[2] = -(B[1] + D[1]*beta[1]) / (C[1] + D[1]*alpha[1])
    beta[2] = -A[1] / (C[1] + D[1]*alpha[1])
    gamma[2] = (H[1] - D[1]*gamma[1]) / (C[1] + D[1]*alpha[1])

    for i in range(2, N-1):
        alpha[i+1] = -(B[i] + D[i]*beta[i] + E[i]*alpha[i-1]*beta[i]) \
            / (C[i] + D[i]*alpha[i] + E[i]*alpha[i-1]*alpha[i] + E[i]*beta[i-1])
        beta[i+1] = -A[i] \
            / (C[i] + D[i]*alpha[i] + E[i]*alpha[i-1]*alpha[i] + E[i]*beta[i-1])
        gamma[i+1] = (H[i] - D[i]*gamma[i] - E[i]*alpha[i-1]*gamma[i] - E[i]*gamma[i-1]) \
            / (C[i] + D[i]*alpha[i] + E[i]*alpha[i-1]*alpha[i] + E[i]*beta[i-1])

    P_new[N-1] = 0
    P_new[N-2] = alpha[N-1]*P_new[N-1] + gamma[N-1]
    for i in range(N-3, -1, -1):
        P_new[i] = alpha[i+1]*P_new[i+1] + beta[i+1]*P_new[i+2] + gamma[i+1]
            
    return P_new
    

dx = 0.01

start_x, end_x = (0, 1)
N = int((end_x - start_x) / dx) + 1
x = start_x + np.arange(start=0, stop=N) * dx

P_old = np.zeros(N)

start_time = perf_counter()

f = sin(x)
P_new = Five_diagonal_matrix_method(P_old, f, dx, a1=0, b1=0, g1=1)
F = Analytical_solution(x)

end_time = perf_counter()

path = "Results"
if not os.path.exists(path):
    os.makedirs(path)

np.savetxt(f"{path}\\HW4_X_py.txt", x, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW4_U_py.txt", P_new, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW4_F_py.txt", F, fmt="%.6f", delimiter="\t")

print(f"Calculation time: {end_time - start_time:.9f} seconds")
print(f"Maximum error (Five points): {np.max(np.abs(F - P_new))}")

plt.title("1D Possion equation")
plt.grid()
plt.plot(x, P_new, label="Five-diagonal method")
plt.plot(x, F, label="Analytical solution")
plt.xlabel("x")
plt.ylabel("P(x)")
plt.legend()
plt.show()
