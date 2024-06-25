import numpy as np
import matplotlib.pyplot as plt
from numpy import (exp, pi, sin, cos)
from time import perf_counter


# 1D Heat equation
# Thomas algorithm
# Variant 2

# Initial condition
# U(t=0, x) = 1 - x^3

# Boundary conditions
# Ux(t, x=0) = 0
# U(t, x=1) = 0

def Analytical_solution(x, t, m=100):
    λ = lambda n: (2*n - 1)*pi / 2
    A_n = lambda n: -12*((-1)**n + 1/λ(n)) / λ(n)**3
    
    F = 0
    for n in range(1, m+1):
        F += A_n(n) * exp(-λ(n)**2*t) * cos(λ(n)*x)
    
    return F

dx = 0.01
dt = 0.01
start_x, end_x = (0, 1)
N = int((end_x - start_x) / dx) + 1
x = start_x + np.arange(start=0, stop=N) * dx

eps = 1e-9
stop_iteration = 5e4

alpha = np.zeros_like(x)
beta = np.zeros_like(x)
U_old = np.zeros_like(x)
U_new = np.zeros_like(x)

# Initial condition
U_old[:] = 1 - x**3

A = -1 / dx**2
B = 1 / dt + 2 / dx**2
C = -1 / dx**2
D = U_old / dt # numpy array

start_time = perf_counter()

iteration = 0
maximum = 1
while maximum > eps and iteration < stop_iteration:
    D[:] = U_old / dt # numpy array
    
    alpha[1] = 1
    beta[1] = 0
    for i in range(1, N-1):
        alpha[i+1] = -A / (B + C*alpha[i])
        beta[i+1] = (D[i] - C*beta[i]) / (B + C*alpha[i])
    
    U_new[N-1] = 0
    for i in range(N-2, -1, -1):
        U_new[i] = alpha[i+1]*U_new[i+1] + beta[i+1]
    
    maximum = np.max(np.abs(U_new - U_old))
    
    U_old = U_new.copy()
    iteration += 1

F = Analytical_solution(x, dt*iteration)
end_time = perf_counter()

print(f"Calculation time: {end_time - start_time:.6f} seconds")
print(f"Maximum difference: {np.max(np.abs(F - U_new)):.9f}")
print(f"Number of iterations: {iteration}")

np.savetxt("HW2_X_py.txt", x, fmt="%.6e", delimiter="\t")
np.savetxt("HW2_U_py.txt", U_new, fmt="%.6e", delimiter="\t")
np.savetxt("HW2_F_py.txt", F, fmt="%.6e", delimiter="\t")

plt.title("Heat equation using tridiagonal matrix method")
plt.grid()
plt.plot(x, U_new, label="Numerical solution")
plt.plot(x, F, ls="--", label="Analytical solution")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.legend()
plt.tight_layout()
plt.show()
