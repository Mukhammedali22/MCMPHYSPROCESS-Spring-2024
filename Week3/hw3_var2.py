import numpy as np
from numpy import (pi, exp, sin, cos)
import matplotlib.pyplot as plt
from time import perf_counter
import os


# 1D Heat conductivity equation
# Homework 3, Problem 2

# Initial condition
# U(t=0, x) = 1 - x^3

# Boundary conditions
# Ux(t, x=0) = 0
# U(t, x=1) = 0

def Analytical_solution(x, t, m=100):
    λ = lambda n: (2*n - 1)*pi/2
    A_n = lambda n: -12*((-1)**n + 1/λ(n)) / λ(n)**3
    
    F = 0
    for n in range(1, m+1):
        F += A_n(n) * exp(-λ(n)**2*t) * cos(λ(n)*x)
    
    return F

def Simple_iterative_method(U:np.ndarray, N, dt, dx, a2=1, 
                                eps=1e-6, stop_iteration=3e4):
    U_old = U.copy()
    U_new = np.zeros_like(U)

    set_boundary_U(U=U_old)

    iteration = 0
    maximum = 1
    while maximum > eps and iteration < stop_iteration:
        U_new[1:N-1] = U_old[1:N-1] + a2 * dt / dx**2 \
            * (U_old[2:N] - 2*U_old[1:N-1] + U_old[0:N-2])
        
        set_boundary_U(U=U_new)
        
        maximum = np.max(np.abs(U_new - U_old))
        U_old = U_new.copy()
        iteration += 1

    return U_new, iteration

def set_boundary_U(U):
    U[0] = U[1]
    U[N-1] = 0

def Thomas_algorithm(U:np.ndarray, N, dt, dx, a2=1, 
                        eps=1e-6, stop_iteration=3e4):
    U_old = U.copy()
    U_new = np.zeros_like(U)

    alpha = np.zeros_like(U)
    beta = np.zeros_like(U)

    A = -a2 / dx**2
    B = 1 / dt + 2*a2 / dx**2
    C = -a2 / dx**2
    D = U_old / dt # numpy 1d array

    iteration = 0
    maximum = 1
    while maximum > eps and iteration < stop_iteration:
        D = U_old / dt
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

    return U_new, iteration

dx = 0.1
dt = 0.001

start_x, end_x = (0, 1)
N = int((end_x - start_x) / dx) + 1
x = start_x + np.arange(start=0, stop=N) * dx

a2 = 1
eps = 1e-6
stop_iteration = 3e4

U_old = np.zeros_like(x)

# Initial condition
U_old[:] = 1 - x**3

start_time = perf_counter()
U_simple, iter_S = Simple_iterative_method(U_old, N, dt, dx, a2=1)
F_S = Analytical_solution(x, t=iter_S*dt)
end_time = perf_counter()

print("Simple Iterative Method")
print(f"Calculation time: {end_time - start_time:.6f} seconds")
print(f"Maximum error: {np.max(np.abs(F_S - U_simple)):e}")
print(f"Number of iterations: {iter_S}")

start_time = perf_counter()
U_Thomas, iter_T = Thomas_algorithm(U_old, N, dt, dx, a2=1)
F_T = Analytical_solution(x, t=iter_T*dt)
end_time = perf_counter()

print("Thomas algorithm")
print(f"Calculation time: {end_time - start_time:.6f} seconds")
print(f"Maximum error: {np.max(np.abs(F_T - U_simple)):e}")
print(f"Number of iterations: {iter_T}")

path = "Results"
# Create folder where we will store solution
if not os.path.exists(path):
    os.makedirs(path)

np.savetxt(f"{path}\\HW3_X_py.txt", x, fmt="%.6e", delimiter="\t")
np.savetxt(f"{path}\\HW3_U_Simple_py.txt", U_simple, fmt="%.6e", delimiter="\t")
np.savetxt(f"{path}\\HW3_F_Simple_py.txt", F_S, fmt="%.6e", delimiter="\t")
np.savetxt(f"{path}\\HW3_U_Thomas_py.txt", U_simple, fmt="%.6e", delimiter="\t")
np.savetxt(f"{path}\\HW3_F_Thomas_py.txt", F_T, fmt="%.6e", delimiter="\t")

plt.title("1D Heat conductivity equation")
plt.grid()
plt.plot(x, U_simple, label="Simple iterative method")
plt.plot(x, F_S, ls="--", label=f"Analytical solution at {dt*iter_S:.3f}")
plt.plot(x, U_Thomas, label="Thomas algorithm")
plt.plot(x, F_T, ls="--", label=f"Analytical solution at {dt*iter_T:.3f}")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.legend()
plt.tight_layout()
plt.show()
