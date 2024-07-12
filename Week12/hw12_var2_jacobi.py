import numpy as np
from numpy import (pi, exp, sin, cos)
import matplotlib.pyplot as plt
from time import perf_counter
import os
import winsound


# 2D Navier-Stokes equation
# Explicit scheme + Jacobi
# Problem 2

# Initial condition
# U(t=0, x, y) = 0
# V(t=0, x, y) = 0
# P(t=0, x, y) = 0

# Boundary conditions
# U(t, x=0, y) = 0
# U(t, x=1, y) = 0
# Uy(t, 0<x<0.2, y=0) = 0
# U(t, 0.2<x<0.8, y=0) = 0
# Uy(t, 0.8<x<1, y=0) = 0
# U(t, 0<x<0.3, y=1) = 0
# U(t, 0.3<x<0.7, y=1) = 0
# U(t, 0.7<x<1, y=1) = 0

# V(t, x=0, y) = 0
# V(t, x=1, y) = 0
# Vy(t, 0<x<0.2, y=0) = 0
# V(t, 0.2<x<0.8, y=0) = 0
# Vy(t, 0.8<x<1, y=0) = 0
# V(t, 0<x<0.3, y=1) = 0
# V(t, 0.3<x<0.7, y=1) = -1
# V(t, 0.7<x<1, y=1) = 0

# Px(t, x=0, y) = 0
# Px(t, x=1, y) = 0
# P(t, 0<x<0.2, y=0) = 0
# Py(t, 0.2<x<0.8, y=0) = 0
# P(t, 0.8<x<1, y=0) = 0
# Py(t, 0<x<0.3, y=1) = 0
# P(t, 0.3<x<0.7, y=1) = 1
# Py(t, 0.7<x<1, y=1) = 0

# Returns numpy array
def Explicit_Burger(S:np.ndarray, U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, nu) -> np.ndarray:
    """Explicit Euler method for solving 2D Burger's equation"""
    # Reference
    U_old = U
    V_old = V
    S_old = S
    
    S_new = np.zeros_like(S)
    
    S_new[1:M-1, 1:N-1] = S_old[1:M-1, 1:N-1] + dt*(
        - U_old[1:M-1, 1:N-1]*(S_old[1:M-1, 2:N] - S_old[1:M-1, 0:N-2]) \
                                / (2*dx) \
        - V_old[1:M-1, 1:N-1]*(S_old[2:M, 1:N-1] - S_old[0:M-2, 1:N-1]) \
                                / (2*dy) \
        + nu*(
            (S_old[1:M-1, 2:N] - 2*S_old[1:M-1, 1:N-1] + S_old[1:M-1, 0:N-2]) \
                / dx**2 \
            + (S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
                / dy**2))
    
    return S_new

def Jacobi_method(P:np.ndarray, U:np.ndarray, V:np.ndarray, N, M, dx, dy, dt, rho) -> np.ndarray:
    """Jacobi method for solving 2D Poisson equation"""
    # Reference
    P_old = P
    U_new = U
    V_new = V

    P_new = np.zeros_like(P)

    P_new[1:M-1, 1:N-1] = (dy**2*(P_old[1:M-1, 2:N] + P_old[1:M-1, 0:N-2]) \
        + dx**2*(P_old[2:M, 1:N-1] + P_old[0:M-2, 1:N-1])) \
            / (2*(dx**2 + dy**2)) \
        - dx**2*dy**2*rho \
            / (2*dt*(dx**2 + dy**2)) \
        * ((U_new[1:M-1, 2:N] - U_new[1:M-1, 0:N-2]) \
            / (2*dx) \
        + (V_new[2:M, 1:N-1] - V_new[0:M-2, 1:N-1]) \
            / (2*dy))
    
    return P_new

# Void function
# Reference
def set_boundary_U(U:np.ndarray):
    U[0:M, 0] = 0
    U[0:M, N-1] = 0
    U[0, 0:N1] = U[1, 0:N1]
    U[0, N1:N4] = 0
    U[0, N4:N] = U[1, N4:N]
    U[M-1, 0:N2] = 0
    U[M-1, N2:N3] = 0
    U[M-1, N3:N] = 0

# Void function
# Reference
def set_boundary_V(V:np.ndarray):
    V[0:M, 0] = 0
    V[0:M, N-1] = 0
    V[0, 0:N1] = V[1, 0:N1]
    V[0, N1:N4] = 0
    V[0, N4:N] = V[1, N4:N]
    V[M-1, 0:N2] = 0
    V[M-1, N2:N3] = -1
    V[M-1, N3:N] = 0

# Void function
# Reference
def set_boundary_P(P:np.ndarray):
    P[0:M, 0] = P[0:M, 1]
    P[0:M, N-1] = P[0:M, N-2]
    P[0, 0:N1] = 0
    P[0, N1:N4] = P[1, N1:N4]
    P[0, N4:N] = 0
    P[M-1, 0:N2] = P[M-2, 0:N2]
    P[M-1, N2:N3] = 1
    P[M-1, N3:N] = P[M-2, N3:N]


start_x, end_x = (0, 1)
start_y, end_y = (0, 1)

N = 101
M = 101
dx = (end_x - start_x) / (N - 1)
dy = (end_y - start_y) / (M - 1)

dt = 0.0001
Re = 100
rho = 1
eps = 1e-6
stop_iteration = 5e4
eps_P = 1e-6
stop_iteration_P = 1e5

N1 = int(0.2 * N)
N2 = int(0.3 * N)
N3 = int(0.7 * N)
N4 = int(0.8 * N)

x = start_x + np.arange(start=0, stop=N) * dx
y = start_y + np.arange(start=0, stop=M) * dy
X, Y = np.meshgrid(x, y)

U_old = np.zeros((M, N))
U_new = np.zeros((M, N))

V_old = np.zeros((M, N))
V_new = np.zeros((M, N))

P_old = np.zeros((M, N))
P_new = np.zeros((M, N))

# Initial condition
# U(t=0, x, y) = 0
# V(t=0, x, y) = 0
# P(t=0, x, y) = 0
U_old[:, :] = 0
V_old[:, :] = 0
P_old[:, :] = 0
U_new[:, :] = 0
V_new[:, :] = 0
P_new[:, :] = 0

start_time = perf_counter()
# Boundary conditions
set_boundary_U(U=U_old)
set_boundary_V(V=V_old)

iteration = 0
maximum = 1
while maximum > eps and iteration < stop_iteration:
    # 1. Solve Burger's equation to find U*, V*
    U_new = Explicit_Burger(U_old, U_old, V_old, N, M, dx, dy, dt, 1/Re)
    V_new = Explicit_Burger(V_old, U_old, V_old, N, M, dx, dy, dt, 1/Re)
    set_boundary_U(U=U_new)
    set_boundary_V(V=V_new)

    # 2. Solve Poisson equation to find P
    set_boundary_P(P=P_old)
    set_boundary_P(P=P_new)

    iteration_P = 0
    maximum_P = 1
    while maximum_P > eps_P and iteration_P < stop_iteration_P:
        P_new = Jacobi_method(P_old, U_new, V_new, N, M, dx, dy, dt, rho)
        set_boundary_P(P=P_new)

        maximum_P = np.max(np.abs(P_new - P_old))

        P_old = P_new.copy()
        iteration_P += 1

    # Do not change
    # 3. Correction
    U_new[1:M-1, 1:N-1] = U_new[1:M-1, 1:N-1] - dt / rho*(
        P_new[1:M-1, 2:N] - P_new[1:M-1, 0:N-2]) \
            / (2*dx)  
    V_new[1:M-1, 1:N-1] = V_new[1:M-1, 1:N-1] - dt / rho*(
        P_new[2:M, 1:N-1] - P_new[0:M-2, 1:N-1]) \
            / (2*dy)

    set_boundary_U(U=U_new)
    set_boundary_V(V=V_new)

    max_U = np.max(np.abs(U_new - U_old))
    max_V = np.max(np.abs(V_new - V_old))
    maximum = max(max_U, max_V)

    U_old = U_new.copy()
    V_old = V_new.copy()

    if iteration % 20 == 0:
        print(f"{iteration}\t{maximum}\t{iteration_P}\t{maximum_P}")

    iteration += 1

end_time = perf_counter()

path = "Results"
if not os.path.exists(path):
    os.makedirs(path)

np.savetxt(f"{path}\\HW12_X_py.txt", X, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW12_Y_py.txt", Y, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW12_U_py.txt", U_new, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW12_V_py.txt", V_new, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW12_P_py.txt", P_new, fmt="%.6f", delimiter="\t")

with open("Results\\HW12.dat", "w") as file:
    file.write(f"VARIABLES = \"X\", \"Y\", \"U\", \"V\", \"P\"\n")
    file.write(f"ZONE I = {N}, J = {M}\n")
    
    for j in range(M):
        for i in range(N):
            file.write(f"{x[i]}\t{y[j]}\t{U_new[j, i]}\t" +
                       f"{V_new[j, i]}\t{P_new[j, i]}\n")
    
    print("Results are recorded")

winsound.Beep(750, 1000)

print("Results are recorded")
print("2D Navier-Stokes equation")
print("Explicit + Jacobi")
print(f"Calculation time: {end_time - start_time:.6f} seconds")
print(f"Number of iterations: {iteration}")
print(f"{N = }\n{M = }\n{dx = }\n{dy = }\n{dt = }")
print(f"{Re = }\n{rho = }")
print(f"{maximum = }\n{maximum_P = }")
print(f"U_absmax = {np.max(np.abs(U_new))}")
print(f"V_absmax = {np.max(np.abs(V_new))}")
print(f"P_absmax = {np.max(np.abs(P_new))}")

# Post Process
def draw_all(results, names, arrow=True):
    fig, ax = plt.subplots(2, 2)
    cnt = 0
    for row in range(2):
        for col in range(2):
            axs = ax[row, col]
            cf = axs.contourf(X, Y, results[cnt % 4])
            fig.colorbar(cf, ax=axs)
            axs.set_title(names[cnt % 4])
            if arrow:
                axs.streamplot(X, Y, U_new, V_new, color="black")
            axs.set_xlabel("x")
            axs.set_ylabel("y")
            cnt += 1
    plt.tight_layout()
    plt.show()
    
def draw_one(Z, name, arrow=False):
    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, Z)
    fig.colorbar(cf, ax=ax)
    if arrow:
        ax.streamplot(X, Y, U_new, V_new, color="black")
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


UV = (U_new**2 + V_new**2)**0.5
data = [U_new, V_new, P_new, UV]
name = ["U", "V", "P", "U + V"]
draw_all(data, name, arrow=True)
