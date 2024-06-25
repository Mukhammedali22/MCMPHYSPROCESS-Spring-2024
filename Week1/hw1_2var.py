import numpy as np
import matplotlib.pyplot as plt
from numpy import (pi, sin, cos)
from time import perf_counter

# Thomas algorithm
# Variant 2
# Boundary conditions
# P(x=0) = 1
# P(x=1) = 0

dx = 0.1
start_x, end_x = (0, 1)
N = int((end_x - start_x) / dx) + 1
x = np.arange(start=0, stop=N) * dx

f = sin(x)

# Analytical solution
F = sin(x) - (sin(1) + 1)*x + 1

A = 1 / dx**2
B = -2 / dx**2
C = 1 / dx**2
D = -f # numpy array

alpha = np.zeros_like(x)
beta = np.zeros_like(x)
P = np.zeros_like(x)

start_time = perf_counter()

alpha[1] = 0
beta[1] = 1
for i in range(1, N-1):
    alpha[i+1] = -A / (B + C*alpha[i])
    beta[i+1] = (D[i] - C*beta[i]) / (B + C*alpha[i])

P[N-1] = 0
for i in range(N-2, -1, -1):
    P[i] = alpha[i+1]*P[i+1] + beta[i+1]

end_time = perf_counter()

print(f"Calculation time: {end_time - start_time:.6f} seconds")
print(f"Maximum difference: {np.max(np.abs(F - P))}")

plt.title("Possion equation using tridiagonal matrix method")
plt.grid()
plt.plot(x, P, label="Numerical solution")
plt.plot(x, F, label="Analytical solution")
plt.legend()
plt.show()
