import numpy as np
from numpy import (pi, exp, sin, cos)
import matplotlib.pyplot as plt
from matplotlib import animation
from time import perf_counter
import os


# 1D Transport equation
# The first scheme against the flow method
# Problem 2

# Initial condition
# U(t=0, x) = cos(pi*x/2)

# Boundary conditions
# U(t, x=0) = 1
# U(t, x=1) = 0

# c = -1
# If c is negative use forward scheme else backward scheme

def set_boundary_U(U:np.ndarray):
    U[0] = 1
    U[N-1] = 0

c = [-1, 1][1]
dx = 0.01
dt = dx / abs(c)
# dt = 0.001

eps = 1e-6
stop_iteration = 10000

start_x, end_x = (0, pi)
N = int((end_x - start_x) / dx) + 1
x = start_x + np.arange(start=0, stop=N) * dx

U_old = np.zeros(N)
U_new = np.zeros(N)
U_gif = []
k = 3 # saving each kth frame to animate

# Initial condition
U_old[0:N] = cos(pi*x/2)
# Boundary conditions
set_boundary_U(U=U_old)

start_time = perf_counter()

iteration = 0
maximum = 1
while maximum > eps and iteration < stop_iteration:
    if(c > 0):
        U_new[1:N-1] = U_old[1:N-1] - c * dt / dx * (U_old[1:N-1] - U_old[0:N-2])
    else:
        U_new[1:N-1] = U_old[1:N-1] - c * dt / dx * (U_old[2:N] - U_old[1:N-1])

    set_boundary_U(U=U_new)

    if iteration % k == 0:
        U_gif.append(U_new.copy())

    maximum = np.max(np.abs(U_new - U_old))
    print(f"{iteration}\t{maximum:.9f}")
    U_old = U_new.copy()
    iteration += 1

end_time = perf_counter()

path = "Results"
if not os.path.exists(path):
    os.makedirs(path)

np.savetxt(f"{path}\\HW5_X_py.txt", x, fmt="%.6f", delimiter="\t")
np.savetxt(f"{path}\\HW5_U_py.txt", U_new, fmt="%.6f", delimiter="\t")

print(f"Calculation time: {end_time - start_time:.9f} seconds")
print(f"Maximum difference: {np.max(np.abs(U_new - U_old))}")
print(f"{N = }\n{dx = }\n{dt = }\n{c = }")

fig, ax = plt.subplots()
def update(frame):
    ax.cla()
    ax.plot(x, U_gif[frame])
    ax.grid()
    ax.set_title(f"U(t={frame*k*dt:.3f}, x), {c=}")
    ax.set_xlabel("x")
    ax.set_ylabel("U(t, x)")

scheme = "backward" if c > 0 else "forward"
ani = animation.FuncAnimation(fig=fig, func=update, frames=len(U_gif), interval=1)
ani.save(filename=f"HW5_2_{scheme}.gif", writer="pillow")
plt.show()

plt.title(f"1D Transport equation t={iteration*dt:.2f} s")
plt.grid()
plt.plot(x, U_new, label="The first scheme againts the flow")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.legend()
plt.show()
