import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def draw_all(arrow=False):
    fig, ax = plt.subplots(2, 2)

    cnt = 0
    for row in range(2):
        for col in range(2):
            axs = ax[row, col]
            cf = axs.contourf(X, Y, results[cnt % 4])
            fig.colorbar(cf, ax=axs)
            axs.set_title(names[cnt % 4])
            if arrow:
                axs.streamplot(X, Y, U, V, color="black")
                # axs.quiver(X[::1, ::1], Y[::1, ::1], U[::1, ::1], V[::1, ::1]) 

            axs.set_xlabel("x")
            axs.set_ylabel("y")
            cnt += 1

    end_time = perf_counter()
    print(f"Elapsed time: {end_time - start_time:.6f} seconds")
    plt.tight_layout()
    plt.show()
    
def draw_one(Z, name, arrow=False):
    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, Z)
    fig.colorbar(cf, ax=ax)
    print(np.min(Z))
    if arrow:
        ax.streamplot(X, Y, U, V, color="black")
        # ax.quiver(X[::1, ::1], Y[::1, ::1], U[::1, ::1], V[::1, ::1]) 

    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

start_time = perf_counter()

lang = ["cpp", "py"][1]
print(lang)
X = np.loadtxt(f"Results\\HW12_X_{lang}.txt")
Y = np.loadtxt(f"Results\\HW12_Y_{lang}.txt")
U = np.loadtxt(f"Results\\HW12_U_{lang}.txt")
V = np.loadtxt(f"Results\\HW12_V_{lang}.txt")
P = np.loadtxt(f"Results\\HW12_P_{lang}.txt")
UV = (U**2 + V**2)**0.5

results = [U, V, P, UV]
names = ["U", "V", "P", "U + V"]

draw_all(arrow=True)
# draw_one(UV, "U + V", arrow=True)
