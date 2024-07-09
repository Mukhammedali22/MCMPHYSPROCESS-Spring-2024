import numpy as np
import matplotlib.pyplot as plt


def plot_result(X, Y, U, lvl=7, name="Numerical method"):
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contourf(X, Y, U, levels=lvl)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


lang = ["py", "cpp"][0]
method = ["Fractional step", "Alternating direction"][0]
print(lang)
path = "Results"
X = np.loadtxt(f"{path}\\HW11_X_{lang}.txt", delimiter="\t", dtype=float)
Y = np.loadtxt(f"{path}\\HW11_Y_{lang}.txt", delimiter="\t", dtype=float)
U = np.loadtxt(f"{path}\\HW11_U_{lang}.txt", delimiter="\t", dtype=float)
V = np.loadtxt(f"{path}\\HW11_V_{lang}.txt", delimiter="\t", dtype=float)

plot_result(X, Y, (U**2 + V**2)**0.5, lvl=20, name=f"{method} method")
plot_result(X, Y, U, lvl=20, name=f"{method} method")
plot_result(X, Y, V, lvl=20, name=f"{method} method")
