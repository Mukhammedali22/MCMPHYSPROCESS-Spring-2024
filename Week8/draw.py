import numpy as np
import matplotlib.pyplot as plt


def plot_result(X, Y, P, name="Numerical method"):
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contourf(X, Y, P)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


lang = ["py", "cpp"][0]
method = ["Fractional step", "Alternating direction"][0]
print(lang)
path = "Results"
X = np.loadtxt(f"{path}\\HW8_X_{lang}.txt", delimiter="\t", dtype=float)
Y = np.loadtxt(f"{path}\\HW8_Y_{lang}.txt", delimiter="\t", dtype=float)
U = np.loadtxt(f"{path}\\HW8_U_{lang}.txt", delimiter="\t", dtype=float)

plot_result(X, Y, U, name=f"{method} method")
