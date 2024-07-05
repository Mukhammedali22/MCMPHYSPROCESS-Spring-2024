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
print(lang)
path = "Results"
X = np.loadtxt(f"{path}\\HW6_X_{lang}.txt", delimiter="\t", dtype=float)
Y = np.loadtxt(f"{path}\\HW6_Y_{lang}.txt", delimiter="\t", dtype=float)
U1 = np.loadtxt(f"{path}\\HW6_P1_{lang}.txt", delimiter="\t", dtype=float)
U2 = np.loadtxt(f"{path}\\HW6_P2_{lang}.txt", delimiter="\t", dtype=float)
U3 = np.loadtxt(f"{path}\\HW6_P3_{lang}.txt", delimiter="\t", dtype=float)

plot_result(X, Y, U1, name="Jacobi method")
plot_result(X, Y, U2, name="Gauss-Seidel method")
plot_result(X, Y, U3, name="Over relaxation method")
