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
X = np.loadtxt(f"{path}\\HW7_X_{lang}.txt", delimiter="\t", dtype=float)
Y = np.loadtxt(f"{path}\\HW7_Y_{lang}.txt", delimiter="\t", dtype=float)
U = np.loadtxt(f"{path}\\HW7_P_{lang}.txt", delimiter="\t", dtype=float)

plot_result(X, Y, U, name="Tridiagonal matrix method")
