import numpy as np
import matplotlib.pyplot as plt


lang = ["py", "cpp"][0]
print(lang)
path = "Results"
x = np.loadtxt(f"{path}\\HW4_X_{lang}.txt", delimiter="\t", dtype=float)
U = np.loadtxt(f"{path}\\HW4_U_{lang}.txt", delimiter="\t", dtype=float)
F = np.loadtxt(f"{path}\\HW4_F_{lang}.txt", delimiter="\t", dtype=float)

print(f"Maximum difference: {np.max(np.abs(F - U)):.9f}")

plt.title("1D Poisson equation")
plt.grid()
plt.plot(x, U, ls="--", label="Five diagonal matrix method")
plt.plot(x, F, label="Analytical solution")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.legend()
plt.tight_layout()
plt.show()
