import numpy as np
import matplotlib.pyplot as plt


lang = ["py", "cpp"][0]
print(lang)
x = np.loadtxt(f"HW2_X_{lang}.txt", delimiter="\t", dtype=float)
U = np.loadtxt(f"HW2_U_{lang}.txt", delimiter="\t", dtype=float)
F = np.loadtxt(f"HW2_F_{lang}.txt", delimiter="\t", dtype=float)

print(f"Maximum difference: {np.max(np.abs(F - U)):.9f}")

plt.title("Heat equation using tridiagonal matrix method")
plt.grid()
plt.plot(x, U, label="Numerical solution")
plt.plot(x, F, ls="--", label="Analytical solution")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.legend()
plt.tight_layout()
plt.show()
