import numpy as np
import matplotlib.pyplot as plt


lang = ["py", "cpp"][0]
path = "Results"
print(lang)
x = np.loadtxt(f"{path}\\HW3_X_{lang}.txt", delimiter="\t", dtype=float)
U_S = np.loadtxt(f"{path}\\HW3_U_Simple_{lang}.txt", delimiter="\t", dtype=float)
F_S = np.loadtxt(f"{path}\\HW3_F_Simple_{lang}.txt", delimiter="\t", dtype=float)
U_T = np.loadtxt(f"{path}\\HW3_U_Thomas_{lang}.txt", delimiter="\t", dtype=float)
F_T = np.loadtxt(f"{path}\\HW3_F_Thomas_{lang}.txt", delimiter="\t", dtype=float)

print(f"Maximum error (Simple): {np.max(np.abs(F_S - U_S)):e}")
print(f"Maximum error (Thomas): {np.max(np.abs(F_T - U_T)):e}")
print(f"Maximum difference (Simple-Thomas): {np.max(np.abs(U_S - U_T)):e}")

plt.title("1D Heat conductivity equation")
plt.grid()
plt.plot(x, U_S, label="Simple iterative method")
plt.plot(x, F_S, ls="--", label=f"Analytical solution")
plt.plot(x, U_T, label="Thomas algorithm")
plt.plot(x, F_T, ls="--", label=f"Analytical solution")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.legend()
plt.tight_layout()
plt.show()
