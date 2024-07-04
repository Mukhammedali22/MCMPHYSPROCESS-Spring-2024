import numpy as np
import matplotlib.pyplot as plt


lang = ["py", "cpp"][0]
print(lang)
path = "Results"
x = np.loadtxt(f"{path}\\HW5_X_{lang}.txt", delimiter="\t", dtype=float)
U = np.loadtxt(f"{path}\\HW5_U_{lang}.txt", delimiter="\t", dtype=float)

plt.title("1D Transport equation")
plt.grid()
plt.plot(x, U, label="The first scheme against the flow")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.legend()
plt.tight_layout()
plt.show()
