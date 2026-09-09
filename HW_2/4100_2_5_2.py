import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(28032008)

N = 10000
R = 1000

C = np.ones(R, dtype=int)

for n in range(3, N):
    grow = rng.random(R) < C / n
    C += grow

z = C / N

theory_mean = 1 / 3
theory_ratio = 1 / np.sqrt(2)

emp_mean = z.mean()
emp_ratio = z.std(ddof=0) / emp_mean

print("empirical mean =", emp_mean)
print("theoretical mean =", theory_mean)
print("empirical std/mean =", emp_ratio)
print("theoretical std/mean =", theory_ratio)
print("smallest core =", C.min())
print("largest core =", C.max())

bw = 0.04
bins = np.arange(0, 1 + bw, bw)

x = np.linspace(0, 1, 1000)
h = 2 * (1 - x)

plt.figure()

plt.hist(
    z,
    bins=bins,
    density=True,
    label=f"empirical, bin width = {bw}"
)

plt.plot(x, h, label="h(z) = 2(1 - z)")

plt.xlabel("z = C/N")
plt.ylabel("density")
plt.title("HW2 Problem 5: Distribution of z = C/N")
plt.legend()

plt.savefig("HW2_P5_core_histogram.png", dpi=300, bbox_inches="tight")
plt.show()