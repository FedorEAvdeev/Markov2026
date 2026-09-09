import numpy as np
import matplotlib.pyplot as plt
import time
import math

rng = np.random.default_rng(28032008)

N = 100000

a = 0.9
lam_f = 1000.0
lam_s = 10.0

fast = rng.random(N) < a
U = rng.random(N)

lam = np.where(fast, lam_f, lam_s)
T = -np.log(1 - U) / lam

mean_theory = a / lam_f + (1 - a) / lam_s
tail_theory = a * np.exp(-lam_f * 0.05) + (1 - a) * np.exp(-lam_s * 0.05)

print("empirical mean =", T.mean())
print("theoretical mean =", mean_theory)
print("empirical P(T > 50 ms) =", np.mean(T > 0.05))
print("theoretical P(T > 50 ms) =", tail_theory)

bw = 0.0005  
bins = np.arange(0, 0.6 + bw, bw)

x = np.linspace(0, 0.6, 3000)
f = a * lam_f * np.exp(-lam_f * x) + (1 - a) * lam_s * np.exp(-lam_s * x)

plt.figure()

plt.hist(
    T,
    bins=bins,
    density=True,
    label=f"empirical, bin width = {1000*bw:.1f} ms"
)

plt.plot(x, f, label="theory")
plt.yscale("log")
plt.xlabel("t [s]")
plt.ylabel("density")
plt.title("HW2 Problem 4: Closed dwell times")
plt.legend()

plt.savefig("HW2_P4_dwell_times.png", dpi=300, bbox_inches="tight")
plt.show()