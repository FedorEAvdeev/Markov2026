from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

a = 1
b = 1

P = np.zeros((5, 5))

for k in range(5):
    if k < 4:
        P[k, k + 1] = a * (4 - k) / 4
    if k > 0:
        P[k, k - 1] = b * k / 4
    P[k, k] = 1 - sum(P[k])

q = np.zeros((61, 5))
q[0, 0] = 1

for n in range(60):
    q[n + 1] = q[n] @ P

print("Transition matrix:")
print(P)
print("q50 =", q[50])
print("q51 =", q[51])

average = np.zeros(61)
total = 0

for n in range(61):
    total += q[n, 2]
    average[n] = total / (n + 1)

steps = range(61)

plt.figure(figsize=(9, 5))
plt.plot(steps, q[:, 2], label="q_n(2)")
plt.plot(steps, q[:, 4], label="q_n(4)")
plt.plot(steps, average, label="Running average of q_n(2)")
plt.axhline(0.375, linestyle="--", color="gray", label="pi(2)")

plt.xlabel("Step n")
plt.ylabel("Probability")
plt.title("Part 1(b)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(Path(__file__).parent / "problem1b.png", dpi=300, bbox_inches="tight")
plt.close()
