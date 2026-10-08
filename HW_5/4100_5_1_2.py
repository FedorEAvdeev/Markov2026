
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from math import comb

# Part 1(b)
P_old = np.zeros((5, 5))

for k in range(5):
    if k < 4:
        P_old[k, k + 1] = (4 - k) / 4
    if k > 0:
        P_old[k, k - 1] = k / 4

q_old = np.zeros((61, 5))
q_old[0, 0] = 1

for n in range(60):
    q_old[n + 1] = q_old[n] @ P_old

average = np.zeros(61)

for n in range(61):
    average[n] = np.sum(q_old[:n + 1, 2]) / (n + 1)


# Part 1(c)
a = 0.3
b = 0.1

P = np.zeros((5, 5))

for k in range(5):
    if k < 4:
        P[k, k + 1] = a * (4 - k) / 4
    if k > 0:
        P[k, k - 1] = b * k / 4
    P[k, k] = 1 - sum(P[k])

# Find stationary distribution
A = P.T - np.eye(5)
A[4, :] = 1

rhs = np.zeros(5)
rhs[4] = 1

pi = np.linalg.solve(A, rhs)

# Check with binomial formula
theta = a / (a + b)
pi_binomial = np.zeros(5)

for k in range(5):
    pi_binomial[k] = comb(4, k) * theta**k * (1 - theta)**(4 - k)

print("Stationary distribution:", pi)
print("Binomial distribution:", pi_binomial)
print("Difference:", np.max(np.abs(pi - pi_binomial)))

# Compute q_n
q = np.zeros((201, 5))
q[0, 0] = 1

for n in range(200):
    q[n + 1] = q[n] @ P

# Find convergence step
for n in range(201):
    error = np.max(np.abs(q[n] - pi))
    if error < 1e-6:
        print("First n with error < 1e-6:", n)
        break

# Eigenvalues
eigenvalues = np.linalg.eigvals(P)
moduli = sorted(np.abs(eigenvalues), reverse=True)

print("Eigenvalues:", eigenvalues)
print("SLEM:", moduli[1])


# Plot both parts
plt.figure(figsize=(12, 5))

# First panel
plt.subplot(1, 2, 1)

plt.plot(range(61), q_old[:, 2], label="q_n(2)")
plt.plot(range(61), q_old[:, 4], label="q_n(4)")
plt.plot(range(61), average, label="Running average")
plt.axhline(0.375, color="gray", linestyle="--", label="pi(2)")

plt.xlabel("Step n")
plt.ylabel("Probability")
plt.title("1(b): a = 1, b = 1")
plt.legend()
plt.grid()

# Second panel
plt.subplot(1, 2, 2)

for k in range(5):
    plt.plot(range(201), q[:, k], label=f"q_n({k})")

plt.xlabel("Step n")
plt.ylabel("Probability")
plt.title("1(c): a = 0.3, b = 0.1")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(Path(__file__).parent / "problem1c.png", dpi=300)
plt.close()
