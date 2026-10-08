
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

P = np.array([
    [0,   1/2, 1/2, 0,   0,   0,   0,   0],
    [0,   0,   1/2, 1/2, 0,   0,   0,   0],
    [1/2, 0,   0,   0,   1/2, 0,   0,   0],
    [1/3, 0,   1/3, 0,   1/3, 0,   0,   0],
    [0,   1/3, 0,   0,   0,   1/3, 1/3, 0],
    [0,   0,   0,   0,   0,   1,   0,   0],
    [0,   0,   0,   0,   0,   0,   0,   1],
    [0,   0,   0,   0,   0,   0,   1,   0]
])

d = 0.85
G = d * P + (1 - d) / 8 * np.ones((8, 8))

# Stationary distribution from 3(b)
pi = np.ones(8) / 8

for n in range(300):
    pi = pi @ G

# Simulate 100 surfers starting from page 1
R = 100
T = 100000
np.random.seed(42)

states = np.zeros(R, dtype=int)
counts = np.zeros((R, 8), dtype=int)

# Cumulative probabilities for inverse transform
cumulative = np.cumsum(G, axis=1)

check_times = np.unique(np.logspace(2, 5, 100).astype(int))
rms_errors = np.zeros(len(check_times))
check_index = 0

for t in range(1, T + 1):
    u = np.random.random(R)

    # Choose next page using cumulative probabilities
    states = np.sum(u[:, None] > cumulative[states], axis=1)

    # Count visits to each page
    counts[np.arange(R), states] += 1

    if t == check_times[check_index]:
        estimates = counts / t
        max_errors = np.max(np.abs(estimates - pi), axis=1)
        rms_errors[check_index] = np.sqrt(np.mean(max_errors**2))
        check_index += 1

# Final estimates
estimates = counts / T

# Fit slope on log-log axes
slope, intercept = np.polyfit(
    np.log10(check_times),
    np.log10(rms_errors),
    1
)

print("Stationary distribution:", pi)
print("First surfer estimate:", estimates[0])
print("Mean estimate:", np.mean(estimates, axis=0))
print("Fitted slope:", slope)
print("RMS error:", rms_errors[-1])

# Compare with independent samples
independent_error = np.sqrt(pi[5] * (1 - pi[5]) / T)

print("Independent error:", independent_error)
print("Ratio:", rms_errors[-1] / independent_error)

folder = Path(__file__).parent

# Bar chart
pages = np.arange(1, 9)

plt.figure(figsize=(8, 5))
plt.bar(pages - 0.2, estimates[0], width=0.4, label="Surfer 1")
plt.bar(pages + 0.2, pi, width=0.4, label="Stationary distribution")

plt.xticks(pages)
plt.xlabel("Page")
plt.ylabel("Probability")
plt.title("PageRank simulation")
plt.legend()
plt.tight_layout()

plt.savefig(folder / "problem3c_bar.png", dpi=300)
plt.close()

# Error plot
plt.figure(figsize=(8, 5))
plt.loglog(check_times, rms_errors, label="RMS maximum error")
plt.loglog(
    check_times,
    10**intercept * check_times**slope,
    "--",
    label=f"Fit: slope = {slope:.3f}"
)

plt.xlabel("Number of steps")
plt.ylabel("RMS maximum error")
plt.title("PageRank convergence")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(folder / "problem3c_error.png", dpi=300)
plt.close()
