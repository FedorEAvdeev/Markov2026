
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

rates = np.array([3.0, 2.0, 1.5, 1.2, 1.0, 0.75])
n = len(rates)
expected = np.zeros(2 ** n)
last = np.zeros((2 ** n, n))

for mask in range(1, 2 ** n):
    remaining = [i for i in range(n) if mask & (1 << i)]
    total = rates[remaining].sum()
    expected[mask] = 1 / total

    for i in remaining:
        smaller = mask ^ (1 << i)
        probability = rates[i] / total
        expected[mask] += probability * expected[smaller]
        if len(remaining) == 1:
            last[mask, i] = 1
        else:
            last[mask] += probability * last[smaller]

all_students = 2 ** n - 1
inclusion = 0

for size in range(1, n + 1):
    for group in combinations(range(n), size):
        inclusion += (-1) ** (size + 1) / rates[list(group)].sum()

print('Expected last finishing time (recursion):', expected[all_students])
print('Expected last finishing time (inclusion-exclusion):', inclusion)
print('Exact probabilities of finishing last:', last[all_students])

rng = np.random.default_rng(21)
times = rng.exponential(scale=1 / rates, size=(200000, n))
maximum = times.max(axis=1)
minimum = times.min(axis=1)
first = times.argmin(axis=1)
last_student = times.argmax(axis=1)

print('Simulated expected last finishing time:', maximum.mean())
print('Simulated probabilities of finishing last:', np.bincount(last_student, minlength=n) / len(times))
print('Theoretical mean first finishing time:', 1 / rates.sum())

for i in range(n):
    print(f'Student {i + 1} first: mean first time = {minimum[first == i].mean():.4f}')

x = np.linspace(0, 10, 500)
density = np.zeros_like(x)

for i in range(n):
    other = [j for j in range(n) if j != i]
    density += rates[i] * np.exp(-rates[i] * x) * np.prod(
        1 - np.exp(-rates[other, None] * x), axis=0
    )

width = 0.1
bins = np.arange(0, maximum.max() + width, width)
plt.hist(maximum, bins=bins, density=True, alpha=0.5, label=f'Simulation (bin width {width} h)')
plt.plot(x, density, label='Exact density')
plt.xlim(0, 9)
plt.xlabel('Time until everyone finishes (hours)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('2d_maximum.png', dpi=180)
plt.show()

ks = np.arange(1, 13)
expected_max = []

for k in ks:
    r = np.arange(1, k + 1)
    value, _ = quad(lambda t: 1 - np.prod(1 - np.exp(-r * t)), 0, np.inf)
    expected_max.append(value)

print('Expected maximum for k = 1,...,12:')
print(np.round(expected_max, 5))

plt.plot(ks, expected_max, 'o-')
plt.axhline(2 * np.log(2), linestyle='--', color='black', label='2 log(2)')
plt.xlabel('Number of students')
plt.ylabel('Expected last finishing time (hours)')
plt.legend()
plt.tight_layout()
plt.savefig('2d_growing_k.png', dpi=180)
plt.show()
