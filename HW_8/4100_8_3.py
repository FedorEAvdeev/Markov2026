import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rng = np.random.default_rng(2027)

days = 30
T = 24 * days

N = rng.poisson(20 * T)
times = rng.uniform(0, T, N)

rates = 12 + 8 * np.cos(np.pi * ((times % 24) - 16) / 12)
keep = rng.random(N) < rates / 20
arrivals = np.sort(times[keep])

print(f'Proposed: {N}')
print(f'Accepted: {len(arrivals)}')
print(f'Fraction kept: {len(arrivals) / N:.6f} (theory: 0.6)')

counts = np.zeros((days, 24), dtype=int)

day = (arrivals // 24).astype(int)
hour = (arrivals % 24).astype(int)
np.add.at(counts, (day, hour), 1)

means = counts.mean(axis=0)

h = np.arange(24)
x = np.linspace(0, 24, 500)

true_rate = 12 + 8 * np.cos(np.pi * (x - 16) / 12)

plt.figure(figsize=(9, 5))
plt.plot(x, true_rate, label='True rate')
plt.plot(h + 0.5, means, 'o', label='Simulation')

plt.xlabel('Hour of day')
plt.ylabel('Arrivals per hour')
plt.xticks(np.arange(0, 25, 2))
plt.legend()
plt.tight_layout()
plt.savefig('4100_8_3_hourly.png', dpi=200)
plt.show()

early = counts[:, 3]

mu = 12 + 96 / np.pi * (
    np.sin(np.pi * (4 - 16) / 12)
    - np.sin(np.pi * (3 - 16) / 12)
)

print('\n3–4 am:')
print(f'Mean: {early.mean():.6f} (theory: {mu:.6f})')
print(f'Variance: {early.var():.6f} (theory: {mu:.6f})')
def neg_loglike(params):
    a, b, phi = params

    if a <= abs(b):
        return np.inf

    rate = a + b * np.cos(np.pi * (arrivals - phi) / 12)
    return a * T - np.sum(np.log(rate))

result = minimize(
    neg_loglike,
    [10, 5, 14],
    method='Nelder-Mead',
    options={'maxiter': 30000}
)

a, b, phi = result.x
phi = phi % 24

ll_fit = -result.fun

constant = len(arrivals) / T
ll_constant = len(arrivals) * np.log(constant) - constant * T

print('\nMLE fit:')
print(f'Optimization successful: {result.success}')
print(f'a = {a:.6f} (true: 12)')
print(f'b = {b:.6f} (true: 8)')
print(f'phi = {phi:.6f} (true: 16)')
print(f'Fitted log-likelihood: {ll_fit:.6f}')
print(f'Constant log-likelihood: {ll_constant:.6f}')
print(f'Difference: {ll_fit - ll_constant:.6f}')
