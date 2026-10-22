
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, gamma, binom

rng = np.random.default_rng(24)
reps = 200000
rate = 4

gaps = rng.exponential(1 / rate, size=(reps, 35))
arrivals = np.cumsum(gaps, axis=1)
assert np.all(arrivals[:, -1] > 1)

count_gaps = (arrivals <= 1).sum(axis=1)
first_gaps = arrivals[:, 0]
third_gaps = arrivals[:, 2]
half_gaps = (arrivals <= 0.5).sum(axis=1)

count_uniform = rng.poisson(rate, reps)
first_uniform = np.full(reps, np.nan)
half_uniform = np.zeros(reps, dtype=int)
first_ten_uniform = []

for i, n in enumerate(count_uniform):
    points = np.sort(rng.random(n))
    if n:
        first_uniform[i] = points[0]
    half_uniform[i] = np.searchsorted(points, 0.5)
    if i < 10:
        first_ten_uniform.append(points)

for label, count, first, half in [
    ('Exponential gaps', count_gaps, first_gaps, half_gaps),
    ('Poisson + uniform', count_uniform, first_uniform, half_uniform)
]:
    selected = count == 3
    print(label)
    print('Mean and variance:', count.mean(), count.var())
    print('P(no meteor in first half | N(1) = 3):', np.mean(half[selected] == 0))
    print('E[T1 | N(1) = 3]:', np.mean(first[selected]))
    print('Correlation of half-hour counts:', np.corrcoef(half, count - half)[0, 1])

values = np.arange(15)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

for ax, count, title in zip(axes, [count_gaps, count_uniform], ['Exponential gaps', 'Poisson + uniform']):
    ax.bar(values, np.bincount(count, minlength=15)[:15] / reps, alpha=0.6, label='Simulation')
    ax.plot(values, poisson.pmf(values, rate), 'ko-', markersize=3, label='Poisson(4)')
    ax.set_title(title)
    ax.set_xlabel('Meteors in one hour')
    ax.legend()

axes[0].set_ylabel('Probability')
plt.tight_layout()
plt.savefig('3d_counts.png', dpi=180)
plt.show()

width = 0.04
x = np.linspace(0, 3, 500)
plt.hist(third_gaps, bins=np.arange(0, third_gaps.max() + width, width),
         density=True, alpha=0.5, label=f'Simulation (bin width {width} h)')
plt.plot(x, gamma.pdf(x, a=3, scale=1 / rate), label='Gamma(3, 4) density')
plt.xlim(0, 2.5)
plt.xlabel('Time of third meteor (hours)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('3d_third_arrival.png', dpi=180)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
axes[0].eventplot([arrivals[i, arrivals[i] <= 1] for i in range(10)])
axes[1].eventplot(first_ten_uniform)

for ax, title in zip(axes, ['Exponential gaps', 'Poisson + uniform']):
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Time (hours)')

axes[0].set_ylabel('Simulated hour')
plt.tight_layout()
plt.savefig('3d_rasters.png', dpi=180)
plt.show()

m_values = np.unique(np.geomspace(10, 2000, 85).astype(int))
dt = 1 / m_values
variations = []

for m in m_values:
    k = np.arange(m + 1)
    difference = np.abs(binom.pmf(k, m, rate / m) - poisson.pmf(k, rate))
    variation = 0.5 * (difference.sum() + poisson.sf(m, rate))
    variations.append(variation)

variations = np.array(variations)
fit = m_values >= 100
slope, intercept = np.polyfit(np.log(dt[fit]), np.log(variations[fit]), 1)
print('TV log-log slope:', slope)
print('TV at dt = 0.1:', variations[0])
print('TV at dt = 0.0005:', variations[-1])

plt.loglog(dt, variations, 'o', markersize=3, label='Total variation')
plt.loglog(dt, np.exp(intercept) * dt ** slope, label=f'Fit (slope {slope:.3f})')
plt.xlabel('Delta t (hours)')
plt.ylabel('Total variation distance')
plt.legend()
plt.tight_layout()
plt.savefig('3d_total_variation.png', dpi=180)
plt.show()
