
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(27)
ns = [10, 100, 1000]
mean_scatter = []
median_scatter = []

for n in ns:
    x = 1.5 + 0.8 * rng.standard_cauchy((2000, n))
    means = x.mean(axis=1)
    medians = np.median(x, axis=1)

    mean_iqr = np.quantile(means, 0.75) - np.quantile(means, 0.25)
    median_iqr = np.quantile(medians, 0.75) - np.quantile(medians, 0.25)
    mean_scatter.append(mean_iqr / 1.349)
    median_scatter.append(median_iqr / 1.349)

    print(f'n = {n}: mean scatter = {mean_scatter[-1]:.4f}, median scatter = {median_scatter[-1]:.4f}')

plt.plot(ns, mean_scatter, 'o-', label='Sample mean')
plt.plot(ns, median_scatter, 'o-', label='Sample median')
plt.xscale('log')
plt.yscale('log')
plt.xticks(ns, [str(n) for n in ns])
plt.xlabel('Number of flashes')
plt.ylabel('IQR / 1.349 (km)')
plt.legend()
plt.tight_layout()
plt.savefig('1d_scatter.png', dpi=180)
plt.show()
