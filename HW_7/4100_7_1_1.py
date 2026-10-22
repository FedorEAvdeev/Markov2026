import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    -1.77, 2.47, 1.34, 2.18, 13.05, 1.60, 1.50, -1.97,
    0.79, 1.50, 2.00, 2.63, 1.19, -2.31, 0.87
])


def log_posterior(alpha, b):
    return len(x) * b - np.log(np.exp(2 * b) + (x - alpha) ** 2).sum()


def run_chain(s, seed):
    rng = np.random.default_rng(seed)
    samples = np.zeros((50000, 2))
    z = np.array([0.0, 0.0])
    current = log_posterior(*z)
    accepted = 0

    for i in range(len(samples)):
        y = z + s * rng.normal(size=2)
        next_value = log_posterior(*y)
        if np.log(rng.random()) < min(0, next_value - current):
            z = y
            current = next_value
            accepted += 1
        samples[i] = z

    return samples, accepted / len(samples)


def autocorrelation_time(values):
    values = values - values.mean()
    n = len(values)
    transform = np.fft.rfft(values, n=2 * n)
    corr = np.fft.irfft(transform * np.conj(transform), n=2 * n)[:n]
    corr /= corr[0]
    tau = 1.0

    for k in range(1, n // 2 - 1, 2):
        pair = corr[k] + corr[k + 1]
        if pair <= 0:
            break
        tau += 2 * pair
        if k > 5 * tau:
            break

    return tau


def describe(values):
    return np.mean(values), *np.quantile(values, [0.05, 0.95])


def grid_summary(values, weights):
    weights = weights / weights.sum()
    cumulative = np.cumsum(weights)
    return np.sum(values * weights), *np.interp([0.05, 0.95], cumulative, values)


scales = [0.02, 0.5, 5]
chains = {}
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

for ax, s in zip(axes, scales):
    samples, rate = run_chain(s, int(1000 * s) + 18)
    chains[s] = samples
    tau = autocorrelation_time(samples[:, 0])
    print(f's = {s}: acceptance = {rate:.3f}, autocorrelation time = {tau:.1f}')
    ax.plot(samples[:400, 0], linewidth=0.9)
    ax.set_ylabel('alpha (km)')
    ax.set_title(f's = {s}', loc='left')

axes[-1].set_xlabel('MCMC step')
plt.tight_layout()
plt.savefig('1c_traces.png', dpi=180)
plt.show()

alpha_grid = np.linspace(-2, 5, 901)
b_grid = np.linspace(-2.8, 2.5, 801)
A, B = np.meshgrid(alpha_grid, b_grid, indexing='ij')
log_values = len(x) * B

for xi in x:
    log_values -= np.log(np.exp(2 * B) + (xi - A) ** 2)

weights = np.exp(log_values - log_values.max())
weights /= weights.sum()

samples = chains[0.5][5000:]
alpha = samples[:, 0]
beta = np.exp(samples[:, 1])

print('\nSample mean:', x.mean())
print('Sample median:', np.median(x))
print('MCMC alpha (mean, 5%, 95%):', describe(alpha))
print('Grid alpha (mean, 5%, 95%):', grid_summary(alpha_grid, weights.sum(axis=1)))
print('MCMC beta (mean, 5%, 95%):', describe(beta))
print('Grid beta (mean, 5%, 95%):', grid_summary(np.exp(b_grid), weights.sum(axis=0)))

plt.figure(figsize=(8, 5))
plt.contour(A, B, np.exp(log_values - log_values.max()), levels=[0.02, 0.08, 0.18, 0.35, 0.58, 0.78])
plt.scatter(alpha[::20], samples[::20, 1], s=4, alpha=0.2)
plt.xlabel('alpha (km)')
plt.ylabel('log(beta)')
plt.xlim(-1, 4.5)
plt.ylim(-2.2, 1.7)
plt.tight_layout()
plt.savefig('1c_posterior.png', dpi=180)
plt.show()
