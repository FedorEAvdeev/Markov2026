import numpy as np

N = 100
K = 10
gamma = 1
dt = 0.005

def expected_steps(R0):
    beta = R0 * gamma / N
    n = np.arange(1, K)

    p = beta * n * (N - n) * dt
    q = gamma * n * dt

    A = np.diag(p + q)
    A += np.diag(-p[:-1], 1)
    A += np.diag(-q[1:], -1)

    e = np.linalg.solve(A, np.ones(K - 1))
    return e

for R0 in (0.5, 0.8, 1.5, 3.0):
    e = expected_steps(R0)
    print(f'R0={R0:.1f}, E[T]={e[0]:.6f} steps, E[T]*dt={e[0]*dt:.6f}')

R0 = 1.5
rng = np.random.default_rng(2026)
trials = 100000

x = np.ones(trials, dtype=int)
times = np.zeros(trials, dtype=int)
active = (x > 0) & (x < K)
beta = R0 * gamma / N

while active.any():
    ix = np.where(active)[0]
    n = x[ix]

    p = beta * n * (N - n) * dt
    q = gamma * n * dt
    u = rng.random(len(ix))

    x[ix] += (u < p).astype(int) - ((u >= p) & (u < p + q)).astype(int)
    times[ix] += 1
    active[ix] = (x[ix] > 0) & (x[ix] < K)

samples = times * dt

print(f'\nSimulation for R0={R0}')
print(f'Exact mean: {expected_steps(R0)[0] * dt:.6f}')
print(f'Simulated mean: {samples.mean():.6f}')
print(f'Standard error: {samples.std(ddof=1)/np.sqrt(trials):.6f}')
