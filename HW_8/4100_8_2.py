import numpy as np

rng = np.random.default_rng(2026)

n = 200000
t = 5
q = 0.4

star = rng.poisson(3 * t, n)
sky = rng.poisson(2 * t, n)

star1 = rng.binomial(star, q)
sky1 = rng.binomial(sky, q)

star2 = rng.poisson(1.2 * t, n)
sky2 = rng.poisson(0.8 * t, n)

for name, s, b in [('Thinning', star1, sky1),
                    ('Direct', star2, sky2)]:
    total = s + b
    mask = total == 10

    print(f'\n{name}:')
    print(f'Star: mean={s.mean():.4f}, var={s.var():.4f}')
    print(f'Sky: mean={b.mean():.4f}, var={b.var():.4f}')
    print(f'Correlation: {np.corrcoef(s, b)[0, 1]:.4f}')
    print(f'Total: mean={total.mean():.4f}, var={total.var():.4f}')
    print(f'P(6 star | 10 total): {np.mean(s[mask] == 6):.4f}')

ts = rng.exponential(1 / 1.2, n)
tb = rng.exponential(1 / 0.8, n)

first = np.minimum(ts, tb)
star_first = ts < tb
M = np.maximum(ts, tb)

print('\nFirst detection:')
print(f'P(star first): {star_first.mean():.6f} (theory: 0.6)')
print(f'P(star first within 1 s): {np.mean(star_first & (first <= 1)):.6f} (theory: {0.6 * (1 - np.exp(-2)):.6f})')
print(f'E[M]: {M.mean():.6f} (theory: {1 / 1.2 + 1 / 0.8 - 1 / 2:.6f})')
print(f'Corr(first time, type): {np.corrcoef(first, star_first.astype(int))[0, 1]:.6f} (theory: 0)')
