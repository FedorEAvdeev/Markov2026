import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

R = 20_000
T = 10_000
d0 = 10


def simulate(N, seed):
    rng = np.random.default_rng(seed)
    D = np.full((R, N), d0, dtype=int)
    alive = np.ones(R, dtype=bool)
    S = np.zeros(T + 1)
    S[0] = 1.0



    for t in range(1, T + 1):
        idx = np.where(alive)[0]

        if len(idx) == 0:
            break
        
        lamb_steps = rng.choice([-1, 1], size=len(idx))

        lion_steps = rng.choice([-1, 1], size=(len(idx), N))

        D[idx] += lion_steps - lamb_steps[:, None]
        caught = np.any(D[idx] == 0, axis=1)
        alive[idx[caught]] = False
        S[t] = np.mean(alive)

    return S



S1 = simulate(N=1, seed=1)
S2 = simulate(N=2, seed=12)

time = np.arange(T + 1)


mask1 = (
    (time >= 100)
    & (time <= 10_000)
    & (S1 > 0)
)

slope1, intercept1 = np.polyfit(
    np.log(time[mask1]),
    np.log(S1[mask1]),
    1
)

beta1 = -slope1

mask2 = (
    (time >= 100)
    & (time <= 10_000)
    & (S2 > 0)
)

slope2, intercept2 = np.polyfit(
    np.log(time[mask2]),
    np.log(S2[mask2]),
    1
)

beta2 = -slope2

print("beta1 =", beta1)
print("beta2 =", beta2)


continuum = np.full(T + 1, np.nan)
continuum[1:] = erf(d0 / (2 * np.sqrt(time[1:])))


print("\nTable:")

for t in [100, 1000, 10000]:
    print(
        f"t = {t:5d}    "
        f"S2 = {S2[t]:.6f}    "
        f"S1^2 = {S1[t]**2:.6f}"
    )


plt.figure(figsize=(8, 6))

plt.loglog(
    time[1:],
    S1[1:],
    label=r"$S_1(t)$"
)

plt.loglog(
    time[1:],
    S2[1:],
    label=r"$S_2(t)$"
)

plt.loglog(
    time[1:],
    S1[1:]**2,
    label=r"$S_1(t)^2$"
)

plt.loglog(
    time[1:],
    continuum[1:],
    "--",
    label=r"$\mathrm{erf}\left(\frac{d_0}{2\sqrt{t}}\right)$"
)

plt.xlabel("t")
plt.ylabel("Survival probability")

plt.title(
    rf"$\beta_1={beta1:.3f},\ \beta_2={beta2:.3f}$, "
    rf"fit window: $10^2 \leq t \leq 10^4$"
)

plt.legend()
plt.tight_layout()

plt.savefig("HW3_P4_C.png", dpi=300, bbox_inches="tight")

plt.show()