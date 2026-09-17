import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

R = 20_000
T = 10_000
d0 = 10
rng = np.random.default_rng(1)
D = np.full(R, d0, dtype=int)
alive = np.ones(R, dtype=bool)
S1 = np.zeros(T + 1)
S1[0] = 1.0




for t in range(1, T + 1):
    # indices of realizations where lamb is alive at t
    idx = np.where(alive)[0]

    if len(idx) == 0:
        break

    lamb_steps = rng.choice([-1, 1], size=len(idx))
    lion_steps = rng.choice([-1, 1], size=len(idx))
    D[idx] += lion_steps - lamb_steps
    caught = (D[idx] == 0)
    alive[idx[caught]] = False



    S1[t] = np.mean(alive)



time = np.arange(T + 1)

fit_mask = (
    (time >= 100)
    & (time <= 10_000)
    & (S1 > 0)
)

log_t = np.log(time[fit_mask])
log_S1 = np.log(S1[fit_mask])
slope, intercept = np.polyfit(log_t, log_S1, 1)
beta1 = -slope
print("beta1 =", beta1)



continuum = np.full(T + 1, np.nan)
continuum[1:] = erf(d0 / (2 * np.sqrt(time[1:])))



plt.figure(figsize=(8, 6))

plt.loglog(
    time[1:],
    S1[1:],
    label="Simulation $S_1(t)$"
)

plt.loglog(
    time[1:],
    continuum[1:],
    "--",
    label=r"$\mathrm{erf}\left(\frac{d_0}{2\sqrt{t}}\right)$"
)

plt.xlabel("t")
plt.ylabel(r"$S_1(t)$")

plt.title(
    rf"$\beta_1 = {beta1:.3f}$, fit over $10^2 \leq t \leq 10^4$"
)

plt.legend()
plt.tight_layout()

plt.savefig("HW3_P4_B.png", dpi=300, bbox_inches="tight")

plt.show()