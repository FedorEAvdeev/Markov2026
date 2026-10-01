import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260930)

P = np.array([
    [0.0, 0.5, 0.5, 0.0, 0.0],
    [0.25, 0.0, 0.0, 0.5, 0.25],
    [0.75, 0.0, 0.0, 0.0, 0.25],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]
])

states = ["U", "I", "M", "F", "A"]
state_to_idx = {s: i for i, s in enumerate(states)}

def inverse_step(x):
    u = rng.random()
    c = np.cumsum(P[x])
    return np.searchsorted(c, u, side="right")

def simulate(start, R=10000):
    x0 = state_to_idx[start]
    times = np.empty(R, dtype=int)
    fates = np.empty(R, dtype=int)
    for r in range(R):
        x = x0
        t = 0
        while x not in (3, 4):
            x = inverse_step(x)
            t += 1
        times[r] = t
        fates[r] = x
    folded = fates == 3
    aggregated = fates == 4
    h_hat = folded.mean()
    g_hat = times.mean()
    tauF_hat = times[folded].mean()
    tauA_hat = times[aggregated].mean()
    return {
        "times": times,
        "fates": fates,
        "h_hat": h_hat,
        "g_hat": g_hat,
        "tauF_hat": tauF_hat,
        "tauA_hat": tauA_hat
    }

exact = {
    "U": {"h": 1/2, "g": 4, "tauF": 4, "tauA": 4},
    "I": {"h": 5/8, "g": 2, "tauF": 9/5, "tauA": 7/3},
    "M": {"h": 3/8, "g": 4, "tauF": 5, "tauA": 17/5}
}

results = {s: simulate(s) for s in ["U", "I", "M"]}

rows = []
for s in ["U", "I", "M"]:
    rows.append([
        s,
        f"{results[s]['h_hat']:.4f}",
        f"{exact[s]['h']:.4f}",
        f"{results[s]['g_hat']:.4f}",
        f"{exact[s]['g']:.4f}",
        f"{results[s]['tauF_hat']:.4f}",
        f"{exact[s]['tauF']:.4f}",
        f"{results[s]['tauA_hat']:.4f}",
        f"{exact[s]['tauA']:.4f}"
    ])
    print(
        s,
        "h_hat =", results[s]["h_hat"], "h_exact =", exact[s]["h"],
        "g_hat =", results[s]["g_hat"], "g_exact =", exact[s]["g"],
        "tauF_hat =", results[s]["tauF_hat"], "tauF_exact =", exact[s]["tauF"],
        "tauA_hat =", results[s]["tauA_hat"], "tauA_exact =", exact[s]["tauA"]
    )

fig, ax = plt.subplots(figsize=(12, 2.8))
ax.axis("off")
col_labels = ["start", "h_hat", "h_exact", "g_hat", "g_exact", "tauF_hat", "tauF_exact", "tauA_hat", "tauA_exact"]
table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.tight_layout()
plt.savefig("summary_table.png", dpi=300, bbox_inches="tight")
plt.close()

Q = P[:3, :3]
R = P[:3, 3:]
times_I = results["I"]["times"]
fates_I = results["I"]["fates"]
fold_times = times_I[fates_I == 3]
agg_times = times_I[fates_I == 4]

n_max = times_I.max()
n_values = np.arange(1, n_max + 1)
pmf_F = np.zeros_like(n_values, dtype=float)
pmf_A = np.zeros_like(n_values, dtype=float)

for j, n in enumerate(n_values):
    M = np.linalg.matrix_power(Q, n - 1) @ R
    pmf_F[j] = M[1, 0] / (5/8)
    pmf_A[j] = M[1, 1] / (3/8)

bins_F = np.arange(0.5, fold_times.max() + 1.5, 1)
plt.figure(figsize=(8, 5))
plt.hist(fold_times, bins=bins_F, density=True, alpha=0.6, label="Simulation")
plt.plot(n_values, pmf_F, "o-", label="Exact conditional PMF")
plt.xlabel("T")
plt.ylabel("Probability")
plt.title("Start I: T conditioned on folding")
plt.legend()
plt.tight_layout()
plt.savefig("I_fold_hist.png", dpi=300, bbox_inches="tight")
plt.close()

bins_A = np.arange(0.5, agg_times.max() + 1.5, 1)
plt.figure(figsize=(8, 5))
plt.hist(agg_times, bins=bins_A, density=True, alpha=0.6, label="Simulation")
plt.plot(n_values, pmf_A, "o-", label="Exact conditional PMF")
plt.xlabel("T")
plt.ylabel("Probability")
plt.title("Start I: T conditioned on aggregation")
plt.legend()
plt.tight_layout()
plt.savefig("I_aggregate_hist.png", dpi=300, bbox_inches="tight")
plt.close()
