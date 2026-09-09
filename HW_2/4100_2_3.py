import numpy as np
import matplotlib.pyplot as plt
import time
import math

rng = np.random.default_rng(28032008)

def rejection_gamma(N, lam):
    c = 1 / (math.e * lam * (1 - lam))

    samples = []
    proposals = 0

    start = time.perf_counter()

    while len(samples) < N:
        v = rng.random()
        u = rng.random()

        x = -np.log(1 - v) / lam
        proposals += 1

        accept_prob = math.e * (1 - lam) * x * np.exp(-(1 - lam) * x)

        if u < accept_prob:
            samples.append(x)

    end = time.perf_counter()

    samples = np.array(samples)

    acceptance = N / proposals
    time_per_sample = (end - start) / N

    return samples, acceptance, time_per_sample, c


for lam in [0.5, 0.2]:

    X, acceptance, time_per_sample, c = rejection_gamma(10000, lam)

    print("lambda =", lam)
    print("empirical acceptance =", acceptance)
    print("theoretical acceptance =", 1 / c)
    print("mean time per accepted sample =", time_per_sample, "s")

    bin_width = 0.2
    bins = np.arange(0, X.max() + bin_width, bin_width)

    x = np.linspace(0, X.max(), 1000)

    plt.figure()

    plt.hist(
        X,
        bins=bins,
        density=True,
        label="simulation"
    )

    plt.plot(
        x,
        x * np.exp(-x),
        label="theory"
    )

    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("lambda = " + str(lam))
    plt.legend()

    if lam == 0.5:
        plt.savefig("HW2_P3_lambda05.png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig("HW2_P3_lambda02.png", dpi=300, bbox_inches="tight")

    plt.show()