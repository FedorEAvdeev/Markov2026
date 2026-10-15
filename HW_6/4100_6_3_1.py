
import numpy as np
import matplotlib.pyplot as plt

def outbreak_probability(N, K, R0):
    rho = 1.0
    total = 1.0

    for k in range(1, K):
        rho *= N / (R0 * (N - k))
        total += rho

    return 1 / total

values = np.linspace(0.5, 3, 251)

plt.figure(figsize=(7, 4.5))

for N in (100, 1000):
    K = N // 10
    probabilities = [outbreak_probability(N, K, x) for x in values]
    plt.plot(values, probabilities, label=f'Exact, N={N}, K={K}')

plt.plot(values, 1 - 1 / values, '--', label='1 - 1/R0')
plt.plot(values, np.maximum(0, 1 - 1 / values), ':', label='Limiting probability')

plt.xlabel('Basic reproduction number R0')
plt.ylabel('Probability of reaching K')
plt.title('Probability of an outbreak from one infection')
plt.legend()
plt.tight_layout()
plt.savefig('problem3_probability.png', dpi=200)
plt.show()
