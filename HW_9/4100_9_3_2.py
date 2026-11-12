import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import brentq

Q = np.array([
    [-2, 2, 0, 0],
    [1, -3, 2, 0],
    [0, 2, -4, 2],
    [0, 0, 3, -3]
], dtype=float)

pi = np.array([3, 6, 6, 4], dtype=float) / 19
initial = np.array([1, 0, 0, 0], dtype=float)

eigenvalues = np.sort(np.linalg.eigvals(Q).real)[::-1]
lambda2 = eigenvalues[1]

t = np.linspace(0, 5, 500)
probabilities = np.array([initial @ expm(s * Q) for s in t])

tv = 0.5 * np.abs(probabilities - pi).sum(axis=1)

def variation(s):
    return 0.5 * np.abs(initial @ expm(s * Q) - pi).sum()

time_001 = brentq(lambda s: variation(s) - 0.01, 0, 5)
spectral_time = np.log(100) / abs(lambda2)

print('Stationary distribution:', pi)
print('Second eigenvalue:', lambda2)
print('Time to TV = 0.01 (hours):', time_001)
print('ln(100)/|lambda2| (hours):', spectral_time)

plt.figure(figsize=(7, 4.5))

for i in range(4):
    plt.plot(t, probabilities[:, i], label=f'State {i}')

plt.xlabel('Time (hours)')
plt.ylabel('Probability')
plt.title('Number of open positions')
plt.legend()
plt.tight_layout()
plt.savefig('4100_9_3_states.png', dpi=180)
plt.show()

plt.figure(figsize=(7, 4.5))

for i in range(4):
    plt.semilogy(t, probabilities[:, i], label=f'State {i}')

plt.xlabel('Time (hours)')
plt.ylabel('Probability (log scale)')
plt.title('State probabilities on log scale')
plt.legend()
plt.tight_layout()
plt.savefig('4100_9_3_states_log.png', dpi=180)
plt.show()

reference = tv[-1] * np.exp(lambda2 * (t - t[-1]))

plt.figure(figsize=(7, 4.5))
plt.semilogy(t, tv, label='Total variation distance')
plt.semilogy(t, reference, '--',
             label=f'Slope lambda2 = {lambda2:.4f}')
plt.axhline(0.01, linestyle=':', color='black',
            label='TV = 0.01')

plt.xlabel('Time (hours)')
plt.ylabel('Total variation distance')
plt.title('Convergence to stationarity')
plt.legend()
plt.tight_layout()
plt.savefig('4100_9_3_tv.png', dpi=180)
plt.show()
