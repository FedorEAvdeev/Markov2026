import numpy as np
import matplotlib.pyplot as plt

ms = np.arange(3, 13)
exponents = []
moduli = []

for m in ms:
    P = np.zeros((m, m))
    for i in range(m - 1):
        P[i, i + 1] = 1
    P[-1, 0] = 0.5
    P[-1, 1] = 0.5

    B = np.eye(m, dtype=bool)
    edges = (P > 0).astype(int)
    n = 0

    while not B.all():
        B = B.astype(int) @ edges > 0
        n += 1

    eigenvalues = np.linalg.eigvals(P)
    second = np.sort(np.abs(eigenvalues))[-2]

    exponents.append(n)
    moduli.append(second)

    print(f"m={m:2d}  M={n:3d}  bound={(m-1)**2+1:3d}  |lambda2|={second:.6f}")

plt.figure(figsize=(7, 4.5))
plt.plot(ms, exponents, 'o-', label='Computed M')
plt.plot(ms, (ms - 1)**2 + 1, '--', label='Wielandt bound')
plt.xlabel('Number of states m')
plt.ylabel('Exponent M')
plt.title('Wielandt chain: exponent')
plt.legend()
plt.tight_layout()
plt.savefig('problem1_exponents.png', dpi=200)
plt.show()

plt.figure(figsize=(7, 4.5))
plt.plot(ms, moduli, 'o-', label='Second-largest eigenvalue modulus')
plt.xlabel('Number of states m')
plt.ylabel('|lambda2|')
plt.title('Wielandt chain: spectral convergence')
plt.legend()
plt.tight_layout()
plt.savefig('problem1_eigenvalues.png', dpi=200)
plt.show()
