import numpy as np

Q = np.array([
    [-2, 2, 0, 0],
    [1, -3, 2, 0],
    [0, 2, -4, 2],
    [0, 0, 3, -3]
], dtype=float)

eigenvalues = np.sort(np.linalg.eigvals(Q).real)[::-1]

print('Eigenvalues:', eigenvalues)
print('Sum of eigenvalues:', eigenvalues.sum())
print('Trace of Q:', np.trace(Q))
print('Second eigenvalue:', eigenvalues[1])
