
import numpy as np

P = np.array([
    [0,   1/2, 1/2, 0,   0,   0,   0, 0],
    [0,   0,   1/2, 1/2, 0,   0,   0, 0],
    [1/2, 0,   0,   0,   1/2, 0,   0, 0],
    [1/3, 0,   1/3, 0,   1/3, 0,   0, 0],
    [0,   1/3, 0,   0,   0,   1/3, 1/3, 0],
    [0,   0,   0,   0,   0,   1,   0, 0],
    [0,   0,   0,   0,   0,   0,   0, 1],
    [0,   0,   0,   0,   0,   0,   1, 0]
])

d = 0.85
G = d * P + (1 - d) / 8 * np.ones((8, 8))

# Power iteration starting from uniform distribution
q = np.ones(8) / 8
iterations = 0

while True:
    q_next = q @ G
    iterations += 1

    if np.sum(np.abs(q_next - q)) < 1e-10:
        q = q_next
        break

    q = q_next

print("Iterations:", iterations)
print("Stationary distribution:", q)

# Check using a linear solve
A = G.T - np.eye(8)
A[-1, :] = 1

b = np.zeros(8)
b[-1] = 1

pi = np.linalg.solve(A, b)

print("Linear solve:", pi)
print("Difference:", np.sum(np.abs(q - pi)))

# Rank pages from highest to lowest
ranking = np.argsort(-q) + 1

print("Ranking:", ranking)
print("Page 1:", q[0])
print("Page 5:", q[4])
print("Pages 1 and 5 tie:", np.isclose(q[0], q[4]))
