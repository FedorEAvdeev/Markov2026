import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(28032008)

U = rng.random(100_000)
z = 1 - np.sqrt(1-U)

print(z.mean())