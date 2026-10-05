import numpy as np

vol = np.zeros((4, 4, 4))
vol[0, 0, 0] = 0.9  # blob 1
vol[3, 3, 3] = 0.8  # blob 2

# Check what flat index 0 corresponds to
print(np.unravel_index(0, (4,4,4)))   # → ?

# Check what flat index 63 corresponds to
print(np.unravel_index(63, (4,4,4)))  # → ?