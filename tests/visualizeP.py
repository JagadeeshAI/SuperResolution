import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# Set the file path here
npz_file_path = "./data/trainPatches/7009.npz"

# Load the NPZ file
data = np.load(npz_file_path)
raw_data = data['raw']
max_val = float(data['max_val'])

# Normalize data correctly
normalized = raw_data.astype(np.float32) / max_val

# Transpose the array properly
normalized_transposed = np.transpose(normalized, (1, 2, 0))

# Try with different value scaling
plt.figure(figsize=(10, 8))
# Use only first 3 channels for RGB
rgb = normalized_transposed[:, :, :3]

rgb_eq = exposure.equalize_hist(rgb)

plt.imshow(rgb_eq)
plt.title("Enhanced RGB Composite")
plt.colorbar()
plt.show()
