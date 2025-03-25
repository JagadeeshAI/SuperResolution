import numpy as np
import matplotlib.pyplot as plt

# Set the file path here
npz_file_path = "./data/valPatches50/29.npz"

# Load the NPZ file
data = np.load(npz_file_path)
raw_data = data['raw']
max_val = float(data['max_val'])

print(f"The shape of the raw is {raw_data.shape}")

normalized = raw_data.astype(np.float32) / max_val

plt.figure(figsize=(10, 8))
rgb = np.clip(normalized[:, :, :3], 0, 1)
plt.imshow(rgb)
plt.title("RGB Composite")

plt.show()
