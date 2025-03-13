import os
import numpy as np


def get_sample_npz_shapes(base_folder):
    print(f"{'Folder':<20} {'File':<30} {'Shape':<30} {'Max Shape?':<10}")
    print("=" * 100)

    max_size = 0
    max_entry = None
    shapes_info = []

    for root, _, files in os.walk(base_folder):
        npz_files = [f for f in files if f.endswith(".npz")]
        if npz_files:  # Take only one file per folder
            file = npz_files[0]  # Select the first .npz file
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)

            try:
                data = np.load(file_path)
                if "raw" in data:
                    raw_shape = data["raw"].shape
                    size = np.prod(raw_shape)  # Calculate total elements
                else:
                    raw_shape = "No 'raw' key"
                    size = 0
            except Exception as e:
                raw_shape = f"Error: {e}"
                size = 0

            shapes_info.append((folder_name, file, raw_shape, size))

            if size > max_size:
                max_size = size
                max_entry = (folder_name, file, raw_shape, size)

    # Print results
    for folder_name, file, raw_shape, size in shapes_info:
        is_max = "Yes" if size == max_size and max_size > 0 else "No"
        print(f"{folder_name:<20} {file:<30} {str(raw_shape):<30} {is_max:<10}")


if __name__ == "__main__":
    base_folder = "./data"
    get_sample_npz_shapes(base_folder)
