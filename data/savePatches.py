import os
import numpy as np
import torch
from glob import glob
import gc
from tqdm import tqdm
from config import Config
from utils.util import extract_patches


def extract_and_save_patches(
    data_dir=Config.train_dir,
    cache_dir="./data/trainPatches",
):
    """
    Extract patches from images and save them as individual .npz files

    Args:
        data_dir: Directory containing the image files (default is Config.train_dir)
        cache_dir: Directory to save patch files
    """
    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)

    # Get all .npz files in the data directory
    data_paths = sorted(glob(os.path.join(data_dir, "*.npz")))
    print(f"Found {len(data_paths)} images in {data_dir}")

    base_name = os.path.basename(data_dir)
    existing_patches = glob(os.path.join(cache_dir, "*.npz"))

    total_patches = 0
    current_patch_idx = 1

    # Iterate through all images
    print(f"Processing {len(data_paths)} images...")
    for file_idx, path in enumerate(tqdm(data_paths)):
        try:
            # Load image data
            with np.load(path) as data:
                hr_img = data["raw"].astype(np.float32)
                max_val = data["max_val"]

                hr_img = np.expand_dims(hr_img, axis=0)
                hr_img = np.transpose(hr_img, (0, 3, 1, 2))
                hr_img = torch.from_numpy(hr_img)

                # Extract patches using the utility function
                hr_patches, total_patches_count = extract_patches(
                    hr_img, patch_size=256
                )

                # Process all patches
                for patch_idx in range(len(hr_patches)):
                    hr_patch = hr_patches[patch_idx]

                    # Skip patches that are too small
                    if hr_patch.shape[-1] < 256 or hr_patch.shape[-2] < 256:
                        continue

                    # Convert patch to numpy for saving
                    hr_patch_np = hr_patch.cpu().numpy()

                    # Prepare patch metadata
                    patch_data = {
                        "raw": hr_patch_np,  # Patch data
                        "max_val": max_val,  # Original max value
                        "source_file": os.path.basename(path),  # Original source file
                        "source_patch_idx": patch_idx,  # Patch index in original image
                    }

                    # Save individual patch
                    patch_filename = os.path.join(cache_dir, f"{current_patch_idx}.npz")
                    np.savez_compressed(patch_filename, **patch_data)

                    # Increment patch tracking
                    current_patch_idx += 1
                    total_patches += 1

                    # Manually trigger garbage collection to manage memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {path}: {str(e)}")

    print(
        f"Finished processing {len(data_paths)} images. Total patches extracted: {total_patches}"
    )
    print(f"Patches are saved in {cache_dir} with names 1.npz, 2.npz, ...")


# Run the function when this script is executed
if __name__ == "__main__":
    print(f"Starting patch extraction from {Config.train_dir}")
    extract_and_save_patches()
    print("Patch extraction complete!")
