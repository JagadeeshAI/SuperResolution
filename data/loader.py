import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from config import Config
from utils.util import extract_patches, select_random_patches, downsample_raw

class LazyRAWDataset(Dataset):
    def __init__(self, data_dir, use_patches=None):
        self.data_paths = sorted(glob(os.path.join(data_dir, "*.npz")))
        print(f"Found {len(self.data_paths)} samples")

        self.use_patches = Config.patches if use_patches is None else use_patches

        if self.use_patches:
            self.sample_map = []
            for i, path in enumerate(self.data_paths):
                try:
                    with np.load(path) as data:
                            hr_img = data["raw"].astype(np.float32)
                            max_val = data["max_val"]

                            hr_img = hr_img / max_val
                            hr_img = np.expand_dims(hr_img, axis=0)
                            hr_img = np.transpose(hr_img, (0, 3, 1, 2))
                            hr_img = torch.from_numpy(hr_img)
                            _,totalPatches = extract_patches(hr_img, patch_size=512)
                            
                            for j in range(min(4, totalPatches)):
                                self.sample_map.append((i, j))
                except Exception as e:
                    print(f"Error inspecting {path}: {str(e)}")
            print(f"Will create {len(self.sample_map)} LR-HR patch pairs on demand")
        else:
            self.sample_map = list(range(len(self.data_paths)))
            print(f"Will process {len(self.sample_map)} full images on demand")

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        try:
            if self.use_patches:
                file_idx, patch_idx = self.sample_map[idx]
                path = self.data_paths[file_idx]
            else:
                path = self.data_paths[idx]
            
            hr_data = np.load(path)
            hr_img = hr_data["raw"].astype(np.float32)
            max_val = hr_data["max_val"]

            hr_img = hr_img / max_val
            hr_img = np.expand_dims(hr_img, axis=0)
            hr_img = np.transpose(hr_img, (0, 3, 1, 2))
            hr_img = torch.from_numpy(hr_img)

            if self.use_patches:
                hr_patches, _ = extract_patches(hr_img, patch_size=512)
                if patch_idx >= len(hr_patches):
                    # Fallback if patch index is out of range
                    patch_idx = 0
                    
                hr_patch = hr_patches[patch_idx]

                # Ensure minimum size for convolution operations
                if hr_patch.shape[-1] < 8 or hr_patch.shape[-2] < 8:
                    # Skip small patches by returning a dummy with proper dimensions
                    dummy_lr = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
                    dummy_hr = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
                    return {
                        "lr": dummy_lr,
                        "hr": dummy_hr,
                        "max": 1.0,
                        "filename": f"dummy_{idx}.npz",
                    }

                hr_patch_for_down = hr_patch.unsqueeze(0)  
                lr_patch = downsample_raw(hr_patch_for_down)
                
                # Print shapes for debugging
                # print(f"LR patch shape before permute: {lr_patch.shape}")
                
                # Fixed permutation based on dimensions - key change here!
                if lr_patch.dim() == 3:
                    lr_patch = lr_patch.permute(2, 0, 1).unsqueeze(0)
                elif lr_patch.dim() == 4:
                    # If 4D, it already has [batch, channels, height, width] format
                    # or we need to permute differently
                    if lr_patch.shape[1] == 3 or lr_patch.shape[1] == 4:  # If channels are already in dim 1
                        lr_patch = lr_patch
                    else:
                        lr_patch = lr_patch.permute(0, 3, 1, 2)
                else:
                    print(f"Warning: Unexpected LR patch dimensions: {lr_patch.shape}")
                    dummy_lr = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
                    dummy_hr = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
                    return {
                        "lr": dummy_lr,
                        "hr": dummy_hr,
                        "max": 1.0,
                        "filename": f"dummy_dim_{idx}.npz",
                    }
                    
                hr_patch = hr_patch.unsqueeze(0)

                return {
                    "lr": lr_patch,
                    "hr": hr_patch,
                    "max": max_val,
                    "filename": os.path.basename(path),
                }
            else:
                # Check if the image is large enough for processing
                if hr_img.shape[-1] < 8 or hr_img.shape[-2] < 8:
                    print(f"Image too small: {path} with shape {hr_img.shape}")
                    # Return dummy with proper dimensions
                    dummy_lr = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
                    dummy_hr = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
                    return {
                        "lr": dummy_lr,
                        "hr": dummy_hr,
                        "max": 1.0,
                        "filename": f"dummy_small_{idx}.npz",
                    }
                
                lr_img = downsample_raw(hr_img)
                
                # DEBUG: Print out the shape
                # print(f"LR full image shape before permute: {lr_img.shape}, dimensions: {lr_img.dim()}")
                
                # Fixed permutation based on dimensions - key change here!
                if lr_img.dim() == 3:
                    lr_img = lr_img.permute(2, 0, 1).unsqueeze(0)
                elif lr_img.dim() == 4:
                    # If 4D, it already has [batch, channels, height, width] format
                    # or we need to permute differently
                    if lr_img.shape[1] == 3 or lr_img.shape[1] == 4:  # If channels are already in dim 1
                        lr_img = lr_img
                    else:
                        lr_img = lr_img.permute(0, 3, 1, 2)
                else:
                    print(f"Warning: LR image has unexpected dimensions: {lr_img.shape}")
                    dummy_lr = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
                    dummy_hr = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
                    return {
                        "lr": dummy_lr,
                        "hr": dummy_hr,
                        "max": 1.0,
                        "filename": f"dummy_dim_{idx}.npz",
                    }

                return {
                    "lr": lr_img,
                    "hr": hr_img,
                    "max": max_val,
                    "filename": os.path.basename(path),
                }
        except Exception as e:
            print(f"Skipping {path if 'path' in locals() else 'unknown path'}: {str(e)}")
            # Increased dummy size to ensure it's large enough for model operations
            dummy_lr = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
            dummy_hr = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
            return {
                "lr": dummy_lr,
                "hr": dummy_hr,
                "max": 1.0,
                "filename": f"dummy_{idx}.npz",
            }


class SubmissionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_paths = sorted(glob(os.path.join(data_dir, "*.npz")))
        print(f"Found {len(self.data_paths)} submission samples")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        try:
            path = self.data_paths[idx]
            
            data = np.load(path)
            raw_img = data["raw"].astype(np.float32)
            max_val = data["max_val"]

            raw_img = raw_img / max_val
            
            raw_img = np.expand_dims(raw_img, axis=0)
            raw_img = np.transpose(raw_img, (0, 3, 1, 2))
            raw_img = torch.from_numpy(raw_img)
            
            # Check if image is too small
            if raw_img.shape[-1] < 8 or raw_img.shape[-2] < 8:
                print(f"Submission image too small: {path} with shape {raw_img.shape}")
                dummy_raw = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
                return {
                    "raw": dummy_raw,
                    "max": 1.0,
                    "filename": f"dummy_{idx}.npz",
                }

            return {
                "raw": raw_img,
                "max": max_val,
                "filename": os.path.basename(path),
            }
        except Exception as e:
            print(f"Error loading submission file {path}: {str(e)}")
            dummy_raw = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
            return {
                "raw": dummy_raw,
                "max": 1.0,
                "filename": f"error_{idx}.npz",
            }


def simple_collate(batch):
    """Stack tensors into batches, with error handling for None items"""
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        return {
            "lr": torch.zeros((1, 4, 64, 64), dtype=torch.float32),  # Increased size
            "hr": torch.zeros((1, 4, 128, 128), dtype=torch.float32),  # Increased size
            "max": 1.0,
            "filename": ["dummy.npz"]
        }
    
    # Check if all tensors have the same shape to prevent issues with torch.cat
    lr_shapes = [item["lr"].shape for item in valid_batch]
    hr_shapes = [item["hr"].shape for item in valid_batch]
    
    # If shapes are inconsistent, process one at a time
    if len(set(str(s) for s in lr_shapes)) > 1 or len(set(str(s) for s in hr_shapes)) > 1:
        print(f"Warning: Inconsistent tensor shapes detected. Falling back to batch size 1.")
        # Return just the first item to maintain batch size of 1
        return {
            "lr": valid_batch[0]["lr"],
            "hr": valid_batch[0]["hr"],
            "max": valid_batch[0]["max"],
            "filename": [valid_batch[0]["filename"]]
        }
    
    try:
        lr_batch = torch.cat([item["lr"] for item in valid_batch])
        hr_batch = torch.cat([item["hr"] for item in valid_batch])
        max_vals = [item["max"] for item in valid_batch]
        filenames = [item["filename"] for item in valid_batch]
        
        # Final shape check to ensure minimum sizes
        if lr_batch.shape[-1] < 8 or lr_batch.shape[-2] < 8:
            print(f"Warning: Collated batch too small: {lr_batch.shape}")
            return {
                "lr": torch.zeros((len(valid_batch), 4, 64, 64), dtype=torch.float32),
                "hr": torch.zeros((len(valid_batch), 4, 128, 128), dtype=torch.float32),
                "max": max_vals[0],
                "filename": filenames
            }
        
        return {"lr": lr_batch, "hr": hr_batch, "max": max_vals[0], "filename": filenames}
    except Exception as e:
        print(f"Error during collation: {str(e)}")
        return {
            "lr": torch.zeros((1, 4, 64, 64), dtype=torch.float32),
            "hr": torch.zeros((1, 4, 128, 128), dtype=torch.float32),
            "max": 1.0,
            "filename": ["collate_error.npz"]
        }


def submission_collate(batch):
    """Simple collate function for submission data"""
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        return {
            "raw": torch.zeros((1, 4, 64, 64), dtype=torch.float32),
            "max": 1.0,
            "filename": ["dummy.npz"]
        }
    
    # Check if all tensors have the same shape
    raw_shapes = [item["raw"].shape for item in valid_batch]
    
    # If shapes are inconsistent, process one at a time
    if len(set(str(s) for s in raw_shapes)) > 1:
        print(f"Warning: Inconsistent tensor shapes in submission. Using batch size 1.")
        return {
            "raw": valid_batch[0]["raw"],
            "max": valid_batch[0]["max"],
            "filename": [valid_batch[0]["filename"]]
        }
    
    try:
        raw_batch = torch.cat([item["raw"] for item in valid_batch])
        max_vals = [item["max"] for item in valid_batch]
        filenames = [item["filename"] for item in valid_batch]
        
        # Final shape check
        if raw_batch.shape[-1] < 8 or raw_batch.shape[-2] < 8:
            print(f"Warning: Collated submission batch too small: {raw_batch.shape}")
            return {
                "raw": torch.zeros((len(valid_batch), 4, 64, 64), dtype=torch.float32),
                "max": max_vals,
                "filename": filenames
            }
        
        return {"raw": raw_batch, "max": max_vals, "filename": filenames}
    except Exception as e:
        print(f"Error during submission collation: {str(e)}")
        return {
            "raw": torch.zeros((1, 4, 64, 64), dtype=torch.float32),
            "max": 1.0,
            "filename": ["collate_error.npz"]
        }


def get_submission_loader(submission_dir=Config.Submission_input, num_workers=10, batch_size=1):
    """
    Creates a data loader for submission data - no patches, no downsampling.
    
    Args:
        submission_dir: Directory containing submission files
        num_workers: Number of worker processes for data loading
        batch_size: Batch size (default 1 to process files individually)
        
    Returns:
        DataLoader for submission data
    """
    print(f"Creating submission dataset from {submission_dir}")
    submission_dataset = SubmissionDataset(submission_dir)
    
    # Force batch_size to 1 for safety with potentially inconsistent tensor shapes
    batch_size = 1
    
    submission_loader = DataLoader(
        submission_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=submission_collate,
        pin_memory=True,
    )
    
    return submission_loader


def get_data_loaders(
    train_data_dir=Config.train_dir, 
    val_data_dir=Config.val_dir, 
    submission_dir=Config.Submission_input,
    num_workers=10,
    Train_also=True,
    force_batch_size=None  # Added parameter to override Config batch size
):
    # Set safer batch sizes
    val_batch_size = 1 if force_batch_size is None else force_batch_size
    
    if Train_also:
        train_use_patches = Config.patches
        print(f"Creating training dataset with patches={train_use_patches}")
        train_dataset = LazyRAWDataset(train_data_dir, use_patches=train_use_patches)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=simple_collate,
            pin_memory=True,
        )

        print(f"Creating validation dataset with NO patches (full images), batch_size={val_batch_size}")
        val_dataset = LazyRAWDataset(val_data_dir, use_patches=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,  # Using safer batch size
            shuffle=False,
            num_workers=num_workers,
            collate_fn=simple_collate,
            pin_memory=True,
        )
        
        if submission_dir:
            print(f"Creating submission dataset from {submission_dir}")
            submission_dataset = SubmissionDataset(submission_dir)
            submission_loader = DataLoader(
                submission_dataset,
                batch_size=1,  # Always use batch_size=1 for submission
                shuffle=False,
                num_workers=num_workers,
                collate_fn=submission_collate,
                pin_memory=True,
            )

        return train_loader, val_loader, submission_loader
    else:
        print(f"Creating validation dataset with NO patches (full images), batch_size={val_batch_size}")
        val_dataset = LazyRAWDataset(val_data_dir, use_patches=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,  # Using safer batch size
            shuffle=False,
            num_workers=num_workers,
            collate_fn=simple_collate,
            pin_memory=True,
        )
        
        if submission_dir:
            print(f"Creating submission dataset from {submission_dir}")
            submission_dataset = SubmissionDataset(submission_dir)
            submission_loader = DataLoader(
                submission_dataset,
                batch_size=1,  # Always use batch_size=1 for submission
                shuffle=False,
                num_workers=num_workers,
                collate_fn=submission_collate,
                pin_memory=True,
            )
        return val_loader, submission_loader
        

def get_safe_data_loaders(
    train_data_dir=Config.train_dir, 
    val_data_dir=Config.val_dir, 
    submission_dir=Config.Submission_input,
    num_workers=0,
    Train_also=True
):
    """
    A safer version of get_data_loaders that forces batch_size=1
    for validation and submission to avoid shape/dimension issues.
    """
    return get_data_loaders(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        submission_dir=submission_dir,
        num_workers=num_workers,
        Train_also=Train_also,
        force_batch_size=1  # Force batch_size=1 for validation
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing data loaders with safe settings...")
    if Config.batch_size > 1:
        print(f"WARNING: Config batch_size is {Config.batch_size}, but using batch_size=1 for safety")
    
    # Add debug mode to print shapes
    debug = False
    if debug:
        # Test a single file to check output shapes
        test_dataset = LazyRAWDataset("data/val", use_patches=False)
        sample = test_dataset[0]
        print("Test sample shapes:")
        print(f"LR shape: {sample['lr'].shape}")
        print(f"HR shape: {sample['hr'].shape}")
    
    train_loader, val_loader, submission_loader = get_safe_data_loaders()

    print("\nTesting train loader:")
    for batch in train_loader:
        lr = batch["lr"]
        hr = batch["hr"]
        print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
        print(f"Filenames: {batch['filename'][:2]}...")
        print(f"HR max value: {batch['max']}")
        break

    if val_loader:
        print("\nTesting validation loader:")
        for batch in val_loader:
            lr = batch["lr"]
            hr = batch["hr"]
            print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
            print(f"Filenames: {batch['filename'][:2]}...")
            print(f"Max value: {batch['max']}")
            break
            
    if submission_loader:
        print("\nTesting submission loader:")
        for batch in submission_loader:
            raw = batch["raw"]
            print(f"Raw shape: {raw.shape}")
            print(f"Filenames: {batch['filename']}")
            print(f"Max value: {batch['max']}")
            break