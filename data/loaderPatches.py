import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from config import Config
from utils.util import pad_and_patch
from data.make_patches import extract_patches_overlapping, select_random_patches

def downsample_raw(raw):
    """
    Downsamples a 4-channel packed RAW image by a factor of 2.
    The input raw should be a [H/2, W/2, 4] tensor -- with respect to its mosaiced version [H,w]
    Output is a [H/4, W/4, 4] tensor, preserving the RGGB pattern.
    """
    downsampled_image = F.avg_pool2d(raw, kernel_size=2, stride=2, padding=0)
    downsampled_image = downsampled_image.squeeze(0).permute(1, 2, 0)
    return downsampled_image

class LazyRAWDataset(Dataset):
    def __init__(self, data_dir, use_patches=None):
        self.data_paths = sorted(glob(os.path.join(data_dir, "*.npz")))        
        print(f"Found {len(self.data_paths)} samples")
        
        # Use provided use_patches or fallback to Config.patches
        self.use_patches = Config.patches if use_patches is None else use_patches
        
        # Store metadata instead of preloading all data
        if self.use_patches:
            # For patch mode, we need to know how many patches each file will generate
            self.sample_map = []
            for i, path in enumerate(self.data_paths):
                try:
                    # Just peek at the file to get its shape
                    with np.load(path) as data:
                        img = data["raw"]
                        h, w = img.shape[0], img.shape[1]
                        stride = self.patch_size // 2  # 50% overlap
                        patches_h = (h - self.patch_size) // stride + 1
                        patches_w = (w - self.patch_size) // stride + 1
                        total_patches = patches_h * patches_w
                        for j in range(total_patches):  # Assuming 4 patches as in original code
                            self.sample_map.append((i, j))
                except Exception as e:
                    print(f"Error inspecting {path}: {str(e)}")
            print(f"Will create {len(self.sample_map)} LR-HR patch pairs on demand")
        else:
            # For full image mode, each file is one sample
            self.sample_map = list(range(len(self.data_paths)))
            print(f"Will process {len(self.sample_map)} full images on demand")
    
    def __len__(self):
        return len(self.sample_map)
    
    def __getitem__(self, idx):
        if self.use_patches:
            file_idx, patch_idx = self.sample_map[idx]
            path = self.data_paths[file_idx]
        else:
            path = self.data_paths[idx]
        
        try:
            # Load the data only when needed
            hr_data = np.load(path)
            hr_img = hr_data["raw"].astype(np.float32)
            max_val = hr_data["max_val"]
            
            hr_img = hr_img / max_val
            hr_img = np.expand_dims(hr_img, axis=0)
            hr_img = np.transpose(hr_img, (0, 3, 1, 2))
            hr_img = torch.from_numpy(hr_img)
            
            if self.use_patches:
                # Extract patches and select just the one we need for this index
                hr_patches = extract_patches_overlapping(hr_img, patch_size=128)
                # random_patches = select_random_patches(hr_patches, num_patches=4)
                # hr_patch = random_patches[patch_idx]
                
                lr_tensor = downsample_raw(hr_patches)
                
                lr_patch = lr_tensor.cpu()
                lr_patch = lr_patch.permute(2, 0, 1).unsqueeze(0)
                hr_patch = hr_patch.unsqueeze(0)
                
                return {
                    "lr": lr_patch,
                    "hr": hr_patch,
                    "max": max_val,
                    "filename": os.path.basename(path)
                }
            else:
                # Process the whole image
                lr_img = downsample_raw(hr_img)
                lr_img = lr_img.permute(2, 0, 1).unsqueeze(0)
                
                return {
                    "lr": lr_img,
                    "hr": hr_img,
                    "max": max_val,
                    "filename": os.path.basename(path)
                }
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            # Return an empty item in case of error
            return {
                "lr": torch.zeros((1, 4, 32, 32)),
                "hr": torch.zeros((1, 4, 64, 64)),
                "max": 1.0,
                "filename": os.path.basename(path)
            }

def simple_collate(batch):
    """Stack tensors into batches"""
    lr_batch = torch.cat([item["lr"] for item in batch])
    hr_batch = torch.cat([item["hr"] for item in batch])
    max_vals = [item["max"] for item in batch]
    filenames = [item["filename"] for item in batch]
    
    return {
        "lr": lr_batch,
        "hr": hr_batch,
        "max": max_vals[0],
        "filename": filenames
    }

def get_data_loaders(train_data_dir=Config.train_dir, val_data_dir=Config.val_dir, num_workers=0):
    # For training data, use Config.patches
    train_use_patches = Config.patches
    print(f"Creating training dataset with patches={train_use_patches}")
    train_dataset = LazyRAWDataset(train_data_dir, use_patches=train_use_patches)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=simple_collate,
        pin_memory=True
    )
    
    if val_data_dir:
        # Force validation to always use full images (no patches)
        print("Creating validation dataset with NO patches (full images)")
        val_dataset = LazyRAWDataset(val_data_dir, use_patches=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=simple_collate,
            pin_memory=True
        )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Print device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_data_dir = Config.train_dir
    val_data_dir = Config.val_dir
    
    print("Initializing data loaders...")
    train_loader, val_loader = get_data_loaders(
        train_data_dir,
        val_data_dir,
    )
    
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