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
                            
                            for j in range(4):

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
                hr_patches,_ = extract_patches(hr_img, patch_size=512)
                hr_patch = hr_patches[patch_idx]

                hr_patch_for_down = hr_patch.unsqueeze(0)  
                lr_patch = downsample_raw(hr_patch_for_down)
                
                lr_patch = lr_patch.permute(2, 0, 1).unsqueeze(0)
                hr_patch = hr_patch.unsqueeze(0)

                return {
                    "lr": lr_patch,
                    "hr": hr_patch,
                    "max": max_val,
                    "filename": os.path.basename(path),
                }
            else:
                lr_img = downsample_raw(hr_img)
                lr_img = lr_img.permute(2, 0, 1).unsqueeze(0)

                return {
                    "lr": lr_img,
                    "hr": hr_img,
                    "max": max_val,
                    "filename": os.path.basename(path),
                }
        except Exception as e:
            print(f"Skipping {path}: {str(e)}")
            dummy_lr = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
            dummy_hr = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
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
            "lr": torch.zeros((1, 4, 32, 32), dtype=torch.float32),
            "hr": torch.zeros((1, 4, 64, 64), dtype=torch.float32),
            "max": 1.0,
            "filename": ["dummy.npz"]
        }
    
    lr_batch = torch.cat([item["lr"] for item in valid_batch])
    hr_batch = torch.cat([item["hr"] for item in valid_batch])
    max_vals = [item["max"] for item in valid_batch]
    filenames = [item["filename"] for item in valid_batch]

    return {"lr": lr_batch, "hr": hr_batch, "max": max_vals[0], "filename": filenames}


def submission_collate(batch):
    """Simple collate function for submission data"""
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        return {
            "raw": torch.zeros((1, 4, 64, 64), dtype=torch.float32),
            "max": 1.0,
            "filename": ["dummy.npz"]
        }
    
    raw_batch = torch.cat([item["raw"] for item in valid_batch])
    max_vals = [item["max"] for item in valid_batch]
    filenames = [item["filename"] for item in valid_batch]

    return {"raw": raw_batch, "max": max_vals, "filename": filenames}


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
    num_workers=0,
    Train_also=True
):
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

        print("Creating validation dataset with NO patches (full images)")
        val_dataset = LazyRAWDataset(val_data_dir, use_patches=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.val_batch_size,
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
                batch_size=1,  
                shuffle=False,
                num_workers=num_workers,
                collate_fn=submission_collate,
                pin_memory=True,
            )

        return train_loader, val_loader, submission_loader
    else:
        print("Creating validation dataset with NO patches (full images)")
        val_dataset = LazyRAWDataset(val_data_dir, use_patches=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.val_batch_size,
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
                batch_size=1,  
                shuffle=False,
                num_workers=num_workers,
                collate_fn=submission_collate,
                pin_memory=True,
            )
        return val_loader ,submission_loader
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing data loaders...")
    train_loader, val_loader, submission_loader = get_data_loaders()

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