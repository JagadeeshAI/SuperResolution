import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from config import Config
from utils.util import downsample_raw


class LazyRAWDataset(Dataset):
    def __init__(self, data_dir, is_val=False):
        self.data_paths = sorted(glob(os.path.join(data_dir, "*.npz")))
        print(f"Found {len(self.data_paths)} samples")
        self.sample_map = list(range(len(self.data_paths)))
        self.is_val = is_val
        print(f"Dataset initialized with is_val={self.is_val}")

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        try:
            path = self.data_paths[idx]

            hr_data = np.load(path)
            hr_img = hr_data["raw"].astype(np.float32)
            max_val = hr_data["max_val"]

            # Normalize
            hr_img = hr_img / max_val

            # Initial shaping
            hr_img = np.expand_dims(hr_img, axis=0)
            hr_img = np.transpose(hr_img, (0, 3, 1, 2))  # [B, C, H, W]
            hr_img = torch.from_numpy(hr_img)

            # For training path
            if not self.is_val:
                hr_img = hr_img.permute(0, 2, 1, 3)  # [B, H, C, W] for training
                lr_img = downsample_raw(hr_img)

                # Handle dimension variations
                if lr_img.dim() == 3:
                    lr_img = lr_img.permute(2, 0, 1).unsqueeze(0)
                elif lr_img.dim() == 4:
                    if lr_img.shape[1] == 3 or lr_img.shape[1] == 4:
                        # Channels already in correct position
                        pass
                    else:
                        lr_img = lr_img.permute(0, 3, 1, 2)
            # For validation path
            else:
                # Keep original format for validation
                lr_img = downsample_raw(hr_img)

                # Only permute if needed
                if not (lr_img.shape[1] == 3 or lr_img.shape[1] == 4):
                    if lr_img.dim() == 3:
                        lr_img = lr_img.permute(2, 0, 1).unsqueeze(0)
                    elif lr_img.dim() == 4:
                        lr_img = lr_img.permute(0, 3, 1, 2)

            # Make sure final tensors are in BCHW format
            # If channels are not in dimension 1, fix
            if hr_img.shape[1] != 3 and hr_img.shape[1] != 4:
                if hr_img.shape[-1] == 3 or hr_img.shape[-1] == 4:
                    hr_img = hr_img.permute(0, 3, 1, 2)
                elif hr_img.shape[2] == 3 or hr_img.shape[2] == 4:
                    hr_img = hr_img.permute(0, 2, 3, 1)

            if lr_img.shape[1] != 3 and lr_img.shape[1] != 4:
                if lr_img.shape[-1] == 3 or lr_img.shape[-1] == 4:
                    lr_img = lr_img.permute(0, 3, 1, 2)
                elif lr_img.shape[2] == 3 or lr_img.shape[2] == 4:
                    lr_img = lr_img.permute(0, 2, 3, 1)

            return {
                "lr": lr_img,
                "hr": hr_img,
                "max": max_val,
                "filename": os.path.basename(path),
            }
        except Exception as e:
            print(
                f"Skipping {path if 'path' in locals() else 'unknown path'}: {str(e)}"
            )


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
            "lr": torch.zeros((1, 4, 64, 64), dtype=torch.float32),
            "hr": torch.zeros((1, 4, 128, 128), dtype=torch.float32),
            "max": 1.0,
            "filename": ["dummy.npz"],
        }

    # For validation/submission, always return the first item (batch_size=1)
    # This bypasses any concatenation issues with inconsistent tensor shapes
    return {
        "lr": valid_batch[0]["lr"],
        "hr": valid_batch[0]["hr"],
        "max": valid_batch[0]["max"],
        "filename": [valid_batch[0]["filename"]],
    }


def submission_collate(batch):
    """Simple collate function for submission data"""
    valid_batch = [item for item in batch if item is not None]

    if not valid_batch:
        return {
            "raw": torch.zeros((1, 4, 64, 64), dtype=torch.float32),
            "max": 1.0,
            "filename": ["dummy.npz"],
        }

    # Always return just the first item for consistency
    return {
        "raw": valid_batch[0]["raw"],
        "max": valid_batch[0]["max"],
        "filename": [valid_batch[0]["filename"]],
    }


def get_submission_loader(
    submission_dir=Config.Submission_input, num_workers=10, batch_size=1
):
    """Creates a data loader for submission data"""
    print(f"Creating submission dataset from {submission_dir}")
    submission_dataset = SubmissionDataset(submission_dir)

    # Force batch_size to 1 for safety
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
    num_workers=0,
    Train_also=True,
    force_batch_size=None,
):
    # Set batch sizes
    batch_size = Config.batch_size
    val_batch_size = 1

    if Train_also:
        print(f"Creating training dataset")
        train_dataset = LazyRAWDataset(train_data_dir, is_val=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=simple_collate,
            pin_memory=True,
        )

        print(f"Creating validation dataset, batch_size={val_batch_size}")
        val_dataset = LazyRAWDataset(val_data_dir, is_val=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
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
        print(f"Creating validation dataset, batch_size={val_batch_size}")
        val_dataset = LazyRAWDataset(val_data_dir, is_val=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
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
        return val_loader, submission_loader


def get_safe_data_loaders(
    train_data_dir=Config.train_dir,
    val_data_dir=Config.val_dir,
    submission_dir=Config.Submission_input,
    num_workers=0,
    Train_also=True,
):
    """Forces batch_size=1 for validation and submission"""
    return get_data_loaders(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        submission_dir=submission_dir,
        num_workers=num_workers,
        Train_also=Train_also,
        force_batch_size=1,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing data loaders with safe settings...")

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
