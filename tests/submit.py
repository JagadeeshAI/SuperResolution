import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import interpolate
import numpy as np


from model.unet import UNet
from config import Config
from model.Restromer import Restormer
from ASID.components.ASID import ASID
from utils.util import (
    save_checkpoint,
    load_checkpoint,
    random_sliding_crop,
    params,
    define_Model,
    update_last_epoch,
    validate,
)
from data.loader import get_data_loaders


# crop an image to the multiple of base
def crop_img(image, base=64):
    """
    Crops a 4D tensor [B, C, H, W] to make H and W multiples of base.
    Returns tensor in the same format.
    """
    b, c, h, w = image.shape
    crop_h = h % base
    crop_w = w % base
    
    h_start = crop_h // 2
    h_end = h - crop_h + crop_h // 2
    w_start = crop_w // 2
    w_end = w - crop_w + crop_w // 2
    
    return image[:, :, h_start:h_end, w_start:w_end]


def generate_submissions():
    model = Restormer().to(Config.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr, weight_decay=Config.lr_decay
    )

    if Config.loss == "mse":
        criterion = nn.MSELoss()
    elif Config.loss == "l1":
        criterion = nn.L1Loss()

    _, val_loader = get_data_loaders()

    load_checkpoint(model, optimizer, Config.device)

    model = model.float()  

    with torch.no_grad():
        for idx, batch in enumerate(
            tqdm(val_loader, desc="Processing validation files")
        ):
            # Initialize hr_raw as None
            hr_raw = None
            
            if (
                isinstance(batch, dict)
                and "lr" in batch
                and len(batch["lr"].shape) == 3
            ):
                lr_raw = batch["lr"].unsqueeze(0).to(Config.device)
                if "hr" in batch:
                    hr_raw = batch["hr"].unsqueeze(0).to(Config.device)
                filename = batch["filename"]
                lr_max = batch["max"] if "max" in batch else 1.0

            else:
                lr_raw = batch["lr"].to(Config.device)
                if "hr" in batch:
                    hr_raw = batch["hr"].to(Config.device)
                filename = (
                    batch["filename"][0]
                    if isinstance(batch["filename"], list)
                    else batch["filename"]
                )
                lr_max = (
                    batch["max"][0]
                    if isinstance(batch["max"], list) and "max" in batch
                    else batch.get("max", 1.0)
                )

            lr_raw = crop_img(lr_raw, base=16)
            
            # Calculate expected 2X dimensions
            expected_h = lr_raw.shape[2] * 2
            expected_w = lr_raw.shape[3] * 2

            # Process through model
            sr_output = model(lr_raw)
            
            # Check if model output is already at 2X scale
            if sr_output.shape[2] != expected_h or sr_output.shape[3] != expected_w:
                print(f"Model output shape {sr_output.shape[2:]} doesn't match 2X scale {(expected_h, expected_w)}. Applying interpolation.")
                # Apply interpolation to get exactly 2X scale
                sr_output = F.interpolate(
                    sr_output, size=(expected_h, expected_w), mode="bilinear", align_corners=False
                )

            sr_output = sr_output.squeeze(0).permute(1, 2, 0)

            raw_img = (sr_output.cpu().numpy() * float(lr_max)).astype(np.uint16)

            if isinstance(filename, str):
                output_filename = os.path.splitext(filename)[0]
            else:
                output_filename = f"{idx}"

            output_path = os.path.join(
                Config.submission_save_dir, f"{output_filename}.npz"
            )
            
            # Make sure submission directory exists
            os.makedirs(Config.submission_save_dir, exist_ok=True)

            np.savez(output_path, raw=raw_img, max_val=lr_max)

    print(f"All files processed and saved to {Config.submission_save_dir}")


if __name__ == "__main__":
    generate_submissions()