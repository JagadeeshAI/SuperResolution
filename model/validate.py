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


from config import Config
from utils.util import (
    save_checkpoint,
    load_checkpoint,
    define_Model,
    update_last_epoch,
    validate,
    calculate_psnr,
)
from data.loaderPatches import get_data_loaders
from model.Restromer import Restormer


def crop_img(image, base=64):
    """
    Crops a 4D tensor [B, C, H, W] to make H and W multiples of base.
    Returns tensor in the same format.
    """
    b, c, h, w = image.shape
    crop_h = h % base
    crop_w = w % base

    # Calculate start and end indices
    h_start = crop_h // 2
    h_end = h - crop_h + crop_h // 2
    w_start = crop_w // 2
    w_end = w - crop_w + crop_w // 2

    # Crop while maintaining batch and channel dimensions
    return image[:, :, h_start:h_end, w_start:w_end]


criterion = nn.L1Loss()


def validate():
    model = Restormer().to(Config.device).eval()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr, weight_decay=Config.lr_decay
    )
    start_epoch, best_val_loss, best_psnr = load_checkpoint(
        model, optimizer, Config.device
    )
    val_loss, psnr_values = [], []
    train_loader, val_loader = get_data_loaders()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            lr_raw = batch["lr"].to(Config.device)
            hr_raw = batch["hr"].to(Config.device)

            lr_raw = crop_img(lr_raw, base=16)
            output = model(lr_raw)
            output = F.interpolate(
                output, size=hr_raw.shape[2:], mode="bilinear", align_corners=False
            )
            loss = criterion(hr_raw, output)
            val_loss.append(loss.item())
            psnr = calculate_psnr(output, hr_raw)
            psnr_values.append(psnr)

        avg_val_loss, avg_psnr = np.mean(val_loss), np.mean(psnr_values)

    print(f"The avergae PSNR is {avg_psnr}")
    return avg_val_loss, avg_psnr


if __name__ == "__main__":
    validate()
