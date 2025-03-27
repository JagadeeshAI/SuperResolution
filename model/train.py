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

from basicsr.archs.mambairv2_arch import MambaIRv2

from config import Config
from utils.util import (
    save_checkpoint,
    load_checkpoint,
    define_Model,
    update_last_epoch,
    validate,
    crop_img,
)
from data.loader import get_data_loaders


# wandb.init(project="RAW-SuperResolution", name="UNet-MSE-Training")


def train():
    """Train the UNet model for RAW Super-Resolution."""
    os.makedirs(Config.out_dir, exist_ok=True)

    model = MambaIRv2(
        upscale=2,
        img_size=128,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=1.,
        upsampler='pixelshuffledirect',
        in_chans=4 
    ).to(Config.device) 

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr, weight_decay=Config.lr_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=1e-6
    )

    if Config.loss == "mse":
        criterion = nn.MSELoss()
    elif Config.loss == "l1":
        criterion = nn.L1Loss()

    train_loader, val_loader, _ = get_data_loaders()

    start_epoch, best_val_loss, best_psnr = load_checkpoint(
        model, optimizer, Config.device
    )

    for epoch in range(start_epoch, Config.epochs):
        model.train()
        model = model.float()
        train_loss = []
        current_lr = optimizer.param_groups[0]["lr"]
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}"):
            lr_raw = batch["lr"].to(Config.device).float()
            hr_raw = batch["hr"].to(Config.device).float()
            
            lr_raw = lr_raw.squeeze(1)
            hr_raw = hr_raw.squeeze(1)

            optimizer.zero_grad()

            output = model(lr_raw)

            loss = criterion(output, hr_raw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss.append(loss.item())

        epoch_loss = np.mean(train_loss)

        avg_val_loss, avg_psnr = validate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{Config.epochs} - Train Loss: {epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}, PSNR: {avg_psnr:.2f} dB "
        )

        update_last_epoch(epoch)

        # wandb.log(
        #     {
        #         "Epoch": epoch + 1,
        #         "Train Loss": epoch_loss,
        #         "Val Loss": avg_val_loss,
        #         "PSNR": avg_psnr,
        #         "Learning Rate": current_lr,
        #     }
        # )

        scheduler.step()

        # Save model if validation loss improves OR PSNR improves
        if avg_val_loss < best_val_loss or avg_psnr > best_psnr:
            best_val_loss = min(best_val_loss, avg_val_loss)
            best_psnr = max(best_psnr, avg_psnr)
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                avg_val_loss,
                avg_psnr,
                best_val_loss,
                best_psnr,
            )


if __name__ == "__main__":
    train()
