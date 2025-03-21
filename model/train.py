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
    crop_img,
)
from data.loader import get_data_loaders


wandb.init(project="RAW-SuperResolution", name="UNet-MSE-Training")


def train():
    """Train the UNet model for RAW Super-Resolution."""
    os.makedirs(Config.out_dir, exist_ok=True)

    model = define_Model()
    model.gradient_checkpointing = True
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

            optimizer.zero_grad()

            output = model(crop_img(lr_raw))

            if output.shape != hr_raw.shape:
                output = interpolate(
                    output,
                    size=(hr_raw.shape[2], hr_raw.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(output, hr_raw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss.append(loss.item())

        epoch_loss = np.mean(train_loss)

        avg_val_loss, avg_psnr = validate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{Config.epochs} - Train Loss: {epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}, PSNR: {avg_psnr:.2f} dB and the curren learning rate is {current_lr}"
        )

        update_last_epoch(epoch)

        wandb.log(
            {
                "Epoch": epoch + 1,
                "Train Loss": epoch_loss,
                "Val Loss": avg_val_loss,
                "PSNR": avg_psnr,
                "Learning Rate": current_lr,
            }
        )

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
