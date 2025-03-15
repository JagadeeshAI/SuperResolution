import numpy as np
import cv2
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import math
import glob

from config import Config
from model.Restromer import Restormer
from model.unet import UNet


def save_checkpoint(
    model, optimizer, epoch, val_loss, psnr, best_val_loss, best_psnr, max_checkpoints=6
):

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(
        Config.out_dir,
        f"best_model_{timestamp}_epoch{epoch}_loss{val_loss:.4f}_PSNR{psnr:.2f}.pth",
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "psnr": psnr,
            "best_val_loss": best_val_loss,
            "best_psnr": best_psnr,
        },
        model_path,
    )

    with open(Config.log_file, "w") as f:
        json.dump(
            {
                "last_epoch": epoch,
                "best_model": model_path,
                "best_val_loss": best_val_loss,
                "best_psnr": best_psnr,
            },
            f,
        )

    # Delete oldest checkpoints if exceeding max_checkpoints
    checkpoint_files = glob.glob(os.path.join(Config.out_dir, "best_model_*.pth"))
    checkpoint_files.sort(key=os.path.getctime)  # Sort by creation time

    # If we have more than max_checkpoints files after adding the new one, delete oldest
    if len(checkpoint_files) > max_checkpoints:
        files_to_delete = checkpoint_files[: len(checkpoint_files) - max_checkpoints]
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted old checkpoint: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    print(f"Model saved: {model_path}")
    wandb.save(model_path)


def calculate_psnr(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    if mse < 1e-10:
        return float("inf")
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def load_checkpoint(model, optimizer, device):
    """Load checkpoint from log file or default checkpoint if available."""
    best_val_loss, best_psnr = float("inf"), float("-inf")

    if Config.RESUME == False:
        print("we are not resuming so , Starting training from scratch.")
        return 0, float("inf"), float("-inf")

    if os.path.exists(Config.log_file):
        with open(Config.log_file, "r") as f:
            logs = json.load(f)
            checkpoint_path = logs.get("best_model", None)
            best_val_loss = logs.get("best_val_loss", float("inf"))
            best_psnr = logs.get("best_psnr", float("-inf"))
            last_epoch = logs.get("last_epoch", 0)

            if checkpoint_path and os.path.exists(checkpoint_path):
                print(checkpoint_path)
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print(
                    f"Resuming training from epoch {last_epoch}, Best Loss: {best_val_loss:.4f}, Best PSNR: {best_psnr:.2f}"
                )
                return last_epoch, best_val_loss, best_psnr

            print("Checkpoint path from logs not found, trying default checkpoint.")

    default_checkpoint_path = os.path.join(Config.out_dir, "unet-raw-sr-best.pt")
    if os.path.exists(default_checkpoint_path):
        checkpoint = torch.load(
            default_checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded default checkpoint: {default_checkpoint_path}")
        return 0, float("inf"), float("-inf")

    print("No valid checkpoint found. Starting training from scratch.")
    return 0, float("inf"), float("-inf")


def define_Model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Config.model == "unet":
        print("we are using unet model")
        model = UNet().to(device)
    elif Config.model == "restormer":
        model = Restormer().to(device)
    if model:
        return model


def update_last_epoch(last_epoch):

    default_data = {
        "last_epoch": last_epoch,
        "best_model": "no valid checkpoint",
        "best_val_loss": float("inf"),
        "best_psnr": 0.0,
    }

    if not os.path.exists(Config.log_file):
        with open(Config.log_file, "w") as f:
            json.dump(default_data, f, indent=4)

    try:
        with open(Config.log_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        data = default_data

    data["last_epoch"] = last_epoch

    with open(Config.log_file, "w") as f:
        json.dump(data, f, indent=4)


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


def validate(model, val_loader, criterion):
    val_loss, psnr_values = [], []
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

    return avg_val_loss, avg_psnr
