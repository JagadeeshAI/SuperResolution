import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import interpolate
from model.Restromer import Restormer

from config import Config
from utils.util import (
    save_checkpoint,
    load_checkpoint,
    define_Model,
    update_last_epoch,
    validate,
    calculate_psnr,
    crop_img,
)
from data.loader import get_data_loaders
from model.unet import UNet

from basicsr.archs.mambairv2_arch import MambaIRv2


criterion = nn.L1Loss()


def validateNow():
    
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
    start_epoch, best_val_loss, best_psnr = load_checkpoint(
        model, optimizer, Config.device
    )

    val_loss, psnr_values = [], []

    val_loader, _ = get_data_loaders(Train_also=False)

    avg_val_loss, avg_psnr = validate(model, val_loader, criterion)

    print(f"The avergae PSNR is {avg_psnr}")
    return avg_val_loss, avg_psnr


if __name__ == "__main__":
    validateNow()
