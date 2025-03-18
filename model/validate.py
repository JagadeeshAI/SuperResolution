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
    crop_img,
)
from data.loader import get_data_loaders
from model.unet import UNet

criterion = nn.L1Loss()


def validateNow():
    model = define_Model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr, weight_decay=Config.lr_decay
    )
    start_epoch, best_val_loss, best_psnr = load_checkpoint(
        model, optimizer, Config.device
    )
    
    val_loss, psnr_values = [], []
    
    val_loader,_ = get_data_loaders(Train_also=False)
    
    avg_val_loss, avg_psnr = validate(model, val_loader, criterion)

    print(f"The avergae PSNR is {avg_psnr}")
    return avg_val_loss, avg_psnr


if __name__ == "__main__":
    validateNow()
