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
from utils.util import (
    load_checkpoint,
    define_Model
)
from data.loader import get_data_loaders


# def crop_img(image, base=64):
#     """
#     Crops a 4D tensor [B, C, H, W] to make H and W multiples of base.
#     Returns tensor in the same format.
#     """
#     b, c, h, w = image.shape
#     crop_h = h % base
#     crop_w = w % base

#     h_start = crop_h // 2
#     h_end = h - crop_h + crop_h // 2
#     w_start = crop_w // 2
#     w_end = w - crop_w + crop_w // 2

#     return image[:, :, h_start:h_end, w_start:w_end]


def generate_submissions():
    model = define_Model()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr, weight_decay=Config.lr_decay
    )

    
    _, _, sub_loader = get_data_loaders()

    load_checkpoint(model, optimizer, Config.device)

    model = model.float()

    with torch.no_grad():
        for idx, batch in enumerate(
            tqdm(sub_loader, desc="Processing validation files")
        ):
            input_raw = batch["raw"].to(Config.device)
            max = batch["max"][0]
            filename = batch["filename"][0]

            sr_output = model(input_raw)

            sr_output = sr_output.squeeze(0).permute(1, 2, 0)
            
            raw_img = (sr_output.cpu().numpy() * float(max)).astype(np.uint16)

            if isinstance(filename, str):
                output_filename = os.path.splitext(filename)[0]
            else:
                output_filename = f"{idx}"

            output_path = os.path.join(
                Config.submission_save_dir, f"{output_filename}.npz"
            )

            os.makedirs(Config.submission_save_dir, exist_ok=True)

            np.savez(output_path, raw=raw_img, max_val=max)

    print(f"All files processed and saved to {Config.submission_save_dir}")


if __name__ == "__main__":
    generate_submissions()