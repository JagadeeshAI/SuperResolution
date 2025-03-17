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

def generate_submissions():
    model = define_Model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr, weight_decay=Config.lr_decay
    )
    
    _, sub_loader = get_data_loaders(val_only=True)
    load_checkpoint(model, optimizer, Config.device)
    model = model.float()
    model.eval()  # Ensure model is in evaluation mode
    
    # Track statistics for reporting
    stats = []
    
    with torch.no_grad():
        for idx, batch in enumerate(
            tqdm(sub_loader, desc="Processing files for submission")
        ):
            input_raw = batch["raw"].to(Config.device)
            max_val = batch["max"][0]
            filename = batch["filename"][0]
            
            # Load original data to get raw_max
            if isinstance(filename, str):
                original_path = os.path.join(Config.Submission_input, filename)
                try:
                    original_data = np.load(original_path)
                    original_raw = original_data["raw"]
                    original_raw_max = np.max(original_raw)
                except Exception as e:
                    print(f"Warning: Could not load original file to get raw_max: {e}")
                    original_raw_max = None
            else:
                original_raw_max = None
            
            sr_output = model(input_raw)
            
            sr_output = sr_output.squeeze(0).permute(1, 2, 0)
            
            sr_output = torch.clamp(sr_output, 0.0, 1.0)
            
            sr_numpy = sr_output.cpu().numpy() * float(max_val)
            
            raw_img = sr_numpy.astype(np.uint16)
            
            output_raw_max = np.max(raw_img)
            
            if isinstance(filename, str):
                output_filename = os.path.splitext(filename)[0]
            else:
                output_filename = f"{idx}"
            
            output_path = os.path.join(
                Config.submission_save_dir, f"{output_filename}.npz"
            )
            os.makedirs(Config.submission_save_dir, exist_ok=True)
            
            np.savez(output_path, raw=raw_img, max_val=max_val)
            
            # Track stats
            stats.append({
                "file": output_filename,
                "max_val": float(max_val),
                "original_raw_max": float(original_raw_max) if original_raw_max is not None else None,
                "output_raw_max": float(output_raw_max)
            })
            
                
    # Save statistics
    stats_path = os.path.join(Config.submission_save_dir, "processing_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"All files processed and saved to {Config.submission_save_dir}")
    print(f"Statistics saved to {stats_path}")

if __name__ == "__main__":
    generate_submissions()