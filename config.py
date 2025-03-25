import torch
import torch.nn as nn
import os

class Config:
    lr = 1e-4
    lr_decay = 1e-6
    epochs = 1000
    loss = "l1"
    batch_size = 16
    out_dir = "./results/mambair"
    log_file = os.path.join(out_dir, "logs.json")
    train_dir = "data/trainPatches"
    val_dir = "data/val"
    Submission_input = "data/Submission_input"
    submission_save_dir = "data/Restomer_results"
    RESUME = True
    model = "restormer"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {Config.device}")
