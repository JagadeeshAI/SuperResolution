import torch
import torch.nn as nn
import os


class Config:
    lr = 1e-6
    lr_decay = 1e-6
    epochs = 1000
    batch_size = 1
    num_workers = 0
    loss = "l1"
    out_dir = "./results/restormer_l1"
    log_file = os.path.join(out_dir, "logs.json")
    train_dir = "data/train"
    val_dir = "data/val"
    patch_size = 128
    Submission_input = "data/Submission_input"
    submission_save_dir = "data/Restomer_results"
    train_batch_size = 1
    val_batch_size = 1
    RESUME = True
    model = "restormer"
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patches = True

print(f"Using device: {Config.device}")
