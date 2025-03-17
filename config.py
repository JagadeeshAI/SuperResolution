import torch
import torch.nn as nn
import os

class Config:
    lr = 1e-6
    lr_decay = 1e-6
    epochs = 1000
    batch_size = 2
    loss = "l1"
    out_dir = "./results/restromer_l1"
    log_file = os.path.join(out_dir, "logs.json")
    train_dir = "data/train"
    val_dir = "data/val"
    Submission_input = "data/Submission_input"
    submission_save_dir = "data/Restomer_results"
    RESUME = True
    model = "restormer"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patches = True
    useHalf=False
    train_batch_size=4

print(f"Using device: {Config.device}")
