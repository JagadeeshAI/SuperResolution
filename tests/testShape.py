import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from model.unet import UNet
from data.loaderPatches import get_data_loaders
from config import Config


def printShapes():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    _, val_loader = get_data_loaders(
        Config.train_data_dir,
        Config.train_gt_dir,
        Config.val_data_dir,
        Config.val_gt_dir,
        Config.num_workers,
    )

    if val_loader:
        for idx, batch in enumerate(
            tqdm(val_loader, desc="Processing validation files")
        ):
            # Add batch dimension if using custom_collate
            if (
                isinstance(batch, dict)
                and "lr" in batch
                and len(batch["lr"].shape) == 3
            ):
                lr_raw = batch["lr"].unsqueeze(0).to(device)
                filename = batch["filename"]
                # Get the max value from the batch
                lr_max = batch["lr_max"]
            else:
                lr_raw = batch["lr"].to(device)
                filename = (
                    batch["filename"][0]
                    if isinstance(batch["filename"], list)
                    else batch["filename"]
                )
                # Get the max value from the batch
                lr_max = (
                    batch["lr_max"][0]
                    if isinstance(batch["lr_max"], list)
                    else batch["lr_max"]
                )

            print("The index of the image is", idx, "and lr shape is ", lr.shape)


if __name__ == "__main__":
    # printShapes()
    import numpy as np

    # /home/jagadeesh/Downloads/Restomer_results_Light_Weight_65/Restomer_results/1.npz

    raw = np.load("data/Restomer_results/1.npz")
    print("The shape of the input is ")
    raw_img = raw["raw"]
    print(raw_img.shape)
    raw_max = raw["max_val"]
    raw_img = (raw_img / raw_max).astype(np.float32)
    print("This is the max value", raw_max)
    # print(raw_img)
