import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.unet import UNet
from config import Config
from model.Restromer import Restormer
from utils.util import define_Model, load_checkpoint

# Submission_input
# Restomer_results_fnn_1_25


for i in range(40):
    if i==0:
        continue
    raw=np.load(f"data/Restomer_results_fnn_1_25/{i}.npz")
    max_val=raw['raw']
    print(f"max of {i}.npz is {raw}")