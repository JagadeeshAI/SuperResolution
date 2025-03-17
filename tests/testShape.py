import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F

def downsample_raw(raw):
    """
    Downsamples a 4-channel packed RAW image by a factor of 2.
    The input raw should be a [H/2, W/2, 4] tensor -- with respect to its mosaiced version [H,w]
    Output is a [H/4, W/4, 4] tensor, preserving the RGGB pattern.
    """
    if len(raw.shape) == 3:  
        raw = raw.permute(2, 0, 1).unsqueeze(0)  
    elif len(raw.shape) == 4 and raw.shape[1] == 4:  
        pass
    else:
        raise ValueError(f"Unexpected shape for raw: {raw.shape}")
        
    downsampled_image = F.avg_pool2d(raw, kernel_size=2, stride=2, padding=0)
    
    if len(raw.shape) == 4 and raw.shape[0] == 1:
        downsampled_image = downsampled_image.squeeze(0).permute(1, 2, 0)
        
    return downsampled_image

if __name__ == "__main__":
    # Load the raw data
    raw = np.load("data/train/500.npz")
    print("The shape of the input is ")
    
    raw_img = raw["raw"]
    print(raw_img.shape)
    
    # Convert NumPy array to PyTorch tensor with float32 type
    # raw_img_tensor = torch.from_numpy(raw_img.astype(np.float32))
    
    # Downsample the raw image
    # downsampled_img = downsample_raw(raw_img_tensor)
    
    # Convert back to NumPy array
    # raw_img = downsampled_img.detach().cpu().numpy()
    
    raw_max = raw["max_val"]
    raw_img = (raw_img / raw_max).astype(np.float32)
    print("This is the max value", raw_max)
    
    # Create a directory for saving if it doesn't exist
    output_dir = "temp_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed raw image for visualization
    # if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
        # Method 1: Simple RGGB to RGB conversion (average the two greens)
    r = raw_img[:, :, 0]
    g = (raw_img[:, :, 1] + raw_img[:, :, 2]) / 2  # Average the two green channels
    b = raw_img[:, :, 3]
    rgb_img = np.stack([r, g, b], axis=2)
        
        # Save as RGB image
    rgb_img_uint8 = (rgb_img * 255).astype(np.uint8)
    img = Image.fromarray(rgb_img_uint8)
    img.save(os.path.join(output_dir, "train_500_hr.png"))
    print(f"RGB image saved to {os.path.join(output_dir, 'temp_rgb_image.png')}")
        
        # # Method 2: Save each channel separately for more detailed analysis
        # for i, channel_name in enumerate(['R', 'G1', 'G2', 'B']):
        #     channel_img = raw_img[:, :, i]
        #     channel_img_uint8 = (channel_img * 255).astype(np.uint8)
        #     channel_pil = Image.fromarray(channel_img_uint8)
        #     channel_path = os.path.join(output_dir, f"temp_{channel_name}_channel.png")
        #     channel_pil.save(channel_path)
        #     print(f"{channel_name} channel saved to {channel_path}")
    # else:
    #     # If it's not a 4-channel image, just save what we have
    #     print(f"Warning: Expected 4 channels but got shape {raw_img.shape}")
    #     # Try to save in whatever format is available
    #     if len(raw_img.shape) == 2 or (len(raw_img.shape) == 3 and raw_img.shape[2] in [1, 3]):
    #         img_uint8 = (raw_img * 255).astype(np.uint8)
    #         img = Image.fromarray(img_uint8)
    #         img.save(os.path.join(output_dir, "temp_image.png"))
    #         print(f"Image saved to {os.path.join(output_dir, 'temp_image.png')}")
    #     else:
    #         print(f"Could not save image with shape {raw_img.shape}")