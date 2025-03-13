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


def extract_padded_patches(tensor, patch_size):
    """
    Divides the input tensor into patches with proper padding for edge cases.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
        patch_size (int): Size of each square patch

    Returns:
        patches: List of patches, each of shape (B, C, patch_size, patch_size)
        patches_info: List of (y, x, padded_h, padded_w) tuples indicating patch position and padding
    """
    B, C, H, W = tensor.shape
    patches = []
    patches_info = []

    # Calculate number of full patches
    h_patches = H // patch_size
    w_patches = W // patch_size

    # Process full patches and edge patches
    for i in range(h_patches + (1 if H % patch_size != 0 else 0)):
        for j in range(w_patches + (1 if W % patch_size != 0 else 0)):
            # Calculate patch coordinates
            y_start = i * patch_size
            x_start = j * patch_size

            # Handle edge cases
            if y_start + patch_size > H or x_start + patch_size > W:
                # This is an edge patch that needs padding
                y_end = min(y_start + patch_size, H)
                x_end = min(x_start + patch_size, W)

                # Extract patch
                patch = tensor[:, :, y_start:y_end, x_start:x_end]

                # Calculate padding
                pad_h = patch_size - (y_end - y_start)
                pad_w = patch_size - (x_end - x_start)

                # Apply padding if needed
                if pad_h > 0 or pad_w > 0:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h))

                patches.append(patch)
                patches_info.append((y_start, x_start, pad_h, pad_w))
            else:
                # This is a full patch
                patch = tensor[
                    :, :, y_start : y_start + patch_size, x_start : x_start + patch_size
                ]
                patches.append(patch)
                patches_info.append((y_start, x_start, 0, 0))

    return patches, patches_info


def merge_patches(patches, patches_info, original_shape, patch_size):
    """
    Reconstruct the original tensor from patches, properly handling padded regions.

    Args:
        patches: List of tensors of shape (B, C, output_h, output_w)
        patches_info: List of (y, x, padded_h, padded_w) tuples from extract_padded_patches
        original_shape: Shape of the original tensor (B, C, H, W)
        patch_size: Size of each patch before processing

    Returns:
        torch.Tensor: Reconstructed tensor with upscaled dimensions
    """
    B, C, H, W = original_shape

    # Determine scale factor from first patch
    output_h, output_w = patches[0].shape[2:4]
    scale_factor = output_h / patch_size

    # Calculate new dimensions
    new_h = int(H * scale_factor)
    new_w = int(W * scale_factor)

    # Initialize output tensor
    output = torch.zeros((B, C, new_h, new_w), device=patches[0].device)

    # Create a weight tensor to handle overlapping regions
    weight = torch.zeros((B, 1, new_h, new_w), device=patches[0].device)

    # Place each patch in the output tensor
    for patch, (y, x, pad_h, pad_w) in zip(patches, patches_info):
        # Scale coordinates
        y_scaled = int(y * scale_factor)
        x_scaled = int(x * scale_factor)

        # Calculate effective patch size after removing padding
        effective_h = output_h - int(pad_h * scale_factor)
        effective_w = output_w - int(pad_w * scale_factor)

        # Add patch to output, excluding padded regions
        output[
            :, :, y_scaled : y_scaled + effective_h, x_scaled : x_scaled + effective_w
        ] += patch[:, :, :effective_h, :effective_w]

        # Update weight tensor
        weight[
            :, :, y_scaled : y_scaled + effective_h, x_scaled : x_scaled + effective_w
        ] += 1.0

    # Average overlapping regions
    weight = torch.clamp(weight, min=1.0)  # Avoid division by zero
    output = output / weight

    return output


def process_single_image(model, image_path, patch_size=128):
    """
    Process a single image using a patch-based approach with proper padding.

    Args:
        model: Model to use for inference
        image_path: Path to the image (.npz file)
        patch_size: Size of patches to extract

    Returns:
        lr_img: Original low-resolution image
        hr_img: Ground truth high-resolution image
        sr_img: Super-resolved image
    """
    device = next(model.parameters()).device

    try:
        # Load the image
        print(f"Loading image: {image_path}")
        data = np.load(image_path)
        raw_img = data["raw"].astype(np.float32)
        max_val = data["max_val"]

        print(f"Image max_val: {max_val}")

        # Normalize the image using max_val
        raw_img = raw_img / max_val

        # Preprocess the image
        raw_img = np.expand_dims(raw_img, axis=0)  # Add batch dimension
        raw_img = np.transpose(raw_img, (0, 3, 1, 2))  # [B, C, H, W]
        hr_img = torch.from_numpy(raw_img)

        # Create LR image (assuming downsampling is part of your pipeline)
        lr_img = F.interpolate(
            hr_img, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        # Store original shape
        original_shape = lr_img.shape

        # Extract patches with padding for edge cases
        patches, patches_info = extract_padded_patches(lr_img, patch_size)
        print(f"Extracted {len(patches)} patches from image of shape {lr_img.shape}")

        # Process patches with model
        model.eval()
        output_patches = []

        with torch.no_grad():
            for i, patch in enumerate(tqdm(patches, desc="Processing patches")):
                # Move to device
                patch = patch.to(device)

                # Model inference
                output = model(patch)

                # Move output back to CPU
                output_patches.append(output.cpu())

                # Debug first patch
                if i == 0:
                    print(f"Input patch shape: {patch.shape}")
                    print(f"Output patch shape: {output.shape}")
                    print(
                        f"Input patch values - min: {patch.min().item()}, max: {patch.max().item()}"
                    )
                    print(
                        f"Output patch values - min: {output.min().item()}, max: {output.max().item()}"
                    )

        # Merge patches back into full image
        print("Merging patches...")
        sr_img = merge_patches(output_patches, patches_info, original_shape, patch_size)

        # Sample values for debugging
        print("HR sample (first pixel):", hr_img[0, :, 0, 0].tolist())
        print("SR sample (first pixel):", sr_img[0, :, 0, 0].tolist())
        print("HR range:", hr_img.min().item(), "to", hr_img.max().item())
        print("SR range:", sr_img.min().item(), "to", sr_img.max().item())

        return lr_img.cpu(), hr_img, sr_img

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = define_Model()
    model = model.to(device)
    model.gradient_checkpointing = False

    # Set loss function
    criterion = nn.MSELoss()

    # Initialize optimizer (needed for loading checkpoint)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr, weight_decay=Config.lr_decay
    )

    # Load checkpoint
    _, best_val_loss, best_psnr = load_checkpoint(model, optimizer, device)

    # Process a single image
    image_path = "data/Submission_input/1.npz"

    try:
        lr_img, hr_img, sr_img = process_single_image(model, image_path, patch_size=128)

        # Print original shapes
        print(f"Original HR shape: {hr_img.shape}")
        print(f"Original SR shape: {sr_img.shape}")

        # Compare properly aligned dimensions
        if hr_img.shape != sr_img.shape:
            print(
                "HR and SR shapes don't match. Checking if channel permutation is needed..."
            )

            # Try permuting if needed
            if hr_img.shape[1:] == sr_img.shape[1:].permute(1, 0, 2):
                print("Permuting HR tensor to match SR tensor")
                hr_img = hr_img.permute(0, 2, 1, 3)

            # If shapes still don't match, resize
            if hr_img.shape != sr_img.shape:
                print(f"Resizing from {sr_img.shape} to {hr_img.shape}")
                sr_img = F.interpolate(
                    sr_img,
                    size=(hr_img.shape[2], hr_img.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

        # Ensure both tensors are in the same value range for fair comparison
        print("\nPerforming final normalization check before metrics calculation:")
        print(f"HR range: {hr_img.min().item():.4f} to {hr_img.max().item():.4f}")
        print(f"SR range: {sr_img.min().item():.4f} to {sr_img.max().item():.4f}")

        # Optional: Normalize SR to match HR range if they're dramatically different
        if sr_img.max() > 10 * hr_img.max() or sr_img.min() < 10 * hr_img.min():
            print("Normalizing SR output to match HR range")
            sr_img = (sr_img - sr_img.min()) / (sr_img.max() - sr_img.min()) * (
                hr_img.max() - hr_img.min()
            ) + hr_img.min()
            print(
                f"After normalization - SR range: {sr_img.min().item():.4f} to {sr_img.max().item():.4f}"
            )

        # Calculate metrics
        loss = criterion(sr_img, hr_img)
        psnr = calculate_psnr(sr_img, hr_img)

        print("\nEvaluation Results:")
        print(f"Loss: {loss.item():.6f}")
        print(f"PSNR: {psnr:.2f} dB")

        print("\nFinal Image shapes:")
        print(f"LR shape: {lr_img.shape}")
        print(f"HR shape: {hr_img.shape}")
        print(f"SR shape: {sr_img.shape}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback

        traceback.print_exc()
