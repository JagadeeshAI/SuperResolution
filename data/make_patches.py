import torch
import torch.nn.functional as F


# def extract_patches(tensor, patch_size):
#     """
#     Divides the input tensor into patches and moves patches to the batch dimension.

#     Args:
#         tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
#         patch_size (int): Size of each square patch

#     Returns:
#         torch.Tensor: Tensor with patches moved to batch dimension (B*num_patches, C, patch_size, patch_size)
#     """
#     B, C, H, W = tensor.shape
#     #assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"

#     # Reshape and permute to extract patches
#     tensor = tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#     tensor = tensor.permute(0, 2, 3, 1, 4, 5).contiguous()
#     tensor = tensor.view(B * (H // patch_size) * (W // patch_size), C, patch_size, patch_size)

#     return tensor


def extract_patches_overlapping(tensor, patch_size, overlap=0.5):
    """
    Divides the input tensor into patches with specified overlap and moves patches to the batch dimension.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
        patch_size (int): Size of each square patch
        overlap (float): Fraction of overlap between patches (default: 0.5 for 50% overlap)

    Returns:
        torch.Tensor: Tensor with patches moved to batch dimension
    """
    B, C, H, W = tensor.shape

    # Calculate stride based on overlap
    stride = int(patch_size * (1 - overlap))

    # Reshape and permute to extract patches
    tensor = tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

    # Calculate new dimensions based on stride
    patches_h = (H - patch_size) // stride + 1
    patches_w = (W - patch_size) // stride + 1

    # Reshape to get the patches in batch dimension
    tensor = tensor.permute(0, 2, 3, 1, 4, 5).contiguous()
    tensor = tensor.view(B * patches_h * patches_w, C, patch_size, patch_size)

    return tensor


def select_random_patches(patches, num_patches=4):
    """
    Selects random patches from the batch of patches.

    Args:
        patches (torch.Tensor): Tensor of patches with shape (B*num_patches, C, patch_size, patch_size)
        num_patches (int): Number of random patches to select

    Returns:
        torch.Tensor: Randomly selected patches
    """
    total_patches = patches.shape[0]
    indices = torch.randperm(total_patches)[:num_patches]
    return patches[indices]


def downsample_raw(raw):
    """
    Downsamples a 4-channel packed RAW image by a factor of 2.
    The input raw should be a [H/2, W/2, 4] tensor -- with respect to its mosaiced version [H,w]
    Output is a [H/4, W/4, 4] tensor, preserving the RGGB pattern.
    """
    avg_pool = torch.nn.AvgPool2d(2, stride=2)
    downsampled_image = avg_pool(raw)
    return downsampled_image


# # Example Usage
# B, C, H, W = 1, 10, 1043, 1023  # Example batch size, channels, height, and width
# patch_size = 128
# tensor = torch.randn(B, C, H, W)  # Random tensor

# # Extract patches and move to batch dimension
# patches = extract_patches(tensor, patch_size)

# # Select 4 random patches
# random_patches = select_random_patches(patches, num_patches=4)
# downsampled_patches = downsample_raw(random_patches)

# print("Original Tensor Shape:", tensor.shape)
# print("Extracted Patches Shape:", patches.shape)
# print("Randomly Selected Patches Shape:", random_patches.shape)
# print('downsampled_patches', downsampled_patches.size())
