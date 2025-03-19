import torch
import os

# Path to the checkpoint file
checkpoint_path = "/media/jagadeesh/New Volume/Jagadeesh/SuperResolution/results/best_model_20250318-085923_epoch123_loss0.0150_PSNR30.42.pth"


# Function to load and print checkpoint information
def extract_checkpoint_info(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    try:
        # Load the checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        )

        # Print the top-level keys
        print("Checkpoint contents:")
        print("=" * 50)

        for key in checkpoint.keys():
            if key == "model_state_dict":
                print(f"model_state_dict: {len(checkpoint[key])} layers")
            elif key == "optimizer_state_dict":
                print(f"optimizer_state_dict: Contains optimizer states")
            else:
                print(f"{key}: {checkpoint[key]}")

        print("=" * 50)

        # Print detailed model structure if present
        if "model_state_dict" in checkpoint:
            print("\nModel layers:")
            for layer_name in list(checkpoint["model_state_dict"].keys())[
                :10
            ]:  # Show first 10 layers
                tensor = checkpoint["model_state_dict"][layer_name]
                print(f"  {layer_name}: {tensor.shape}, {tensor.dtype}")

            if len(checkpoint["model_state_dict"]) > 10:
                print(
                    f"  ... and {len(checkpoint['model_state_dict']) - 10} more layers"
                )

    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")


# Run the function
if __name__ == "__main__":
    extract_checkpoint_info(checkpoint_path)
