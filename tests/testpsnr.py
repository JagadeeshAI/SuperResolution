def test_psnr():
    import numpy as np
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from config import Config
    from utils.util import define_Model, load_checkpoint
    from data.loader import get_data_loaders

    # Initialize model
    model = define_Model()

    # Initialize optimizer (needed for load_checkpoint)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr, weight_decay=Config.lr_decay
    )

    # Load model weights
    load_checkpoint(model, optimizer, Config.device)

    # Set model to evaluation mode
    model.eval()

    # Get validation data loader
    _, val_loader = get_data_loaders()

    if not val_loader:
        print("Error: Validation loader is not available.")
        return

    # Initialize metrics
    total_psnr = 0.0
    total_samples = 0

    # PSNR calculation function
    def calculate_psnr(img1, img2, max_val=1.0):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(max_val / torch.sqrt(mse))

    # Process validation data
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing PSNR on validation set"):
            # Get input and target
            lr = batch["lr"].to(Config.device)
            hr = batch["hr"].to(Config.device)
            max_val = batch["max"]

            # Forward pass
            sr_output = model(lr)

            # Calculate PSNR for each image in batch
            for i in range(sr_output.size(0)):
                # Calculate MSE loss
                mse_loss = nn.MSELoss()(sr_output[i], hr[i])

                # Also calculate L1 loss
                l1_loss = nn.L1Loss()(sr_output[i], hr[i])

                # Calculate PSNR
                psnr = calculate_psnr(sr_output[i], hr[i])
                total_psnr += psnr.item()
                total_samples += 1

                # Print detail for this sample
                print(
                    f"Sample {total_samples}: PSNR={psnr.item():.2f} dB, MSE={mse_loss.item():.6f}, L1={l1_loss.item():.6f}"
                )

    # Calculate average PSNR
    avg_psnr = total_psnr / total_samples
    print(f"Average PSNR on validation set: {avg_psnr:.2f} dB")
    return avg_psnr


if __name__ == "__main__":
    test_psnr()
