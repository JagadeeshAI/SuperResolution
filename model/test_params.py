# test_params.py
import torch
from basicsr.archs.mambairv2_arch import MambaIRv2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Create the model
    # Create the model
    model = MambaIRv2(
        upscale=2,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=1.,
        upsampler='pixelshuffledirect',
        in_chans=3
    )


    # Create a 4-channel input (instead of 3)
    input_tensor = torch.randn(1, 3, 64, 64)

    model = model.cuda()
    input_tensor = input_tensor.cuda()

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Print input and output shapes
    print(f"Input shape: {input_tensor.shape}")  
    print(f"Output shape: {output.shape}")  # Should be [1, 4, 128, 128]
    
    # Print only total trainable parameters
    params = count_parameters(model)
    print(f"Total trainable parameters: {params:,}")
    print(f"In millions: {params/1e6:.2f}M")

if __name__ == "__main__":
    main()