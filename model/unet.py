import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, chs=(4, 64, 128, 256, 512)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i + 1] + chs[i + 1], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]

            # Handle size mismatch for skip connections
            if x.size()[2:] != enc_ftrs.size()[2:]:
                # Use center crop
                target_h, target_w = x.size()[2], x.size()[3]
                enc_h, enc_w = enc_ftrs.size()[2], enc_ftrs.size()[3]

                h_diff = enc_h - target_h
                w_diff = enc_w - target_w

                h_start = h_diff // 2
                w_start = w_diff // 2

                enc_ftrs = enc_ftrs[
                    :, :, h_start : h_start + target_h, w_start : w_start + target_w
                ]

            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(self, double_precision=True):
        super().__init__()
        # self.encoder = Encoder(chs=(4, 64, 128, 256, 512))
        # self.decoder = Decoder(chs=(512, 256, 128, 64))
        # self.upsample = nn.PixelShuffle(2)  
        # self.pre_pixelshuffle = nn.Conv2d(64, 16, 1)  
        # self.final = nn.Conv2d(4, 4, 3, padding=1)
        
        self.encoder = Encoder(chs=(4, 32, 64, 128, 256))  # Reduced from (4, 64, 128, 256, 512)
        self.decoder = Decoder(chs=(256, 128, 64, 32))    # Reduced from (512, 256, 128, 64)
        self.upsample = nn.PixelShuffle(2)  
        self.pre_pixelshuffle = nn.Conv2d(32, 16, 1)      # Reduced from 64 to 32
        self.final = nn.Conv2d(4, 4, 3, padding=1)   

        

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        target_h, target_w = input_h * 2, input_w * 2

        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        # Pixel shuffle upsampling (x2)
        out = self.pre_pixelshuffle(out)
        out = self.upsample(out)
        out = self.final(out)

        # Ensure output has exactly 2x the input dimensions
        # if out.shape[2] != target_h or out.shape[3] != target_w:
        #     out = F.interpolate(out, size=(target_h, target_w), mode='bilinear', align_corners=False)

        out = torch.clamp(out, min=0.0, max=1.0)
        return out


if __name__ == "__main__":
    model = UNet().to(Config.device)

    # Check model size
    model_size = (
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    )
    print(f"Model Size: {model_size:.2f} MB")

    # Memory before creating input
    print(
        f"GPU Memory Before Input: {torch.cuda.memory_allocated(Config.device)/1024/1024:.2f} MB"
    )

    input_sizes = [(1, 4, 730, 1096), (1, 4, 512, 512)]

    for size in input_sizes:
        x = torch.randn(*size).to(Config.device)

        # Memory after creating input
        print(
            f"GPU Memory After Input {size}: {torch.cuda.memory_allocated(Config.device)/1024/1024:.2f} MB"
        )

        output = model(x)

        # Memory after model forward pass
        print(
            f"GPU Memory After Forward Pass {size}: {torch.cuda.memory_allocated(Config.device)/1024/1024:.2f} MB"
        )

        expected_h, expected_w = size[2] * 2, size[3] * 2
        print(
            f"Input: {size}, Output: {output.shape}, Expected: (1, 4, {expected_h}, {expected_w})"
        )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params / 1e6:.2f}M")
