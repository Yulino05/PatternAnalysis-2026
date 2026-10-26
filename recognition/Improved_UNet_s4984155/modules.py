"""
Model Architecture Definitions for Medical Image Segmentation.

Optimized for:
- Task: HipMRI 2D Prostate Slice Segmentation (Normal Difficulty, Target Dice >= 0.75).
- Reference: Isensee et al. (2018) Brain tumor segmentation / nnU-Net foundational architecture.
- Hardware Target: UQ Rangpur HPC (Nvidia A100 GPU).
- Resolution: Naturally supports HipMRI (256x128) and OASIS (256x256) via pure 2x scaling.

Author: s4984155
Course: COMP3710 / COMP7710
NOTE: Strictly no NumPy dependencies in this module (PyTorch-only).
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Component Blocks for Improved U-Net (Isensee et al., 2018)
# ==============================================================================

class ContextModule(nn.Module):
    """
    Residual Context Module from Isensee et al. (2018).
    Consists of two 3x3 convolutions with InstanceNorm and LeakyReLU,
    plus a spatial dropout layer, wrapped with a residual identity skip connection.
    """
    def __init__(self, channels: int, dropout_p: float = 0.3):
        super(ContextModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.lrelu1(self.in1(self.conv1(x)))
        out = self.dropout(out)
        out = self.lrelu2(self.in2(self.conv2(out)))
        # Element-wise residual addition (Skip Connection)
        return out + residual


class LocalizationModule(nn.Module):
    """
    Localization Module from Isensee et al. (2018).
    Recombines concatenated decoder and skip-connection features,
    then compresses channels by half using a 1x1 convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(LocalizationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lrelu1(self.in1(self.conv1(x)))
        out = self.lrelu2(self.in2(self.conv2(out)))
        return out


class UpsampleBlock(nn.Module):
    """
    Upsampling module that performs 2x spatial upsampling using
    bilinear interpolation followed by a 3x3 convolution to eradicate
    checkerboard artifacts commonly caused by transposed convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.in_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.lrelu(self.in_norm(self.conv(out)))
        return out


# ==============================================================================
# 2. Main Architecture: 2D Improved U-Net (Normal Difficulty)
# ==============================================================================

class ImprovedUNet(nn.Module):
    """
    2D Improved U-Net Architecture (Isensee et al., 2018).
    
    Features:
    - Residual context blocks in the encoder for unobstructed gradient backpropagation.
    - Instance Normalization + LeakyReLU (alpha=0.01) for small-batch medical stability.
    - Interpolation-based upsampling to eliminate checkerboard artifacts.
    - Multi-scale deep supervision heads for robust boundary contouring on edge slices.
    - Kaiming (He) weight initialization for accelerated convergence.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 16,
        dropout_p: float = 0.3
    ):
        super(ImprovedUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters

        # Encoder
        # Level 1 (base_filters = 16)
        self.in_conv = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False)
        self.in_norm = nn.InstanceNorm2d(base_filters, affine=True)
        self.in_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context1 = ContextModule(base_filters, dropout_p=dropout_p)

        # Level 2 (32 filters)
        self.down1 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.down1_norm = nn.InstanceNorm2d(base_filters * 2, affine=True)
        self.down1_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context2 = ContextModule(base_filters * 2, dropout_p=dropout_p)

        # Level 3 (64 filters)
        self.down2 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.down2_norm = nn.InstanceNorm2d(base_filters * 4, affine=True)
        self.down2_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context3 = ContextModule(base_filters * 4, dropout_p=dropout_p)

        # Level 4 (128 filters)
        self.down3 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.down3_norm = nn.InstanceNorm2d(base_filters * 8, affine=True)
        self.down3_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context4 = ContextModule(base_filters * 8, dropout_p=dropout_p)

        # Level 5 / Bottleneck (256 filters)
        self.down4 = nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.down4_norm = nn.InstanceNorm2d(base_filters * 16, affine=True)
        self.down4_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context5 = ContextModule(base_filters * 16, dropout_p=dropout_p)

        # Decoder
        self.up4 = UpsampleBlock(base_filters * 16, base_filters * 8)
        self.loc4 = LocalizationModule(base_filters * 16, base_filters * 8)

        self.up3 = UpsampleBlock(base_filters * 8, base_filters * 4)
        self.loc3 = LocalizationModule(base_filters * 8, base_filters * 4)

        self.up2 = UpsampleBlock(base_filters * 4, base_filters * 2)
        self.loc2 = LocalizationModule(base_filters * 4, base_filters * 2)

        self.up1 = UpsampleBlock(base_filters * 2, base_filters)
        self.loc1 = LocalizationModule(base_filters * 2, base_filters)

        # Deep Supervision Heads
        self.seg_head3 = nn.Conv2d(base_filters * 4, num_classes, kernel_size=1)
        self.seg_head2 = nn.Conv2d(base_filters * 2, num_classes, kernel_size=1)
        self.seg_head1 = nn.Conv2d(base_filters, num_classes, kernel_size=1)

        # Initialize network weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) normal initialization for LeakyReLU layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)) and m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        enable_deep_supervision: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        :param x: Input tensor [B, in_channels, H, W] (supports 256x128 or arbitrary multiples of 16).
        :param enable_deep_supervision: If True during training, returns multi-scale predictions.
        :return: Segmentation logits [B, num_classes, H, W] or tuple of 3 multi-scale logits.
        """
        # Encoder
        x1 = self.in_lrelu(self.in_norm(self.in_conv(x)))
        c1 = self.context1(x1)  # Level 1

        x2 = self.down1_lrelu(self.down1_norm(self.down1(c1)))
        c2 = self.context2(x2)  # Level 2

        x3 = self.down2_lrelu(self.down2_norm(self.down2(c2)))
        c3 = self.context3(x3)  # Level 3

        x4 = self.down3_lrelu(self.down3_norm(self.down3(c3)))
        c4 = self.context4(x4)  # Level 4

        x5 = self.down4_lrelu(self.down4_norm(self.down4(c4)))
        c5 = self.context5(x5)  # Level 5 (Bottleneck)

        # Decoder with Skip Connections (Pure 2x scaling)
        u4 = self.up4(c5)
        l4 = self.loc4(torch.cat([u4, c4], dim=1))

        u3 = self.up3(l4)
        l3 = self.loc3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(l3)
        l2 = self.loc2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(l2)
        l1 = self.loc1(torch.cat([u1, c1], dim=1))

        # Final full-resolution prediction
        out1 = self.seg_head1(l1)

        # Multi-scale deep supervision (activated only in training mode)
        if enable_deep_supervision and self.training:
            out2 = self.seg_head2(l2)
            out3 = self.seg_head3(l3)
            # Bilinear upsample intermediate heads to full output resolution
            out2_up = F.interpolate(out2, size=out1.shape[2:], mode="bilinear", align_corners=False)
            out3_up = F.interpolate(out3, size=out1.shape[2:], mode="bilinear", align_corners=False)
            return out1, out2_up, out3_up

        return out1


# ==============================================================================
# 3. Baseline Architecture: Standard 2D U-Net (Ronneberger et al., 2015)
# ==============================================================================

class DoubleConv(nn.Module):
    """Standard Double 3x3 Conv + BatchNorm + ReLU block."""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineUNet(nn.Module):
    """
    Classical 2D U-Net (Ronneberger et al., 2015) serving as the baseline benchmark.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_filters: int = 16):
        super(BaselineUNet, self).__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 4, base_filters * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 8, base_filters * 16))

        self.up1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_filters * 16, base_filters * 8)

        self.up2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up4 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_filters * 2, base_filters)

        self.outc = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv_up1(torch.cat([x, x4], dim=1))

        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x2], dim=1))

        x = self.up4(x)
        x = self.conv_up4(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        return logits


# ==============================================================================
# 4. Smoke Test Module Verification
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Testing modules.py on HipMRI Resolution (256 x 128)...")
    print("=" * 65)

    # 1. Simulate a HipMRI pelvic slice batch [Batch=2, Channels=1, H=256, W=128]
    hipmri_sample = torch.randn(2, 1, 256, 128)

    # 2. Test BaselineUNet
    baseline = BaselineUNet(in_channels=1, num_classes=1, base_filters=16)
    baseline_out = baseline(hipmri_sample)
    print(f"[OK] BaselineUNet forward pass successful!")
    print(f"     Input shape:  {hipmri_sample.shape}")
    print(f"     Output shape: {baseline_out.shape}")
    print(f"     Parameters:   {sum(p.numel() for p in baseline.parameters()):,}")

    # 3. Test ImprovedUNet (Inference Mode)
    improved = ImprovedUNet(in_channels=1, num_classes=1, base_filters=16)
    improved.eval()
    improved_out = improved(hipmri_sample, enable_deep_supervision=False)
    print(f"\n[OK] ImprovedUNet (Inference) forward pass successful!")
    print(f"     Output shape: {improved_out.shape}")
    print(f"     Parameters:   {sum(p.numel() for p in improved.parameters()):,}")

    # 4. Test ImprovedUNet (Deep Supervision Mode)
    improved.train()
    heads = improved(hipmri_sample, enable_deep_supervision=True)
    print(f"\n[OK] ImprovedUNet (Deep Supervision) forward pass successful!")
    print(f"     Number of supervision heads: {len(heads)}")
    for idx, h in enumerate(heads):
        print(f"     Head {idx+1} shape: {h.shape}")

    print("=" * 65)
    print("All HipMRI module smoke tests PASSED cleanly with pure 2x scaling!")
    print("=" * 65)
