import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Baseline Architecture: Standard 2D U-Net 
#（Use HipMRI 2D slices， Target Dice≥0.75）
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
# 2.1 Improved 2D U-Net Component Blocks
# ==============================================================================

# ContextModule
class ContextModule(nn.Module):
    """
    Residual Context Module from Isensee et al. (2018).
    Consists of two 3x3 convolutions with InstanceNorm and LeakyReLU,
    plus a dropout layer, wrapped with a residual identity skip connection.
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
        # Element-wise residual addition
        return out + residual

# LocalizationModule
class LocalizationModule(nn.Module):
    """
    Localization Module from Isensee et al. (2018).
    Consists of a 3x3 convolution followed by a 1x1 convolution
    to recombine concatenated features and reduce channels by half.
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

# UpsampleBlock
class UpsampleBlock(nn.Module):
    """
    Upsampling module that performs 2x spatial upsampling using
    bilinear interpolation followed by a 3x3 convolution.
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
# 2. Main Architecture: 2D Improved U-Net 
#（Use HipMRI 2D slices， Target Dice≥0.75） 
# ==============================================================================
class ImprovedUNet(nn.Module):
    """
    2D Improved U-Net Architecture (Isensee et al., 2018).
    
    Features:
    - Residual context blocks in the encoder to prevent gradient degradation.
    - Instance Normalization + LeakyReLU (alpha=0.01) for small-batch medical stability.
    - Upsampling with convolution to reduce checkerboard artifacts.
    - Deep supervision option across multiple decoder levels for robust boundary learning.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_filters: int = 16):
        super(ImprovedUNet, self).__init__()
        self.num_classes = num_classes

        # Encoder Level 1 (16 filters)
        self.in_conv = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False)
        self.in_norm = nn.InstanceNorm2d(base_filters, affine=True)
        self.in_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context1 = ContextModule(base_filters)

        # Encoder Level 2 (32 filters)
        self.down1 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.down1_norm = nn.InstanceNorm2d(base_filters * 2, affine=True)
        self.down1_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context2 = ContextModule(base_filters * 2)

        # Encoder Level 3 (64 filters)
        self.down2 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.down2_norm = nn.InstanceNorm2d(base_filters * 4, affine=True)
        self.down2_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context3 = ContextModule(base_filters * 4)

        # Encoder Level 4 (128 filters)
        self.down3 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.down3_norm = nn.InstanceNorm2d(base_filters * 8, affine=True)
        self.down3_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context4 = ContextModule(base_filters * 8)

        # Encoder Level 5 / Bottleneck (256 filters)
        self.down4 = nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.down4_norm = nn.InstanceNorm2d(base_filters * 16, affine=True)
        self.down4_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context5 = ContextModule(base_filters * 16)

        # Decoder / Localization Pathway
        self.up4 = UpsampleBlock(base_filters * 16, base_filters * 8)
        self.loc4 = LocalizationModule(base_filters * 16, base_filters * 8)
        self.up3 = UpsampleBlock(base_filters * 8, base_filters * 4)
        self.loc3 = LocalizationModule(base_filters * 8, base_filters * 4)
        self.up2 = UpsampleBlock(base_filters * 4, base_filters * 2)
        self.loc2 = LocalizationModule(base_filters * 4, base_filters * 2)
        self.up1 = UpsampleBlock(base_filters * 2, base_filters)
        self.loc1 = LocalizationModule(base_filters * 2, base_filters)

        # Deep Supervision Segmentation Heads (1x1 Convolutions)
        self.seg_head3 = nn.Conv2d(base_filters * 4, num_classes, kernel_size=1)
        self.seg_head2 = nn.Conv2d(base_filters * 2, num_classes, kernel_size=1)
        self.seg_head1 = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, enable_deep_supervision: bool = False):
        """
        Forward pass.
        :param x: Input tensor of shape [B, C, H, W]
        :param enable_deep_supervision: If True (typically during training), returns 
                                        a tuple of multi-scale segmentation maps.
        :return: Final segmentation map [B, num_classes, H, W], or multi-scale list.
        """
        # Encoder
        x1 = self.in_lrelu(self.in_norm(self.in_conv(x)))
        c1 = self.context1(x1)  # [B, 16, H, W]
        x2 = self.down1_lrelu(self.down1_norm(self.down1(c1)))
        c2 = self.context2(x2)  # [B, 32, H/2, W/2]
        x3 = self.down2_lrelu(self.down2_norm(self.down2(c2)))
        c3 = self.context3(x3)  # [B, 64, H/4, W/4]
        x4 = self.down3_lrelu(self.down3_norm(self.down3(c3)))
        c4 = self.context4(x4)  # [B, 128, H/8, W/8]
        x5 = self.down4_lrelu(self.down4_norm(self.down4(c4)))
        c5 = self.context5(x5)  # [B, 256, H/16, W/16]

        # Decoder with Skip Connections
        u4 = self.up4(c5)
        l4 = self.loc4(torch.cat([u4, c4], dim=1))
        u3 = self.up3(l4)
        l3 = self.loc3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(l3)
        l2 = self.loc2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(l2)
        l1 = self.loc1(torch.cat([u1, c1], dim=1))

        # Deep supervision feature projections
        out1 = self.seg_head1(l1)
        if enable_deep_supervision and self.training:
            out2 = self.seg_head2(l2)
            out3 = self.seg_head3(l3)
            # Upsample lower heads to match output resolution
            out2_up = F.interpolate(out2, size=out1.shape[2:], mode="bilinear", align_corners=False)
            out3_up = F.interpolate(out3, size=out1.shape[2:], mode="bilinear", align_corners=False)
            # Multi-scale combination for deep supervision
            return out1, out2_up, out3_up
        return out1


# ==============================================================================
# 3. self-test for modules
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing modules.py architectures (Smoke Test)...")
    print("=" * 60)
    # 1. Test BaselineUNet
    sample_tensor = torch.randn(2, 1, 256, 256)
    baseline = BaselineUNet(in_channels=1, num_classes=1, base_filters=16)
    baseline_out = baseline(sample_tensor)
    print(f"[OK] BaselineUNet forward pass successful!")
    print(f"     Input shape:  {sample_tensor.shape}")
    print(f"     Output shape: {baseline_out.shape}")
    print(f"     Parameters:   {sum(p.numel() for p in baseline.parameters()):,}")
    # 2. Test ImprovedUNet (Standard Inference mode)
    improved = ImprovedUNet(in_channels=1, num_classes=1, base_filters=16)
    improved.eval()
    improved_out = improved(sample_tensor, enable_deep_supervision=False)
    print(f"\n[OK] ImprovedUNet (Inference) forward pass successful!")
    print(f"     Output shape: {improved_out.shape}")
    print(f"     Parameters:   {sum(p.numel() for p in improved.parameters()):,}")
    # 3. Test ImprovedUNet (Deep Supervision Training mode)
    improved.train()
    heads = improved(sample_tensor, enable_deep_supervision=True)
    print(f"\n[OK] ImprovedUNet (Deep Supervision) forward pass successful!")
    print(f"     Number of supervision heads: {len(heads)}")
    for idx, h in enumerate(heads):
        print(f"     Head {idx+1} shape: {h.shape}")
    print("=" * 60)
    print("All module smoke tests passed successfully! No NumPy dependency detected.")
    print("=" * 60)