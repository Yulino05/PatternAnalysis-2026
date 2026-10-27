"""
Training Pipeline for 2D Prostate Slice Segmentation on HipMRI.

Features:
- Supports training both 2D Improved U-Net and Baseline U-Net.
- Combined Loss Formulation: Binary Cross-Entropy (BCE) + Soft Dice Loss.
- Multi-scale deep supervision handling during training.
- Evaluates per-epoch validation Sørensen-Dice Similarity Coefficient (DSC) and IoU.
- Checkpoint management: Automatically saves the best model weights.
- Generates and saves training/validation loss and metric curves.
- GPU profiling: Logs peak VRAM and execution runtime for reporting.

Author: s4984155
Course: COMP3710 / COMP7710
Target Domain: Section 2.2 HipMRI 2D Prostate Segmentation
"""

import os
import sys
import time
import argparse
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import components from mandatory files
from modules import ImprovedUNet, BaselineUNet
from dataset import get_data_loaders, HipMRI2DSliceDataset


# ==============================================================================
# 1. Loss Formulations & Quantitative Metrics
# ==============================================================================

class SoftDiceLoss(nn.Module):
    """
    Differentiable Soft Sørensen-Dice Loss for binary medical segmentation.
    Counteracts severe class imbalance (e.g., prostate < 0.5% slice volume).
    """
    def __init__(self, smooth: float = 1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        denominator = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross-Entropy and Soft Dice Loss.
    Combines smooth gradient flow (BCE) with boundary overlap optimization (Dice).
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-5
) -> Tuple[float, float]:
    """
    Computes Sørensen-Dice Similarity Coefficient (DSC) and Intersection over Union (IoU).
    Accurately handles background-only boundary slices (Apex and Base).
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total_pred = preds_flat.sum().item()
    total_target = targets_flat.sum().item()

    # Handing cases where prostate is absent in the slice
    if total_target == 0 and total_pred == 0:
        return 1.0, 1.0  # Perfect true negative
    if total_target == 0 and total_pred > 0:
        return 0.0, 0.0  # False alarm on background slice

    dice = (2.0 * intersection + eps) / (total_pred + total_target + eps)
    union = total_pred + total_target - intersection
    iou = (intersection + eps) / (union + eps)

    return float(dice), float(iou)


# ==============================================================================
# 2. Training and Validation Engine
# ==============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_deep_supervision: bool = True
) -> float:
    """Runs a single training epoch across all mini-batches."""
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if use_deep_supervision and isinstance(model, ImprovedUNet):
            out1, out2, out3 = model(images, enable_deep_supervision=True)
            loss1 = criterion(out1, masks)
            loss2 = criterion(out2, masks)
            loss3 = criterion(out3, masks)
            # Weighted multi-scale loss formulation
            loss = loss1 + 0.5 * loss2 + 0.25 * loss3
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluates the model on validation data in evaluation mode."""
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_samples = len(loader)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)  # Single full-resolution output
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Accumulate batch metrics
            b_dice, b_iou = compute_metrics(outputs, masks)
            total_dice += b_dice
            total_iou += b_iou

    epoch_loss = running_loss / len(loader.dataset)
    mean_dice = total_dice / num_samples
    mean_iou = total_iou / num_samples

    return epoch_loss, mean_dice, mean_iou


# ==============================================================================
# 3. Visualization Function
# ==============================================================================

def plot_training_history(history: Dict, save_path: str, model_name: str):
    """Plots and saves loss and metric convergence curves across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    axes[0].set_title(f"{model_name}: Combined BCE + Dice Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Dice & IoU Curves
    axes[1].plot(epochs, history["val_dice"], label="Val Dice (DSC)", color="forestgreen", lw=2)
    axes[1].plot(epochs, history["val_iou"], label="Val IoU", color="darkorange", lw=2, linestyle="--")
    axes[1].axhline(y=0.75, color="gray", linestyle="-.", label="Target Dice = 0.75")
    axes[1].set_title(f"{model_name}: Validation Segmentation Metrics", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Training convergence curves saved to: {save_path}")


# ==============================================================================
# 4. Main Training Driver
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train U-Net on HipMRI 2D Prostate Data")
    parser.add_argument("--model", type=str, default="improved", choices=["improved", "baseline"],
                        help="Choose model architecture to train ('improved' or 'baseline')")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--data_dir", type=str, default="", help="Path to keras_slices_data directory")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save checkpoints and plots")
    parser.add_argument("--no_deep_supervision", action="store_true", help="Disable deep supervision")

    args = parser.parse_args()

    # Determine hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"================================================================")
    print(f"Starting Training Process on Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model.upper()} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"================================================================")

    # Determine dataset root path
    if args.data_dir:
        data_path = args.data_dir
    else:
        # Auto-detect Rangpur HPC or local
        rangpur_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
        local_path = r"C:\Users\29916\Desktop\code\COMP3710\keras_slices_data"
        data_path = rangpur_path if os.path.exists(rangpur_path) else local_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data root directory not found: {data_path}")

    # Load data loaders
    print(f"[Data Loading] Initializing loaders from: {data_path}")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_root=data_path,
        batch_size=args.batch_size,
        num_workers=4 if torch.cuda.is_available() else 0,
        target_size=(256, 128),
        prostate_only=True
    )

    # Instantiate model
    if args.model == "improved":
        model = ImprovedUNet(in_channels=1, num_classes=1, base_filters=16)
        model_name = "Improved_UNet"
        use_deep_supervision = not args.no_deep_supervision
    else:
        model = BaselineUNet(in_channels=1, num_classes=1, base_filters=16)
        model_name = "Baseline_UNet"
        use_deep_supervision = False

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Loaded] {model_name} initialized with {num_params:,} trainable parameters.")

    # Setup loss, optimizer, and scheduler
    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Checkpoint and history tracking
    best_val_dice = 0.0
    checkpoint_path = os.path.join(args.save_dir, f"best_{args.model}_unet.pth")
    plot_path = os.path.join(args.save_dir, f"training_curves_{args.model}.png")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    start_time = time.time()

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_deep_supervision)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        ep_duration = time.time() - ep_start

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        # Save checkpoint if best validation Dice is achieved
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> [Saved Best Model] Val Dice improved to {val_dice:.4f}! Weights saved to {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"================================================================")
    print(f"Training Complete! Total Elapsed Time: {total_time/60:.2f} minutes.")
    print(f"Best Validation Prostate Dice (DSC): {best_val_dice:.4f}")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        print(f"Peak GPU VRAM Usage: {peak_vram:.2f} MB")
    print(f"================================================================")

    # Generate and save convergence plot
    plot_training_history(history, plot_path, model_name)


if __name__ == "__main__":
    main()
