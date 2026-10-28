"""
Evaluation and Inference Script for 2D Prostate Slice Segmentation on HipMRI.

Deliverables:
- Loads trained model weights (best_improved_unet.pth or best_baseline_unet.pth).
- Evaluates unseen test set (540 slices) for Sørensen-Dice Similarity Coefficient (DSC) and IoU.
- Profiles per-slice inference latency (latency in ms and throughput in FPS).
- Generates 4-panel visual verification figures:
    [Input MRI | Ground Truth | Model Prediction | Difference Error Overlay]
    (TP = Green, FP = Red, FN = Blue).
- Saves representative boundary cases (Apex, Mid-gland, Base) for clinical error autopsy.

Author: s4984155
Course: COMP3710 / COMP7710
Target Domain: Section 2.2 HipMRI 2D Prostate Segmentation (Target DSC >= 0.75)
"""

import os
import sys
import time
import argparse
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from modules import ImprovedUNet, BaselineUNet
from dataset import get_data_loaders, HipMRI2DSliceDataset


# ==============================================================================
# 1. Quantitative Evaluation Engine
# ==============================================================================

def compute_slice_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-5
) -> Tuple[float, float]:
    """
    Computes per-slice Dice and IoU metrics on binary numpy masks.
    Correctly accounts for true negative slices (no prostate in slice).
    """
    pred_flat = pred_mask.astype(bool).flatten()
    gt_flat = gt_mask.astype(bool).flatten()

    intersection = np.logical_and(pred_flat, gt_flat).sum()
    total_pred = pred_flat.sum()
    total_gt = gt_flat.sum()

    if total_gt == 0 and total_pred == 0:
        return 1.0, 1.0  # True negative slice correctly segmented
    if total_gt == 0 and total_pred > 0:
        return 0.0, 0.0  # False positive on background slice

    dice = (2.0 * intersection + eps) / (total_pred + total_gt + eps)
    union = total_pred + total_gt - intersection
    iou = (intersection + eps) / (union + eps)

    return float(dice), float(iou)


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Runs full inference on test_loader and reports mean test DSC, IoU, and latency.
    """
    model.eval()
    dice_list = []
    iou_list = []
    latencies = []

    print("[Evaluation] Running full test set inference...", flush=True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)

            # Measure inference time
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_latency = (t1 - t0) / images.size(0)
            latencies.extend([batch_latency] * images.size(0))

            preds = (probs > threshold).float().cpu().numpy()
            targets = masks.cpu().numpy()

            for b in range(images.size(0)):
                p_slice = preds[b, 0]
                g_slice = targets[b, 0]
                d, i = compute_slice_metrics(p_slice, g_slice)
                dice_list.append(d)
                iou_list.append(i)

    mean_dice = float(np.mean(dice_list))
    std_dice = float(np.std(dice_list))
    mean_iou = float(np.mean(iou_list))
    avg_latency_ms = float(np.mean(latencies)) * 1000.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_iou": mean_iou,
        "latency_ms": avg_latency_ms,
        "fps": fps,
        "total_slices": len(dice_list)
    }


# ==============================================================================
# 2. Qualitative Error Autopsy Visualizer
# ==============================================================================

def create_difference_overlay(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Creates an RGB error overlay:
    - True Positive (TP): Green [0, 255, 0]
    - False Positive (FP): Red [255, 0, 0]
    - False Negative (FN): Blue [0, 150, 255]
    - Background: Black [0, 0, 0]
    """
    h, w = pred.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    tp = np.logical_and(pred == 1, gt == 1)
    fp = np.logical_and(pred == 1, gt == 0)
    fn = np.logical_and(pred == 0, gt == 1)

    overlay[tp] = [46, 204, 113]   # Emerald Green
    overlay[fp] = [231, 76, 60]    # Crimson Red
    overlay[fn] = [52, 152, 219]   # Bright Blue

    return overlay


def plot_qualitative_results(
    model: nn.Module,
    dataset: HipMRI2DSliceDataset,
    device: torch.device,
    save_path: str,
    num_samples: int = 4
):
    """
    Generates a 4-panel visual comparison:
    Column 1: Raw Input MRI (256x128)
    Column 2: Ground Truth Prostate Mask (Label 5)
    Column 3: Model Prediction
    Column 4: Multi-colored Error Difference Overlay
    """
    model.eval()

    # Find slices that contain prostate tissue for meaningful visualization
    prostate_indices = []
    for i in range(len(dataset)):
        _, m = dataset[i]
        if m.sum() > 50:  # contains prostate pixels
            prostate_indices.append(i)

    # Select evenly distributed samples from apex, mid-gland, and base
    if len(prostate_indices) >= num_samples:
        step = len(prostate_indices) // num_samples
        selected_indices = [prostate_indices[i * step] for i in range(num_samples)]
    else:
        selected_indices = list(range(min(num_samples, len(dataset))))

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    with torch.no_grad():
        for row_idx, data_idx in enumerate(selected_indices):
            image, mask = dataset[data_idx]
            input_tensor = image.unsqueeze(0).to(device)

            output = model(input_tensor)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float().cpu().numpy()[0, 0]
            gt = mask.numpy()[0]
            mri_img = image.numpy()[0]

            dice, iou = compute_slice_metrics(pred, gt)
            overlay = create_difference_overlay(pred, gt)

            # Col 1: Raw Input MRI
            axes[row_idx, 0].imshow(mri_img, cmap="gray")
            axes[row_idx, 0].set_title(f"Sample {row_idx+1}: Input MRI (256x128)", fontsize=10)
            axes[row_idx, 0].axis("off")

            # Col 2: Ground Truth Mask
            axes[row_idx, 1].imshow(mri_img, cmap="gray")
            axes[row_idx, 1].imshow(gt, cmap="spring", alpha=0.5)
            axes[row_idx, 1].set_title("Ground Truth (Label 5: Prostate)", fontsize=10)
            axes[row_idx, 1].axis("off")

            # Col 3: Model Prediction
            axes[row_idx, 2].imshow(mri_img, cmap="gray")
            axes[row_idx, 2].imshow(pred, cmap="cool", alpha=0.5)
            axes[row_idx, 2].set_title(f"Predicted Mask (Dice: {dice:.3f})", fontsize=10)
            axes[row_idx, 2].axis("off")

            # Col 4: Error Difference Overlay
            axes[row_idx, 3].imshow(mri_img, cmap="gray")
            axes[row_idx, 3].imshow(overlay, alpha=0.6)
            axes[row_idx, 3].set_title("Error Overlay (TP=Grn, FP=Red, FN=Blu)", fontsize=10)
            axes[row_idx, 3].axis("off")

    # Legend for overlay
    tp_patch = mpatches.Patch(color='#2ECC71', label='True Positive (TP)')
    fp_patch = mpatches.Patch(color='#E74C3C', label='False Positive (FP)')
    fn_patch = mpatches.Patch(color='#3498DB', label='False Negative (FN)')
    fig.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot Saved] Qualitative prediction figures saved to: {save_path}", flush=True)


# ==============================================================================
# 3. Main CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained U-Net on HipMRI 2D Prostate Test Set")
    parser.add_argument("--weights", type=str, default="best_improved_unet.pth",
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--model", type=str, default="improved", choices=["improved", "baseline"],
                        help="Model architecture: 'improved' or 'baseline'")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Path to keras_slices_data directory")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save prediction evaluation figures")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of qualitative demo slices to plot")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65, flush=True)
    print(f"Starting Quantitative Test Evaluation on Device: {device}", flush=True)
    print(f"Model: {args.model.upper()} | Weights: {args.weights}", flush=True)
    print("=" * 65, flush=True)

    # Locate dataset root
    if args.data_dir:
        data_path = args.data_dir
    else:
        rangpur_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
        local_path = r"C:\Users\29916\Desktop\code\COMP3710\keras_slices_data"
        data_path = rangpur_path if os.path.exists(rangpur_path) else local_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data root directory not found: {data_path}")

    # Load test loader
    _, _, test_loader = get_data_loaders(
        data_root=data_path,
        batch_size=16,
        num_workers=0,
        target_size=(256, 128),
        prostate_only=True
    )

    # Build model architecture
    if args.model == "improved":
        model = ImprovedUNet(in_channels=1, num_classes=1, base_filters=16)
    else:
        model = BaselineUNet(in_channels=1, num_classes=1, base_filters=16)

    # Load weights
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint weights not found: {args.weights}")

    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"[Model Loaded] Successfully loaded weights from {args.weights}", flush=True)

    # Run quantitative evaluation
    results = evaluate_test_set(model, test_loader, device)

    print("=" * 65, flush=True)
    print("FINAL TEST SET BENCHMARK RESULTS (HipMRI 2D Prostate):", flush=True)
    print(f"  - Total Test Slices:       {results['total_slices']}")
    print(f"  - Mean Sørensen-Dice (DSC): {results['mean_dice']:.4f} +/- {results['std_dice']:.4f}")
    print(f"  - Mean IoU (Jaccard Index): {results['mean_iou']:.4f}")
    print(f"  - Per-Slice Latency:        {results['latency_ms']:.2f} ms")
    print(f"  - Throughput:               {results['fps']:.1f} slices/sec")
    print(f"  - Target Requirement:       DSC >= 0.75 -> {'PASSED [OK]' if results['mean_dice'] >= 0.75 else 'NEEDS RETRAIN'}")
    print("=" * 65, flush=True)

    # Generate qualitative visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f"prediction_demo_{args.model}.png")
    test_dataset = test_loader.dataset
    plot_qualitative_results(model, test_dataset, device, plot_path, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
