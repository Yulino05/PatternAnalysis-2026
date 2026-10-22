"""
Data loading and preprocessing pipeline for 2D Medical Slice Segmentation.

Features:
- Loads paired 2D slices and segmentation masks.
- Enforces strict patient-level separation across train/val/test splits (zero leakage).
- Data augmentation: Random horizontal/vertical flip, random rotation.
- Intensity normalization: Standardized scaling [0, 1] or Z-score normalization.
- Mask mapping: Converts raw multi-class pixel values [0, 85, 170, 255] -> class indices [0, 1, 2, 3].

Author: s4984155
Course: COMP3710 / COMP7710
"""

import os
import glob
from typing import Tuple, List, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


# ==============================================================================
# 1. Custom Medical 2D Slice Dataset
# ==============================================================================

class Medical2DSliceDataset(Dataset):
    """
    PyTorch Dataset for loading 2D medical slice images and corresponding masks.
    """
    # Mapping raw mask pixel values to class indices [0, 1, 2, 3]
    LABEL_MAPPING = {
        0: 0,      # Background
        85: 1,     # CSF (Cerebrospinal Fluid)
        170: 2,    # White Matter (WM)
        255: 3     # Gray Matter (GM)
    }

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        is_train: bool = False,
        target_size: Tuple[int, int] = (256, 256)
    ):
        """
        :param image_dir: Directory containing input MRI slice images.
        :param mask_dir: Directory containing corresponding segmentation masks.
        :param is_train: Whether training augmentations should be applied.
        :param target_size: Output spatial resolution (H, W).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.target_size = target_size

        # Retrieve and sort all image files
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {image_dir}")

        # Pair images with masks based on filename convention:
        # e.g., 'case_001_slice_10.nii.png' matches 'seg_001_slice_10.nii.png'
        self.mask_files = []
        for img_path in self.image_files:
            filename = os.path.basename(img_path)
            mask_filename = filename.replace("case_", "seg_")
            mask_path = os.path.join(mask_dir, mask_filename)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Corresponding mask not found: {mask_path}")
            self.mask_files.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def _map_labels(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """
        Maps raw intensity values [0, 85, 170, 255] to categorical indices [0, 1, 2, 3].
        """
        mapped_mask = torch.zeros_like(mask_tensor, dtype=torch.long)
        for raw_val, class_idx in self.LABEL_MAPPING.items():
            # Allow small tolerance for antialiasing/interpolation artifacts
            mapped_mask[torch.isclose(mask_tensor, torch.tensor(raw_val / 255.0), atol=0.05)] = class_idx
        return mapped_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # Load grayscale images
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Resize if needed
        if image.size != self.target_size:
            image = TF.resize(image, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.target_size, interpolation=TF.InterpolationMode.NEAREST)

        # Convert to Tensor [0.0, 1.0]
        image_tensor = TF.to_tensor(image)  # Shape: [1, H, W]
        mask_tensor = TF.to_tensor(mask)    # Shape: [1, H, W]

        # Synchronized Data Augmentation for Training
        if self.is_train:
            # 1. Random Horizontal Flip (50% probability)
            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            # 2. Random Vertical Flip (50% probability)
            if random.random() > 0.5:
                image_tensor = TF.vflip(image_tensor)
                mask_tensor = TF.vflip(mask_tensor)

            # 3. Random Slight Rotation (-10 to +10 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-10.0, 10.0)
                image_tensor = TF.rotate(image_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_tensor = TF.rotate(mask_tensor, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Z-score Normalization on image intensity: (x - mean) / (std + 1e-8)
        mean = image_tensor.mean()
        std = image_tensor.std() + 1e-8
        image_tensor = (image_tensor - mean) / std

        # Map mask intensities to class indices [H, W] as LongTensor
        mask_indices = self._map_labels(mask_tensor.squeeze(0))

        return image_tensor, mask_indices


# ==============================================================================
# 2. DataLoader Constructor with Patient-Level Audit
# ==============================================================================

def get_data_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Instantiates DataLoader objects for train, validation, and test splits.
    Audits patient IDs to guarantee strict patient-level separation.

    :param data_root: Root path containing 'keras_png_slices_train', 'keras_png_slices_validate', etc.
    :param batch_size: Mini-batch size.
    :param num_workers: Number of background worker processes.
    :return: Tuple of (train_loader, val_loader, test_loader).
    """
    train_img_dir = os.path.join(data_root, "keras_png_slices_train")
    train_mask_dir = os.path.join(data_root, "keras_png_slices_seg_train")

    val_img_dir = os.path.join(data_root, "keras_png_slices_validate")
    val_mask_dir = os.path.join(data_root, "keras_png_slices_seg_validate")

    test_img_dir = os.path.join(data_root, "keras_png_slices_test")
    test_mask_dir = os.path.join(data_root, "keras_png_slices_seg_test")

    # Patient-level leakage audit
    def extract_patients(folder: str) -> set:
        files = glob.glob(os.path.join(folder, "*.png"))
        return set(os.path.basename(f).split("_slice")[0] for f in files)

    train_pts = extract_patients(train_img_dir)
    val_pts = extract_patients(val_img_dir)
    test_pts = extract_patients(test_img_dir)

    assert not (train_pts & val_pts), "Data Leakage Alert: Overlap detected between train and val splits!"
    assert not (train_pts & test_pts), "Data Leakage Alert: Overlap detected between train and test splits!"
    assert not (val_pts & test_pts), "Data Leakage Alert: Overlap detected between val and test splits!"

    print(f"[Dataset Audit] Patient-level separation verified:")
    print(f"  - Training unique patients:   {len(train_pts)}")
    print(f"  - Validation unique patients: {len(val_pts)}")
    print(f"  - Testing unique patients:    {len(test_pts)}")

    train_dataset = Medical2DSliceDataset(train_img_dir, train_mask_dir, is_train=True)
    val_dataset = Medical2DSliceDataset(val_img_dir, val_mask_dir, is_train=False)
    test_dataset = Medical2DSliceDataset(test_img_dir, test_mask_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


# ==============================================================================
# 3. Smoke Test Verification Block
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing dataset.py Data Pipeline (Smoke Test)...")
    print("=" * 60)

    # Path to local dataset
    default_data_dir = r"C:\Users\29916\Desktop\code\COMP3710\keras_png_slices_data"

    if os.path.exists(default_data_dir):
        train_loader, val_loader, test_loader = get_data_loaders(
            default_data_dir, batch_size=4, num_workers=0
        )
        print(f"\n[OK] DataLoaders constructed successfully:")
        print(f"     Train batches: {len(train_loader)} (Total: {len(train_loader.dataset)} slices)")
        print(f"     Val batches:   {len(val_loader)} (Total: {len(val_loader.dataset)} slices)")
        print(f"     Test batches:  {len(test_loader)} (Total: {len(test_loader.dataset)} slices)")

        # Inspect a single mini-batch
        images, masks = next(iter(train_loader))
        print(f"\n[OK] Sample batch extracted:")
        print(f"     Image Tensor shape: {images.shape} (dtype: {images.dtype})")
        print(f"     Mask Tensor shape:  {masks.shape} (dtype: {masks.dtype})")
        print(f"     Unique mask labels: {torch.unique(masks).tolist()}")
        print(f"     Image range: [min={images.min():.2f}, max={images.max():.2f}]")
        print("=" * 60)
        print("Data pipeline smoke test PASSED with ZERO leakage!")
        print("=" * 60)
    else:
        print(f"Dataset path not found at: {default_data_dir}. Please verify path.")
