"""
Data loading and preprocessing pipeline for HipMRI 2D Prostate Slice Segmentation.

Incorporates Nifti loading principles from Appendix B (Listing 1) of the course specification:
- Uses Nibabel to load 2D Nifti slices (.nii / .nii.gz).
- Squeezes trailing dimensions for HipMRI slices (if len(shape) == 3: img = img[:, :, 0]).
- Standardizes intensities via Z-score normalization: (x - mean) / (std + 1e-8).
- Extracts Prostate target mask (Label 5) for high-precision clinical contouring.
- Enforces strict patient-level separation across splits to ensure zero data leakage.
- Synchronized PyTorch data augmentation (spatial flips, rotations).

Author: s4984155
Course: COMP3710 / COMP7710
Target: HipMRI 2D slices (keras_slices_data), Target Prostate Dice >= 0.75.
"""

import os
import glob
from typing import Tuple, List, Optional
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


# ==============================================================================
# 1. Custom HipMRI 2D Slice Dataset (Nifti-based)
# ==============================================================================

class HipMRI2DSliceDataset(Dataset):
    """
    PyTorch Dataset for loading 2D HipMRI Nifti slices (.nii / .nii.gz)
    and extracting the prostate clinical target volume (Label 5).
    """
    PROSTATE_LABEL = 5  # Confirmed anatomical label index for prostate in HipMRI

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        is_train: bool = False,
        target_size: Optional[Tuple[int, int]] = (256, 128),
        prostate_only: bool = True
    ):
        """
        :param image_dir: Directory containing input MRI slice Nifti files.
        :param mask_dir: Directory containing corresponding segmentation Nifti files.
        :param is_train: Whether training augmentations should be applied.
        :param target_size: Output spatial resolution (H, W). Default (256, 128).
        :param prostate_only: If True, returns binary prostate mask [1, H, W] for Dice >= 0.75 target.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.target_size = target_size
        self.prostate_only = prostate_only

        # Find all .nii or .nii.gz files
        self.image_files = sorted(
            glob.glob(os.path.join(image_dir, "*.nii*"))
        )
        if not self.image_files:
            raise FileNotFoundError(f"No Nifti files found in directory: {image_dir}")

        # Pair images with masks:
        # e.g., 'case_040_week_0_slice_10.nii.gz' matches 'seg_040_week_0_slice_10.nii.gz'
        self.mask_files = []
        for img_path in self.image_files:
            fname = os.path.basename(img_path)
            mask_fname = fname.replace("case_", "seg_")
            mask_path = os.path.join(mask_dir, mask_fname)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Corresponding mask not found: {mask_path}")
            self.mask_files.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_nifti(self, path: str) -> np.ndarray:
        """
        Loads a 2D Nifti slice, handling extra dimensions as specified in Appendix B Listing 1.
        """
        nii = nib.load(path)
        data = nii.get_fdata(caching="unchanged")
        # Remove extra trailing dimensions if present in HipMRI data
        if len(data.shape) == 3:
            data = data[:, :, 0]
        return data.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # 1. Load Nifti slices via Nibabel
        image_np = self._load_nifti(img_path)
        mask_np = self._load_nifti(mask_path)

        # 2. Extract prostate mask if binary target
        if self.prostate_only:
            # Binary mask: 1.0 for prostate (label 5), 0.0 otherwise
            mask_np = (mask_np == self.PROSTATE_LABEL).astype(np.float32)

        # 3. Intensity normalization (Listing 1, lines 46-47)
        mean = image_np.mean()
        std = image_np.std() + 1e-8
        image_np = (image_np - mean) / std

        # 4. Convert NumPy arrays to PyTorch Tensors [1, H, W]
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)   # [1, H, W]

        # 5. Resize to standard target size if specified
        if self.target_size is not None and image_tensor.shape[1:] != self.target_size:
            image_tensor = TF.resize(image_tensor, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
            mask_tensor = TF.resize(mask_tensor, self.target_size, interpolation=TF.InterpolationMode.NEAREST)

        # 6. Synchronized Data Augmentation for Training
        if self.is_train:
            # Horizontal Flip (50% prob)
            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            # Vertical Flip (50% prob)
            if random.random() > 0.5:
                image_tensor = TF.vflip(image_tensor)
                mask_tensor = TF.vflip(mask_tensor)

            # Random Rotation (-10 to +10 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-10.0, 10.0)
                image_tensor = TF.rotate(image_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_tensor = TF.rotate(mask_tensor, angle, interpolation=TF.InterpolationMode.NEAREST)

        return image_tensor, mask_tensor


# ==============================================================================
# 2. DataLoader Constructor with Patient-Level Audit
# ==============================================================================

def get_data_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 2,
    target_size: Tuple[int, int] = (256, 128),
    prostate_only: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs train, validation, and test DataLoaders for HipMRI 2D slices.
    Verifies strict patient-level separation to prevent data leakage.

    :param data_root: Path containing 'keras_slices_train', 'keras_slices_test', etc.
    :param batch_size: Mini-batch size (e.g. 16 on A100 GPU).
    :param num_workers: Background DataLoader worker processes.
    :param target_size: (H, W) spatial dimensions.
    :param prostate_only: If True, targets the prostate (Label 5) as binary segmentation.
    :return: (train_loader, val_loader, test_loader).
    """
    train_img_dir = os.path.join(data_root, "keras_slices_train")
    train_mask_dir = os.path.join(data_root, "keras_slices_seg_train")

    val_img_dir = os.path.join(data_root, "keras_slices_validate")
    val_mask_dir = os.path.join(data_root, "keras_slices_seg_validate")

    test_img_dir = os.path.join(data_root, "keras_slices_test")
    test_mask_dir = os.path.join(data_root, "keras_slices_seg_test")

    # Helper function to extract patient IDs (e.g., 'case_040' from 'case_040_week_0_slice_10.nii.gz')
    def extract_patients(folder: str) -> set:
        if not os.path.exists(folder):
            return set()
        files = glob.glob(os.path.join(folder, "*.nii*"))
        return set(os.path.basename(f).split("_week")[0] for f in files)

    train_pts = extract_patients(train_img_dir)
    val_pts = extract_patients(val_img_dir)
    test_pts = extract_patients(test_img_dir)

    # Enforce zero leakage when folders exist
    if train_pts and val_pts:
        assert not (train_pts & val_pts), "Leakage Error: Patient overlap between train and val splits!"
    if train_pts and test_pts:
        assert not (train_pts & test_pts), "Leakage Error: Patient overlap between train and test splits!"
    if val_pts and test_pts:
        assert not (val_pts & test_pts), "Leakage Error: Patient overlap between val and test splits!"

    print("[HipMRI Data Hygiene Audit] Patient-level separation verified:")
    print(f"  - Unique Train patients: {len(train_pts)}")
    print(f"  - Unique Val patients:   {len(val_pts)}")
    print(f"  - Unique Test patients:  {len(test_pts)}")

    # Instantiate datasets
    train_dataset = HipMRI2DSliceDataset(train_img_dir, train_mask_dir, is_train=True, target_size=target_size, prostate_only=prostate_only)
    val_dataset = HipMRI2DSliceDataset(val_img_dir, val_mask_dir, is_train=False, target_size=target_size, prostate_only=prostate_only)
    test_dataset = HipMRI2DSliceDataset(test_img_dir, test_mask_dir, is_train=False, target_size=target_size, prostate_only=prostate_only)

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
# 3. Smoke Test Verification
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Testing HipMRI dataset.py Pipeline with Nifti Loader (Smoke Test)...")
    print("=" * 65)

    # 1. Determine data root (Supports both Rangpur HPC and Local)
    rangpur_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    local_path = r"C:\Users\29916\Desktop\code\COMP3710\keras_slices_data"

    data_dir = rangpur_path if os.path.exists(rangpur_path) else local_path

    if os.path.exists(data_dir):
        test_img_dir = os.path.join(data_dir, "keras_slices_test")
        test_mask_dir = os.path.join(data_dir, "keras_slices_seg_test")

        if os.path.exists(test_img_dir) and os.path.exists(test_mask_dir):
            test_ds = HipMRI2DSliceDataset(test_img_dir, test_mask_dir, is_train=False, target_size=(256, 128), prostate_only=True)
            test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

            images, masks = next(iter(test_loader))
            print(f"[OK] Test Nifti dataset loaded successfully:")
            print(f"     Total test slices:   {len(test_ds)}")
            print(f"     Batch images shape:  {images.shape} (dtype: {images.dtype})")
            print(f"     Batch masks shape:   {masks.shape} (dtype: {masks.dtype})")
            print(f"     Mask values:         {torch.unique(masks).tolist()} (0: background, 1: prostate)")
            print(f"     Image intensity range: [min={images.min():.2f}, max={images.max():.2f}]")
            print("=" * 65)
            print("HipMRI Nifti Data Pipeline Smoke Test PASSED!")
            print("=" * 65)
        else:
            print(f"Test subfolders not found under {data_dir}.")
    else:
        print(f"Data root not found at {data_dir}.")
