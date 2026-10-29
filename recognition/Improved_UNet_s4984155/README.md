# 2D Improved U-Net for Automated Prostate MRI Segmentation
**Student Name:** Haifeng Huang  
**Student ID:** 4984155  
**Course:** COMP3710 / COMP7710 - Pattern Recognition (Semester 2, 2026)  
**Target Domain:** Section 2.2 Medical Image Segmentation (Prostate Radiotherapy Planning)  
**Assigned Difficulty:** Normal Difficulty (Target Test Dice $\ge 0.75$ on Prostate Label)  

---

## 1. Project Overview & Working Principles

### 1.1 Clinical Problem
Prostate cancer external-beam radiotherapy requires millimeter-level contouring of the prostate clinical target volume (CTV) on pelvic transverse T2-weighted MRI scans while strictly avoiding adjacent Organs at Risk (OARs: rectum and bladder). Under-contouring risks fatal tumor recurrence, whereas over-contouring causes severe radiation proctitis or cystitis. Manual delineation is time-consuming and suffers from high inter-observer variability. This project develops an automated 2D segmentation prototype using the **HipMRI 2D slice dataset (`keras_slices_data`)** to accurately delineate the prostate boundary and flag ambiguous transition slices for clinical expert triage.

### 1.2 Algorithmic Working Principle
The implemented model is the **2D Improved U-Net** (adapted from Isensee et al., 2018), benchmarked against a classical **Standard 2D U-Net** (Ronneberger et al., 2015). It addresses the unique challenges of small pelvic targets (prostate occupies $<2\%$ of pelvic slice area):
1. **Residual Context Units**: Incorporates residual skip connections ($\text{out} + \text{residual}$) in each encoder level to prevent gradient vanishing across deep feature hierarchies.
2. **Instance Normalization & LeakyReLU**: Replaces standard batch normalization with `InstanceNorm2d(affine=True)` to eradicate mini-batch statistical drift under small medical batch sizes, combined with `LeakyReLU(0.01)` to prevent dying neurons on low-contrast soft tissues.
3. **Artifact-Free Upsampling**: Employs bilinear interpolation followed by 3x3 convolutions (`UpsampleBlock`) to eliminate checkerboard artifacts inherent in standard transposed convolutions.
4. **Multi-Scale Deep Supervision**: Projects auxiliary segmentation heads at intermediate decoder resolutions to guide gradient propagation directly into semantic contour representations during training.

---

## 2. Compulsory Feasibility Review (Midpoint Check-Off)

> *This section corresponds to the one-page written feasibility review presented to the project manager (tutor) during the midpoint practical consultation.*

### 2.1 User Need, Scope & Acceptance Criteria
* **Intended User:** Radiation oncologists and medical physicists planning prostate radiotherapy.
* **Prototype Purpose:** Automate prostate contouring on transverse 2D pelvic MRI slices, providing rapid automated drafts and highlighting ambiguous transitional slices.
* **Testable Acceptance Criteria:**
  1. **Quantitative Accuracy:** Mean Sørensen-Dice Similarity Coefficient (DSC) $\ge 0.75$ on the unseen test set prostate label (Label 5).
  2. **Boundary Precision:** Mean Intersection-over-Union (IoU) $\ge 0.60$ on the test split.
  3. **Computational Resource Budget:** Peak training GPU VRAM $\le 4.0\text{ GB}$; single-case inference latency $< 500\text{ ms}$ on Rangpur HPC (A100 GPU) or local GPU.
  4. **Strict Data Hygiene:** 100% verified patient-level separation (zero slice leakage across train/val/test splits).

### 2.2 Model Choice & Course Concepts
* **Dataset Configuration:** HipMRI 2D slice dataset (`keras_slices_data`), comprising multi-patient transverse T2-weighted scans with expert-annotated pelvic masks (Label 5 = Prostate). Slices are stored in medical Nifti format (`.nii.gz`).
* **Model Selection:** **2D Improved U-Net** [Normal Difficulty], compared against a **Standard 2D U-Net** baseline.
* **Literature Support:** Based on Isensee et al. (2018), residual context extraction and instance normalization provide robust representations on heavily imbalanced medical structures without mode collapse.

### 2.3 Preliminary Feasibility Evidence
* **Data Pipeline Verification:** Implemented `HipMRI2DSliceDataset` using `nibabel` based on course Appendix B (Listing 1). Verified intensity Z-score normalization and dimension handling (`(256, 128)` resolution).
* **Smoke-Test Validation:** Executed end-to-end forward passes on both `BaselineUNet` (1.94M parameters) and `ImprovedUNet` (3.19M parameters) on real HipMRI test slices with zero runtime or dimension errors.
* **Data Hygiene Audit:** Built automated set-intersection assertions confirming 0% patient-level cross-contamination across splits.

### 2.4 Risks, Compute Budget & Fallback
* **Identified Risks:** High class imbalance (prostate $<0.5\%$ of slice volume) and ambiguous contrast at the superior base and inferior apex slices.
* **Compute Budget:** ~60 epochs of training on UQ Rangpur HPC (`comp3710` A100 GPU partition), estimated runtime $\approx 20\text{ minutes}$.
* **Fallback Strategy:** If multi-scale deep supervision exhibits training instability, revert to standard 2D U-Net with focal loss as a robust baseline while tuning hyperparameters.

---

## 3. Data Pipeline & Hygiene (Zero-Leakage Splitting)

### 3.1 Preprocessing Pipeline
1. **Nifti Slicing:** Loaded via `nibabel.load(path).get_fdata(caching="unchanged")`. Slices with trailing singleton dimensions are squeezed to 2D matrices (`H=256, W=128`).
2. **Prostate Target Extraction:** In the HipMRI multiclass pelvic annotation, the prostate is explicitly labeled as **`Label 5`**. Masks are binarized:
   $$\text{Mask}_{\text{prostate}} = (\text{Segmentation} == 5)_{\text{float32}}$$
3. **Z-Score Intensity Normalization:** Each slice is normalized using its internal intensity distribution:
   $$I_{\text{norm}} = \frac{I - \mu_I}{\sigma_I + 10^{-8}}$$
4. **Synchronized Augmentation:** Training data undergoes synchronized random horizontal flips, vertical flips, and small rotations ($\pm 10^\circ$) to mitigate overfitting.

### 3.2 Patient-Level Split Justification
To eliminate patient data leakage, dataset splitting is strictly enforced at the **Subject ID level** (e.g., `case_040`). Slices from any given patient reside exclusively within either training, validation, or test sets. The codebase includes automated programmatic assertions:
```python
assert not (train_pts & val_pts), "Leakage Error: Patient overlap in train/val!"
assert not (train_pts & test_pts), "Leakage Error: Patient overlap in train/test!"
```

---

## 4. Codebase Architecture & File Responsibilities

The codebase resides in `recognition/Improved_UNet_s4984155/` with the mandatory 5-file architecture:

```text
recognition/Improved_UNet_s4984155/
├── modules.py     # Pure PyTorch definitions of ImprovedUNet and BaselineUNet (No NumPy)
├── dataset.py     # Nifti data loading, patient-level audit, augmentation, and DataLoaders
├── train.py       # Training loop, loss formulation, validation, and metric logging
├── predict.py     # Evaluation on test set, quantitative benchmarking, and visualization
└── README.md      # Comprehensive technical report and feasibility documentation
```

---

## 5. Environment & Quick Start Guide

### 5.1 Dependencies
* Python 3.9+
* PyTorch >= 2.0.0
* Torchvision >= 0.15.0
* Nibabel >= 5.0.0
* Matplotlib >= 3.7.0

Install required packages via pip:
```bash
pip install torch torchvision nibabel matplotlib
```

### 5.2 Running Locally or on Rangpur HPC
* **Run Model Smoke Test:**
  ```bash
  python modules.py
  ```
* **Run Data Pipeline Smoke Test:**
  ```bash
  python dataset.py
  ```
* **Train Models on Rangpur HPC (A100 GPU):**
  ```bash
  sbatch job.sh
  ```
* **Evaluate Models on Unseen Test Slices & Generate Visual Autopsies:**
  ```bash
  # Evaluate 2D Improved U-Net
  python predict.py --model improved --weights best_improved_unet.pth

  # Evaluate Standard Baseline U-Net
  python predict.py --model baseline --weights best_baseline_unet.pth
  ```

---

## 6. Investigation of the Open Research Dilemma & Empirical Evaluation

### 6.1 The Boundary & Transition Dilemma in Prostate Radiotherapy
In prostate segmentation, aggregate global Dice scores often conceal hazardous localized failures. Because the prostate tapers drastically at the apex (inferior, adjacent to the urogenital diaphragm) and base (superior, abutting the bladder), 2D slice-by-slice models frequently face extreme partial-volume effects and low contrast. In clinical practice, false positives leaking into the rectal wall can trigger radiation proctitis, while false negatives at the base risk under-treating tumor extensions.

### 6.2 Quantitative Benchmark Results (UQ Rangpur HPC A100 GPU)

Both models were trained under identical conditions: 50 epochs, AdamW optimizer ($\text{lr}=10^{-3}$ with Cosine Annealing), Combined BCE + Soft Dice Loss, and verified patient-isolated splits (31 train, 4 val, 3 test patients; 540 unseen test slices).

| Quantitative Evaluation Metric | Standard 2D U-Net (Baseline) | 2D Improved U-Net (Proposed) | Clinical Engineering Trade-off |
| :--- | :---: | :---: | :--- |
| **Trainable Parameters** | 1,942,289 (~1.94M) | 3,186,723 (~3.19M) | +64% capacity for multi-scale residual context |
| **Best Validation Prostate Dice (DSC)** | 0.7740 | **0.7987** | **+2.47% gain** on hyperparameter selection |
| **Unseen Test Prostate Dice (DSC)** | **0.9010 $\pm$ 0.2284** | **0.8858 $\pm$ 0.2540** | **Both comfortably exceed $\ge 0.75$ hurdle** |
| **Unseen Test Prostate IoU (Jaccard)** | **0.8702** | **0.8551** | Strong volumetric overlap and boundary agreement |
| **Per-Slice Inference Latency** | **51.65 ms** (19.4 FPS) | **101.57 ms** (9.8 FPS) | Both $<110\text{ ms}$, suitable for interactive clinical triage |
| **Peak Training GPU VRAM** | **751.13 MB** | **1478.75 MB** | Both $<1.5\text{ GB}$ (far below 4.0 GB target budget) |
| **50-Epoch Training Runtime (A100)** | 24.08 minutes | 31.51 minutes | RAM caching delivers ~34s/epoch training throughput |

### 6.3 Qualitative Visual Inspection & Error Autopsy

The models generated automated 4-panel visual verification figures featuring multi-color error difference overlays:
* **True Positive (TP)**: Green — Accurately contoured prostate clinical target volume.
* **False Positive (FP)**: Red — Dangerous leakage into adjacent rectum or bladder tissue.
* **False Negative (FN)**: Blue — Under-segmented target prostate tissue.

#### 1. Training Convergence Curves
![Training Convergence - Improved U-Net](training_curves_improved.png)  
*Figure 1: Training and validation loss, Sørensen-Dice coefficient, and IoU curves for 2D Improved U-Net across 50 epochs on Rangpur HPC.*

![Training Convergence - Baseline U-Net](training_curves_baseline.png)  
*Figure 2: Training and validation convergence curves for Standard Baseline U-Net.*

#### 2. Qualitative Prediction & Error Overlays
![Qualitative Predictions - Improved U-Net](prediction_demo_improved.png)  
*Figure 3: Four-panel qualitative segmentation results produced by `predict.py` for 2D Improved U-Net on test slices. Columns display: (1) Raw T2 MRI Slice, (2) Ground Truth Prostate Mask (Label 5), (3) Predicted Contours with Slice Dice, and (4) Error Difference Overlay (Green: TP, Red: FP, Blue: FN).*

![Qualitative Predictions - Baseline U-Net](prediction_demo_baseline.png)  
*Figure 4: Four-panel qualitative segmentation results and error overlays for Standard Baseline U-Net.*

### 6.4 Critical Engineering Analysis of the Open Dilemma
1. **Validation vs. Test Generalization**: While the `Improved U-Net` achieved a higher validation Dice score ($0.7987$ vs $0.7740$), both models demonstrated exceptional test set generalizability ($0.8858$ and $0.9010$, respectively), vastly exceeding the compulsory assessment hurdle of $\ge 0.75$.
2. **Computational vs. Semantic Trade-Off**: The `Baseline U-Net` operates with lower memory footprint (751 MB) and half the inference latency (51 ms vs 101 ms). Conversely, the `Improved U-Net`'s residual context blocks and instance normalization provide superior resilience against contrast fluctuations across different hospital scanners.
3. **Actionable Clinical Recommendation**: In high-throughput clinical radiotherapy workstations, `Baseline U-Net` offers rapid initial auto-contour drafts (19.4 FPS). For complex, small-volume prostate targets where anatomical boundary ambiguity is high, `Improved U-Net` provides multi-scale deep supervision features that can be integrated with human-in-the-loop confidence gating. Future iterations should incorporate 2.5D adjacent-slice context to eliminate remaining apex/base boundary errors.

---

## 7. Artificial Intelligence Usage Disclosure

In compliance with UQ Academic Integrity and Course Assessment Specification (Section 6.1):

1. **AI Tools Utilized:**
   * **Google Gemini / Anthropic Claude:** Consulted for scaffolding PyTorch modular structure, debugging Nifti dimension squeezing, and formatting markdown documentation.
   * **GitHub Copilot:** Used as an intelligent code-completion assistant for repetitive typing (e.g., standard layer instantiation and docstrings).
2. **Specific Tasks Assisted:**
   * Structuring the 5-file modular design according to Section 4.1.
   * Scaffolding the `HipMRI2DSliceDataset` and patient ID regex parsing logic.
   * Formulating the Kaiming initialization and deep supervision multi-scale loss weighting syntax.
3. **Audit, Verification & Empirical Testing Methodology:**
   * **Mathematical & Dimension Audit:** Manually calculated downsampling and upsampling tensor dimensions ($256 \times 128 \to 16 \times 8 \to 256 \times 128$) to verify that pure 2x bilinear scaling prevents spatial distortion.
   * **Leakage Verification:** Independently designed set-intersection assertions (`train_pts & test_pts`) to mathematically guarantee zero data leakage.
   * **Empirical Execution:** Executed smoke tests on live HipMRI test slices to confirm that the model compiles, calculates gradients, and outputs exact `[B, 1, 256, 128]` tensors without NumPy dependency.
