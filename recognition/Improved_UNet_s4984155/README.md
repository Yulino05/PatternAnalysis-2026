# Feasibility Review: 2D Improved U-Net for Automated Prostate MRI Segmentation
**Author:** Haifeng Huang / 4984155 | **Course:** COMP3710 / COMP7710  
**Target Domain:** Section 2.2 Segmentation (Prostate Radiotherapy Planning) | **Difficulty Band:** Normal Difficulty

---

### 1. User Need, Scope & Acceptance Criteria
* **Intended User & Clinical Need:** Radiation oncologists and medical physicists planning external-beam prostate radiotherapy. Manual contouring of the prostate from pelvic MRI is labor-intensive and suffers from inter-observer variability. Accurate automated segmentation accelerates treatment planning while tightly sparing adjacent Organs at Risk (OARs: rectum and bladder).
* **Prototype Purpose:** Develop a 2D deep learning segmentation prototype to automatically delineate the prostate boundary on transverse MRI slices, highlighting ambiguous boundary slices for clinician review.
* **Acceptable Quantitative Acceptance Criteria:**
  1. **Accuracy Threshold:** Mean Sørensen-Dice Similarity Coefficient (DSC) $\ge 0.75$ and mean IoU $\ge 0.60$ on the unseen test set.
  2. **Boundary Integrity:** Minimize fatal anatomical leakage into rectal walls on transitional slices (prostate apex and base false positive rate $< 5\%$).
  3. **Resource Budget:** Peak training GPU VRAM $\le 4.0\text{ GB}$; single-case inference latency $< 500\text{ ms}$, ensuring real-time clinical review feasibility.
  4. **Data Hygiene:** Strictly 0% cross-patient slice leakage across train/val/test splits.

---

### 2. Model Choice & Course Concepts
* **Dataset Configuration:** HipMRI 2D slice dataset (`keras_slices_data`), comprising multi-patient transverse T2-weighted scans with expert-annotated prostate masks. Patients are partitioned at the subject level (Patient-level Split: 70% train, 15% val, 15% test).
* **Proposed Architecture:** **2D Improved U-Net** (Isensee et al. [2]), benchmarked against a canonical **Standard 2D U-Net** [1] baseline.
* **Justification via Course Concepts & Literature:**
  * Standard U-Net lacks capacity for subtle soft-tissue contrast differences. The Improved U-Net incorporates **Residual Context Modules** in the encoder to facilitate stable gradient flow without degradation.
  * **Instance Normalization** replaces Batch Normalization to prevent mini-batch statistics drift caused by small medical batch sizes.
  * **Deep Supervision** integrates multi-scale feature maps from intermediate decoder stages to guide gradient propagation directly to semantic boundary edges.
  * Combined **Dice Loss + Binary Cross-Entropy (BCE)** is utilized to counteract the extreme foreground-background class imbalance (prostate $< 2\%$ volume).

---

### 3. Preliminary Feasibility Evidence
* **Data Pipeline Verification:** Verified Nifti slice parsing via Nibabel. Normalized slice intensities using z-score normalization. Confirmed spatial alignment between images and ground-truth masks.
* **Smoke-Test Experiment:** Constructed prototype tensor pipelines in PyTorch. Executed a mini-batch forward/backward pass through both the Baseline U-Net and Improved U-Net without gradient anomalies or dimension mismatches.
* **Resource Profile:** Preliminary single-batch forward pass consumes $\sim 1.2\text{ GB}$ VRAM, well within the target $4.0\text{ GB}$ hardware budget.

---

### 4. Technical Risks, Compute Budget & Fallback Plan
* **Identified Risks:**
  * *Boundary Ambiguity at Apex/Base:* Superior and inferior slices have weak contrast against bladder/rectal boundaries.
  * *Overfitting on Small Patient Cohort:* Risk of memorizing patient-specific pelvic shapes.
* **Compute Budget & Timeline:** 80 epochs of training on Rangpur HPC / local GPU ($\approx 45\text{ minutes}$ runtime per full experiment).
* **Fallback & Mitigation Plan:**
  * Apply spatial data augmentation (random affine, elastic deformation, horizontal flips) to alleviate small-cohort overfitting.
  * If multi-scale deep supervision causes training instability, fallback to standard 2D U-Net with focal Tversky loss as a reliable Easy-tier baseline while debugging.