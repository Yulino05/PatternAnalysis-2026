#!/bin/bash
#SBATCH --job-name=prostate_unet
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# Activate conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

# Flush stdout immediately so logs appear in real-time
export PYTHONUNBUFFERED=1

# Dataset path on Rangpur HPC
DATA_DIR="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

# 1. Train Primary Model: 2D Improved U-Net (Normal Difficulty)
echo "=== Step 1: Training 2D Improved U-Net ==="
python -u train.py --model improved --epochs 50 --batch_size 16 --data_dir "$DATA_DIR"

# 2. Train Benchmark Baseline: Standard 2D U-Net (For report comparison)
echo "=== Step 2: Training 2D Baseline U-Net ==="
python -u train.py --model baseline --epochs 50 --batch_size 16 --data_dir "$DATA_DIR"

echo "Job finished at $(date)"
