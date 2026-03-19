#!/bin/bash

echo "============================================"
echo "Installing Perceptual Quality Metrics"
echo "============================================"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate dcvc_rt

echo ""
echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

echo "Installing pytorch-msssim (for SSIM and MS-SSIM)..."
pip install pytorch-msssim -q

echo ""
echo "Installing lpips (for perceptual similarity)..."
pip install lpips -q

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "Installed packages:"
pip list | grep -E "pytorch-msssim|lpips"
echo ""
echo "You can now run the visualization tests with:"
echo "  bash test_stage1.sh"
echo "  bash test_stage2.sh"
echo "  bash test_both_stages.sh"
