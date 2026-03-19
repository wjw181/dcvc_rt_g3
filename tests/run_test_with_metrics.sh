#!/bin/bash

# ============================================
# 安装依赖并运行可视化测试（带感知质量指标）
# ============================================

echo "============================================"
echo "Installing dependencies and running test"
echo "============================================"

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dcvc_rt

echo ""
echo "Current Python: $(which python3)"
echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# 安装依赖
echo "Installing pytorch-msssim..."
pip install pytorch-msssim -q

echo "Installing lpips..."
pip install lpips -q

echo ""
echo "✅ Dependencies installed"
echo ""

# 运行测试 - 第二阶段
echo "============================================"
echo "Running Stage 2 visualization test"
echo "============================================"

python3 visualize_comparison.py \
    --wan_checkpoint pretrained/DMC_WAN/1/checkpoint_stage2_wan.pth.tar \
    --dcvc_checkpoint pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
    --wan_vae Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
    --i_frame_model checkpoints/cvpr2025_image.pth.tar \
    --test_dir /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60 \
    --output_dir visualization_results_stage2_metrics \
    --qp 37 \
    --num_frames 5

echo ""
echo "============================================"
echo "Test completed!"
echo "Results saved in: visualization_results_stage2_metrics"
echo "============================================"
