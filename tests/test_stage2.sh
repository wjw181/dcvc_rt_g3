#!/bin/bash

# ============================================
# 阶段二测试脚本（带感知质量指标）
# 测试第二阶段训练的 checkpoint（端到端微调）
# ============================================

echo "============================================"
echo "Stage 2 Visualization Test (with Perceptual Metrics)"
echo "============================================"

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate dcvc_rt

echo ""
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Testing Stage 2 checkpoint (end-to-end fine-tuned)..."
echo ""

# 配置参数
WAN_CHECKPOINT="pretrained/DMC_WAN/1/checkpoint_stage2_wan.pth.tar"
DCVC_CHECKPOINT="pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar"
WAN_VAE="Wan2.2-main/checkpoints/Wan2.2_VAE.pth"
I_FRAME_MODEL="checkpoints/cvpr2025_image.pth.tar"
TEST_DIR="/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/BasketballDrive_1920x1080_50"
OUTPUT_DIR="tests/results_stage2_metrics"  # 保存在 tests 文件夹下
QP=37
NUM_FRAMES=5
MATCH_BPP=true
QP_LIST="0,2,4,6,8,10,12,14,16,18,20,22,27,32,37"

echo "Configuration:"
echo "  WAN checkpoint: $WAN_CHECKPOINT"
echo "  DCVC checkpoint: $DCVC_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  QP: $QP"
echo "  Number of frames: $NUM_FRAMES"
echo "  Match BPP: $MATCH_BPP"
echo "  QP candidates: $QP_LIST"
echo "============================================"
echo ""

# 运行测试
python3 visualize_comparison.py \
    --wan_checkpoint $WAN_CHECKPOINT \
    --dcvc_checkpoint $DCVC_CHECKPOINT \
    --wan_vae $WAN_VAE \
    --i_frame_model $I_FRAME_MODEL \
    --test_dir $TEST_DIR \
    --output_dir $OUTPUT_DIR \
    --qp $QP \
    --num_frames $NUM_FRAMES \
    $( [ "$MATCH_BPP" = true ] && echo "--match_bpp --qp_list $QP_LIST" )

echo ""
echo "============================================"
echo "Stage 2 test completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "============================================"
