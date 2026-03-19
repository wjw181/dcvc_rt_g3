#!/bin/bash

# ============================================
# 可视化对比测试脚本 - 使用第二阶段 checkpoint
# WAN 集成模型 vs 原始 DCVC_RT 模型
# ============================================

echo "============================================"
echo "WAN vs DCVC_RT 可视化对比测试（第二阶段）"
echo "============================================"

# 配置参数 - 使用第二阶段的 checkpoint
WAN_CHECKPOINT="pretrained/DMC_WAN/1/checkpoint_best_stage2_wan.pth.tar"
DCVC_CHECKPOINT="pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar"
WAN_VAE="Wan2.2-main/checkpoints/Wan2.2_VAE.pth"
I_FRAME_MODEL="checkpoints/cvpr2025_image.pth.tar"
TEST_DIR="/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60"
OUTPUT_DIR="visualization_results_stage2"
QP=37
NUM_FRAMES=5

echo ""
echo "配置信息:"
echo "  WAN 模型: $WAN_CHECKPOINT (第二阶段 - 端到端微调)"
echo "  DCVC 模型: $DCVC_CHECKPOINT"
echo "  测试目录: $TEST_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  QP: $QP"
echo "  测试帧数: $NUM_FRAMES"
echo "============================================"
echo ""

# 检查 checkpoint 是否存在
if [ ! -f "$WAN_CHECKPOINT" ]; then
    echo "❌ 错误: WAN checkpoint 不存在: $WAN_CHECKPOINT"
    exit 1
fi

if [ ! -f "$DCVC_CHECKPOINT" ]; then
    echo "❌ 错误: DCVC checkpoint 不存在: $DCVC_CHECKPOINT"
    exit 1
fi

# 运行测试
python3 visualize_comparison.py \
    --wan_checkpoint $WAN_CHECKPOINT \
    --dcvc_checkpoint $DCVC_CHECKPOINT \
    --wan_vae $WAN_VAE \
    --i_frame_model $I_FRAME_MODEL \
    --test_dir $TEST_DIR \
    --output_dir $OUTPUT_DIR \
    --qp $QP \
    --num_frames $NUM_FRAMES

echo ""
echo "============================================"
echo "测试完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "============================================"
