#!/bin/bash
# 优化的训练启动脚本 - OOM错误修复版本

# 设置显存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 清理CUDA缓存
echo "清理CUDA缓存..."
python -c "import torch; torch.cuda.empty_cache(); print('CUDA缓存已清理')"

# 显示GPU信息
echo "GPU信息:"
nvidia-smi

# 训练参数
MODE="pixel"                    # 训练模式: pixel 或 latent
BATCH_SIZE=1                    # 降低batch size以节省显存
NUM_WORKERS=2                   # 减少数据加载workers
LEARNING_RATE=1e-5
QUALITY_LEVEL=1
EPOCHS=40
NUM_GPUS=2

# 模型路径
PRETRAINED_DCVC="pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar"
I_FRAME_MODEL="checkpoints/cvpr2025_image.pth.tar"

# 数据集路径
TEST_DATASET="/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60/"
TEST_FILELIST="/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/test.txt"

# GLC路径 (用于感知损失)
GLC_ROOT="/home/serverdn/hdd-0/wjw/GLC"

echo "========================================"
echo "训练配置:"
echo "  模式: $MODE"
echo "  Batch Size: $BATCH_SIZE"
echo "  混合精度: 启用"
echo "  GPU数量: $NUM_GPUS"
echo "  学习率: $LEARNING_RATE"
echo "========================================"

# 启动训练
torchrun --nproc_per_node=$NUM_GPUS train_vd_phase_1_wan.py \
    --mode $MODE \
    --batch-size $BATCH_SIZE \
    --use_amp \
    --num-workers $NUM_WORKERS \
    --learning-rate $LEARNING_RATE \
    --quality-level $QUALITY_LEVEL \
    --epochs $EPOCHS \
    --pretrained_dcvc $PRETRAINED_DCVC \
    --model_path_i $I_FRAME_MODEL \
    --test_dataset $TEST_DATASET \
    --test_filelist $TEST_FILELIST \
    --glc_root $GLC_ROOT \
    --clip_max_norm 0.5 \
    --save

echo "训练完成!"
