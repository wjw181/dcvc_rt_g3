#!/bin/bash

# ============================================
# 改进的训练脚本 - 解决过拟合问题
# ============================================

# 设置GPU数量
NUM_GPUS=8

# 基础配置
MODEL="DMC"
QUALITY_LEVEL=1
EPOCHS=50
BATCH_SIZE=4
NUM_WORKERS=2

# 改进的超参数（防止过拟合）
LEARNING_RATE=5e-5          # 降低学习率
WEIGHT_DECAY=1e-5           # 添加L2正则化
CLIP_MAX_NORM=0.5           # 更严格的梯度裁剪
EARLY_STOPPING_PATIENCE=7   # 早停耐心值

# 多个测试集路径（重要改进！）
TEST_DATASETS=(
    "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60"
    "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/BasketballDrive_1920x1080_50"
    "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_C/RaceHorses_832x480_30"
)

TEST_FILELISTS=(
    "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/test.txt"
    "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/test.txt"
    "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_C/test.txt"
)

# 模型路径
MODEL_PATH_I="checkpoints/cvpr2025_image.pth.tar"
CHECKPOINT="pretrained/DMC_WAN/1/checkpoint_best_stage1_wan.pth.tar"

# 构建测试集参数
TEST_DATASETS_STR=""
for dataset in "${TEST_DATASETS[@]}"; do
    TEST_DATASETS_STR="$TEST_DATASETS_STR $dataset"
done

TEST_FILELISTS_STR=""
for filelist in "${TEST_FILELISTS[@]}"; do
    TEST_FILELISTS_STR="$TEST_FILELISTS_STR $filelist"
done

# 打印配置信息
echo "============================================"
echo "启动改进的训练 - 防止过拟合"
echo "============================================"
echo "GPU数量: $NUM_GPUS"
echo "学习率: $LEARNING_RATE (降低)"
echo "权重衰减: $WEIGHT_DECAY (新增)"
echo "梯度裁剪: $CLIP_MAX_NORM (更严格)"
echo "早停耐心: $EARLY_STOPPING_PATIENCE"
echo "测试集数量: ${#TEST_DATASETS[@]}"
echo "============================================"

# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_vd_phase_2_improved.py \
    -m $MODEL \
    -q $QUALITY_LEVEL \
    -e $EPOCHS \
    -lr $LEARNING_RATE \
    -n $NUM_WORKERS \
    --batch-size $BATCH_SIZE \
    --test-batch-size 1 \
    --test_datasets $TEST_DATASETS_STR \
    --test_filelists $TEST_FILELISTS_STR \
    --model_path_i $MODEL_PATH_I \
    --checkpoint $CHECKPOINT \
    --clip_max_norm $CLIP_MAX_NORM \
    --weight_decay $WEIGHT_DECAY \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --save

echo "训练完成！"
