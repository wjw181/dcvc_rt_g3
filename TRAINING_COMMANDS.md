# DMC_WAN 训练指令指南

## 📋 前置准备检查清单

✅ **所有checkpoint文件已就绪:**
- Wan VAE: `Wan2.2-main/checkpoints/Wan2.2_VAE.pth` (2.7G)
- DCVC_RT: `pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar` (238M)
- I-frame: `checkpoints/cvpr2025_image.pth.tar` (175M)

## 🚀 训练指令

### 阶段1: Pixel Mode 训练 (可选，用于对比基线)

**目的**: 在像素空间训练，作为基线对比

```bash
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2

python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_vd_phase_1_wan.py \
    --model DMC_WAN \
    --mode pixel \
    --test_dataset /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60/ \
    --test_filelist /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/test.txt \
    --epochs 120 \
    --learning-rate 1e-5 \
    --batch-size 2 \
    --quality-level 1 \
    --model_path_i checkpoints/cvpr2025_image.pth.tar \
    --pretrained_dcvc pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
    --freeze_dcvc \
    --freeze_vae \
    --clip_max_norm 0.5 \
    --num-workers 4 \
    --save
```

**特点:**
- 标准DCVC_RT流程
- 冻结DCVC和VAE（如果提供）
- 用于对比基准性能

---

### 阶段2: Latent Mode 训练 (主要训练阶段) ⭐

**目的**: 在生成式潜空间训练，实现GLC范式

```bash
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2

python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_vd_phase_1_wan.py \
    --model DMC_WAN \
    --mode latent \
    --test_dataset /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60/ \
    --test_filelist /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/test.txt \
    --epochs 120 \
    --learning-rate 1e-4 \
    --batch-size 2 \
    --quality-level 1 \
    --model_path_i checkpoints/cvpr2025_image.pth.tar \
    --pretrained_dcvc pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
    --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
    --freeze_dcvc \
    --freeze_vae \
    --clip_max_norm 0.5 \
    --num-workers 4 \
    --save
```

**特点:**
- ✅ GLC范式（在潜空间压缩）
- ✅ 训练adaptation layers (~277K参数)
- ✅ 冻结DCVC和VAE，只训练adaptation
- ✅ 学习率: 1e-4 (adaptation layers需要较高学习率)

**输出位置:**
```
pretrained/DMC_WAN/1/
├── checkpoint_wan.pth.tar              # 每个epoch保存
└── checkpoint_best_loss_wan.pth.tar    # 最佳模型
```

---

### 阶段3: End-to-End 微调 (可选)

**目的**: 解冻部分参数，进行端到端微调

```bash
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2

python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_vd_phase_1_wan.py \
    --model DMC_WAN \
    --mode latent \
    --test_dataset /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60/ \
    --test_filelist /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/test.txt \
    --epochs 50 \
    --learning-rate 1e-5 \
    --batch-size 1 \
    --quality-level 1 \
    --model_path_i checkpoints/cvpr2025_image.pth.tar \
    --pretrained_dcvc pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
    --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
    --clip_max_norm 0.5 \
    --num-workers 4 \
    --save \
    --checkpoint pretrained/DMC_WAN/1/checkpoint_best_loss_wan.pth.tar
```

**特点:**
- ⚠️ **不加 `--freeze_dcvc` 和 `--freeze_vae`** 参数
- 解冻所有参数进行微调
- 需要更大GPU内存 (建议40GB+)
- 更小的batch size和较低学习率

---

## 📊 参数说明

### 必需参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `pixel` | 模式: `pixel` 或 `latent` |
| `--wan_vae_checkpoint` | `None` | Wan VAE权重路径 (latent mode必需) |
| `--pretrained_dcvc` | `pretrained/DMC_SLF/...` | DCVC_RT预训练权重 |
| `--model_path_i` | `checkpoints/...` | I帧模型权重 |

### 训练参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--epochs` | `120` | `120` | 训练轮数 |
| `--learning-rate` | `1e-5` | `1e-4` (latent) | 学习率 |
| `--batch-size` | `2` | `2` (latent), `1` (e2e) | 批次大小 |
| `--clip_max_norm` | `0.5` | `0.5` | 梯度裁剪 |
| `--num-workers` | `4` | `4` | 数据加载线程数 |

### 冻结参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--freeze_dcvc` | `True` | 冻结DCVC_RT参数 |
| `--freeze_vae` | `True` | 冻结Wan VAE参数 |

**注意**: 
- Latent mode训练时，建议同时使用 `--freeze_dcvc` 和 `--freeze_vae`
- 只训练adaptation layers (~277K参数)

---

## 🎯 推荐训练流程

### 方案A: 直接Latent Mode训练 (推荐)

```bash
# 直接开始latent mode训练（最快）
# 第2阶段的命令
```

### 方案B: 渐进式训练

```bash
# 1. 先pixel mode训练50个epoch (可选)
# 第1阶段命令，但只训练50个epoch

# 2. 然后latent mode训练120个epoch
# 第2阶段命令

# 3. 最后end-to-end微调50个epoch (可选)
# 第3阶段命令
```

---

## 📈 训练监控

### 查看日志

```bash
# 实时查看训练日志
tail -f pretrained/DMC_WAN/1/20260113_*.log

# 查看最新日志
ls -lth pretrained/DMC_WAN/1/*.log | head -1
```

### 关键指标

训练日志会输出：
- `Loss`: 总损失
- `PSNR`: 峰值信噪比 (越高越好)
- `BPP`: 比特每像素 (越低越好)

测试日志会输出：
- `Average Loss`: 平均损失
- `Average PSNR`: 平均PSNR
- `Average BPP`: 平均码率

### 学习率调度

学习率会在以下epoch自动衰减:
- Epoch 10: LR × 0.4
- Epoch 20: LR × 0.1
- Epoch 40: LR × 0.04
- Epoch 60: LR × 0.01

---

## 🔧 常见问题

### 1. OOM (内存不足)

**解决方案:**
```bash
# 减小batch size
--batch-size 1

# 减少worker数量
--num-workers 2

# 使用单GPU训练 (如果有多GPU卡)
--nproc_per_node=1
```

### 2. 训练很慢

**检查:**
- GPU使用率 (`nvidia-smi`)
- 数据加载速度 (是否使用SSD)
- 是否使用了梯度累积

### 3. Loss不下降

**检查:**
- 学习率是否合适 (latent mode建议1e-4)
- 是否冻结了太多参数
- 数据是否正常加载

### 4. 从checkpoint继续训练

```bash
# 添加 --checkpoint 参数
--checkpoint pretrained/DMC_WAN/1/checkpoint_wan.pth.tar
```

---

## 📝 完整命令模板

### 单GPU训练 (如果内存足够)

```bash
# 使用 torchrun (推荐)
torchrun --standalone --nproc_per_node=1 \
    train_vd_phase_1_wan.py \
    --mode latent \
    --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
    --pretrained_dcvc pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
    --model_path_i checkpoints/cvpr2025_image.pth.tar \
    --freeze_dcvc \
    --freeze_vae \
    --epochs 120 \
    --learning-rate 1e-4 \
    --batch-size 2 \
    --save
```

### 多GPU训练 (4卡)

```bash
# 第2阶段命令 (推荐)
python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_vd_phase_1_wan.py \
    --mode latent \
    --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
    --pretrained_dcvc pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
    --model_path_i checkpoints/cvpr2025_image.pth.tar \
    --freeze_dcvc \
    --freeze_vae \
    --epochs 120 \
    --learning-rate 1e-4 \
    --batch-size 2 \
    --clip_max_norm 0.5 \
    --num-workers 4 \
    --save
```

---

## ⚡ 快速开始

**最简单的启动方式 (推荐):**

```bash
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2

python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_vd_phase_1_wan.py \
    --mode latent \
    --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
    --freeze_dcvc \
    --freeze_vae \
    --epochs 120 \
    --learning-rate 1e-4 \
    --batch-size 2 \
    --save
```

**其他参数使用默认值即可！**

---

## 📌 训练输出

训练完成后，最佳模型会保存在:
```
pretrained/DMC_WAN/1/checkpoint_best_loss_wan.pth.tar
```

使用该模型进行测试或推理。

---

**最后更新**: 2026-01-13  
**版本**: 1.0
