# DCVC_RT + Wan VAE Integration for Generative Latent Coding (GLC)

## 概述

本项目将 Wan2.2 的生成式 VAE 集成到 DCVC_RT 视频压缩框架中，实现了 **Generative Latent Coding (GLC)** 范式，用于超低码率视频压缩。

参考论文：
- [Generative Latent Coding for Ultra-Low Bitrate Image and Video Compression](https://arxiv.org/abs/...)
- [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314)

## 核心思想

### 传统方法 vs GLC

**传统视频压缩（DCVC_RT）:**
```
RGB Video → YUV → DCVC_RT Encoder → Compressed Bitstream
                                    ↓
Compressed Bitstream → DCVC_RT Decoder → YUV → RGB Video
```

**Generative Latent Coding (GLC):**
```
RGB Video → Wan VAE Encoder → Latent Space (16x16x4 compression)
                                    ↓
            Latent → Adaptation → DCVC_RT → Compressed Latent (ultra-low bitrate)
                                    ↓
            Compressed Latent → DCVC_RT → Adaptation → Decoded Latent
                                    ↓
            Decoded Latent → Wan VAE Decoder → RGB Video
```

### 主要优势

1. **语义感知压缩**: 在生成式潜空间中进行压缩，保留语义信息
2. **更好的感知质量**: 在超低码率下获得更高的视觉质量
3. **人类感知对齐**: VAE 潜空间与人类感知更加一致
4. **更高的压缩比**: Wan VAE 提供 16×16×4 的压缩，叠加 DCVC_RT 压缩

## 架构设计

### 1. Wan VAE Wrapper (`src/models/wan_vae_wrapper.py`)

封装 Wan2.2 VAE，提供以下功能：
- RGB → Latent 编码
- Latent → RGB 解码
- RGB ↔ YUV 转换（与 DCVC_RT 兼容）
- 支持冻结 VAE 参数（全部/仅Encoder）

**关键参数:**
- 输入: RGB [B, C, H, W], 值域 [0, 1]
- 输出 Latent: [B, 48, T', H/16, W/16]
- 时间压缩比: 4x (T' = T/4)
- 空间压缩比: 16×16

### 2. DMC_WAN Model (`src/models/video_t_g_wan.py`)

集成 DCVC_RT 和 Wan VAE 的混合模型。

**Latent Mode（潜空间模式）- GLC 完整流程:**
- 在潜空间进行压缩
- 通过 Adaptation Layers 进行维度转换

**Adaptation Layers (增强型5层设计):**
```
Encoder: Latent (48ch) → 96 → 64 → 32 → 16 → Pseudo-YUV (3ch)
Decoder: Pseudo-YUV (3ch) → 16 → 32 → 64 → 96 → Latent (48ch)
```

### 3. 训练脚本 (`train_vd_phase_1_wan.py`)

支持分布式训练的完整训练流程，包含：
- 多阶段冻结策略
- Learning Rate Warmup + Step Decay
- Lambda Ramp (30 epochs)
- LPIPS 感知损失

---

## 🚀 完整训练方案：实现 PSNR 30+ dB

### 📊 训练阶段总览

| 阶段 | 训练内容 | 冻结组件 | 训练参数 | 预期PSNR | 预期LPIPS | 时间(8卡) |
|------|---------|---------|---------|----------|-----------|----------|
| **Stage 1** | Adaptation Layers | DCVC + VAE | 72万 | 26-27 dB | 0.45-0.50 | 2-3天 |
| **Stage 2** | DCVC + Adaptation | VAE | 2100万 | 28-29 dB | 0.35-0.40 | 3-4天 |
| **Stage 3** | VAE Decoder + 上述 | VAE Encoder | ~7000万 | 29-30 dB | 0.25-0.30 | 2-3天 |
| **Stage 4** | 全部组件 | 无 | 1.6亿 | 30-32 dB | 0.22-0.25 | 3-4天 |
| **Stage 5** | 感知优化 (可选) | 无 | 1.6亿 | 31+ dB | 0.18-0.20 | 1-2天 |

**总计**: 14-16天（8卡全天候训练）

---

## 🎯 阶段转换标准（何时进入下一阶段）

### ⚠️ 重要提示

**不要过早进入下一阶段！** 每个阶段必须充分训练，否则后续阶段将无法达到预期性能。

### Stage 1 → Stage 2 转换标准

**必须满足以下全部条件：**

| 指标 | 最低要求 | 推荐目标 | 当前状态 (Epoch 22) |
|------|---------|---------|-------------------|
| **测试PSNR** | ≥ 25 dB | ≥ 26 dB | ❌ 19.6 dB |
| **训练PSNR** | ≥ 26 dB | ≥ 27 dB | ❌ 22.4 dB |
| **测试LPIPS** | ≤ 0.50 | ≤ 0.45 | ✅ (需确认) |
| **训练Loss** | 稳定收敛 | < 3.0 | ❌ 71.5 |
| **LatentMSE** | < 0.15 | < 0.12 | ✅ 0.12 |
| **训练稳定性** | 连续5 epochs无上升 | 连续10 epochs | ⚠️ 需观察 |

**额外检查点：**
- ✅ PseudoYUV_MSE > 0.0001（确保梯度流畅通）
- ✅ 测试BPP稳定在0.10-0.15范围
- ✅ 训练曲线平滑，无异常抖动

**当前建议**: ⚠️ **继续Stage 1训练**，至少需要再训练20-30个epoch

**预计达标时间**: Epoch 45-50左右（基于当前收敛速度）

---

### Stage 2 → Stage 3 转换标准

**必须满足以下全部条件：**

| 指标 | 最低要求 | 推荐目标 |
|------|---------|---------|
| **测试PSNR** | ≥ 27 dB | ≥ 28 dB |
| **训练PSNR** | ≥ 28 dB | ≥ 29 dB |
| **测试LPIPS** | ≤ 0.42 | ≤ 0.38 |
| **训练Loss** | 稳定收敛 | 连续降低 |
| **PSNR提升** | 相比Stage 1 +2 dB | +3 dB |

**额外检查点：**
- ✅ DCVC参数已适应latent压缩
- ✅ 测试集性能未出现过拟合
- ✅ BPP相比Stage 1略有优化

**预计训练时间**: 20-30 epochs

---

### Stage 3 → Stage 4 转换标准

**必须满足以下全部条件：**

| 指标 | 最低要求 | 推荐目标 |
|------|---------|---------|
| **测试PSNR** | ≥ 28.5 dB | ≥ 29.5 dB |
| **训练PSNR** | ≥ 29.5 dB | ≥ 30.5 dB |
| **测试LPIPS** | ≤ 0.32 | ≤ 0.28 |
| **视觉质量** | 明显改善 | 接近原图 |
| **PSNR提升** | 相比Stage 2 +1.5 dB | +2 dB |

**额外检查点：**
- ✅ VAE Decoder重建质量显著提升
- ✅ 高频细节恢复良好
- ✅ 色彩保真度高

**预计训练时间**: 15-20 epochs

---

### Stage 4 → Stage 5 转换标准

**必须满足以下全部条件：**

| 指标 | 最低要求 | 推荐目标 |
|------|---------|---------|
| **测试PSNR** | ≥ 30 dB | ≥ 31 dB |
| **训练PSNR** | ≥ 31 dB | ≥ 32 dB |
| **测试LPIPS** | ≤ 0.28 | ≤ 0.25 |
| **系统稳定性** | 全流程稳定 | 无异常 |

**额外检查点：**
- ✅ 所有组件协同工作良好
- ✅ 测试集性能接近训练集
- ✅ 达到SOTA水平

**预计训练时间**: 15-20 epochs

**Stage 5 可选条件**: 如果测试LPIPS > 0.25，建议进行Stage 5感知优化

---

### Stage 5 结束标准

**目标达成条件：**

| 指标 | 最低要求 | 推荐目标 |
|------|---------|---------|
| **测试PSNR** | ≥ 30 dB | ≥ 31.5 dB |
| **测试LPIPS** | ≤ 0.22 | ≤ 0.18 |
| **主观质量** | FID < 20 | FID < 15 |

**预计训练时间**: 5-10 epochs

---

## 📋 阶段转换决策流程图

```
Start Stage 1
     ↓
训练50个epochs
     ↓
检查指标 ────→ 未达标 ────→ 继续训练10 epochs ───┐
     ↓ 达标                                      │
     ↓                                          │
进入 Stage 2                                     │
     ↓                                          │
训练30个epochs                                   │
     ↓                                          │
检查指标 ────→ 未达标 ────→ 继续训练5 epochs ────┤
     ↓ 达标                                      │
     ↓                                          │
进入 Stage 3                                     │
     ↓                                          │
训练20个epochs                                   │
     ↓                                          │
检查指标 ────→ 未达标 ────→ 调整超参数 ─────────┤
     ↓ 达标                                      │
     ↓                                          │
进入 Stage 4                                     │
     ↓                                          │
训练20个epochs                                   │
     ↓                                          │
检查指标 ────→ 未达标 ────→ 延长训练 ───────────┤
     ↓ 达标                                      │
     ↓                                          │
LPIPS > 0.25? ─Yes→ Stage 5 ──→ 训练10 epochs   │
     ↓ No                          ↓            │
     ↓                       达标/退出           │
完成训练 ←──────────────────────────────────────┘
```

---

## 🔍 训练进度监控脚本

### 快速检查当前阶段是否可以进入下一阶段

```bash
# 查看最近5个epoch的测试结果
tail -100 pretrained/DMC_WAN/1/*.log | grep "Test epoch"

# 检查PSNR趋势
grep "Test epoch" pretrained/DMC_WAN/1/*.log | tail -10 | awk '{print $8, $10}'

# 检查是否满足Stage 1结束条件
echo "请手动检查："
echo "1. 最近测试PSNR是否 >= 25 dB"
echo "2. 训练PSNR是否 >= 26 dB"
echo "3. Loss是否已稳定收敛"
echo "4. 最近5个epoch测试PSNR是否持续上升或稳定"
```

### Python自动检查脚本

创建 `check_stage_transition.py`:

```python
import re
import sys

def check_stage1_ready(log_file):
    """检查Stage 1是否可以进入Stage 2"""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 提取最近10个epoch的测试结果
    test_results = []
    for line in lines:
        if 'Test epoch' in line:
            match = re.search(r'PSNR: ([\d.]+)', line)
            if match:
                test_results.append(float(match.group(1)))
    
    if len(test_results) < 10:
        print("❌ 测试数据不足（需要至少10个epoch）")
        return False
    
    recent_psnr = test_results[-5:]  # 最近5个epoch
    avg_psnr = sum(recent_psnr) / len(recent_psnr)
    max_psnr = max(recent_psnr)
    
    print(f"\n{'='*60}")
    print("Stage 1 → Stage 2 转换检查")
    print(f"{'='*60}")
    print(f"最近5个epoch平均测试PSNR: {avg_psnr:.2f} dB")
    print(f"最近5个epoch最高测试PSNR: {max_psnr:.2f} dB")
    
    # 检查条件
    checks = {
        "平均PSNR >= 25 dB": avg_psnr >= 25.0,
        "最高PSNR >= 26 dB": max_psnr >= 26.0,
        "PSNR稳定（最近5个epoch波动<1dB）": (max(recent_psnr) - min(recent_psnr)) < 1.0,
    }
    
    print(f"\n检查结果:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        all_passed = all_passed and passed
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ Stage 1 已就绪，可以进入 Stage 2！")
        print("\n建议命令:")
        print("  保存当前最佳模型为 checkpoint_best_stage1_wan.pth.tar")
        print("  然后执行 Stage 2 训练命令")
    else:
        print("⚠️  Stage 1 尚未就绪，建议继续训练")
        print(f"  预计还需: {max(0, int((25-avg_psnr)*5))} epochs")
    print(f"{'='*60}\n")
    
    return all_passed

if __name__ == "__main__":
    log_file = "pretrained/DMC_WAN/1/20260125_161914_wan.log"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    check_stage1_ready(log_file)
```

**使用方法:**
```bash
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2
python check_stage_transition.py pretrained/DMC_WAN/1/20260125_161914_wan.log
```

---

### 🔧 Stage 1: Adaptation 预热

**目标**: 学习最优的 Latent (48ch) ↔ Pseudo-YUV (3ch) 映射

**训练配置**:
- 冻结: DCVC + VAE
- 训练: Adaptation Layers (~72万参数)
- 学习率: 1e-5 (含5 epochs warmup)

**输出 Checkpoint**:
- `checkpoint_stage1_wan.pth.tar` (每个epoch)
- `checkpoint_best_stage1_wan.pth.tar` (最佳模型)

```bash
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  train_vd_phase_1_wan.py \
  --mode latent \
  --stage 1 \
  --pretrained_dcvc pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
  --model_path_i checkpoints/cvpr2025_image.pth.tar \
  --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
  --epochs 50 \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --clip_max_norm 0.5 \
  --freeze_dcvc \
  --freeze_vae \
  --num-workers 2 \
  --perceptual_weight 0.02 \
  --perceptual_net alex \
  --use_amp \
  --save
```

**监控指标**:
- ✅ Loss 持续下降 (14 → 5)
- ✅ PSNR 持续上升 (14 → 20 dB)
- ✅ LatentMSE 下降 (0.48 → 0.18)
- ✅ PseudoYUV_MSE > 0 (不能是0.0000)
- ❌ 如果 `Training dcvc parameters` 出现 → 命令缺少 `--freeze_dcvc`

---

### 🔧 Stage 2: DCVC 微调

**目标**: 让 DCVC 学习 latent-friendly 的压缩策略

**训练配置**:
- 冻结: VAE (仅)
- 训练: DCVC + Adaptation (~2100万参数)
- 学习率: 5e-6 (更小，避免破坏预训练)

**输出 Checkpoint**:
- `checkpoint_stage2_wan.pth.tar` (每个epoch)
- `checkpoint_best_stage2_wan.pth.tar` (最佳模型)

```bash
# Stage 1 完成后执行
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
  --nproc_per_node=6 \
  --master_port=29500 \
  train_vd_phase_1_wan.py \
  --mode latent \
  --stage 2 \
  --checkpoint /home/serverdn/hdd-0/wjw/dcvc_rt_g2/pretrained/DMC_WAN/1/checkpoint_best_stage2_wan.pth.tar \
  --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
  --epochs 60 \
  --learning-rate 5e-6 \
  --batch-size 1 \
  --clip_max_norm 1.0 \
  --freeze_vae \
  --num-workers 4 \
  --perceptual_weight 0.03 \
  --perceptual_net alex \
  --use_amp \
  --save
```

**关键变化**:
- ✅ `--stage 2` (指定阶段，区分checkpoint)
- ❌ 移除 `--freeze_dcvc` (解冻 DCVC)
- ✅ `--checkpoint` 加载 Stage 1 最佳模型
- 📉 学习率降低到 5e-6
- 📉 batch_size=1 (显存需求增加)
- 📈 perceptual_weight=0.03

---

### 🔧 Stage 3: VAE Decoder 微调

**目标**: 优化从 Latent 到 RGB 的重建质量

**训练配置**:
- 冻结: VAE Encoder (仅)
- 训练: VAE Decoder + DCVC + Adaptation
- 学习率: 2e-6 (VAE decoder 非常敏感)

**输出 Checkpoint**:
- `checkpoint_stage3_wan.pth.tar` (每个epoch)
- `checkpoint_best_stage3_wan.pth.tar` (最佳模型)

```bash
# Stage 2 完成后执行
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  train_vd_phase_1_wan.py \
  --mode latent \
  --stage 3 \
  --checkpoint pretrained/DMC_WAN/1/checkpoint_best_stage2_wan.pth.tar \
  --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
  --epochs 20 \
  --learning-rate 2e-6 \
  --batch-size 1 \
  --clip_max_norm 1.5 \
  --freeze_vae_encoder_only \
  --num-workers 2 \
  --perceptual_weight 0.05 \
  --perceptual_net alex \
  --use_amp \
  --save


  CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun \
  --nproc_per_node=5 \
  --master_port=29500 \
  train_vd_phase_1_wan.py \
  --mode latent \
  --stage 3 \
  --checkpoint pretrained/DMC_WAN/1/checkpoint_best_stage2_wan.pth.tar \
  --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
  --epochs 20 \
  --learning-rate 2e-6 \
  --batch-size 1 \
  --clip_max_norm 1.5 \
  --freeze_vae_encoder_only \
  --num-workers 2 \
  --perceptual_weight 0.05 \
  --perceptual_net alex \
  --use_amp \
  --save
```

**关键变化**:
- ✅ `--stage 3` (指定阶段，区分checkpoint)
- ✅ `--checkpoint` 加载 Stage 2 最佳模型
- ✅ `--freeze_vae_encoder_only` (只冻结 VAE encoder，训练 decoder)
- 📉 学习率降低到 2e-6
- 📈 perceptual_weight=0.05 (加强 LPIPS 优化)
- 📈 clip_max_norm=1.5 (VAE decoder 梯度较大)

---

### 🔧 Stage 4: 端到端微调

**目标**: 全系统协同优化，榨取最后性能

**训练配置**:
- 冻结: 无
- 训练: 全部组件 (~1.6亿参数)
- 学习率: 1e-6 (非常小的微调)

**输出 Checkpoint**:
- `checkpoint_stage4_wan.pth.tar` (每个epoch)
- `checkpoint_best_stage4_wan.pth.tar` (最佳模型)

```bash
# Stage 3 完成后执行
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  train_vd_phase_1_wan.py \
  --mode latent \
  --stage 4 \
  --checkpoint pretrained/DMC_WAN/1/checkpoint_best_stage3_wan.pth.tar \
  --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
  --epochs 20 \
  --learning-rate 1e-6 \
  --batch-size 1 \
  --clip_max_norm 2.0 \
  --num-workers 2 \
  --perceptual_weight 0.05 \
  --perceptual_net alex \
  --use_amp \
  --save

  CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun \
  --nproc_per_node=5 \
  --master_port=29500 \
  train_vd_phase_1_wan.py \
  --mode latent \
  --stage 4 \
  --checkpoint pretrained/DMC_WAN/1/checkpoint_stage4_wan.pth.tar \
  --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
  --epochs 40 \
  --learning-rate 1e-6 \
  --batch-size 1 \
  --clip_max_norm 2.0 \
  --num-workers 2 \
  --perceptual_weight 0.05 \
  --perceptual_net alex \
  --use_amp \
  --save
```

**关键变化**:
- ✅ `--stage 4` (指定阶段，区分checkpoint)
- ✅ `--checkpoint` 加载 Stage 3 最佳模型
- ❌ 移除所有 freeze 参数 (全部解冻)
- 📉 学习率降低到 1e-6
- 📈 clip_max_norm=2.0 (全局梯度裁剪)

---

### 🔧 Stage 5: 感知质量优化 (可选)

**目标**: 进一步降低 LPIPS，牺牲少量 PSNR

**训练配置**:
- 冻结: 无
- 训练: 全部组件
- 学习率: 5e-7 (精细调整)
- perceptual_weight: 0.10 (翻倍，强调感知质量)

**输出 Checkpoint**:
- `checkpoint_stage5_wan.pth.tar` (每个epoch)
- `checkpoint_best_stage5_wan.pth.tar` (最佳模型)

```bash
# Stage 4 完成后，如果 LPIPS > 0.25 执行
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  train_vd_phase_1_wan.py \
  --mode latent \
  --stage 5 \
  --checkpoint pretrained/DMC_WAN/1/checkpoint_best_stage4_wan.pth.tar \
  --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
  --epochs 10 \
  --learning-rate 5e-7 \
  --batch-size 1 \
  --clip_max_norm 2.0 \
  --num-workers 2 \
  --perceptual_weight 0.10 \
  --perceptual_net alex \
  --use_amp \
  --save
```

**关键变化**:
- ✅ `--stage 5` (指定阶段，区分checkpoint)
- ✅ `--checkpoint` 加载 Stage 4 最佳模型
- 📈 perceptual_weight=0.10 (翻倍，强调感知质量)
- 📉 学习率降低到 5e-7

---

## 📋 训练参数速查表

### Checkpoint 文件命名

| 阶段 | `--stage` | 输出文件 | 输入 `--checkpoint` |
|------|-----------|---------|---------------------|
| Stage 1 | `1` | `checkpoint_best_stage1_wan.pth.tar` | (无，从头训练) |
| Stage 2 | `2` | `checkpoint_best_stage2_wan.pth.tar` | `checkpoint_best_stage1_wan.pth.tar` |
| Stage 3 | `3` | `checkpoint_best_stage3_wan.pth.tar` | `checkpoint_best_stage2_wan.pth.tar` |
| Stage 4 | `4` | `checkpoint_best_stage4_wan.pth.tar` | `checkpoint_best_stage3_wan.pth.tar` |
| Stage 5 | `5` | `checkpoint_best_stage5_wan.pth.tar` | `checkpoint_best_stage4_wan.pth.tar` |

**重要**: 每个阶段使用不同的 `--stage` 参数，确保 checkpoint 不会互相覆盖！

### 冻结策略参数

| 参数 | 说明 | 使用阶段 |
|------|------|---------|
| `--freeze_dcvc` | 冻结 DCVC_RT 全部参数 | Stage 1 |
| `--freeze_vae` | 冻结 VAE 全部参数 | Stage 1, 2 |
| `--freeze_vae_encoder_only` | 只冻结 VAE encoder，训练 decoder | Stage 3 |
| (无 freeze 参数) | 全部解冻 | Stage 4, 5 |

### 各阶段学习率

| 阶段 | 学习率 | 原因 |
|------|--------|------|
| Stage 1 | 1e-5 | Adaptation 随机初始化，需要较大学习率 |
| Stage 2 | 5e-6 | DCVC 预训练，避免破坏权重 |
| Stage 3 | 2e-6 | VAE decoder 非常敏感 |
| Stage 4 | 1e-6 | 端到端精细调整 |
| Stage 5 | 5e-7 | 感知优化微调 |

### perceptual_weight 设置

| 阶段 | 权重 | 说明 |
|------|------|------|
| Stage 1 | 0.02 | 基础感知约束 |
| Stage 2 | 0.03 | 略微增加 |
| Stage 3 | 0.05 | 开始关注感知质量 |
| Stage 4 | 0.05 | 保持 |
| Stage 5 | 0.10 | 强调感知质量 |

---

## 🎯 训练监控

### 健康训练标志

| 阶段 | Loss变化 | PSNR变化 | LPIPS变化 | 备注 |
|------|---------|---------|-----------|------|
| Stage 1 | 持续下降 | +6 dB | -0.4 | 快速收敛 |
| Stage 2 | 缓慢下降 | +2 dB | -0.1 | DCVC适应 |
| Stage 3 | 稳定下降 | +1 dB | -0.1 | Decoder优化 |
| Stage 4 | 微小下降 | +1 dB | 稳定 | 全局微调 |
| Stage 5 | LPIPS主导 | ±0.5 dB | -0.05 | 感知优化 |

### 异常标志

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss 突然上升 > 20% | 学习率过高 / 梯度爆炸 | 降低学习率 / 增加 clip_max_norm |
| PSNR 下降 | 过拟合 / 配置错误 | 检查 checkpoint 加载 / 减少 epochs |
| LPIPS 持续上升 | perceptual_weight 过低 | 增加 perceptual_weight |
| PseudoYUV_MSE = 0.0000 | 梯度流断开 | 检查代码 / detach 问题 |
| 测试集远低于训练集 | 过拟合 | 减少 epochs / 数据增强 |

---

## 🔬 性能对标

### SOTA 方法对比 (UVG 数据集)

| 方法 | BPP | PSNR | LPIPS | 说明 |
|------|-----|------|-------|------|
| H.265 | 0.25 | 28 dB | 0.45 | 传统编码器 |
| DCVC-HEM | 0.15 | 37 dB | 0.15 | SOTA 神经编码 |
| GLC-Paper | 0.04 | - | 0.15 | 论文报告 |
| **本方案 (Stage 4)** | 0.22 | **30-32 dB** | **0.22-0.25** | 目标性能 |

---

## 📊 参数统计

### 模型大小

```
DCVC_RT: ~20M 参数
Wan VAE (Encoder + Decoder): ~140M 参数
Adaptation Layers: ~0.72M 参数
--------------------------------------
Total: ~160M 参数
```

### 各阶段训练参数

| 阶段 | 训练参数量 | 显存需求 (每卡) |
|------|-----------|----------------|
| Stage 1 | 0.72M | ~8GB |
| Stage 2 | 21M | ~14GB |
| Stage 3 | ~70M | ~18GB |
| Stage 4 | 160M | ~22GB |

---

## 🚨 故障排查

### 1. OOM (Out of Memory)

**解决方案优先级:**
1. ✅ 启用 AMP (`--use_amp`)
2. ✅ 减小 batch_size 到 1
3. ✅ 减少 GPU 数量
4. ✅ 关闭 LPIPS (`--perceptual_weight 0`)

### 2. "unrecognized arguments" 错误

**原因**: 多行命令格式错误

**解决**: 确保反斜杠 `\` 后无任何字符（包括空格和注释）

```bash
# ✅ 正确
torchrun \
  --nproc_per_node=8 \
  train.py

# ❌ 错误（反斜杠后有注释）
torchrun \  # 这里有注释
  --nproc_per_node=8 \
  train.py
```

### 3. DCVC 意外参与训练

**检查**: 日志应该显示
```
Training adaptation parameters: 720,243
```

**如果显示** `Training dcvc parameters: 20,691,456` → 缺少 `--freeze_dcvc`

### 4. VAE 加载失败

```python
import sys
sys.path.insert(0, '/home/serverdn/hdd-0/wjw/Wan2.2-main')
from wan.modules.vae2_2 import Wan2_2_VAE

# 测试 VAE 是否可以加载
vae = Wan2_2_VAE(vae_pth="path/to/Wan2.2_VAE.pth")
```

---

## 📁 文件结构

```
dcvc_rt_g2/
├── src/
│   └── models/
│       ├── wan_vae_wrapper.py        # Wan VAE 封装
│       ├── video_t_g_wan.py          # DMC_WAN 主模型
│       ├── video_t.py                # 原始 DCVC_RT
│       └── image_model.py            # I帧模型
├── train_vd_phase_1_wan.py           # GLC 训练脚本
├── README_WAN_INTEGRATION.md         # 本文档
├── Wan2.2-main/
│   └── checkpoints/
│       └── Wan2.2_VAE.pth            # VAE 权重
└── pretrained/
    ├── DMC_SLF/
    │   └── checkpoint_best_loss_vd.pth.tar  # DCVC 预训练
    └── DMC_WAN/
        └── 1/
            ├── checkpoint_wan.pth.tar
            └── checkpoint_best_loss_wan.pth.tar
```

---

## 📖 引用

```bibtex
@article{qi2024glc,
  title={Generative Latent Coding for Ultra-Low Bitrate Image and Video Compression},
  author={Qi, Linfeng and Jia, Zhaoyang and Li, Jiahao and Li, Bin and Li, Houqiang and Lu, Yan},
  journal={arXiv preprint},
  year={2024}
}

@article{wan2025,
  title={Wan: Open and Advanced Large-Scale Video Generative Models},
  author={Team Wan and Wang, Ang and ...},
  journal={arXiv preprint arXiv:2503.20314},
  year={2025}
}
```

---

## 📜 许可证

- DCVC_RT: MIT License
- Wan2.2: Apache 2.0 License



测试脚本：


python3 visualize_comparison.py \
  --dcvc_checkpoint pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
  --test_dir /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60 \
  --qp 37 \
  --match_bpp \
  --qp_list "0,2,4,6,8,10,12,14,16,18,20,25,30,35,40" \
  --num_frames 10 \
  --output_dir visualization_results_stage4_v2




python3 evaluate_paper.py \
  --wan_qps "55,59,63,65,67,69,71" \
  --dcvc_qps "0,4,8,12,16,20,25,30,40,50,63" \
  --num_frames 33 \
  --output_dir paper_results


cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun \
  --nproc_per_node=5 \
  --master_port=29500 \
  train_vd_phase_1_wan.py \
  --mode latent \
  --stage 4 \
  --checkpoint pretrained/DMC_WAN/1/checkpoint_stage4_wan.pth.tar \
  --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
  --epochs 40 \
  --learning-rate 5e-5 \
  --batch-size 1 \
  --clip_max_norm 2.0 \
  --freeze_vae \
  --num-workers 2 \
  --perceptual_weight 0.05 \
  --perceptual_net alex \
  --use_amp \
  --save \
  --lr_schedule cosine \
  --warmup_epochs 3 \
  --reset_epoch \
  --rgb_weight 0.3 \
  --adaptation_lr_scale 5.0 \
  --dcvc_lr_scale 0.5


  更新后指令：
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=4 train_vd_phase_1_wan.py \
  --model DMC_WAN \
  --mode latent \
  --pretrained_dcvc pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
  --wan_vae_checkpoint Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
  --freeze_vae \
  --perceptual_weight 0.05 \
  --perceptual_net alex \
  --glc_root /home/serverdn/hdd-0/wjw/GLC \
  --epochs 40 \
  --learning-rate 5e-5 \
  --clip_max_norm 2.0 \
  --batch-size 1 \
  --num-workers 2 \
  --quality-level 1 \
  --use_amp \
  --test_dataset /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/ \
  --test_filelist /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/test.txt \
  --checkpoint pretrained/DMC_WAN/1/checkpoint_stage4_wan.pth.tar \
  --stage 4 \
  --lr_schedule cosine \
  --warmup_epochs 3 \
  --reset_epoch \
  --rgb_weight 0.3 \
  --dcvc_lr_scale 0.5 \
  --adaptation_lr_scale 5.0 \
  --ramp_start 1.0 \
  --ramp_epochs 10