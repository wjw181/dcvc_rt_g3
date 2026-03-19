# DCVC RT 训练指南

本项目包含两个训练阶段，用于训练视频压缩模型。

## 训练阶段概述

### 第一阶段：3-7帧训练 (`train_vd_phase_1.py`)

第一阶段负责短序列（3-7帧）的多阶段训练，逐步增加训练帧数。

**特点：**
- 训练轮数：120 epochs
- 帧数范围：3-7帧
- 帧数增长策略：
  - Epoch 0-50: 固定3帧
  - Epoch 50-60: 3帧 → 4帧
  - Epoch 60-70: 4帧 → 5帧
  - Epoch 70-80: 5帧 → 6帧
  - Epoch 80-120: 6帧 → 7帧
- Batch size策略：
  - 帧数 ≤ 5: 使用完整 batch_size (默认4)
  - 帧数 > 5: 使用 batch_size/2 (默认2)
- 学习率衰减：
  - Epoch 60: 衰减至 0.4 倍
  - Epoch 100: 衰减至 0.1 倍
  - Epoch 110: 衰减至 0.04 倍
  - Epoch 115: 衰减至 0.01 倍

**训练数据集：**
- 使用 Vimeo-90k 数据集（短序列）
- 数据集路径在 `src/dataload.py` 中配置

**使用方法：**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29601 train_vd_phase_1.py 

```

### 第二阶段：长序列微调 (`train_vd_phase_2.py`)

第二阶段在第一阶段的基础上进行长序列（8-32帧）的微调训练。

**特点：**
- 训练轮数：180 epochs
- 帧数范围：3-32帧
- 帧数增长策略：
  - Epoch 0-120: 3-7帧（与第一阶段相同）
  - Epoch 120+: 8-32帧，每2个epoch增加4帧
    - Epoch 120: 8帧
    - Epoch 122: 12帧
    - Epoch 124: 16帧
    - ... 最多到32帧
- Batch size策略：
  - 帧数 ≤ 5: batch_size = 4
  - 帧数 6-7: batch_size = 2
  - 帧数 ≥ 8: batch_size = 1
- 学习率衰减：
  - Epoch 100: 衰减至 0.4 倍
  - Epoch 160: 衰减至 0.1 倍
  - Epoch 170: 衰减至 0.04 倍
  - Epoch 175: 衰减至 0.01 倍
- **默认加载第一阶段检查点**：`pretrained/DMC_slf_yuv420/1/checkpoint_vd.pth.tar`（加载第一阶段训练120个epoch之后的检查点）

**训练数据集：**
- Epoch < 120: 使用 Vimeo-90k 数据集（短序列）
- Epoch ≥ 120: 使用长序列数据集
- 数据集路径在 `src/dataload.py` 中配置

**使用方法：**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29601 train_vd_phase_2.py 
```

**注意：** 第二阶段默认加载第一阶段训练完成（120个epoch）后的检查点，从epoch 120开始继续训练。

## 数据集配置

### 训练数据集路径

训练数据集路径在 `src/dataload.py` 的 `DataSet` 类中配置，支持通过初始化参数自定义：

**默认路径：**
- **短序列数据集（3-7帧）：**
  - 数据根目录：`/home/admin1/Data/data/vimeo_septuplet/sequences/`
  - 文件列表：`/home/admin1/Data/data/vimeo_septuplet/test.txt`
- **长序列数据集（≥8帧）：**
  - 数据根目录：`/home/admin1/Data/data/`
  - 文件列表：`/home/admin1/Data/data/output_paths_no_transitions.txt`

**修改方法：**

方法1：修改训练脚本中的 `DataSet()` 初始化（推荐）
```python
# 在 train_vd_phase_1.py 或 train_vd_phase_2.py 中
train_dataset = DataSet(
    short_seq_root="/your/path/to/vimeo_septuplet/sequences/",
    short_seq_filelist="/your/path/to/vimeo_septuplet/test.txt",
    long_seq_root="/your/path/to/long_sequences/",
    long_seq_filelist="/your/path/to/long_sequences/filelist.txt"
)
```

方法2：直接编辑 `src/dataload.py` 中的默认路径参数

### 测试数据集路径

测试数据集路径通过训练脚本的命令行参数指定：

- `--test_dataset`: 测试数据集根目录
- `--test_filelist`: 测试文件列表路径

**默认值：**
- `--test_dataset`: `/home/admin1/Data/data/testdata/HEVC_E/Johnny_1280x720_60`
- `--test_filelist`: `/home/admin1/Data/data/testdata/HEVC_E/image_paths.txt`

## 主要参数说明

### 通用参数

- `--model`: 模型架构（默认：DMC）
- `--quality-level`: 质量级别（phase_1: 1, phase_2: 2）
- `--epochs`: 训练轮数（phase_1: 120, phase_2: 180）
- `--batch-size`: 批处理大小（默认：4）
- `--learning-rate`: 初始学习率（默认：1e-4）
- `--test-batch-size`: 测试批处理大小（默认：1）
- `--num-workers`: 数据加载线程数（默认：4）
- `--clip_max_norm`: 梯度裁剪最大值（默认：1.0）
- `--local-rank`: 分布式训练本地rank（由torch.distributed.launch自动设置）

### 模型相关参数

- `--model_path_i`: I帧模型检查点路径（默认：`checkpoints/cvpr2025_image.pth.tar`）
- `--checkpoint`: 继续训练的检查点路径
  - phase_1: 默认 None（从头训练）
  - phase_2: 默认 `pretrained/DMC_slf_yuv420/1/checkpoint_vd.pth.tar`（从phase_1继续）

### 测试相关参数

- `--test_dataset`: 测试数据集根目录
- `--test_filelist`: 测试文件列表路径

## 损失函数

训练使用率失真损失函数（Rate-Distortion Loss），定义如下：

### 损失函数组成

1. **BPP损失（Bit Per Pixel Loss）**：衡量压缩比特率
   ```python
   bpp_loss = result["bpp"]
   ```

2. **MSE损失（Mean Squared Error Loss）**：衡量重建质量
   ```python
   mse_loss = result["mse"]
   ```

3. **综合损失**：根据训练阶段动态调整
   - **Epoch 0-19**：仅优化质量（MSE权重放大）
     ```python
     loss = mse_loss * 100000
     ```
   - **Epoch ≥ 20**：率失真优化
     ```python
     loss = lambda * mse_loss + bpp_loss
     ```

### Lambda参数计算

Lambda参数通过QP（Quantization Parameter）映射得到：

- **QP范围**：[0, 71]（共72个QP值）
- **Lambda范围**：[1, 768]（对数映射）
- **映射公式**：
  ```python
  scale = qp / (q_num - 1)  # q_num = 72
  ln_lambda = ln(1) + scale * (ln(768) - ln(1))
  lambda = exp(ln_lambda)
  ```

### 训练时的Lambda策略

- **第一帧（idx=1）**：使用 `1.0 * lambda_qs`
- **其他帧**：使用 `lambda_qs`
- **帧权重**：不同帧位置应用不同权重 `weights[idx % 8]`
  ```python
  weights = [0.5, 1.2, 0.5, 0.9, 0.5, 1.2, 0.5, 0.9]
  ```

### QP采样策略

- **Epoch < 48**：固定使用 QP=71（最高质量）
- **Epoch ≥ 48**：
  - 每3个batch使用一次 QP=71
  - 其他batch随机采样 QP ∈ [0, 70]

## 训练流程

1. **准备数据**：确保训练和测试数据集路径正确配置
2. **第一阶段训练**：运行 `train_vd_phase_1.py` 进行3-7帧训练
3. **第二阶段训练**：运行 `train_vd_phase_2.py` 进行长序列微调（自动加载第一阶段检查点）

## 测试脚本

项目提供了两个测试脚本用于评估训练好的模型，它们的主要区别在于使用的模型版本不同：

### `test_video_yuv_o.py` - 新版本模型测试

使用改进版本的DMC模型（`src/models/video_t.py`），

**使用方法：**
```bash
python test_video_yuv_o.py 

### `test_video_yuv_ori.py` - 原始版本模型测试

使用原始版本的DMC模型（`src/models/video_t_ori.py`），

**使用方法：**
```bash
python test_video_yuv_ori.py \
```

### 测试参数说明

- `--yuv`: YUV420格式视频文件路径
- `--width`: 视频宽度（像素）
- `--height`: 视频高度（像素）
- `--ckpt_i`: I帧模型（DMCI）检查点路径
- `--ckpt_p`: P帧模型（DMC）检查点路径
- `--QP`: 量化参数（0-71，值越大质量越高）
- `--max_frames`: 测试的最大帧数（包括I帧），-1表示测试全部帧
- `--device`: 运行设备（默认：cuda）

### 测试输出

两个测试脚本都会输出：
- 每帧的加权PSNR、Y/U/V通道PSNR和BPP
- 整个序列的平均PSNR、MSE和BPP

## 输出文件

训练过程中会生成以下文件：

- `pretrained/{model}/{quality_level}/checkpoint_vd.pth.tar`: 每个epoch的检查点
- `pretrained/{model}/{quality_level}/checkpoint_best_loss_vd.pth.tar`: 最佳损失检查点
- `pretrained/{model}/{quality_level}/{timestamp}.log`: 训练日志

## 注意事项

1. **分布式训练配置**：
   - 使用 `CUDA_VISIBLE_DEVICES` 指定使用的GPU
   - `--nproc_per_node` 必须与可见GPU数量一致
   - `--master_port` 用于多机训练时的端口通信，单机训练时建议使用不同端口避免冲突

2. **检查点加载**：
   - 第一阶段训练完成后，检查点保存在 `pretrained/DMC_slf_yuv420/1/` 目录
   - 第二阶段默认加载第一阶段120个epoch后的检查点，从epoch 120开始继续训练
   - 如需从头训练可设置 `--checkpoint None`

3. **数据集路径**：
   - 训练数据集路径需要根据实际环境修改 `src/dataload.py` 中的默认路径参数
   - 测试数据集路径可通过命令行参数 `--test_dataset` 和 `--test_filelist` 灵活指定

4. **训练策略**：
   - 前20个epoch专注于质量优化（MSE权重放大）
   - 20个epoch后开始率失真优化（平衡质量和比特率）
   - QP采样策略在epoch 48后启用随机采样，增加训练的鲁棒性

