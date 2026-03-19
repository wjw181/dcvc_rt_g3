# 测试脚本说明

本目录包含用于测试和对比 WAN 集成模型的各种脚本。

## 📁 文件列表

### 主要测试脚本

1. **`test_stage1.sh`** - 测试阶段一模型
   - 使用 checkpoint: `checkpoint_best_stage1_wan.pth.tar`
   - 输出目录: `tests/results_stage1_metrics/`

2. **`test_stage2.sh`** - 测试阶段二模型
   - 使用 checkpoint: `checkpoint_stage2_wan.pth.tar`
   - 输出目录: `tests/results_stage2_metrics/`

3. **`test_both_stages.sh`** ⭐ **推荐使用**
   - 自动运行阶段一和阶段二的测试
   - 方便对比两个阶段的效果

### 辅助脚本

4. **`install_perceptual_metrics.sh`** - 安装感知质量指标库
   - 安装 pytorch-msssim (SSIM, MS-SSIM)
   - 安装 lpips (感知相似性)

5. **`run_test_with_metrics.sh`** - 完整测试流程
   - 自动安装依赖
   - 运行阶段二测试

## 🚀 快速开始

### 方法一：运行完整对比测试（推荐）

```bash
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2/tests
./test_both_stages.sh
```

这将：
1. 自动安装所需依赖
2. 测试阶段一模型
3. 测试阶段二模型
4. 生成详细的对比报告

### 方法二：单独测试某个阶段

```bash
# 只测试阶段一
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2/tests
./test_stage1.sh

# 或只测试阶段二
./test_stage2.sh
```

### 方法三：手动安装依赖后测试

```bash
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2/tests

# 1. 安装依赖
./install_perceptual_metrics.sh

# 2. 运行测试
./test_stage2.sh
```

## 📊 测试指标说明

测试脚本会计算以下质量指标：

| 指标 | 说明 | 范围 | 越高越好？ |
|------|------|------|-----------|
| **PSNR** | 峰值信噪比 | 0-∞ dB | ✅ 是 |
| **BPP** | 每像素比特数（码率） | 0-∞ | ❌ 否（越低越好）|
| **SSIM** | 结构相似性 | 0-1 | ✅ 是 |
| **MS-SSIM** | 多尺度结构相似性 | 0-1 | ✅ 是 |
| **LPIPS** | 感知相似性 | 0-∞ | ❌ 否（越低越好）|
| **MSE** | 均方误差 | 0-∞ | ❌ 否（越低越好）|

## 📈 查看结果

测试完成后，结果保存在 tests 文件夹内：

```bash
# 查看阶段一统计结果
cat results_stage1_metrics/statistics.txt

# 查看阶段二统计结果
cat results_stage2_metrics/statistics.txt

# 查看可视化对比图
ls results_stage1_metrics/*.png
ls results_stage2_metrics/*.png
```

## 🔧 自定义测试参数

如果需要修改测试参数（如 QP、测试帧数等），可以编辑对应的脚本文件：

```bash
# 编辑阶段二测试脚本
vim test_stage2.sh

# 可修改的参数：
# - QP: 量化参数（默认 37）
# - NUM_FRAMES: 测试帧数（默认 5）
# - TEST_DIR: 测试视频目录
# - OUTPUT_DIR: 输出目录
```

## 📝 注意事项

1. **环境要求**：
   - 需要激活 `dcvc_rt` conda 环境
   - 需要安装 PyTorch 和相关依赖

2. **GPU 要求**：
   - 测试需要 GPU 支持
   - 确保有足够的显存（建议 8GB+）

3. **数据要求**：
   - 确保测试数据目录存在
   - 默认使用 Johnny_1280x720_60 序列

4. **首次运行**：
   - 首次运行会自动下载 LPIPS 预训练模型
   - 可能需要几分钟时间

## 🐛 故障排除

### 问题：找不到 conda 命令
```bash
# 手动激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dcvc_rt
```

### 问题：缺少依赖库
```bash
# 手动安装
pip install pytorch-msssim lpips
```

### 问题：CUDA 内存不足
```bash
# 减少测试帧数
# 编辑脚本，将 NUM_FRAMES=5 改为 NUM_FRAMES=2
```

## 📂 结果目录结构

测试完成后，tests 文件夹结构如下：

```
tests/
├── README.md                          # 本文件
├── test_stage1.sh                     # 测试脚本
├── test_stage2.sh
├── test_both_stages.sh
├── install_perceptual_metrics.sh
├── run_test_with_metrics.sh
│
├── results_stage1_metrics/            # 阶段一测试结果
│   ├── statistics.txt                 # 统计数据
│   ├── comparison_frame_001.png       # 对比图
│   ├── comparison_frame_002.png
│   └── ...
│
└── results_stage2_metrics/            # 阶段二测试结果
    ├── statistics.txt                 # 统计数据
    ├── comparison_frame_001.png       # 对比图
    ├── comparison_frame_002.png
    └── ...
```

## 📧 联系方式

如有问题，请查看主目录的 `README_WAN_INTEGRATION.md` 文档。
