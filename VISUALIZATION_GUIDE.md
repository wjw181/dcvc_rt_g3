# 可视化对比测试指南

## 功能说明

这个工具可以对比 **WAN 集成模型** 和 **原始 DCVC_RT 模型** 的压缩效果，生成可视化对比图。

## 快速开始

### 方法1：使用默认配置（推荐）

```bash
cd /home/serverdn/hdd-0/wjw/dcvc_rt_g2
bash run_visualization.sh
```

### 方法2：自定义参数

```bash
python3 visualize_comparison.py \
    --wan_checkpoint pretrained/DMC_WAN/1/checkpoint_best_stage1_wan.pth.tar \
    --dcvc_checkpoint pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar \
    --wan_vae Wan2.2-main/checkpoints/Wan2.2_VAE.pth \
    --test_dir /path/to/test/images \
    --output_dir visualization_results \
    --qp 37 \
    --num_frames 5
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wan_checkpoint` | `pretrained/DMC_WAN/1/checkpoint_best_stage1_wan.pth.tar` | WAN 模型 checkpoint |
| `--dcvc_checkpoint` | `pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar` | DCVC_RT 模型 checkpoint |
| `--wan_vae` | `Wan2.2-main/checkpoints/Wan2.2_VAE.pth` | WAN VAE checkpoint |
| `--i_frame_model` | `checkpoints/cvpr2025_image.pth.tar` | I帧模型 |
| `--test_dir` | `/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60` | 测试图像目录 |
| `--output_dir` | `visualization_results` | 输出目录 |
| `--qp` | `37` | 量化参数 (QP) |
| `--num_frames` | `5` | 测试帧数 |

## 输出结果

### 1. 可视化图像

每一帧会生成一张对比图，包含：

**第一行：**
- 原始图像
- WAN 重建图像（带 PSNR 和 BPP）
- DCVC_RT 重建图像（带 PSNR 和 BPP）

**第二行：**
- WAN 误差图（热力图）
- DCVC_RT 误差图（热力图）
- 误差差异图（蓝色=WAN更好，红色=DCVC更好）

文件名：`comparison_frame_001.png`, `comparison_frame_002.png`, ...

### 2. 统计结果

`statistics.txt` 文件包含：
- 平均 PSNR 对比
- 平均 BPP 对比
- 平均 MSE 对比

### 3. 控制台输出

实时显示每一帧的对比结果：
```
测试帧 1: im00002.png
WAN 模型压缩中...
  PSNR: 28.45 dB
  BPP: 0.1234
  MSE: 0.001234

DCVC_RT 模型压缩中...
  PSNR: 27.89 dB
  BPP: 0.1456
  MSE: 0.001456

对比结果:
  PSNR 差异: +0.56 dB
  BPP 差异: -0.0222
  MSE 差异: -0.000222
```

## 示例输出

### 统计结果示例

```
WAN vs DCVC_RT 对比统计
============================================================

测试帧数: 5
QP: 37

平均 PSNR:
  WAN:      28.45 dB
  DCVC_RT:  27.89 dB
  差异:     +0.56 dB

平均 BPP:
  WAN:      0.1234
  DCVC_RT:  0.1456
  差异:     -0.0222

平均 MSE:
  WAN:      0.001234
  DCVC_RT:  0.001456
  差异:     -0.000222
```

## 测试不同的 QP 值

```bash
# 低码率 (QP=42)
python3 visualize_comparison.py --qp 42 --output_dir results_qp42

# 中码率 (QP=37, 默认)
python3 visualize_comparison.py --qp 37 --output_dir results_qp37

# 高码率 (QP=32)
python3 visualize_comparison.py --qp 32 --output_dir results_qp32
```

## 测试不同的视频序列

```bash
# HEVC_B 序列
python3 visualize_comparison.py \
    --test_dir /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/BasketballDrive_1920x1080_50 \
    --output_dir results_basketball

# HEVC_C 序列
python3 visualize_comparison.py \
    --test_dir /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_C/RaceHorses_832x480_30 \
    --output_dir results_racehorses
```

## 查看结果

```bash
# 查看生成的图像
ls -lh visualization_results/

# 查看统计结果
cat visualization_results/statistics.txt

# 在图形界面中查看图像
# (如果有 X11 转发)
eog visualization_results/comparison_frame_001.png
```

## 预期结果

### WAN 模型的优势

- ✅ **更低的 BPP**：在相同 QP 下，WAN 模型通常有更低的码率
- ✅ **更好的感知质量**：在低码率下，WAN 模型的视觉质量更好
- ✅ **更好的语义保留**：WAN 在潜空间压缩，更好地保留语义信息

### DCVC_RT 模型的优势

- ✅ **更高的 PSNR**：在高码率下，DCVC_RT 可能有更高的 PSNR
- ✅ **更快的速度**：DCVC_RT 不需要 VAE 编解码
- ✅ **更稳定**：DCVC_RT 是成熟的模型

## 故障排查

### 问题1: 找不到 checkpoint

```bash
# 检查文件是否存在
ls -lh pretrained/DMC_WAN/1/checkpoint_best_stage1_wan.pth.tar
ls -lh pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar
```

### 问题2: CUDA out of memory

```bash
# 减少测试帧数
python3 visualize_comparison.py --num_frames 2
```

### 问题3: 找不到测试图像

```bash
# 检查测试目录
ls /home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60/

# 或使用其他测试目录
python3 visualize_comparison.py --test_dir /path/to/your/images
```

### 问题4: 缺少依赖

```bash
# 安装 matplotlib
pip install matplotlib

# 安装 imageio
pip install imageio
```

## 高级用法

### 批量测试多个 QP

```bash
#!/bin/bash
for qp in 32 37 42 47; do
    echo "Testing QP=$qp"
    python3 visualize_comparison.py \
        --qp $qp \
        --output_dir results_qp${qp} \
        --num_frames 10
done
```

### 生成 RD 曲线数据

```bash
# 测试多个 QP 并收集数据
for qp in 22 27 32 37 42 47; do
    python3 visualize_comparison.py \
        --qp $qp \
        --output_dir results_qp${qp} \
        --num_frames 20
    
    # 提取统计数据
    grep "平均" results_qp${qp}/statistics.txt >> rd_curve_data.txt
done
```

## 注意事项

1. **确保有足够的 GPU 内存**：两个模型都需要加载到 GPU
2. **测试图像格式**：目前只支持 PNG 格式
3. **图像命名**：图像文件名应该可以按字母顺序排序
4. **输出目录**：会自动创建，如果已存在会覆盖

## 相关文件

- `visualize_comparison.py` - 主测试脚本
- `run_visualization.sh` - 快速启动脚本
- `VISUALIZATION_GUIDE.md` - 本文档

---

**创建时间**: 2026-02-01  
**版本**: 1.0
