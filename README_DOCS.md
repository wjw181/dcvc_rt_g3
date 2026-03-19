# 项目文档说明

本项目的所有文档已整合，保持简洁清晰。

## 📚 主要文档

### 1. README_WAN_INTEGRATION.md（完整指南）⭐
**这是最重要的文档，包含所有内容：**
- ✅ Wan VAE 集成说明
- ✅ 完整训练方案（Stage 1-5）
- ✅ 阶段转换标准
- ✅ 过拟合问题解决方案
- ✅ 训练命令大全
- ✅ 故障排查指南
- ✅ Checkpoint兼容性说明

**查看方式**：
```bash
cat README_WAN_INTEGRATION.md
# 或在编辑器中打开
```

### 2. README.md（项目概述）
- 项目基本介绍
- 快速开始指南

## 🚀 快速开始

### 训练 Stage 1
```bash
# 查看完整命令
grep -A 20 "Stage 1: Adaptation" README_WAN_INTEGRATION.md
```

### 训练 Stage 2（改进版，解决过拟合）
```bash
# 8卡服务器
bash run_improved_training.sh
```

### 检查是否可以进入下一阶段
```bash
# 查看阶段转换标准
grep -A 30 "阶段转换标准" README_WAN_INTEGRATION.md
```

## 🔧 辅助工具

### 训练脚本
- `train_vd_phase_1_wan.py` - Stage 1-5 训练
- `train_vd_phase_2_improved.py` - Stage 2 改进训练（解决过拟合）
- `run_improved_training.sh` - 一键启动脚本（8卡优化）

### 检查工具
- `check_before_training.py` - 训练前环境检查
- `compare_training_results.py` - 训练结果对比
- `check_stage_transition.py` - 阶段转换检查

## 📋 其他文档（可选）

这些文档包含特定主题的详细信息，可根据需要查阅：

- `ADAPTATION_LAYERS_REDESIGN.md` - Adaptation层设计细节
- `TRAINING_COMMANDS.md` - 训练命令参考
- `Loss上升分析报告.md` - Loss异常分析
- `训练分析报告.md` - 训练过程分析
- `训练分析报告_Stage转换标准.md` - 阶段转换详细标准

## ❓ 常见问题

### Q1: 我应该从哪里开始？
**A**: 阅读 `README_WAN_INTEGRATION.md`，它包含了所有你需要的信息。

### Q2: 如何解决过拟合问题？
**A**: 查看 `README_WAN_INTEGRATION.md` 中的"过拟合问题解决方案"章节。

### Q3: 如何判断是否可以进入下一阶段？
**A**: 查看 `README_WAN_INTEGRATION.md` 中的"阶段转换标准"章节。

### Q4: Stage 2 改进的训练脚本和原始脚本兼容吗？
**A**: 完全兼容！checkpoint格式完全相同，可以无缝衔接。详见 `README_WAN_INTEGRATION.md` 中的"Checkpoint兼容性"章节。

### Q5: 8卡服务器如何配置？
**A**: 使用 `run_improved_training.sh`，已针对8卡优化（BATCH_SIZE=8, NUM_WORKERS=4）。

## 📞 需要帮助？

1. **查看完整文档**：`cat README_WAN_INTEGRATION.md`
2. **搜索关键词**：`grep -i "关键词" README_WAN_INTEGRATION.md`
3. **查看训练命令**：`grep -A 15 "Stage X:" README_WAN_INTEGRATION.md`

---

**文档版本**: 2.0（整合版）
**最后更新**: 2026-02-01
