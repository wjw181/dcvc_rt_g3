#!/bin/bash

# ============================================
# 对比测试脚本
# 依次运行阶段一和阶段二的测试，并生成对比报告
# ============================================

echo "============================================"
echo "Running Both Stage 1 and Stage 2 Tests"
echo "============================================"

# 切换到 tests 目录
cd "$(dirname "$0")" || exit 1

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate dcvc_rt

echo ""
echo "Step 1: Installing dependencies (if needed)..."
pip install pytorch-msssim lpips -q 2>/dev/null || echo "Dependencies already installed"

echo ""
echo "============================================"
echo "Step 2: Testing Stage 1 checkpoint..."
echo "============================================"
bash test_stage1.sh

echo ""
echo "============================================"
echo "Step 3: Testing Stage 2 checkpoint..."
echo "============================================"
bash test_stage2.sh

echo ""
echo "============================================"
echo "All tests completed!"
echo "============================================"
echo ""
echo "Results:"
echo "  Stage 1: tests/results_stage1_metrics/"
echo "  Stage 2: tests/results_stage2_metrics/"
echo ""
echo "To compare results, check:"
echo "  - tests/results_stage1_metrics/statistics.txt"
echo "  - tests/results_stage2_metrics/statistics.txt"
echo ""
echo "View comparison images:"
echo "  - tests/results_stage1_metrics/comparison_frame_*.png"
echo "  - tests/results_stage2_metrics/comparison_frame_*.png"
echo "============================================"
