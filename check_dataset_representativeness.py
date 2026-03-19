#!/usr/bin/env python3
"""
检查训练集和测试集的代表性
用于诊断训练/测试分布不一致问题
"""

import os
import sys
import numpy as np
from PIL import Image
from collections import defaultdict


def get_image_stats(image_path):
    """获取单张图片的统计信息"""
    img = Image.open(image_path)
    arr = np.array(img) / 255.0
    
    return {
        'size': img.size,  # (width, height)
        'mean': arr.mean(axis=(0, 1)),  # RGB通道均值
        'std': arr.std(axis=(0, 1)),    # RGB通道标准差
        'min': arr.min(),
        'max': arr.max(),
    }


def analyze_dataset(root_dir, filelist_path=None, num_samples=50, name="Dataset"):
    """分析数据集统计特征"""
    print(f"\n{'='*60}")
    print(f"📊 分析: {name}")
    print(f"{'='*60}")
    
    # 收集图片路径
    image_paths = []
    
    if filelist_path and os.path.exists(filelist_path):
        # 从文件列表读取
        with open(filelist_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        for line in lines[:num_samples]:
            if line.endswith('.png') or line.endswith('.jpg'):
                path = os.path.join(root_dir, line)
            else:
                # Vimeo格式: 00001/0001
                path = os.path.join(root_dir, line, 'im1.png')
            
            if os.path.exists(path):
                image_paths.append(path)
    else:
        # 直接从目录读取
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.png') or f.endswith('.jpg'):
                    image_paths.append(os.path.join(root, f))
                    if len(image_paths) >= num_samples:
                        break
            if len(image_paths) >= num_samples:
                break
    
    if not image_paths:
        print(f"❌ 未找到图片！路径: {root_dir}")
        return None
    
    print(f"📁 路径: {root_dir}")
    print(f"📄 文件列表: {filelist_path or '无'}")
    print(f"🖼️  采样数量: {len(image_paths)}")
    
    # 统计分析
    sizes = defaultdict(int)
    means = []
    stds = []
    
    for path in image_paths:
        try:
            stats = get_image_stats(path)
            sizes[stats['size']] += 1
            means.append(stats['mean'])
            stds.append(stats['std'])
        except Exception as e:
            print(f"⚠️  无法读取 {path}: {e}")
    
    if not means:
        print(f"❌ 无法分析任何图片！")
        return None
    
    means = np.array(means)
    stds = np.array(stds)
    
    # 输出结果
    print(f"\n📐 分辨率分布:")
    for size, count in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"   {size[0]}×{size[1]}: {count} 张 ({count/len(image_paths)*100:.1f}%)")
    
    print(f"\n🎨 RGB 通道统计:")
    print(f"   Mean (R,G,B): [{means[:,0].mean():.4f}, {means[:,1].mean():.4f}, {means[:,2].mean():.4f}]")
    print(f"   Std  (R,G,B): [{stds[:,0].mean():.4f}, {stds[:,1].mean():.4f}, {stds[:,2].mean():.4f}]")
    
    print(f"\n📊 图像亮度分布:")
    luminance = means.mean(axis=1)
    print(f"   亮度均值: {luminance.mean():.4f}")
    print(f"   亮度标准差: {luminance.std():.4f}")
    print(f"   亮度范围: [{luminance.min():.4f}, {luminance.max():.4f}]")
    
    # 返回统计数据供对比
    return {
        'name': name,
        'num_samples': len(image_paths),
        'primary_size': max(sizes.items(), key=lambda x: x[1])[0],
        'mean_rgb': means.mean(axis=0),
        'std_rgb': stds.mean(axis=0),
        'luminance_mean': luminance.mean(),
        'luminance_std': luminance.std(),
    }


def compare_datasets(train_stats, test_stats):
    """对比训练集和测试集"""
    if train_stats is None or test_stats is None:
        print("\n❌ 无法对比，缺少数据集统计信息")
        return
    
    print(f"\n{'='*60}")
    print(f"🔍 数据集对比: {train_stats['name']} vs {test_stats['name']}")
    print(f"{'='*60}")
    
    issues = []
    
    # 分辨率对比
    train_size = train_stats['primary_size']
    test_size = test_stats['primary_size']
    print(f"\n📐 分辨率:")
    print(f"   训练集: {train_size[0]}×{train_size[1]}")
    print(f"   测试集: {test_size[0]}×{test_size[1]}")
    if train_size != test_size:
        issues.append(f"分辨率不匹配: {train_size} vs {test_size}")
        print(f"   ⚠️  不匹配！")
    else:
        print(f"   ✅ 匹配")
    
    # RGB均值对比
    train_mean = train_stats['mean_rgb']
    test_mean = test_stats['mean_rgb']
    mean_diff = np.abs(train_mean - test_mean).mean()
    print(f"\n🎨 RGB均值:")
    print(f"   训练集: [{train_mean[0]:.4f}, {train_mean[1]:.4f}, {train_mean[2]:.4f}]")
    print(f"   测试集: [{test_mean[0]:.4f}, {test_mean[1]:.4f}, {test_mean[2]:.4f}]")
    print(f"   差异: {mean_diff:.4f}")
    if mean_diff > 0.1:
        issues.append(f"RGB均值差异过大: {mean_diff:.4f}")
        print(f"   ⚠️  差异较大！")
    else:
        print(f"   ✅ 差异可接受")
    
    # 亮度对比
    train_lum = train_stats['luminance_mean']
    test_lum = test_stats['luminance_mean']
    lum_diff = abs(train_lum - test_lum)
    print(f"\n💡 亮度均值:")
    print(f"   训练集: {train_lum:.4f}")
    print(f"   测试集: {test_lum:.4f}")
    print(f"   差异: {lum_diff:.4f}")
    if lum_diff > 0.1:
        issues.append(f"亮度差异过大: {lum_diff:.4f}")
        print(f"   ⚠️  差异较大！")
    else:
        print(f"   ✅ 差异可接受")
    
    # 亮度多样性对比
    train_lum_std = train_stats['luminance_std']
    test_lum_std = test_stats['luminance_std']
    print(f"\n📊 亮度多样性 (标准差):")
    print(f"   训练集: {train_lum_std:.4f}")
    print(f"   测试集: {test_lum_std:.4f}")
    if test_lum_std < train_lum_std * 0.3:
        issues.append(f"测试集多样性不足: std {test_lum_std:.4f} vs {train_lum_std:.4f}")
        print(f"   ⚠️  测试集多样性不足！")
    else:
        print(f"   ✅ 多样性可接受")
    
    # 总结
    print(f"\n{'='*60}")
    print(f"📋 总结")
    print(f"{'='*60}")
    
    if issues:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print(f"\n⚠️  测试集可能不具代表性，训练/测试分布不一致可能导致:")
        print(f"   - 测试集性能波动大")
        print(f"   - 训练集/测试集性能差距大")
        print(f"   - 看起来像'过拟合'，实际是分布不匹配")
    else:
        print(f"\n✅ 训练集和测试集具有较好的代表性匹配")
    
    return issues


def main():
    print("🔍 数据集代表性检查工具")
    print("="*60)
    
    # 训练集配置
    train_root = "/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/sequences/"
    train_list = "/home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/test.txt"
    
    # 测试集配置（当前使用的）
    test_root = "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60/"
    test_list = "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/test.txt"
    
    # 分析训练集
    train_stats = analyze_dataset(
        train_root, train_list, 
        num_samples=100, 
        name="Vimeo-Septuplet (训练集)"
    )
    
    # 分析测试集
    test_stats = analyze_dataset(
        test_root, test_list,
        num_samples=100,
        name="HEVC E - Johnny (测试集)"
    )
    
    # 对比
    issues = compare_datasets(train_stats, test_stats)
    
    # 建议
    if issues:
        print(f"\n{'='*60}")
        print(f"💡 建议")
        print(f"{'='*60}")
        print("""
1. 使用Vimeo验证集作为测试集:
   --test_dataset /home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/sequences/
   --test_filelist /home/serverdn/hdd-0/slf_data/data/vimeo_septuplet/sep_testlist.txt

2. 或者使用更多HEVC测试序列:
   创建包含 FourPeople + Johnny + KristenAndSara 的测试列表

3. 或者在训练集中加入HEVC类型数据进行微调
""")


if __name__ == "__main__":
    main()
