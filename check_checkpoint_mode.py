#!/usr/bin/env python3
"""
检查 checkpoint 的训练模式（pixel 或 latent）
"""
import torch
import sys
import os

def check_checkpoint_mode(ckpt_path):
    """检查 checkpoint 的训练模式"""
    if not os.path.exists(ckpt_path):
        print(f"❌ 错误: 文件不存在: {ckpt_path}")
        return None
    
    print(f"\n📦 加载 checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None
    
    print("\n" + "="*60)
    print("Checkpoint 信息:")
    print("="*60)
    
    # 基本信息
    epoch = ckpt.get('epoch', 'unknown')
    loss = ckpt.get('loss', 'unknown')
    mode = ckpt.get('mode', 'unknown')
    
    print(f"  Epoch: {epoch}")
    print(f"  Loss: {loss:.4f}" if isinstance(loss, (int, float)) else f"  Loss: {loss}")
    print(f"  Mode (saved): {mode}")
    
    # 检查 state_dict 中的 keys
    if 'state_dict' not in ckpt:
        print(f"  ⚠️  警告: checkpoint 中没有 'state_dict' 字段")
        return mode
    
    state_dict = ckpt['state_dict']
    print(f"\n  参数总数: {len(state_dict)} 个 keys")
    
    # 检查是否有 latent mode 特有的层
    latent_keys = [k for k in state_dict.keys() if 'latent_to_pseudo_yuv' in k or 'pseudo_yuv_to_latent' in k]
    has_latent_layers = len(latent_keys) > 0
    
    print(f"  Latent adaptation layers: {'✅ 存在' if has_latent_layers else '❌ 不存在'}")
    if has_latent_layers:
        print(f"    - 找到 {len(latent_keys)} 个 latent 相关参数")
    
    # 判断模式
    print("\n" + "="*60)
    if has_latent_layers:
        print("🎯 结论: 这是一个 LATENT MODE checkpoint")
        print("="*60)
        print("\n使用方法:")
        print("  torchrun --nproc_per_node=2 train_vd_phase_1_wan.py \\")
        print("      --mode latent \\")
        print(f"      --checkpoint {ckpt_path} \\")
        print("      --wan_vae_checkpoint path/to/vae.pth \\")
        print("      --batch-size 1 --use_amp")
        detected_mode = "latent"
    else:
        print("🎯 结论: 这是一个 PIXEL MODE checkpoint")
        print("="*60)
        print("\n使用方法:")
        print("  torchrun --nproc_per_node=2 train_vd_phase_1_wan.py \\")
        print("      --mode pixel \\")
        print(f"      --checkpoint {ckpt_path} \\")
        print("      --batch-size 1 --use_amp")
        detected_mode = "pixel"
    
    # 检查模式一致性
    if mode != 'unknown' and mode != detected_mode:
        print(f"\n⚠️  警告: 保存的模式 ({mode}) 与检测到的模式 ({detected_mode}) 不一致！")
    
    print()
    return detected_mode


def main():
    if len(sys.argv) < 2:
        print("用法: python check_checkpoint_mode.py <checkpoint_path>")
        print("\n示例:")
        print("  python check_checkpoint_mode.py pretrained/DMC_WAN/1/checkpoint_wan.pth.tar")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    check_checkpoint_mode(ckpt_path)


if __name__ == "__main__":
    main()
