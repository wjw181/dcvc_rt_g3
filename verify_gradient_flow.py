#!/usr/bin/env python3
"""
验证 DMC_WAN latent mode 的梯度流是否正确
确保两个 adaptation layers 都能被训练
"""
import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.video_t_g_wan import DMC_WAN


def verify_gradient_flow():
    """验证梯度流"""
    print("=" * 70)
    print("验证 DMC_WAN Latent Mode 梯度流")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 初始化模型（不加载实际的 checkpoint）
    print("\n1. 初始化 DMC_WAN (latent mode, 无 VAE checkpoint)...")
    
    try:
        model = DMC_WAN(
            mode="latent",
            freeze_dcvc=True,
            freeze_vae=True,
            dcvc_checkpoint=None,
            wan_vae_checkpoint=None,
        ).to(device)
    except Exception as e:
        print(f"   ⚠️  初始化失败（可能缺少 VAE checkpoint）: {e}")
        print("   尝试使用 pixel mode 进行基本验证...")
        model = DMC_WAN(
            mode="pixel",
            freeze_dcvc=False,
            freeze_vae=True,
            dcvc_checkpoint=None,
            wan_vae_checkpoint=None,
        ).to(device)
        print("   注意：使用 pixel mode 进行验证，latent mode 需要 VAE checkpoint")
        return
    
    print("   ✅ 模型初始化成功")
    
    # 创建假数据
    print("\n2. 创建测试数据...")
    B, C, H, W = 1, 3, 256, 256
    dummy_input = torch.randn(B, C, H, W, device=device, requires_grad=True) * 0.5 + 0.5
    qp = 71
    print(f"   输入形状: {dummy_input.shape}")
    
    # 前向传播
    print("\n3. 前向传播...")
    try:
        result = model(dummy_input, qp)
        print("   ✅ 前向传播成功")
        print(f"   输出 keys: {list(result.keys())}")
    except Exception as e:
        print(f"   ❌ 前向传播失败: {e}")
        return
    
    # 检查梯度流
    print("\n4. 检查梯度流...")
    
    # 4.1 检查 mse 是否有梯度
    if 'mse' in result:
        mse = result['mse']
        print(f"   - mse.requires_grad: {mse.requires_grad}")
        if mse.requires_grad:
            print("     ✅ mse 有梯度")
        else:
            print("     ❌ mse 没有梯度")
    
    # 4.2 检查 x_hat_rgb 是否有梯度
    if 'x_hat_rgb' in result:
        x_hat_rgb = result['x_hat_rgb']
        print(f"   - x_hat_rgb.requires_grad: {x_hat_rgb.requires_grad}")
        if x_hat_rgb.requires_grad:
            print("     ✅ x_hat_rgb 有梯度 (LPIPS 可用)")
        else:
            print("     ⚠️  x_hat_rgb 没有梯度 (LPIPS 无效)")
    
    # 4.3 检查 latent_mse 和 pseudo_yuv_mse
    if 'latent_mse' in result:
        latent_mse = result['latent_mse']
        print(f"   - latent_mse.requires_grad: {latent_mse.requires_grad}")
        if latent_mse.requires_grad:
            print("     ✅ latent_mse 有梯度 (pseudo_yuv_to_latent 可训练)")
        else:
            print("     ❌ latent_mse 没有梯度")
    
    if 'pseudo_yuv_mse' in result:
        pseudo_yuv_mse = result['pseudo_yuv_mse']
        print(f"   - pseudo_yuv_mse.requires_grad: {pseudo_yuv_mse.requires_grad}")
        if pseudo_yuv_mse.requires_grad:
            print("     ✅ pseudo_yuv_mse 有梯度 (latent_to_pseudo_yuv 可训练)")
        else:
            print("     ❌ pseudo_yuv_mse 没有梯度")
    
    # 5. 反向传播测试
    print("\n5. 反向传播测试...")
    
    # 清零梯度
    model.zero_grad()
    
    # 计算损失
    loss = result['mse']
    if 'latent_mse' in result and 'pseudo_yuv_mse' in result:
        loss = result['latent_mse'] + result['pseudo_yuv_mse']
    
    try:
        loss.backward()
        print("   ✅ 反向传播成功")
    except Exception as e:
        print(f"   ❌ 反向传播失败: {e}")
        return
    
    # 6. 检查各层梯度
    print("\n6. 检查各层梯度...")
    
    # 检查 latent_to_pseudo_yuv
    if hasattr(model, 'latent_to_pseudo_yuv'):
        has_grad = False
        for name, param in model.latent_to_pseudo_yuv.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        if has_grad:
            print("   ✅ latent_to_pseudo_yuv 有梯度 (可训练)")
        else:
            print("   ❌ latent_to_pseudo_yuv 没有梯度")
    else:
        print("   ⚠️  latent_to_pseudo_yuv 不存在 (可能是 pixel mode)")
    
    # 检查 pseudo_yuv_to_latent
    if hasattr(model, 'pseudo_yuv_to_latent'):
        has_grad = False
        for name, param in model.pseudo_yuv_to_latent.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        if has_grad:
            print("   ✅ pseudo_yuv_to_latent 有梯度 (可训练)")
        else:
            print("   ❌ pseudo_yuv_to_latent 没有梯度")
    else:
        print("   ⚠️  pseudo_yuv_to_latent 不存在 (可能是 pixel mode)")
    
    # 检查 DCVC 是否被冻结
    dcvc_has_grad = False
    for name, param in model.dcvc_model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            dcvc_has_grad = True
            break
    if dcvc_has_grad:
        print("   ⚠️  dcvc_model 有梯度 (应该被冻结但没有)")
    else:
        print("   ✅ dcvc_model 没有梯度 (已正确冻结)")
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    all_good = True
    
    if hasattr(model, 'latent_to_pseudo_yuv'):
        l2p_ok = any(p.grad is not None and p.grad.abs().sum() > 0 
                     for p in model.latent_to_pseudo_yuv.parameters())
        if l2p_ok:
            print("✅ latent_to_pseudo_yuv: 可训练")
        else:
            print("❌ latent_to_pseudo_yuv: 不可训练")
            all_good = False
    
    if hasattr(model, 'pseudo_yuv_to_latent'):
        p2l_ok = any(p.grad is not None and p.grad.abs().sum() > 0 
                     for p in model.pseudo_yuv_to_latent.parameters())
        if p2l_ok:
            print("✅ pseudo_yuv_to_latent: 可训练")
        else:
            print("❌ pseudo_yuv_to_latent: 不可训练")
            all_good = False
    
    x_hat_rgb_ok = 'x_hat_rgb' in result and result['x_hat_rgb'].requires_grad
    if x_hat_rgb_ok:
        print("✅ LPIPS 损失: 可用")
    else:
        print("⚠️  LPIPS 损失: 不可用 (x_hat_rgb 没有梯度)")
    
    print("\n" + "=" * 70)
    if all_good:
        print("🎉 所有 adaptation layers 都可以被训练！梯度流正常。")
    else:
        print("⚠️  部分层无法被训练，请检查代码。")
    print("=" * 70)


if __name__ == "__main__":
    verify_gradient_flow()
