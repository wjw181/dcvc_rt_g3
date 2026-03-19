#!/usr/bin/env python3
"""
可视化对比测试脚本
对比 WAN 集成模型 vs 原始 DCVC_RT 模型
添加感知质量指标：SSIM, MS-SSIM, LPIPS
"""

import os
import sys
import argparse
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
import matplotlib
# 解决中文乱码问题
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.video_t_g_wan import DMC_WAN
from src.models.video_t import DMC
from src.models.image_model import DMCI
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, yuv_420_to_444

# 尝试导入 LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not installed, will skip LPIPS calculation")


def load_wan_model(checkpoint_path, wan_vae_path, device='cuda'):
    """加载 WAN 集成模型"""
    print(f"Loading WAN model: {checkpoint_path}")
    
    # 初始化模型
    model = DMC_WAN(
        mode='latent',
        wan_vae_checkpoint=wan_vae_path,
        freeze_vae=True
    )
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除 'module.' 前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print("✅ WAN model loaded successfully")
    return model


def load_dcvc_model(checkpoint_path, device='cuda'):
    """加载原始 DCVC_RT 模型"""
    print(f"Loading DCVC_RT model: {checkpoint_path}")
    
    # 初始化模型
    model = DMC()
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除 'module.' 前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print("✅ DCVC_RT model loaded successfully")
    return model


def _pad_to_multiple(x, multiple=16):
    """将张量 H/W 填充到 multiple 的倍数"""
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, H, W
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x, H + pad_h, W + pad_w


def load_image(image_path):
    """加载图像并转换为 YUV"""
    img = imageio.imread(image_path).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # RGB to YUV
    img_yuv = rgb2ycbcr(img.squeeze(0)).unsqueeze(0)
    img_y, img_uv = yuv_444_to_420(img_yuv)
    img_yuv = yuv_420_to_444(img_y, img_uv)
    
    return img_yuv


def calculate_psnr(img1, img2):
    """计算 PSNR"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def calculate_ssim(img1, img2):
    """计算 SSIM"""
    # SSIM 需要 4D tensor [B, C, H, W]
    return ssim(img1, img2, data_range=1.0, size_average=True).item()


def calculate_ms_ssim(img1, img2):
    """计算 MS-SSIM"""
    # MS-SSIM 需要 4D tensor [B, C, H, W]
    try:
        return ms_ssim(img1, img2, data_range=1.0, size_average=True).item()
    except:
        # 如果图像太小，MS-SSIM 可能失败
        return calculate_ssim(img1, img2)


def calculate_lpips(img1, img2, lpips_model):
    """计算 LPIPS (感知相似性)
    
    输入假设为 YUV444 (或近似 YUV) [B,3,H,W] in [0,1]。
    LPIPS 基于 AlexNet/VGG 预训练于 RGB，因此先用 ycbcr2rgb 转换到 RGB 再计算。
    """
    if not LPIPS_AVAILABLE or lpips_model is None:
        return None
    
    # YUV → RGB（对于 WAN 的 x_hat_rgb 是"近似 YUV"，转换后近似 RGB）
    img1_rgb = ycbcr2rgb(img1).clamp(0.0, 1.0)
    img2_rgb = ycbcr2rgb(img2).clamp(0.0, 1.0)
    
    # LPIPS expects [-1, 1]
    img1_rgb = img1_rgb * 2.0 - 1.0
    img2_rgb = img2_rgb * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_val = lpips_model(img1_rgb, img2_rgb)
    
    return lpips_val.item()


def parse_qp_list(qp_list_str, default_qp):
    if not qp_list_str:
        return [default_qp]
    qp_list = []
    for part in qp_list_str.split(','):
        part = part.strip()
        if part:
            qp_list.append(int(part))
    return qp_list if qp_list else [default_qp]


def compress_and_reconstruct(model, ref_frame, current_frame, i_frame_net, qp=37, device='cuda'):
    """压缩并重建单帧（仅用于 BPP 探测等独立评测场景）"""
    with torch.no_grad():
        ref_frame = ref_frame.to(device)
        current_frame = current_frame.to(device)
        _, _, H_orig, W_orig = current_frame.shape

        ref_frame, H_pad, W_pad = _pad_to_multiple(ref_frame, 16)
        current_frame, _, _ = _pad_to_multiple(current_frame, 16)
        
        # I帧压缩
        if qp > 63:
            ref_compressed = i_frame_net.compress_(ref_frame, 63)
        else:
            ref_compressed = i_frame_net.compress_(ref_frame, qp)
        
        # I帧重建可能被上采样到16的倍数，裁剪回 padding 后的尺寸以避免特征尺寸不匹配
        if ref_compressed.shape[2] != H_pad or ref_compressed.shape[3] != W_pad:
            ref_compressed = ref_compressed[:, :, :H_pad, :W_pad]
        
        # 清空 DPB 并添加参考帧
        model.clear_dpb()
        model.add_ref_frame(None, ref_compressed)
        
        # P帧压缩
        result = model(current_frame, qp)
        
        # 输出裁剪回原始尺寸，便于后续指标计算
        if 'x_hat' in result:
            result['x_hat'] = result['x_hat'][:, :, :H_orig, :W_orig]
        if 'x_hat_rgb' in result:
            result['x_hat_rgb'] = result['x_hat_rgb'][:, :, :H_orig, :W_orig]
        
        return result


def run_model_on_frames(model_name, model, frames, qp, device, i_frame_net, lpips_model):
    """序列化评估：I帧初始化 DPB 后连续编码 P 帧（不清空 DPB），与训练行为一致。

    训练时 DCVC 内部通过 feature 管理参考链，不需要在像素/YUV 空间传递参考帧，
    从而避免了 WAN 的 x_hat_rgb（"类YUV"）作为 I帧参考时的色彩空间不匹配问题。
    """
    per_frame = []
    metrics_acc = {
        'psnr': [], 'bpp': [], 'mse': [],
        'ssim': [], 'ms_ssim': []
    }
    if LPIPS_AVAILABLE:
        metrics_acc['lpips'] = []

    _, _, H_orig, W_orig = frames[0].shape

    with torch.no_grad():
        # ── 1. I帧压缩并初始化 DPB（仅一次）──
        ref_frame = frames[0].to(device)
        ref_padded, _, _ = _pad_to_multiple(ref_frame, 16)
        i_qp = min(qp, 63)  # I帧 QP 上限 63
        ref_compressed = i_frame_net.compress_(ref_padded, i_qp)
        # I帧重建可能因内部 padding 导致尺寸偏大，裁剪回对齐尺寸
        if ref_compressed.shape[2] != ref_padded.shape[2] or ref_compressed.shape[3] != ref_padded.shape[3]:
            ref_compressed = ref_compressed[:, :, :ref_padded.shape[2], :ref_padded.shape[3]]

        model.clear_dpb()
        model.add_ref_frame(None, ref_compressed)

        # ── 2. 逐帧 P 帧压缩（不清空 DPB，DCVC 内部 feature 管理参考链）──
        for idx in range(1, len(frames)):
            current_frame = frames[idx]
            current_device = current_frame.to(device)
            current_padded, _, _ = _pad_to_multiple(current_device, 16)

            print(f"\n{'='*60}")
            print(f"{model_name} | Testing frame {idx}")
            print(f"{'='*60}")
            print(f"Compressing with {model_name} model (QP={qp})...")

            result = model(current_padded, qp)

            # WAN: 优先取 x_hat_rgb（VAE 解码输出，与训练 PSNR 计算一致）
            # DCVC: 无 x_hat_rgb，退回 x_hat（纯 YUV）
            recon = result.get('x_hat_rgb', result['x_hat'])[:, :, :H_orig, :W_orig]
            bpp = result['bpp'].mean().item()

            # PSNR / MSE / SSIM 在模型输出空间计算（与训练一致）
            psnr = calculate_psnr(current_device, recon)
            mse = F.mse_loss(current_device, recon).item()
            ssim_val = calculate_ssim(current_device, recon)
            ms_ssim_val = calculate_ms_ssim(current_device, recon)
            # LPIPS：内部先 YUV→RGB 再计算
            lpips_val = calculate_lpips(current_device, recon, lpips_model)

            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  BPP: {bpp:.4f}")
            print(f"  SSIM: {ssim_val:.4f}")
            print(f"  MS-SSIM: {ms_ssim_val:.4f}")
            if lpips_val is not None:
                print(f"  LPIPS: {lpips_val:.4f}")

            per_frame.append({
                'current': current_device,
                'recon': recon,
                'psnr': psnr,
                'bpp': bpp,
                'mse': mse,
                'ssim': ssim_val,
                'ms_ssim': ms_ssim_val,
                'lpips': lpips_val,
            })

            metrics_acc['psnr'].append(psnr)
            metrics_acc['bpp'].append(bpp)
            metrics_acc['mse'].append(mse)
            metrics_acc['ssim'].append(ssim_val)
            metrics_acc['ms_ssim'].append(ms_ssim_val)
            if lpips_val is not None:
                metrics_acc['lpips'].append(lpips_val)

    avg_metrics = {k: float(np.mean(v)) for k, v in metrics_acc.items()}
    return {
        'qp': qp,
        'per_frame': per_frame,
        'avg': avg_metrics,
    }


def probe_bpp_sequential(model, frames, i_frame_net, qp, device, num_probe_frames=3):
    """用少量帧的序列化编码探测某个QP对应的BPP（与 run_model_on_frames 一致）"""
    n = min(num_probe_frames + 1, len(frames))  # +1 因为第0帧是I帧
    with torch.no_grad():
        ref = frames[0].to(device)
        ref_padded, _, _ = _pad_to_multiple(ref, 16)
        i_qp = min(qp, 63)
        ref_compressed = i_frame_net.compress_(ref_padded, i_qp)
        if ref_compressed.shape[2] != ref_padded.shape[2] or ref_compressed.shape[3] != ref_padded.shape[3]:
            ref_compressed = ref_compressed[:, :, :ref_padded.shape[2], :ref_padded.shape[3]]

        model.clear_dpb()
        model.add_ref_frame(None, ref_compressed)

        bpp_list = []
        for idx in range(1, n):
            current = frames[idx].to(device)
            current_padded, _, _ = _pad_to_multiple(current, 16)
            result = model(current_padded, qp)
            bpp_list.append(result['bpp'].mean().item())

    return sum(bpp_list) / len(bpp_list) if bpp_list else 0.0


def select_dcvc_by_target_bpp(dcvc_model, frames, qp_list, target_bpp, device, i_frame_net, lpips_model):
    """两阶段码率匹配：先用序列化探测找最近QP，再用全帧精测"""
    qp_list = sorted(qp_list)

    print(f"\n--- Phase 1: Sequential BPP probe (3 frames) ---")
    print(f"  Target BPP: {target_bpp:.4f}")
    print(f"  Candidates: {qp_list}")

    qp_bpp = []
    for qp in qp_list:
        bpp = probe_bpp_sequential(dcvc_model, frames, i_frame_net, qp, device)
        qp_bpp.append((qp, bpp))
        print(f"  QP={qp:2d} -> BPP={bpp:.4f}")

    # 找到最接近target_bpp的QP
    qp_bpp.sort(key=lambda x: abs(x[1] - target_bpp))
    best_qp = qp_bpp[0][0]
    best_bpp = qp_bpp[0][1]
    print(f"\n  Best match: QP={best_qp} (BPP={best_bpp:.4f}, diff={abs(best_bpp - target_bpp):.4f})")

    # 如果最近的两个QP分别在target两侧，可以报告一下
    below = [(q, b) for q, b in qp_bpp if b <= target_bpp]
    above = [(q, b) for q, b in qp_bpp if b > target_bpp]
    if below and above:
        below_best = min(below, key=lambda x: abs(x[1] - target_bpp))
        above_best = min(above, key=lambda x: abs(x[1] - target_bpp))
        print(f"  Bracket: QP={below_best[0]}(BPP={below_best[1]:.4f}) < target < QP={above_best[0]}(BPP={above_best[1]:.4f})")

    print(f"\n--- Phase 2: Full evaluation at QP={best_qp} ---")
    result = run_model_on_frames("DCVC_RT", dcvc_model, frames, best_qp, device, i_frame_net, lpips_model)
    return result


def visualize_comparison(original, wan_recon, dcvc_recon, save_path, metrics):
    """可视化对比结果（Y 通道灰度显示 — 视频压缩标准做法）
    
    注意：WAN 的 x_hat_rgb 不是标准 YUV（VAE 预训练在 RGB 上），
    用 ycbcr2rgb 显示会出现黄色偏色。因此统一用第 0 通道灰度显示，
    这是视频压缩论文的标准做法，且 PSNR 主要由亮度决定。
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle('WAN Integrated Model vs Original DCVC_RT Model Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 统一取第 0 通道（原图/DCVC 为 Y 亮度，WAN 为 VAE 解码第 0 通道）
    original_y = original[0, 0].cpu().numpy()
    wan_y = wan_recon[0, 0].cpu().numpy()
    dcvc_y = dcvc_recon[0, 0].cpu().numpy()
    
    # 第一行：原图、WAN重建、DCVC重建（灰度）
    axes[0, 0].imshow(original_y, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original (Y channel)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    
    # WAN 重建
    wan_title = f'WAN Reconstruction\n'
    wan_title += f'PSNR: {metrics["wan_psnr"]:.2f} dB | BPP: {metrics["wan_bpp"]:.4f}\n'
    wan_title += f'SSIM: {metrics["wan_ssim"]:.4f} | MS-SSIM: {metrics["wan_ms_ssim"]:.4f}'
    if metrics.get("wan_lpips") is not None:
        wan_title += f'\nLPIPS: {metrics["wan_lpips"]:.4f}'
    axes[0, 1].imshow(wan_y, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(wan_title, fontsize=11)
    axes[0, 1].axis('off')
    
    # DCVC_RT 重建
    dcvc_title = f'DCVC_RT Reconstruction\n'
    dcvc_title += f'PSNR: {metrics["dcvc_psnr"]:.2f} dB | BPP: {metrics["dcvc_bpp"]:.4f}\n'
    dcvc_title += f'SSIM: {metrics["dcvc_ssim"]:.4f} | MS-SSIM: {metrics["dcvc_ms_ssim"]:.4f}'
    if metrics.get("dcvc_lpips") is not None:
        dcvc_title += f'\nLPIPS: {metrics["dcvc_lpips"]:.4f}'
    axes[0, 2].imshow(dcvc_y, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(dcvc_title, fontsize=11)
    axes[0, 2].axis('off')
    
    wan_error = np.abs(original_y - wan_y)
    dcvc_error = np.abs(original_y - dcvc_y)
    error_diff = wan_error - dcvc_error  # 负值表示 WAN 更好
    
    im1 = axes[1, 0].imshow(wan_error, cmap='hot', vmin=0, vmax=0.1)
    axes[1, 0].set_title(f'WAN Error Map (ch0)\nMSE: {metrics["wan_mse"]:.6f}', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[1, 1].imshow(dcvc_error, cmap='hot', vmin=0, vmax=0.1)
    axes[1, 1].set_title(f'DCVC_RT Error Map (Y)\nMSE: {metrics["dcvc_mse"]:.6f}', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    im3 = axes[1, 2].imshow(error_diff, cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    axes[1, 2].set_title('Error Difference\n(Blue=WAN better, Red=DCVC better)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize WAN vs DCVC_RT comparison')
    parser.add_argument('--wan_checkpoint', type=str, 
                       default='pretrained/DMC_WAN/1/checkpoint_stage4_wan.pth.tar',
                       help='WAN model checkpoint')
    parser.add_argument('--dcvc_checkpoint', type=str,
                       default='pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar',
                       help='DCVC_RT model checkpoint')
    parser.add_argument('--wan_vae', type=str,
                       default='Wan2.2-main/checkpoints/Wan2.2_VAE.pth',
                       help='WAN VAE checkpoint')
    parser.add_argument('--i_frame_model', type=str,
                       default='checkpoints/cvpr2025_image.pth.tar',
                       help='I-frame model checkpoint')
    parser.add_argument('--test_dir', type=str,
                       default='/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60',
                       help='Test image directory')
    parser.add_argument('--output_dir', type=str,
                       default='visualization_results_stage4',
                       help='Output directory')
    parser.add_argument('--qp', type=int, default=37,
                       help='Quantization parameter')
    parser.add_argument('--match_bpp', action='store_true',
                       help='Match DCVC_RT bitrate to WAN (or target_bpp)')
    parser.add_argument('--target_bpp', type=float, default=None,
                       help='Target BPP for bitrate matching')
    parser.add_argument('--qp_list', type=str, default=None,
                       help='Comma-separated QP candidates for matching, e.g. "22,27,32,37,42"')
    parser.add_argument('--num_frames', type=int, default=5,
                       help='Number of frames to test')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 初始化 LPIPS 模型
    lpips_model = None
    if LPIPS_AVAILABLE:
        print("\nInitializing LPIPS model...")
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        print("✅ LPIPS model loaded")
    
    # 加载 I 帧模型
    print("\nLoading I-frame model...")
    i_frame_net = DMCI()
    i_checkpoint = torch.load(args.i_frame_model, map_location='cpu')
    if 'state_dict' in i_checkpoint:
        i_checkpoint = i_checkpoint['state_dict']
    i_frame_net.load_state_dict(i_checkpoint, strict=False)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    print("✅ I-frame model loaded")
    
    # 加载模型
    print("\n" + "="*60)
    wan_model = load_wan_model(args.wan_checkpoint, args.wan_vae, device)
    print("\n" + "="*60)
    dcvc_model = load_dcvc_model(args.dcvc_checkpoint, device)
    print("="*60 + "\n")
    
    # 获取测试图像
    image_files = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.png')])[:args.num_frames + 1]
    
    if len(image_files) < 2:
        print("❌ Not enough images in test directory")
        return
    
    print(f"Found {len(image_files)} test images")
    
    # 预加载帧
    frame_paths = [os.path.join(args.test_dir, f) for f in image_files]
    frames = [load_image(p) for p in frame_paths]
    print(f"\nReference frame: {image_files[0]}")

    # WAN 固定使用 args.qp
    wan_results = run_model_on_frames("WAN", wan_model, frames, args.qp, device, i_frame_net, lpips_model)

    # DCVC 可能使用匹配码率
    dcvc_qp_list = parse_qp_list(args.qp_list, args.qp)
    if args.match_bpp:
        target_bpp = args.target_bpp if args.target_bpp is not None else wan_results['avg']['bpp']
        print(f"\nTarget BPP for matching: {target_bpp:.4f}")
        dcvc_results = select_dcvc_by_target_bpp(
            dcvc_model, frames, dcvc_qp_list, target_bpp, device, i_frame_net, lpips_model
        )
    else:
        dcvc_results = run_model_on_frames("DCVC_RT", dcvc_model, frames, args.qp, device, i_frame_net, lpips_model)

    # 统计结果
    all_metrics = {
        'wan_psnr': [], 'dcvc_psnr': [],
        'wan_bpp': [], 'dcvc_bpp': [],
        'wan_mse': [], 'dcvc_mse': [],
        'wan_ssim': [], 'dcvc_ssim': [],
        'wan_ms_ssim': [], 'dcvc_ms_ssim': [],
    }
    
    if LPIPS_AVAILABLE:
        all_metrics['wan_lpips'] = []
        all_metrics['dcvc_lpips'] = []
    
    # 对齐帧序列并可视化
    num_frames = min(len(wan_results['per_frame']), len(dcvc_results['per_frame']))
    for idx in range(num_frames):
        wan_f = wan_results['per_frame'][idx]
        dcvc_f = dcvc_results['per_frame'][idx]

        print(f"\nComparison for frame {idx + 1}:")
        print(f"  PSNR diff: {wan_f['psnr'] - dcvc_f['psnr']:+.2f} dB")
        print(f"  BPP diff: {wan_f['bpp'] - dcvc_f['bpp']:+.4f}")
        print(f"  SSIM diff: {wan_f['ssim'] - dcvc_f['ssim']:+.4f}")
        print(f"  MS-SSIM diff: {wan_f['ms_ssim'] - dcvc_f['ms_ssim']:+.4f}")
        if wan_f['lpips'] is not None and dcvc_f['lpips'] is not None:
            print(f"  LPIPS diff: {wan_f['lpips'] - dcvc_f['lpips']:+.4f} (lower is better)")
        
        metrics = {
            'wan_psnr': wan_f['psnr'], 'dcvc_psnr': dcvc_f['psnr'],
            'wan_bpp': wan_f['bpp'], 'dcvc_bpp': dcvc_f['bpp'],
            'wan_mse': wan_f['mse'], 'dcvc_mse': dcvc_f['mse'],
            'wan_ssim': wan_f['ssim'], 'dcvc_ssim': dcvc_f['ssim'],
            'wan_ms_ssim': wan_f['ms_ssim'], 'dcvc_ms_ssim': dcvc_f['ms_ssim'],
        }
        
        if wan_f['lpips'] is not None:
            metrics['wan_lpips'] = wan_f['lpips']
            metrics['dcvc_lpips'] = dcvc_f['lpips']
        
        for key in metrics:
            if key in all_metrics:
                all_metrics[key].append(metrics[key])
        
        save_path = os.path.join(args.output_dir, f'comparison_frame_{idx + 1:03d}.png')
        visualize_comparison(wan_f['current'], wan_f['recon'], dcvc_f['recon'], save_path, metrics)
    
    # 打印总结
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")

    print(f"\nQP Settings:")
    print(f"  WAN QP:     {wan_results['qp']}")
    print(f"  DCVC_RT QP: {dcvc_results['qp']}")
    if args.match_bpp:
        target_bpp = args.target_bpp if args.target_bpp is not None else wan_results['avg']['bpp']
        print(f"  Target BPP: {target_bpp:.4f}")
    
    print(f"\nAverage PSNR:")
    print(f"  WAN:      {np.mean(all_metrics['wan_psnr']):.2f} dB")
    print(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_psnr']):.2f} dB")
    print(f"  Diff:     {np.mean(all_metrics['wan_psnr']) - np.mean(all_metrics['dcvc_psnr']):+.2f} dB")
    
    print(f"\nAverage BPP:")
    print(f"  WAN:      {np.mean(all_metrics['wan_bpp']):.4f}")
    print(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_bpp']):.4f}")
    print(f"  Diff:     {np.mean(all_metrics['wan_bpp']) - np.mean(all_metrics['dcvc_bpp']):+.4f}")
    
    print(f"\nAverage SSIM:")
    print(f"  WAN:      {np.mean(all_metrics['wan_ssim']):.4f}")
    print(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_ssim']):.4f}")
    print(f"  Diff:     {np.mean(all_metrics['wan_ssim']) - np.mean(all_metrics['dcvc_ssim']):+.4f}")
    
    print(f"\nAverage MS-SSIM:")
    print(f"  WAN:      {np.mean(all_metrics['wan_ms_ssim']):.4f}")
    print(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_ms_ssim']):.4f}")
    print(f"  Diff:     {np.mean(all_metrics['wan_ms_ssim']) - np.mean(all_metrics['dcvc_ms_ssim']):+.4f}")
    
    if 'wan_lpips' in all_metrics and len(all_metrics['wan_lpips']) > 0:
        print(f"\nAverage LPIPS (lower is better):")
        print(f"  WAN:      {np.mean(all_metrics['wan_lpips']):.4f}")
        print(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_lpips']):.4f}")
        print(f"  Diff:     {np.mean(all_metrics['wan_lpips']) - np.mean(all_metrics['dcvc_lpips']):+.4f}")
    
    # 保存统计结果
    stats_path = os.path.join(args.output_dir, 'statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("WAN vs DCVC_RT Comparison Statistics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test frames: {len(all_metrics['wan_psnr'])}\n")
        f.write(f"WAN QP: {wan_results['qp']}\n")
        f.write(f"DCVC_RT QP: {dcvc_results['qp']}\n")
        if args.match_bpp:
            target_bpp = args.target_bpp if args.target_bpp is not None else wan_results['avg']['bpp']
            f.write(f"Target BPP: {target_bpp:.4f}\n")
        f.write("\n")
        
        f.write(f"Average PSNR:\n")
        f.write(f"  WAN:      {np.mean(all_metrics['wan_psnr']):.2f} dB\n")
        f.write(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_psnr']):.2f} dB\n")
        f.write(f"  Diff:     {np.mean(all_metrics['wan_psnr']) - np.mean(all_metrics['dcvc_psnr']):+.2f} dB\n\n")
        
        f.write(f"Average BPP:\n")
        f.write(f"  WAN:      {np.mean(all_metrics['wan_bpp']):.4f}\n")
        f.write(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_bpp']):.4f}\n")
        f.write(f"  Diff:     {np.mean(all_metrics['wan_bpp']) - np.mean(all_metrics['dcvc_bpp']):+.4f}\n\n")
        
        f.write(f"Average SSIM:\n")
        f.write(f"  WAN:      {np.mean(all_metrics['wan_ssim']):.4f}\n")
        f.write(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_ssim']):.4f}\n")
        f.write(f"  Diff:     {np.mean(all_metrics['wan_ssim']) - np.mean(all_metrics['dcvc_ssim']):+.4f}\n\n")
        
        f.write(f"Average MS-SSIM:\n")
        f.write(f"  WAN:      {np.mean(all_metrics['wan_ms_ssim']):.4f}\n")
        f.write(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_ms_ssim']):.4f}\n")
        f.write(f"  Diff:     {np.mean(all_metrics['wan_ms_ssim']) - np.mean(all_metrics['dcvc_ms_ssim']):+.4f}\n\n")
        
        if 'wan_lpips' in all_metrics and len(all_metrics['wan_lpips']) > 0:
            f.write(f"Average LPIPS (lower is better):\n")
            f.write(f"  WAN:      {np.mean(all_metrics['wan_lpips']):.4f}\n")
            f.write(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_lpips']):.4f}\n")
            f.write(f"  Diff:     {np.mean(all_metrics['wan_lpips']) - np.mean(all_metrics['dcvc_lpips']):+.4f}\n\n")
        
        f.write(f"Average MSE:\n")
        f.write(f"  WAN:      {np.mean(all_metrics['wan_mse']):.6f}\n")
        f.write(f"  DCVC_RT:  {np.mean(all_metrics['dcvc_mse']):.6f}\n")
        f.write(f"  Diff:     {np.mean(all_metrics['wan_mse']) - np.mean(all_metrics['dcvc_mse']):+.6f}\n")
    
    print(f"\n✅ Statistics saved: {stats_path}")
    print(f"✅ All visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
