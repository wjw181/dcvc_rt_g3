#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多QP测试和可视化工具
Evaluate YUV420 video sequences with DMCI (I-frame) + DMC (P-frame) models
across multiple QP values, with Rate-Distortion curve visualization.

功能：
- 支持单个QP、QP范围或QP列表测试
- 自动生成RD曲线可视化图表
- 保存测试结果到JSON文件

Author: Auto-generated
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import matplotlib.pyplot as plt
import json

# ===== 工程依赖 =====
from src.utils.transforms import (
    ycbcr420_to_444_np,
    yuv_444_to_420,
)
from src.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from src.utils.video_reader import YUV420Reader
from src.models.image_model import DMCI
from src.models.video_t import DMC
# ====================

# ────────────────────────── 通用工具 ──────────────────────────

def np2tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """(3,H,W) np.uint8 → (1,3,H,W) float32 [0,1]"""
    return (
        torch.from_numpy(arr)
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .div_(255.0)
    )


# ---------------------------------------------------------------------
# PSNR identical to get_distortion:   x_hat, y, u, v  → weighted PSNR
# ---------------------------------------------------------------------

def psnr(x_hat: torch.Tensor, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Compute 6:1:1 weighted PSNR given reconstructed tensor and GT planes."""
    # Convert reconstruction to YUV420 planes
    y_rec, uv_rec = yuv_444_to_420(x_hat)
    y_rec = (
        torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()[0]
    )  # (H,W)
    uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
    u_rec, v_rec = uv_rec[0], uv_rec[1]
    y = y.squeeze(0).cpu().numpy()[0]
    u = u.squeeze(0).cpu().numpy()[0]
    v = v.squeeze(0).cpu().numpy()[0]
    # Channel PSNRs
    
    psnr_y = calc_psnr(y, y_rec)
    psnr_u = calc_psnr(u, u_rec)
    psnr_v = calc_psnr(v, v_rec)

    # Weighted overall
    return (6 * psnr_y + psnr_u + psnr_v) / 8.0, psnr_y, psnr_u, psnr_v


def Var(x):
    return Variable(x.cuda() if torch.cuda.is_available() else x)


class RateDistortionLoss(nn.Module):
    """λ · MSE  +  bpp"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, result, target, lamada):
        out = {
            "bpp_loss": result["bpp"],
            "mse_loss": result["mse"],
        }
        out["loss"] = lamada * out["mse_loss"] + out["bpp_loss"]
        return out


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def get_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt = ckpt.get("state_dict", ckpt.get("net", ckpt))
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


# ──────────────────────── 单文件 Dataset ────────────────────────
class SingleYuvDataset(Dataset):
    def __init__(
        self,
        yuv_path: str,
        width: int,
        height: int,
        device: str = "cpu",
        max_frames: int = -1,
    ):
        self.yuv_path, self.device = yuv_path, device
        reader = YUV420Reader(yuv_path, width, height)

        frames = []  # 存储 YUV444 Tensor
        y_list = []  # 存储 原始 Y 平面 (H,W)
        uv_list = []  # 存储 原始 UV 平面 (2,H/2,W/2)

        while True:
            if max_frames > 0 and len(frames) >= max_frames:
                break
            got = reader.read_one_frame()
            if got is None:
                break
            y, uv = got  # Y:(H,W)  UV:(2,H/2,W/2)
            y_list.append(y)
            uv_list.append(uv)
            yuv444 = ycbcr420_to_444_np(y, uv)
            frames.append(np2tensor(yuv444, device))
        reader.close()

        if len(frames) < 2:
            raise RuntimeError(
                f"YUV 文件至少包含 2 帧（1×I + ≥1×P），但只读到 {len(frames)} 帧。"
            )
        self.frames = torch.cat(frames, dim=0)  # (N,3,H,W)
        self.y_list = y_list   # len=N, each (H,W)
        self.uv_list = uv_list  # len=N, each (2,H/2,W/2)
        self.name = os.path.basename(yuv_path)

    def __len__(self):
        return 1

    def __getitem__(self, _):
        ref = self.frames[:1]  # (1,3,H,W)
        gop = self.frames.unsqueeze(0)  # (1,N,3,H,W)
        return ref, gop, self.y_list, self.uv_list, self.name
    
index_map = [0, 1, 0, 2, 0, 1, 0, 2]

# ───────────────────────────── 测试循环 ─────────────────────────────
def Test(epoch, model_i, test_dataloader, model, criterion, test_num, QP, verbose=True):
    """
    测试单个QP值，返回结果字典
    
    Args:
        epoch: epoch编号
        model_i: I-frame模型
        test_dataloader: 测试数据加载器
        model: P-frame模型
        criterion: 损失函数
        test_num: 测试帧数
        QP: QP值
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含测试结果的字典
    """
    model.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "psnr", "mse", "y", "u", "v")}

    with torch.no_grad():
        for ref, gop, y_list, uv_list, name in test_dataloader:
            ref = ref.squeeze(1)
            gop = gop.squeeze(1)
            ref, gop = Var(ref), Var(gop)
            lamada, qs_global = 3600, QP # λ, QP
            # I-frame
            ref = ref.to(torch.float16)

            if qs_global > 63:
                ref = model_i.compress_(ref, 63)
            else:
                ref = model_i.compress_(ref, qs_global)

            # P-frames
            model.clear_dpb()
            model.add_ref_frame(None, ref)
            last_qp = 63
            totals = {k: 0.0 for k in ("loss", "bpp", "psnr", "y", "u", "v")}
            for j in range(1, test_num + 1):
                cur = gop[:, j].to(torch.float16)
                if j%32== 1:
                    model.prepare_feature_adaptor_i(last_qp)

                fa_idx = index_map[j % 8]
                
                curr_qp = model.shift_qp(qs_global, fa_idx)
                last_qp = curr_qp
                if QP > 63:
                  curr_qp = QP
                out = model(cur, curr_qp)
                

                out_loss = criterion(out, cur, lamada)
                
                y_gt_np = y_list[j]
                uv_gt = uv_list[j]
                u_gt_np = uv_gt[:, 0:1, :, :]
                v_gt_np = uv_gt[:, 1:2, :, :]
          
                weighted_psnr, psnr_y, psnr_u, psnr_v = psnr(out["x_hat"], y_gt_np, u_gt_np, v_gt_np)

                # 打印每帧结果（如果verbose=True）
                if verbose:
                    print(
                        f"QP={QP:2d} | Frame {j:02d}: Weighted PSNR: {weighted_psnr:.3f} dB | "
                        f"Y: {psnr_y:.3f} | U: {psnr_u:.3f} | V: {psnr_v:.3f} | "
                        f"Bpp: {out['bpp'].mean().item():.4f}"
                    )

                totals["loss"] += out_loss["loss"].mean().item()
                totals["bpp"] += out["bpp"].mean().item()
                totals["psnr"] += weighted_psnr
                totals["y"] += psnr_y
                totals["u"] += psnr_u
                totals["v"] += psnr_v

            # Update meters (averaging over P-frames)
            for k in ("loss", "bpp", "psnr", "y", "u", "v"):
                meters[k].update(totals[k] / test_num)
            meters["mse"].update(out_loss["mse_loss"].mean().item())

            if verbose:
                print(
                    f"QP={QP:2d} | [{name[0]}] Weighted PSNR: {meters['psnr'].avg:.3f} dB | "
                    f"Y-PSNR: {meters['y'].avg:.3f} dB | "
                    f"U-PSNR: {meters['u'].avg:.3f} dB | "
                    f"V-PSNR: {meters['v'].avg:.3f} dB | "
                    f"MSE: {meters['mse'].avg:.5f} | "
                    f"Bpp: {meters['bpp'].avg:.4f}\n"
                )
            
            # 返回结果字典
            return {
                "qp": QP,
                "name": name[0],
                "weighted_psnr": meters['psnr'].avg,
                "psnr_y": meters['y'].avg,
                "psnr_u": meters['u'].avg,
                "psnr_v": meters['v'].avg,
                "bpp": meters['bpp'].avg,
                "mse": meters['mse'].avg,
                "loss": meters['loss'].avg,
            }


# ────────────────────────────── 可视化函数 ──────────────────────────────

def plot_rd_curves(results, save_path=None):
    """
    绘制Rate-Distortion曲线
    results: list of dict, 每个dict包含qp, bpp, weighted_psnr等
    """
    if not results:
        print("没有结果可绘制")
        return
    
    # 按QP排序
    results = sorted(results, key=lambda x: x['qp'])
    
    qps = [r['qp'] for r in results]
    bpps = [r['bpp'] for r in results]
    psnrs = [r['weighted_psnr'] for r in results]
    psnr_y = [r['psnr_y'] for r in results]
    psnr_u = [r['psnr_u'] for r in results]
    psnr_v = [r['psnr_v'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Rate-Distortion Curves: {results[0]["name"]}', fontsize=14, fontweight='bold')
    
    # 1. Weighted PSNR vs BPP
    ax1 = axes[0, 0]
    ax1.plot(bpps, psnrs, 'o-', linewidth=2, markersize=8, label='Weighted PSNR (6:1:1)')
    for i, qp in enumerate(qps):
        ax1.annotate(f'QP={qp}', (bpps[i], psnrs[i]), 
                    textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
    ax1.set_xlabel('Bitrate (bpp)', fontsize=11)
    ax1.set_ylabel('PSNR (dB)', fontsize=11)
    ax1.set_title('Weighted PSNR vs Bitrate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Y-PSNR vs BPP
    ax2 = axes[0, 1]
    ax2.plot(bpps, psnr_y, 's-', linewidth=2, markersize=8, color='green', label='Y-PSNR')
    for i, qp in enumerate(qps):
        ax2.annotate(f'QP={qp}', (bpps[i], psnr_y[i]), 
                    textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
    ax2.set_xlabel('Bitrate (bpp)', fontsize=11)
    ax2.set_ylabel('PSNR (dB)', fontsize=11)
    ax2.set_title('Y-PSNR vs Bitrate', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. UV-PSNR vs BPP
    ax3 = axes[1, 0]
    ax3.plot(bpps, psnr_u, '^-', linewidth=2, markersize=8, color='orange', label='U-PSNR')
    ax3.plot(bpps, psnr_v, 'v-', linewidth=2, markersize=8, color='red', label='V-PSNR')
    for i, qp in enumerate(qps):
        ax3.annotate(f'QP={qp}', (bpps[i], (psnr_u[i] + psnr_v[i])/2), 
                    textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
    ax3.set_xlabel('Bitrate (bpp)', fontsize=11)
    ax3.set_ylabel('PSNR (dB)', fontsize=11)
    ax3.set_title('UV-PSNR vs Bitrate', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. QP vs Metrics
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(qps, psnrs, 'o-', linewidth=2, markersize=8, color='blue', label='Weighted PSNR')
    line2 = ax4_twin.plot(qps, bpps, 's-', linewidth=2, markersize=8, color='red', label='BPP')
    ax4.set_xlabel('QP', fontsize=11)
    ax4.set_ylabel('PSNR (dB)', fontsize=11, color='blue')
    ax4_twin.set_ylabel('Bitrate (bpp)', fontsize=11, color='red')
    ax4.set_title('QP vs PSNR & Bitrate', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_results_to_json(results, save_path):
    """保存结果到JSON文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {save_path}")


# ────────────────────────────── 主入口 ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="多QP测试和可视化工具 - Evaluate YUV420 video with DMCI/DMC across multiple QP values")
    parser.add_argument("--yuv",     default="/home/serverdn/hdd-0/wjw/DCVC_RT_TRAIN_LAST_DANCE/test_data/HEVC_D/BasketballPass_416x240_50.yuv", help=".yuv file path")
    parser.add_argument("--width",   type=int, default=416, help="video width")
    parser.add_argument("--height",  type=int, default=240, help="video height")
    parser.add_argument("--ckpt_i",  default="checkpoints/cvpr2025_image.pth.tar", help="DMCI checkpoint")
    parser.add_argument("--ckpt_p",  default="pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar", help="DMC checkpoint")

    parser.add_argument("--QP",  type=str, default="70", 
                        help="QP值，支持三种格式:\n"
                             "  1. 单个值: 70\n"
                             "  2. 范围: 50:60:5 (从50到60，步长5)\n"
                             "  3. 列表: 50,55,60,65,70\n"
                             "注意: QP越小码率越高，建议高码率测试使用: 20:40:5")
    parser.add_argument("--high_bitrate", action="store_true",
                        help="高码率测试模式，自动测试QP范围 20:50:5（QP越小码率越高）")
    parser.add_argument("--max_frames", type=int, default=32,
                        help="要测试的最大帧数（包括 I 帧）。传 -1 或不传则读取全部帧。")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="结果保存目录（图表和JSON文件）")
    parser.add_argument("--no_plot", action="store_true",
                        help="不生成可视化图表")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式，不打印每帧详细信息")
    args = parser.parse_args() 
    
    torch.backends.cudnn.benchmark = True   
    device = args.device

    # 解析QP值
    def parse_qp(qp_str):
        """解析QP字符串，支持单个值、范围或列表"""
        if ':' in qp_str:
            # 范围格式: start:end:step 或 start:end
            parts = qp_str.split(':')
            start = int(parts[0])
            end = int(parts[1])
            step = int(parts[2]) if len(parts) > 2 else 1
            return list(range(start, end + 1, step))
        elif ',' in qp_str:
            # 列表格式: 50,55,60,65,70
            return [int(x.strip()) for x in qp_str.split(',')]
        else:
            # 单个值
            return [int(qp_str)]
    
    # 高码率测试模式
    if args.high_bitrate:
        qp_str = "20:50:5"
        print(f"高码率测试模式已启用，使用QP范围: {qp_str}")
        print("提示: QP越小码率越高，此范围将测试高码率场景")
    else:
        qp_str = args.QP
    
    qp_list = parse_qp(qp_str)
    
    # QP范围检查：根据模型设计
    # - get_qp_num() = 64
    # - extra_qp = max(qp_shift) = 8 (qp_shift = [0, 8, 4])
    # - 参数数组大小 = 64 + 8 = 72，索引范围 [0, 71]
    # 所以最大支持的QP值是71
    # 注意: QP越小码率越高，QP越大码率越低
    MAX_SAFE_QP = 71  # 根据模型参数，最大安全QP值（对应低码率）
    MIN_QP = 0        # 最小QP值（对应高码率）
    
    # 过滤和警告超出范围的QP值
    original_qp_list = qp_list.copy()
    qp_list = [qp for qp in qp_list if MIN_QP <= qp <= MAX_SAFE_QP]
    skipped_qps = [qp for qp in original_qp_list if qp not in qp_list]
    
    if skipped_qps:
        print(f"警告: 以下QP值超出安全范围 [{MIN_QP}, {MAX_SAFE_QP}]，将被跳过: {skipped_qps}")
    
    if not qp_list:
        raise ValueError(f"没有有效的QP值可以测试。所有QP值都超出了范围 [{MIN_QP}, {MAX_SAFE_QP}]")
    
    print(f"将测试以下QP值: {qp_list}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = SingleYuvDataset(
        args.yuv, args.width, args.height, device, args.max_frames
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    i_net = DMCI().to(device).half()
    i_net.load_state_dict(get_state_dict(args.ckpt_i))
    i_net.eval()

    p_net = DMC().to(device).half()
    p_net.load_state_dict(get_state_dict(args.ckpt_p))
    p_net.eval()
    
    criterion = RateDistortionLoss()

    total_frames = dataset.frames.size(0)
    use_frames = (
        min(total_frames, args.max_frames)
        if args.max_frames > 0
        else total_frames
    )
    test_num = use_frames - 1
    if test_num < 1:
        raise RuntimeError(
            f"读到的帧数只有 {use_frames}，不足以形成 1×I + ≥1×P 帧。"
        )

    # 测试所有QP值
    all_results = []
    failed_qps = []
    print("\n" + "="*80)
    print("开始多QP测试")
    print("="*80 + "\n")
    
    for qp in qp_list:
        print(f"\n{'='*80}")
        print(f"测试 QP = {qp}")
        print(f"{'='*80}\n")
        try:
            # 确保模型状态重置（特别是DPB）
            p_net.clear_dpb()
            p_net.eval()  # 确保在eval模式
            i_net.eval()  # 确保在eval模式
            
            result = Test(0, i_net, loader, p_net, criterion, test_num, qp, verbose=not args.quiet)
            if result is not None:
                all_results.append(result)
                print(f"✓ QP={qp} 测试完成")
            else:
                print(f"✗ QP={qp} 测试返回None，跳过")
                failed_qps.append(qp)
        except Exception as e:
            print(f"✗ QP={qp} 测试失败: {str(e)}")
            print(f"  错误类型: {type(e).__name__}")
            import traceback
            if not args.quiet:
                traceback.print_exc()
            failed_qps.append(qp)
            # 继续测试下一个QP值
            continue
    
    if failed_qps:
        print(f"\n警告: 以下QP值测试失败: {failed_qps}")
    
    if not all_results:
        raise RuntimeError("所有QP值测试都失败了，没有可用的结果。")
    
    # 打印汇总
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    print(f"{'QP':<6} {'Weighted PSNR':<18} {'Y-PSNR':<12} {'U-PSNR':<12} {'V-PSNR':<12} {'BPP':<10}")
    print("-"*80)
    for r in sorted(all_results, key=lambda x: x['qp']):
        print(f"{r['qp']:<6} {r['weighted_psnr']:<18.3f} {r['psnr_y']:<12.3f} "
              f"{r['psnr_u']:<12.3f} {r['psnr_v']:<12.3f} {r['bpp']:<10.4f}")
    print("="*80 + "\n")
    
    # 保存结果
    video_name = os.path.splitext(os.path.basename(args.yuv))[0]
    json_path = os.path.join(args.save_dir, f"{video_name}_results.json")
    save_results_to_json(all_results, json_path)
    
    # 生成可视化
    if not args.no_plot and len(all_results) > 0:
        plot_path = os.path.join(args.save_dir, f"{video_name}_rd_curves.png")
        plot_rd_curves(all_results, save_path=plot_path)


if __name__ == "__main__":
    main()

