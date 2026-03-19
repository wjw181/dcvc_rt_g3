"""
eval_rd_paper.py
================
论文 R-D 曲线评测脚本

功能：
  - 测试 WAN 和 DCVC_RT 在多个 QP 下的率失真性能
  - 覆盖所有 HEVC Class B 序列（1920x1080）
  - 输出 CSV 结果 + 自动绘制 R-D 曲线

用法：
  python eval_rd_paper.py [--wan_qp_list QP1,QP2,...] [--dcvc_qp_list QP1,QP2,...] \
                          [--num_frames N] [--output_dir DIR] [--no_plot]
"""

import argparse
import csv
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import imageio

warnings.filterwarnings("ignore")

# ── 路径初始化 ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Wan2.2-main'))

# 直接复用已验证可用的推理函数和模型加载函数
from visualize_comparison import (
    compress_and_reconstruct,
    calculate_psnr,
    calculate_ssim,
    calculate_ms_ssim,
    calculate_lpips,
    load_image,
    load_wan_model   as _load_wan_model,
    load_dcvc_model  as _load_dcvc_model,
)
from src.models.image_model import DMCI

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# ── 测试序列 ────────────────────────────────────────────────
HEVC_B_SEQUENCES = [
    "BasketballDrive_1920x1080_50",
    "BQTerrace_1920x1080_60",
    "Cactus_1920x1080_50",
    "Kimono1_1920x1080_24",
    "ParkScene_1920x1080_24",
]
HEVC_B_ROOT = "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B"

# ── 默认 QP 列表 ────────────────────────────────────────────
DEFAULT_WAN_QP   = [22, 27, 32, 37, 42, 47]   # WAN 测试点（更多码率）
DEFAULT_DCVC_QP  = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 27, 32, 37]

# ── 模型路径 ────────────────────────────────────────────────
WAN_CHECKPOINT  = "pretrained/DMC_WAN/1/checkpoint_stage3_wan.pth.tar"
DCVC_CHECKPOINT = "pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar"
WAN_VAE         = "Wan2.2-main/checkpoints/Wan2.2_VAE.pth"
I_FRAME_MODEL   = "checkpoints/cvpr2025_image.pth.tar"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  模型加载
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_i_frame_model(device):
    model = DMCI()
    ckpt  = torch.load(I_FRAME_MODEL, map_location='cpu')
    sd    = ckpt.get('state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


def load_wan_model(device, checkpoint=None):
    ckpt = checkpoint or WAN_CHECKPOINT
    return _load_wan_model(ckpt, WAN_VAE, device)


def load_dcvc_model(device):
    return _load_dcvc_model(DCVC_CHECKPOINT, device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  推理核心（与 visualize_comparison.py 保持一致）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def infer_sequence(model, model_name, frames, qp, i_frame_net, lpips_fn, device):
    """
    对一组帧（列表，每个 [1,3,H,W] CPU tensor）执行压缩并收集指标。
    完全复用 compress_and_reconstruct 逻辑，与 visualize_comparison.py 一致。
    返回 dict: {psnr, bpp, ssim, ms_ssim, lpips}，每项为平均值。
    """
    psnrs, bpps, ssims, ms_ssims, lpipss = [], [], [], [], []

    # 始终以 frame 0 作为 I 帧参考，各 P 帧独立评估，结果更稳定
    i_frame = frames[0]
    for cur_frame in frames[1:]:
        result = compress_and_reconstruct(model, i_frame, cur_frame, i_frame_net, qp, device)

        # 与训练对齐：优先使用 x_hat_rgb（WAN VAE 解码输出）
        recon = result.get('x_hat_rgb', result['x_hat'])
        bpp   = result['bpp'].mean().item()

        cur_dev = cur_frame.to(device)
        _, _, H_orig, W_orig = cur_dev.shape
        recon = recon[:, :, :H_orig, :W_orig]

        psnrs.append(calculate_psnr(cur_dev, recon))
        bpps.append(bpp)
        ssims.append(calculate_ssim(cur_dev, recon))
        ms_ssims.append(calculate_ms_ssim(cur_dev, recon))
        lp = calculate_lpips(cur_dev, recon, lpips_fn)
        if lp is not None:
            lpipss.append(lp)

    out = {
        'psnr':     float(np.mean(psnrs)),
        'bpp':      float(np.mean(bpps)),
        'ssim':     float(np.mean(ssims)),
        'ms_ssim':  float(np.mean(ms_ssims)),
    }
    if lpipss:
        out['lpips'] = float(np.mean(lpipss))
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  序列加载
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_sequence_frames(seq_dir, num_frames):
    pngs = sorted([
        os.path.join(seq_dir, f)
        for f in os.listdir(seq_dir)
        if f.endswith('.png')
    ])
    # ref + num_frames P-frames
    selected = pngs[:num_frames + 1]
    # 使用与 visualize_comparison.py 相同的 load_image 函数（RGB→YUV）
    return [load_image(p) for p in selected]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  绘图
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_rd_curves(results, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available, skipping plots.")
        return

    metrics = ['psnr', 'ssim', 'ms_ssim', 'lpips']
    ylabels = ['PSNR (dB)', 'SSIM', 'MS-SSIM', 'LPIPS ↓']

    # 聚合每个模型在所有序列上的平均值
    def aggregate(model_name):
        """返回 {qp: {metric: val}} 按 bpp 排序"""
        from collections import defaultdict
        by_qp = defaultdict(lambda: defaultdict(list))
        for row in results:
            if row['model'] == model_name:
                for m in metrics + ['bpp']:
                    if m in row:
                        by_qp[row['qp']][m].append(row[m])
        # 对每个 qp 取均值
        pts = []
        for qp, d in by_qp.items():
            entry = {'qp': qp}
            for k, v in d.items():
                entry[k] = float(np.mean(v))
            pts.append(entry)
        pts.sort(key=lambda x: x['bpp'])
        return pts

    wan_pts  = aggregate('WAN')
    dcvc_pts = aggregate('DCVC_RT')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('R-D Curves — HEVC Class B (Average)', fontsize=14, fontweight='bold')

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        # WAN
        if wan_pts:
            bpps = [p['bpp'] for p in wan_pts if metric in p]
            vals = [p[metric] for p in wan_pts if metric in p]
            ax.plot(bpps, vals, 'b-o', linewidth=2, markersize=6, label='WAN (Ours)')

        # DCVC_RT
        if dcvc_pts:
            bpps = [p['bpp'] for p in dcvc_pts if metric in p]
            vals = [p[metric] for p in dcvc_pts if metric in p]
            ax.plot(bpps, vals, 'r--s', linewidth=2, markersize=6, label='DCVC_RT')

        ax.set_xlabel('BPP', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if metric == 'lpips':
            ax.invert_yaxis()  # LPIPS 越低越好，坐标轴翻转方便看

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rd_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ R-D 曲线保存至: {plot_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    parser = argparse.ArgumentParser(description='论文 R-D 曲线评测')
    parser.add_argument('--wan_qp_list',  default=','.join(map(str, DEFAULT_WAN_QP)))
    parser.add_argument('--dcvc_qp_list', default=','.join(map(str, DEFAULT_DCVC_QP)))
    parser.add_argument('--num_frames',   type=int, default=7,
                        help='每条序列测试的 P 帧数（参考帧不计入）')
    parser.add_argument('--sequences',    default='all',
                        help='逗号分隔的序列名，或 "all"')
    parser.add_argument('--output_dir',   default='tests/results_rd_paper')
    parser.add_argument('--no_plot',      action='store_true')
    parser.add_argument('--skip_dcvc',    action='store_true', help='跳过 DCVC_RT 评测（节省时间）')
    parser.add_argument('--skip_wan',     action='store_true', help='跳过 WAN 评测')
    parser.add_argument('--wan_checkpoint', default=None,
                        help='WAN 模型 checkpoint 路径（默认使用 Stage2 best）')
    return parser.parse_args()


def main():
    args  = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    wan_qps  = [int(q) for q in args.wan_qp_list.split(',')]
    dcvc_qps = [int(q) for q in args.dcvc_qp_list.split(',')]

    seqs = HEVC_B_SEQUENCES if args.sequences == 'all' else \
           [s.strip() for s in args.sequences.split(',')]

    print("=" * 60)
    print("论文 R-D 曲线评测")
    print(f"  序列: {seqs}")
    print(f"  WAN QP:    {wan_qps}")
    print(f"  DCVC_RT QP: {dcvc_qps}")
    print(f"  P 帧数: {args.num_frames}")
    print(f"  输出目录: {args.output_dir}")
    print("=" * 60)

    # ── 加载公共模型 ──────────────────────────────────────────
    print("\n[1/3] 加载 I-frame 模型...")
    i_frame_net = load_i_frame_model(device)

    print("[2/3] 加载感知评测模型...")
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()

    results = []  # 每行: {model, seq, qp, bpp, psnr, ssim, ms_ssim, lpips}
    csv_path = os.path.join(args.output_dir, 'rd_results.csv')

    # ── WAN 评测 ──────────────────────────────────────────────
    if not args.skip_wan:
        print("\n[3a] 加载 WAN 模型...")
        wan_model = load_wan_model(device, checkpoint=args.wan_checkpoint)

        for seq_name in seqs:
            seq_dir = os.path.join(HEVC_B_ROOT, seq_name)
            if not os.path.exists(seq_dir):
                print(f"  ⚠️  序列目录不存在: {seq_dir}")
                continue
            print(f"\n  序列: {seq_name}")
            frames = load_sequence_frames(seq_dir, args.num_frames)

            for qp in wan_qps:
                print(f"    WAN QP={qp} ...", end='', flush=True)
                try:
                    met = infer_sequence(
                        wan_model, 'WAN', frames, qp,
                        i_frame_net, lpips_fn, device
                    )
                    row = {'model': 'WAN', 'seq': seq_name, 'qp': qp, **met}
                    results.append(row)
                    print(f"  BPP={met['bpp']:.4f}  PSNR={met['psnr']:.2f}  "
                          f"SSIM={met['ssim']:.4f}  LPIPS={met.get('lpips', 'N/A')}")
                except Exception as e:
                    print(f"  ❌ 失败: {e}")

        del wan_model
        torch.cuda.empty_cache()

    # ── DCVC_RT 评测 ─────────────────────────────────────────
    if not args.skip_dcvc:
        print("\n[3b] 加载 DCVC_RT 模型...")
        dcvc_model = load_dcvc_model(device)

        for seq_name in seqs:
            seq_dir = os.path.join(HEVC_B_ROOT, seq_name)
            if not os.path.exists(seq_dir):
                continue
            print(f"\n  序列: {seq_name}")
            frames = load_sequence_frames(seq_dir, args.num_frames)

            for qp in dcvc_qps:
                print(f"    DCVC_RT QP={qp} ...", end='', flush=True)
                try:
                    met = infer_sequence(
                        dcvc_model, 'DCVC_RT', frames, qp,
                        i_frame_net, lpips_fn, device
                    )
                    row = {'model': 'DCVC_RT', 'seq': seq_name, 'qp': qp, **met}
                    results.append(row)
                    print(f"  BPP={met['bpp']:.4f}  PSNR={met['psnr']:.2f}  "
                          f"SSIM={met['ssim']:.4f}  LPIPS={met.get('lpips', 'N/A')}")
                except Exception as e:
                    print(f"  ❌ 失败: {e}")

        del dcvc_model
        torch.cuda.empty_cache()

    # ── 保存 CSV ──────────────────────────────────────────────
    if results:
        fieldnames = ['model', 'seq', 'qp', 'bpp', 'psnr', 'ssim', 'ms_ssim', 'lpips']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ 详细结果保存至: {csv_path}")

        # ── 打印汇总表 ─────────────────────────────────────────
        print("\n" + "=" * 70)
        print("汇总（所有序列平均）")
        print("=" * 70)
        from collections import defaultdict
        agg = defaultdict(lambda: defaultdict(list))
        for row in results:
            key = (row['model'], row['qp'])
            for m in ['bpp', 'psnr', 'ssim', 'ms_ssim', 'lpips']:
                if m in row:
                    agg[key][m].append(row[m])

        print(f"{'Model':<12} {'QP':>4} {'BPP':>7} {'PSNR':>8} {'SSIM':>7} {'MS-SSIM':>9} {'LPIPS':>7}")
        print("-" * 60)
        for (model, qp), d in sorted(agg.items(), key=lambda x: (x[0][0], np.mean(x[1]['bpp']))):
            bpp      = np.mean(d['bpp'])
            psnr     = np.mean(d['psnr'])
            ssim     = np.mean(d['ssim'])
            ms_ssim  = np.mean(d['ms_ssim'])
            lpips_v  = np.mean(d['lpips']) if d['lpips'] else float('nan')
            print(f"{model:<12} {qp:>4} {bpp:>7.4f} {psnr:>8.2f} {ssim:>7.4f} {ms_ssim:>9.4f} {lpips_v:>7.4f}")

        # ── 绘图 ─────────────────────────────────────────────────
        if not args.no_plot:
            plot_rd_curves(results, args.output_dir)

    print("\n✅ 评测完成！")


if __name__ == '__main__':
    main()
