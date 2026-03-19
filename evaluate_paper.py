#!/usr/bin/env python3
"""
Publication-quality evaluation: WAN Generative Video Codec vs DCVC_RT

Features:
  - Multi-QP RD curve evaluation (independent sweep per model)
  - Multi-sequence testing (HEVC B/C/D/E)
  - BD-rate / BD-metric (Bjontegaard Delta) calculation
  - Publication-quality RD curve plots (PDF + PNG)
  - Metrics: PSNR, SSIM, MS-SSIM, LPIPS
  - Visual comparison at matched BPP
  - Detailed CSV + JSON output
  - LaTeX-ready summary table

Usage:
  python evaluate_paper.py                       # default settings
  python evaluate_paper.py --num_frames 49       # more frames
  python evaluate_paper.py --test_dirs /path/to/seq1 /path/to/seq2
"""

import os
import sys
import argparse
import json
import time
import csv
import math

import torch
import numpy as np
import imageio
import torch.nn.functional as F
from collections import OrderedDict
from pytorch_msssim import ssim as calc_ssim_fn, ms_ssim as calc_ms_ssim_fn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Publication matplotlib style
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.video_t_g_wan import DMC_WAN
from src.models.video_t import DMC
from src.models.image_model import DMCI
from src.utils.transforms import rgb2ycbcr, yuv_444_to_420, yuv_420_to_444

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[WARN] LPIPS not installed, perceptual metric will be skipped")


# ═══════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════

def _strip_module(state_dict):
    return OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())


def load_wan_model(ckpt_path, vae_path, device):
    model = DMC_WAN(mode='latent', wan_vae_checkpoint=vae_path, freeze_vae=True)
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = _strip_module(sd.get('state_dict', sd))
    model.load_state_dict(sd, strict=False)
    print(f"[OK] WAN model loaded: {ckpt_path}")
    return model.to(device).eval()


def load_dcvc_model(ckpt_path, device):
    model = DMC()
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = _strip_module(sd.get('state_dict', sd))
    model.load_state_dict(sd, strict=False)
    print(f"[OK] DCVC_RT model loaded: {ckpt_path}")
    return model.to(device).eval()


def load_i_frame_model(ckpt_path, device):
    model = DMCI()
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd.get('state_dict', sd)
    model.load_state_dict(sd, strict=False)
    print(f"[OK] I-frame model loaded: {ckpt_path}")
    return model.to(device).eval()


# ═══════════════════════════════════════════════════════════════════════
#  Data Loading & Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_frames_from_dir(directory, max_frames=None):
    """Load PNG frames → YUV444 tensors [1,3,H,W], matching training pipeline."""
    files = sorted(f for f in os.listdir(directory) if f.lower().endswith('.png'))
    if max_frames is not None:
        files = files[:max_frames]
    frames = []
    for f in files:
        img = imageio.imread(os.path.join(directory, f)).astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W] RGB
        yuv = rgb2ycbcr(t.squeeze(0)).unsqueeze(0)
        y, uv = yuv_444_to_420(yuv)
        yuv = yuv_420_to_444(y, uv)
        frames.append(yuv)
    return frames


def _pad(x, m=16):
    _, _, h, w = x.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return x
    return F.pad(x, (0, pw, 0, ph), mode='replicate')


# ═══════════════════════════════════════════════════════════════════════
#  Metric Computation
# ═══════════════════════════════════════════════════════════════════════

def _psnr(a, b):
    mse = F.mse_loss(a, b).item()
    if mse < 1e-10:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _mse(a, b):
    return F.mse_loss(a, b).item()


def _ssim(a, b):
    return calc_ssim_fn(a, b, data_range=1.0, size_average=True).item()


def _ms_ssim(a, b):
    try:
        return calc_ms_ssim_fn(a, b, data_range=1.0, size_average=True).item()
    except Exception:
        return _ssim(a, b)


def _lpips(a, b, fn):
    if fn is None:
        return None
    a = torch.clamp(a, 0.0, 1.0) * 2.0 - 1.0
    b = torch.clamp(b, 0.0, 1.0) * 2.0 - 1.0
    return fn(a, b).item()


# ═══════════════════════════════════════════════════════════════════════
#  Sequence Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_at_qp(model, model_name, frames, qp, i_frame_net, device,
                   lpips_fn=None, verbose=False):
    """
    Evaluate *model* on *frames* at a single QP.
    Mimics training: I-frame init → sequential P-frame coding, DPB maintained internally.
    
    For WAN:  reconstruction = x_hat_rgb (pseudo-YUV, matches training PSNR)
    For DCVC: reconstruction = x_hat    (pure YUV)
    """
    _, _, H, W = frames[0].shape

    # ── I-frame ──
    ref = _pad(frames[0].to(device))
    i_qp = min(qp, 63)
    ref_hat = i_frame_net.compress_(ref, i_qp)
    _, _, rH, rW = ref.shape
    if ref_hat.shape[2] != rH or ref_hat.shape[3] != rW:
        ref_hat = ref_hat[:, :, :rH, :rW]

    model.clear_dpb()
    model.add_ref_frame(None, ref_hat)

    # ── P-frames ──
    records = []
    for idx in range(1, len(frames)):
        cur = frames[idx].to(device)
        cur_pad = _pad(cur)

        result = model(cur_pad, qp)

        # WAN → x_hat_rgb (pseudo-YUV);  DCVC → x_hat (YUV)
        recon = result.get('x_hat_rgb', result['x_hat'])[:, :, :H, :W]
        bpp = result['bpp'].mean().item()

        psnr_val = _psnr(cur, recon)
        psnr_y   = _psnr(cur[:, 0:1], recon[:, 0:1])
        mse_val  = _mse(cur, recon)
        ssim_val = _ssim(cur, recon)
        ms_val   = _ms_ssim(cur, recon)
        lp_val   = _lpips(cur, recon, lpips_fn)

        records.append(dict(
            frame=idx, bpp=bpp,
            psnr=psnr_val, psnr_y=psnr_y, mse=mse_val,
            ssim=ssim_val, ms_ssim=ms_val, lpips=lp_val,
        ))
        if verbose:
            lp_str = f" LPIPS={lp_val:.4f}" if lp_val is not None else ""
            print(f"    Frame {idx:02d}: BPP={bpp:.4f} PSNR={psnr_val:.2f} "
                  f"SSIM={ssim_val:.4f} MS-SSIM={ms_val:.4f}{lp_str}")

    # ── Average ──
    keys = ['bpp', 'psnr', 'psnr_y', 'mse', 'ssim', 'ms_ssim']
    avg = {k: float(np.mean([r[k] for r in records])) for k in keys}
    if records[0]['lpips'] is not None:
        avg['lpips'] = float(np.mean([r['lpips'] for r in records]))

    return {'qp': qp, 'avg': avg, 'per_frame': records}


def sweep_qps(model, model_name, frames, qp_list, i_frame_net, device,
              lpips_fn=None, verbose=False):
    """Evaluate model at multiple QPs, return list of result dicts."""
    results = []
    for qp in qp_list:
        t0 = time.time()
        r = evaluate_at_qp(model, model_name, frames, qp, i_frame_net,
                           device, lpips_fn, verbose)
        dt = time.time() - t0
        a = r['avg']
        lp = f" LPIPS={a['lpips']:.4f}" if 'lpips' in a else ""
        print(f"  {model_name:8s} QP={qp:3d}  "
              f"BPP={a['bpp']:.4f}  PSNR={a['psnr']:.2f}  "
              f"SSIM={a['ssim']:.4f}  MS-SSIM={a['ms_ssim']:.4f}{lp}  "
              f"({dt:.1f}s)")
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════
#  BD-rate / BD-metric (Bjontegaard Delta)
# ═══════════════════════════════════════════════════════════════════════

def _bd_rate_core(rate_a, dist_a, rate_t, dist_t):
    """
    BD-Rate: average log-rate difference between two RD curves.
    anchor = a, test = t.  Negative → test saves bitrate.
    Requires ≥ 4 points each with overlapping distortion range.
    """
    if len(rate_a) < 4 or len(rate_t) < 4:
        return None
    lR_a = np.log10(np.array(rate_a, dtype=np.float64))
    lR_t = np.log10(np.array(rate_t, dtype=np.float64))
    D_a  = np.array(dist_a, dtype=np.float64)
    D_t  = np.array(dist_t, dtype=np.float64)

    ia = np.argsort(D_a); D_a, lR_a = D_a[ia], lR_a[ia]
    it = np.argsort(D_t); D_t, lR_t = D_t[it], lR_t[it]

    lo = max(D_a[0], D_t[0])
    hi = min(D_a[-1], D_t[-1])
    if lo >= hi:
        return None

    deg = min(3, len(D_a) - 1, len(D_t) - 1)
    p_a = np.polyfit(D_a, lR_a, deg)
    p_t = np.polyfit(D_t, lR_t, deg)

    x = np.linspace(lo, hi, 1000)
    int_a = np.trapz(np.polyval(p_a, x), x)
    int_t = np.trapz(np.polyval(p_t, x), x)

    avg = (int_t - int_a) / (hi - lo)
    return (10.0 ** avg - 1.0) * 100.0


def _bd_metric_core(rate_a, dist_a, rate_t, dist_t):
    """
    BD-PSNR / BD-SSIM etc.: average distortion difference.
    Positive → test is better.
    """
    if len(rate_a) < 4 or len(rate_t) < 4:
        return None
    lR_a = np.log10(np.array(rate_a, dtype=np.float64))
    lR_t = np.log10(np.array(rate_t, dtype=np.float64))
    D_a  = np.array(dist_a, dtype=np.float64)
    D_t  = np.array(dist_t, dtype=np.float64)

    ia = np.argsort(lR_a); lR_a, D_a = lR_a[ia], D_a[ia]
    it = np.argsort(lR_t); lR_t, D_t = lR_t[it], D_t[it]

    lo = max(lR_a[0], lR_t[0])
    hi = min(lR_a[-1], lR_t[-1])
    if lo >= hi:
        return None

    deg = min(3, len(D_a) - 1, len(D_t) - 1)
    p_a = np.polyfit(lR_a, D_a, deg)
    p_t = np.polyfit(lR_t, D_t, deg)

    x = np.linspace(lo, hi, 1000)
    int_a = np.trapz(np.polyval(p_a, x), x)
    int_t = np.trapz(np.polyval(p_t, x), x)

    return (int_t - int_a) / (hi - lo)


def compute_bd_metrics(wan_results, dcvc_results):
    """
    Compute BD-Rate for PSNR / SSIM / MS-SSIM / LPIPS.
    anchor = DCVC_RT,  test = WAN (Ours).
    """
    wan_bpp  = [r['avg']['bpp']  for r in wan_results]
    dcvc_bpp = [r['avg']['bpp'] for r in dcvc_results]

    bd = {}

    for metric in ['psnr', 'ssim', 'ms_ssim']:
        wan_d  = [r['avg'][metric] for r in wan_results]
        dcvc_d = [r['avg'][metric] for r in dcvc_results]
        bd[f'bd_rate_{metric}']   = _bd_rate_core(dcvc_bpp, dcvc_d, wan_bpp, wan_d)
        bd[f'bd_metric_{metric}'] = _bd_metric_core(dcvc_bpp, dcvc_d, wan_bpp, wan_d)

    # LPIPS: lower is better → negate for BD computation (so "higher = better")
    if 'lpips' in wan_results[0]['avg']:
        wan_lp  = [-r['avg']['lpips'] for r in wan_results]
        dcvc_lp = [-r['avg']['lpips'] for r in dcvc_results]
        bd['bd_rate_lpips']   = _bd_rate_core(dcvc_bpp, dcvc_lp, wan_bpp, wan_lp)
        bd['bd_metric_lpips'] = _bd_metric_core(dcvc_bpp, dcvc_lp, wan_bpp, wan_lp)
        # negate bd_metric back so positive means lower LPIPS (better)
        if bd['bd_metric_lpips'] is not None:
            bd['bd_metric_lpips'] = -bd['bd_metric_lpips']

    return bd


# ═══════════════════════════════════════════════════════════════════════
#  RD Curve Plotting (Publication Quality)
# ═══════════════════════════════════════════════════════════════════════

WAN_STYLE  = dict(color='#E74C3C', marker='o', linestyle='-',  label='WAN (Ours)')
DCVC_STYLE = dict(color='#2980B9', marker='s', linestyle='--', label='DCVC_RT')


def plot_rd_curves(wan_results, dcvc_results, save_dir, seq_name, bd=None):
    """4-subplot RD curves: PSNR, SSIM, MS-SSIM, LPIPS."""
    w_bpp = [r['avg']['bpp'] for r in wan_results]
    d_bpp = [r['avg']['bpp'] for r in dcvc_results]

    has_lpips = 'lpips' in wan_results[0]['avg']
    ncols = 4 if has_lpips else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8))

    # ── PSNR ──
    ax = axes[0]
    ax.plot(w_bpp, [r['avg']['psnr'] for r in wan_results], **WAN_STYLE)
    ax.plot(d_bpp, [r['avg']['psnr'] for r in dcvc_results], **DCVC_STYLE)
    ax.set_xlabel('BPP'); ax.set_ylabel('PSNR (dB)'); ax.set_title('Rate–PSNR')
    if bd and bd.get('bd_rate_psnr') is not None:
        ax.text(0.05, 0.05, f'BD-Rate: {bd["bd_rate_psnr"]:+.1f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    ax.legend(loc='lower right'); ax.grid(True)

    # ── SSIM ──
    ax = axes[1]
    ax.plot(w_bpp, [r['avg']['ssim'] for r in wan_results], **WAN_STYLE)
    ax.plot(d_bpp, [r['avg']['ssim'] for r in dcvc_results], **DCVC_STYLE)
    ax.set_xlabel('BPP'); ax.set_ylabel('SSIM'); ax.set_title('Rate–SSIM')
    if bd and bd.get('bd_rate_ssim') is not None:
        ax.text(0.05, 0.05, f'BD-Rate: {bd["bd_rate_ssim"]:+.1f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    ax.legend(loc='lower right'); ax.grid(True)

    # ── MS-SSIM ──
    ax = axes[2]
    ax.plot(w_bpp, [r['avg']['ms_ssim'] for r in wan_results], **WAN_STYLE)
    ax.plot(d_bpp, [r['avg']['ms_ssim'] for r in dcvc_results], **DCVC_STYLE)
    ax.set_xlabel('BPP'); ax.set_ylabel('MS-SSIM'); ax.set_title('Rate–MS-SSIM')
    if bd and bd.get('bd_rate_ms_ssim') is not None:
        ax.text(0.05, 0.05, f'BD-Rate: {bd["bd_rate_ms_ssim"]:+.1f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    ax.legend(loc='lower right'); ax.grid(True)

    # ── LPIPS ──
    if has_lpips:
        ax = axes[3]
        ax.plot(w_bpp, [r['avg']['lpips'] for r in wan_results], **WAN_STYLE)
        ax.plot(d_bpp, [r['avg']['lpips'] for r in dcvc_results], **DCVC_STYLE)
        ax.set_xlabel('BPP'); ax.set_ylabel('LPIPS ↓'); ax.set_title('Rate–LPIPS')
        if bd and bd.get('bd_rate_lpips') is not None:
            ax.text(0.05, 0.95, f'BD-Rate: {bd["bd_rate_lpips"]:+.1f}%',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
        ax.legend(loc='upper right'); ax.grid(True)

    fig.suptitle(f'Rate–Distortion Curves — {seq_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        p = os.path.join(save_dir, f'rd_{seq_name}.{ext}')
        plt.savefig(p, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [PLOT] rd_{seq_name}.pdf/png")


def plot_averaged_rd(all_wan, all_dcvc, save_dir, all_bd=None):
    """Average RD curves across sequences."""
    # Gather per-QP averages across sequences
    def _avg_by_qp(all_results):
        by_qp = {}
        for seq_res in all_results.values():
            for r in seq_res:
                qp = r['qp']
                if qp not in by_qp:
                    by_qp[qp] = []
                by_qp[qp].append(r['avg'])
        out = []
        for qp in sorted(by_qp):
            mlist = by_qp[qp]
            avg = {}
            for k in mlist[0]:
                avg[k] = float(np.mean([m[k] for m in mlist]))
            out.append({'qp': qp, 'avg': avg})
        return out

    wan_avg  = _avg_by_qp(all_wan)
    dcvc_avg = _avg_by_qp(all_dcvc)

    # Compute BD on averaged curves
    bd_avg = compute_bd_metrics(wan_avg, dcvc_avg) if len(wan_avg) >= 4 and len(dcvc_avg) >= 4 else None
    plot_rd_curves(wan_avg, dcvc_avg, save_dir, 'Average', bd_avg)
    return wan_avg, dcvc_avg, bd_avg


# ═══════════════════════════════════════════════════════════════════════
#  Visual Comparison at Matched BPP
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_visual_comparison(wan_model, dcvc_model, i_frame_net, frames,
                               wan_qp, dcvc_qp, device, save_path,
                               target_frame_idx=8, crop_size=256):
    """
    Generate a publication-quality visual comparison figure.
    Top row: full frame (Original | WAN | DCVC_RT)
    Bottom row: zoomed crop + metrics.
    """
    _, _, H, W = frames[0].shape

    # ── Run WAN ──
    ref = _pad(frames[0].to(device))
    ref_hat = i_frame_net.compress_(ref, min(wan_qp, 63))
    _, _, rH, rW = ref.shape
    if ref_hat.shape[2] != rH or ref_hat.shape[3] != rW:
        ref_hat = ref_hat[:, :, :rH, :rW]
    wan_model.clear_dpb()
    wan_model.add_ref_frame(None, ref_hat)
    for idx in range(1, min(target_frame_idx + 1, len(frames))):
        cur_pad = _pad(frames[idx].to(device))
        wan_result = wan_model(cur_pad, wan_qp)
    wan_recon = wan_result.get('x_hat_rgb', wan_result['x_hat'])[:, :, :H, :W]
    wan_bpp = wan_result['bpp'].mean().item()

    # ── Run DCVC ──
    ref_hat2 = i_frame_net.compress_(ref, min(dcvc_qp, 63))
    if ref_hat2.shape[2] != rH or ref_hat2.shape[3] != rW:
        ref_hat2 = ref_hat2[:, :, :rH, :rW]
    dcvc_model.clear_dpb()
    dcvc_model.add_ref_frame(None, ref_hat2)
    for idx in range(1, min(target_frame_idx + 1, len(frames))):
        cur_pad = _pad(frames[idx].to(device))
        dcvc_result = dcvc_model(cur_pad, dcvc_qp)
    dcvc_recon = dcvc_result.get('x_hat_rgb', dcvc_result['x_hat'])[:, :, :H, :W]
    dcvc_bpp = dcvc_result['bpp'].mean().item()

    orig = frames[min(target_frame_idx, len(frames) - 1)].to(device)

    # Metrics
    w_psnr = _psnr(orig, wan_recon);  d_psnr = _psnr(orig, dcvc_recon)
    w_ssim = _ssim(orig, wan_recon);  d_ssim = _ssim(orig, dcvc_recon)
    w_ms   = _ms_ssim(orig, wan_recon); d_ms = _ms_ssim(orig, dcvc_recon)

    # To numpy (Y channel)
    orig_y = orig[0, 0].cpu().numpy()
    wan_y  = wan_recon[0, 0].cpu().numpy()
    dcvc_y = dcvc_recon[0, 0].cpu().numpy()

    # Crop region (center)
    ch, cw = H // 2 - crop_size // 2, W // 2 - crop_size // 2
    ch = max(0, ch); cw = max(0, cw)
    cs = crop_size

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Full frames
    for ax, img, title in [
        (axes[0, 0], orig_y, 'Original'),
        (axes[0, 1], wan_y,  f'WAN (Ours)\n'
                             f'PSNR={w_psnr:.2f} dB | BPP={wan_bpp:.4f}\n'
                             f'SSIM={w_ssim:.4f} | MS-SSIM={w_ms:.4f}'),
        (axes[0, 2], dcvc_y, f'DCVC_RT\n'
                             f'PSNR={d_psnr:.2f} dB | BPP={dcvc_bpp:.4f}\n'
                             f'SSIM={d_ssim:.4f} | MS-SSIM={d_ms:.4f}'),
    ]:
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        # Draw crop rectangle
        rect = plt.Rectangle((cw, ch), cs, cs, linewidth=2,
                              edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # Row 2: Cropped regions
    for ax, img, label in [
        (axes[1, 0], orig_y[ch:ch+cs, cw:cw+cs], 'Original (crop)'),
        (axes[1, 1], wan_y[ch:ch+cs, cw:cw+cs],  'WAN (crop)'),
        (axes[1, 2], dcvc_y[ch:ch+cs, cw:cw+cs],  'DCVC_RT (crop)'),
    ]:
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')

    fig.suptitle(f'Visual Comparison — Frame {target_frame_idx}  |  '
                 f'WAN QP={wan_qp}, DCVC QP={dcvc_qp}',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(save_path.replace('.png', f'.{ext}'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  [VISUAL] {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Result Export
# ═══════════════════════════════════════════════════════════════════════

def save_csv(results, model_name, path):
    """Save per-QP average metrics as CSV."""
    fieldnames = ['model', 'qp', 'bpp', 'psnr', 'psnr_y', 'mse',
                  'ssim', 'ms_ssim', 'lpips']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {'model': model_name, 'qp': r['qp']}
            row.update({k: f'{v:.6f}' for k, v in r['avg'].items()})
            writer.writerow(row)


def save_json(all_wan, all_dcvc, all_bd, path):
    """Save complete results as JSON."""
    data = {
        'wan': {seq: [{'qp': r['qp'], 'avg': r['avg']} for r in res]
                for seq, res in all_wan.items()},
        'dcvc': {seq: [{'qp': r['qp'], 'avg': r['avg']} for r in res]
                 for seq, res in all_dcvc.items()},
        'bd_rates': all_bd,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [JSON] {path}")


def generate_latex_table(all_bd, path):
    """Generate a LaTeX-ready BD-rate summary table."""
    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \caption{BD-Rate (\%) of WAN (Ours) vs.\ DCVC\_RT. Negative values indicate bitrate savings at equal quality.}',
        r'  \label{tab:bdrate}',
        r'  \begin{tabular}{l|cccc}',
        r'    \toprule',
        r'    Sequence & PSNR & SSIM & MS-SSIM & LPIPS \\',
        r'    \midrule',
    ]

    for seq, bd in all_bd.items():
        if seq == 'Average':
            lines.append(r'    \midrule')
        vals = []
        for m in ['psnr', 'ssim', 'ms_ssim', 'lpips']:
            v = bd.get(f'bd_rate_{m}')
            if v is not None:
                vals.append(f'{v:+.1f}')
            else:
                vals.append('--')
        lines.append(f'    {seq} & {" & ".join(vals)} \\\\')

    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  [LaTeX] {path}")


def generate_summary(all_wan, all_dcvc, all_bd, path):
    """Human-readable summary."""
    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Publication Evaluation Summary: WAN (Ours) vs DCVC_RT\n")
        f.write("=" * 80 + "\n\n")

        for seq_name in list(all_wan.keys()):
            f.write(f"{'─' * 60}\n")
            f.write(f"Sequence: {seq_name}\n")
            f.write(f"{'─' * 60}\n\n")

            for label, results in [('WAN', all_wan[seq_name]), ('DCVC_RT', all_dcvc[seq_name])]:
                f.write(f"  {label}:\n")
                f.write(f"    {'QP':>4s}  {'BPP':>8s}  {'PSNR':>7s}  {'SSIM':>7s}  {'MS-SSIM':>8s}  {'LPIPS':>7s}\n")
                for r in results:
                    a = r['avg']
                    lp = f"{a['lpips']:.4f}" if 'lpips' in a else '  --  '
                    f.write(f"    {r['qp']:4d}  {a['bpp']:8.4f}  {a['psnr']:7.2f}  "
                            f"{a['ssim']:7.4f}  {a['ms_ssim']:8.4f}  {lp:>7s}\n")
                f.write("\n")

            if seq_name in all_bd:
                bd = all_bd[seq_name]
                f.write(f"  BD-Rates (WAN vs DCVC_RT anchor):\n")
                for m in ['psnr', 'ssim', 'ms_ssim', 'lpips']:
                    v = bd.get(f'bd_rate_{m}')
                    vstr = f'{v:+.2f}%' if v is not None else 'N/A (insufficient overlap)'
                    f.write(f"    BD-Rate ({m:>8s}): {vstr}\n")
                f.write("\n")

        # Average BD-rates
        if 'Average' in all_bd:
            f.write(f"\n{'=' * 60}\n")
            f.write("Overall Average BD-Rates\n")
            f.write(f"{'=' * 60}\n")
            bd = all_bd['Average']
            for m in ['psnr', 'ssim', 'ms_ssim', 'lpips']:
                v = bd.get(f'bd_rate_{m}')
                vstr = f'{v:+.2f}%' if v is not None else 'N/A'
                f.write(f"  BD-Rate ({m:>8s}): {vstr}\n")

    print(f"  [TXT] {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Publication-quality evaluation: WAN vs DCVC_RT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (HEVC_E, all 3 sequences)
  python evaluate_paper.py

  # Custom QP range + more frames
  python evaluate_paper.py --wan_qps 59,63,65,67,69,71 --num_frames 49

  # Single sequence, verbose
  python evaluate_paper.py --test_dirs /path/to/seq --verbose
""")
    p.add_argument('--wan_checkpoint', default='pretrained/DMC_WAN/1/checkpoint_stage4_wan.pth.tar')
    p.add_argument('--dcvc_checkpoint', default='pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar')
    p.add_argument('--wan_vae', default='Wan2.2-main/checkpoints/Wan2.2_VAE.pth')
    p.add_argument('--i_frame_model', default='checkpoints/cvpr2025_image.pth.tar')
    p.add_argument('--test_dirs', nargs='+', default=[
        '/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60',
        '/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/FourPeople_1280x720_60',
        '/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/KristenAndSara_1280x720_60',
    ])
    p.add_argument('--wan_qps', default='55,59,63,65,67,69,71',
                   help='WAN QP list (training domain: 63-71, extend to 55+ for RD overlap)')
    p.add_argument('--dcvc_qps', default='0,4,8,12,16,20,25,30,40,50,63',
                   help='DCVC_RT QP list (wide range to cover WAN BPP range)')
    p.add_argument('--num_frames', type=int, default=33,
                   help='Total frames per sequence (1 I-frame + N P-frames)')
    p.add_argument('--output_dir', default='paper_results')
    p.add_argument('--no_lpips', action='store_true')
    p.add_argument('--no_visual', action='store_true', help='Skip visual comparison generation')
    p.add_argument('--visual_frame', type=int, default=8, help='Frame index for visual comparison')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wan_qps  = sorted([int(q) for q in args.wan_qps.split(',')])
    dcvc_qps = sorted([int(q) for q in args.dcvc_qps.split(',')])

    print(f"{'=' * 70}")
    print(f"  WAN QPs:  {wan_qps}")
    print(f"  DCVC QPs: {dcvc_qps}")
    print(f"  Frames:   {args.num_frames}")
    print(f"  Seqs:     {len(args.test_dirs)}")
    print(f"{'=' * 70}\n")

    # ── Load models ──
    print("Loading models...")
    wan_model  = load_wan_model(args.wan_checkpoint, args.wan_vae, device)
    dcvc_model = load_dcvc_model(args.dcvc_checkpoint, device)
    i_frame_net = load_i_frame_model(args.i_frame_model, device)

    lpips_fn = None
    if LPIPS_AVAILABLE and not args.no_lpips:
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
        print("[OK] LPIPS model loaded")

    all_wan  = {}
    all_dcvc = {}
    all_bd   = {}

    for test_dir in args.test_dirs:
        seq_name = os.path.basename(test_dir)
        print(f"\n{'═' * 70}")
        print(f"  Sequence: {seq_name}")
        print(f"{'═' * 70}")

        frames = load_frames_from_dir(test_dir, args.num_frames)
        print(f"  Loaded {len(frames)} frames, shape {list(frames[0].shape)}")

        # ── WAN sweep ──
        print(f"\n  ▶ WAN ({len(wan_qps)} QPs)")
        wan_results = sweep_qps(wan_model, 'WAN', frames, wan_qps,
                                i_frame_net, device, lpips_fn, args.verbose)
        all_wan[seq_name] = wan_results

        # ── DCVC sweep ──
        print(f"\n  ▶ DCVC_RT ({len(dcvc_qps)} QPs)")
        dcvc_results = sweep_qps(dcvc_model, 'DCVC_RT', frames, dcvc_qps,
                                 i_frame_net, device, lpips_fn, args.verbose)
        all_dcvc[seq_name] = dcvc_results

        # ── BD-rate ──
        bd = compute_bd_metrics(wan_results, dcvc_results)
        all_bd[seq_name] = bd
        print(f"\n  BD-Rates (WAN vs DCVC_RT):")
        for m in ['psnr', 'ssim', 'ms_ssim', 'lpips']:
            v = bd.get(f'bd_rate_{m}')
            print(f"    {m:>8s}: {v:+.2f}%" if v is not None else f"    {m:>8s}: N/A")

        # ── Per-sequence plots & CSV ──
        plot_rd_curves(wan_results, dcvc_results, args.output_dir, seq_name, bd)
        save_csv(wan_results, 'WAN', os.path.join(args.output_dir, f'{seq_name}_wan.csv'))
        save_csv(dcvc_results, 'DCVC_RT', os.path.join(args.output_dir, f'{seq_name}_dcvc.csv'))

        # ── Visual comparison ──
        if not args.no_visual and len(frames) > args.visual_frame:
            # Find DCVC QP closest to WAN's mid-QP BPP
            mid_wan = wan_results[len(wan_results) // 2]
            target_bpp = mid_wan['avg']['bpp']
            closest_dcvc = min(dcvc_results, key=lambda r: abs(r['avg']['bpp'] - target_bpp))
            generate_visual_comparison(
                wan_model, dcvc_model, i_frame_net, frames,
                wan_qp=mid_wan['qp'], dcvc_qp=closest_dcvc['qp'],
                device=device,
                save_path=os.path.join(args.output_dir, f'visual_{seq_name}.png'),
                target_frame_idx=args.visual_frame,
            )

        torch.cuda.empty_cache()

    # ── Averaged RD curves ──
    if len(args.test_dirs) > 1:
        print(f"\n{'═' * 70}")
        print("  Computing averaged RD curves...")
        _, _, bd_avg = plot_averaged_rd(all_wan, all_dcvc, args.output_dir)
        if bd_avg:
            all_bd['Average'] = bd_avg

    # ── Export ──
    print(f"\n{'═' * 70}")
    print("  Saving results...")
    save_json(all_wan, all_dcvc, all_bd,
              os.path.join(args.output_dir, 'all_results.json'))
    generate_summary(all_wan, all_dcvc, all_bd,
                     os.path.join(args.output_dir, 'summary.txt'))
    generate_latex_table(all_bd,
                         os.path.join(args.output_dir, 'table_bdrate.tex'))

    print(f"\n{'═' * 70}")
    print(f"  ✅ All results saved to: {args.output_dir}/")
    print(f"{'═' * 70}")


if __name__ == '__main__':
    main()
