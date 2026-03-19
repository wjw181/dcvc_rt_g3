# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Training script for DMC_WAN (DCVC_RT + Wan VAE)
Implements Generative Latent Coding (GLC) for ultra-low bitrate video compression
"""

import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist

from src.models.video_t_g_wan import DMC_WAN
from src.models.image_model import DMCI
from src.dataload import DataSet, TetsDataSet
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def adjust_learning_rate(optimizer, epoch, initial_lr, factors, total_epochs=None, 
                         schedule_type="cosine", warmup_epochs=3):
    """调整学习率 - 支持多种策略
    
    Args:
        schedule_type: "step" (旧版阶梯衰减), "cosine" (余弦退火, 推荐)
        warmup_epochs: warmup 轮数
        total_epochs: 总轮数 (cosine 需要)
    """
    if schedule_type == "cosine":
        # ✨ 余弦退火 + warmup (推荐: 更平滑的衰减, 避免突然跳变)
        if total_epochs is None:
            total_epochs = 60
        if epoch < warmup_epochs:
            lr = initial_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            lr = initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = max(lr, initial_lr * 0.01)  # 最低不低于初始LR的1%
    else:
        # 旧版阶梯衰减
        if epoch < 5:
            warmup_factor = (epoch + 1) / 5.0
            lr = initial_lr * warmup_factor
        else:
            lr = initial_lr
            if epoch >= 15:
                lr *= factors[0]
            if epoch >= 30:
                lr *= factors[1]
            if epoch >= 45:
                lr *= factors[2]
            if epoch >= 65:
                lr *= factors[3]
    
    for param_group in optimizer.param_groups:
        # ✨ 差异化学习率：保持各组的相对比例
        if 'name' in param_group and param_group.get('_base_lr') is not None:
            param_group['lr'] = lr * (param_group['_base_lr'] / initial_lr)
        else:
            param_group['lr'] = lr


class RateDistortionLoss(nn.Module):
    """率失真损失 + 感知损失"""
    def __init__(self, lamada=3600, perceptual_weight=0.0, perceptual_net="vgg", glc_root=None, device=None,
                 ramp_start=0.0, ramp_epochs=10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lamada = lamada
        self.perceptual_weight = perceptual_weight
        self.perceptual_net = perceptual_net
        self.glc_root = glc_root
        self.device = device
        self.ramp_start = ramp_start  # lambda ramp 起始比例 (0.0=从0.1开始, 1.0=无ramp)
        self.ramp_epochs = ramp_epochs  # ramp 持续的 epoch 数
        self.lpips_model = None
    
    def _init_lpips(self, device):
        if not self.glc_root:
            raise RuntimeError("GLC root path is required for perceptual loss.")
        glc_src = os.path.join(self.glc_root, "src")
        if glc_src not in sys.path:
            sys.path.insert(0, glc_src)
        from utils.lpips import LPIPS
        lpips = LPIPS(net=self.perceptual_net, version="0.1")
        lpips = lpips.to(device)
        lpips.eval()
        for p in lpips.parameters():
            p.requires_grad = False
        return lpips
    
    def forward(self, epoch, result, target, lamada):
        """
        Args:
            epoch: 当前epoch
            result: 模型输出字典
            target: RGB目标 [B, 3, H, W]
            lamada: lambda参数
        """
        # 如果模型已经提供了MSE（在latent空间计算，有梯度），直接使用
        if 'mse' in result and result['mse'].requires_grad:
            mse_loss = result['mse']
        else:
            # 否则重新计算MSE
            x_hat = result['x_hat']  # YUV format
            x_hat_rgb = result.get('x_hat_rgb', None)  # RGB format if available
            
            # 计算MSE
            if x_hat_rgb is not None and x_hat_rgb.requires_grad:
                mse_loss = self.mse(x_hat_rgb, target)
            else:
                from src.models.wan_vae_wrapper import WanVAEWrapper
                wrapper = WanVAEWrapper()
                target_yuv = wrapper.rgb_to_yuv444(target)
                mse_loss = self.mse(x_hat, target_yuv)
        
        # ✨ 新增：RGB域直接MSE（直接优化PSNR）
        rgb_mse_loss = torch.tensor(0.0, device=target.device)
        x_hat_rgb = result.get('x_hat_rgb', None)
        if x_hat_rgb is not None and x_hat_rgb.requires_grad:
            rgb_mse_loss = self.mse(x_hat_rgb, target)
        
        # BPP损失
        bpp_loss = result['bpp'].mean()
        
        # 感知损失 (LPIPS from GLC)
        perceptual_loss = torch.zeros_like(mse_loss)
        if self.perceptual_weight > 0:
            x_hat_rgb = result.get('x_hat_rgb', None)
            if x_hat_rgb is not None:
                device = x_hat_rgb.device
                if self.lpips_model is None or next(self.lpips_model.parameters()).device != device:
                    self.lpips_model = self._init_lpips(device)
                # LPIPS expects inputs in [-1, 1]
                pred = torch.clamp(x_hat_rgb, 0.0, 1.0) * 2.0 - 1.0
                tgt = torch.clamp(target, 0.0, 1.0) * 2.0 - 1.0
                perceptual_loss = self.lpips_model(tgt.contiguous(), pred.contiguous()).reshape(-1)
            else:
                if dist.is_initialized() and dist.get_rank() == 0:
                    logging.warning("Perceptual loss enabled but x_hat_rgb is missing; skipping LPIPS.")
        
        # ✨ 综合损失（改进版）：
        # 组合 latent MSE + RGB MSE，让损失与PSNR更直接关联
        # rgb_weight 控制 RGB 域损失的权重（推荐 0.3-0.5）
        rgb_weight = getattr(self, 'rgb_weight', 0.3)
        combined_distortion = (1.0 - rgb_weight) * mse_loss + rgb_weight * rgb_mse_loss
        
        # Lambda ramp (逐步增加lambda，之后保持)
        # ramp_start=1.0 时完全禁用ramp（推荐fine-tuning时使用）
        # ramp_start=0.0 时从 1/ramp_epochs 开始线性增到 1.0
        ramp_epochs = self.ramp_epochs
        ramp_start = self.ramp_start
        if ramp_start >= 1.0 or epoch >= ramp_epochs:
            # 无 ramp，直接使用完整 lambda
            ramp = 1.0
        else:
            # 线性从 ramp_start 增长到 1.0
            ramp = ramp_start + (1.0 - ramp_start) * float(epoch + 1) / float(ramp_epochs)
            ramp = min(ramp, 1.0)
        
        loss = (lamada * ramp) * combined_distortion + bpp_loss + self.perceptual_weight * perceptual_loss
        
        return {
            'loss': loss,
            'mse_loss': mse_loss,
            'bpp_loss': bpp_loss,
            'perceptual_loss': perceptual_loss,
            'rgb_mse_loss': rgb_mse_loss,
        }


class AverageMeter:
    """计算平均值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init(args):
    """初始化目录"""
    base_dir = f'./pretrained/{args.model}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_dir):
    """设置日志"""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)
    
    logging.info('Logging file is %s' % log_dir)


def calculate_psnr(x, x_hat, max_val=1.0):
    """计算PSNR"""
    with torch.no_grad():
        mse = F.mse_loss(x.detach(), x_hat.detach(), reduction='mean')
        if mse == 0:
            return float('inf')
        psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.item()


def get_state_dict(ckpt_path):
    """加载checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


def qp_to_lambda(qp, q_num=72, lam_min=1, lam_max=768):
    """QP映射到lambda"""
    scale = qp / (q_num - 1)
    ln_lam_min = math.log(lam_min)
    ln_lam_max = math.log(lam_max)
    ln_lambda = ln_lam_min + scale * (ln_lam_max - ln_lam_min)
    return math.exp(ln_lambda)


def train_one_epoch(epoch, model, i_frame_net, criterion, train_dataloader, 
                    optimizer, gpu_per_batch, clip_max_norm, mode="pixel", scaler=None):
    """训练一个epoch"""
    model.train()
    device = next(model.parameters()).device
    i_frame_net = i_frame_net.to(device)
    
    # 清理显存，为新的epoch做准备
    torch.cuda.empty_cache()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    psnr_meter = AverageMeter()
    perceptual_meter = AverageMeter()
    # ⭐ 新增：latent mode 专用的监控指标
    latent_mse_meter = AverageMeter()
    pseudo_yuv_mse_meter = AverageMeter()
    
    use_amp = scaler is not None
    
    for i, d in enumerate(train_dataloader):
        ref, input_images = d[0].cuda(non_blocking=True), d[1].cuda(non_blocking=True)
        input_images = list(input_images.split(3, dim=1))
        
        optimizer.zero_grad()
        
        # 随机选择QP
        qs_global = random.randint(63, 71) if epoch >= 48 else 71
        lamada_qs = qp_to_lambda(qs_global)
        
        # 压缩参考帧 (使用I帧编码器)
        with torch.no_grad():
            ref_compressed = i_frame_net.compress_(ref, min(qs_global, 63))
        
        # 清空缓冲区并添加参考帧
        model.module.clear_dpb()
        model.module.add_ref_frame(None, ref_compressed)
        
        total_loss = None
        psnr_list = []
        bpp_list = []
        perceptual_list = []
        latent_mse_list = []
        pseudo_yuv_mse_list = []
        
        # 遍历视频帧
        for idx, current_rgb in enumerate(input_images, 1):
            # 模型前向传播 (使用混合精度)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_net = model(current_rgb, qs_global)
                
                # 计算损失
                out_criterion = criterion(epoch, out_net, current_rgb, lamada_qs)
                loss_i = out_criterion["loss"]
                # Ensure loss_i is scalar by taking mean if needed
                if loss_i.dim() > 0:
                    loss_i = loss_i.mean()
                total_loss = loss_i if total_loss is None else total_loss + loss_i
            
            # 记录指标 (使用detach并立即转为标量以节省显存)
            with torch.no_grad():
                psnr_val = calculate_psnr(current_rgb, out_net.get('x_hat_rgb', out_net['x_hat']))
                psnr_list.append(psnr_val)
                bpp_list.append(out_net["bpp"].mean().item())
                if "perceptual_loss" in out_criterion:
                    p_loss = out_criterion["perceptual_loss"]
                    if p_loss.dim() > 0:
                        p_loss = p_loss.mean()
                    perceptual_list.append(p_loss.item())
                # ⭐ 记录 latent mode 专用指标
                if "latent_mse" in out_net:
                    latent_mse_list.append(out_net["latent_mse"].item())
                if "pseudo_yuv_mse" in out_net:
                    pseudo_yuv_mse_list.append(out_net["pseudo_yuv_mse"].item())
        
        # 反向传播 (使用GradScaler if AMP enabled)
        if use_amp:
            scaler.scale(total_loss).backward()
            if clip_max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
        
        # 更新统计 (所有值已经是标量)
        loss_val = total_loss.item()
        bpp_val = sum(bpp_list) / len(bpp_list)
        psnr_val = sum(psnr_list) / len(psnr_list)
        perceptual_val = sum(perceptual_list) / len(perceptual_list) if perceptual_list else 0
        
        loss_meter.update(loss_val)
        bpp_meter.update(bpp_val)
        psnr_meter.update(psnr_val)
        if perceptual_list:
            perceptual_meter.update(perceptual_val)
        # ⭐ 更新 latent mode 专用指标
        if latent_mse_list:
            latent_mse_meter.update(sum(latent_mse_list) / len(latent_mse_list))
        if pseudo_yuv_mse_list:
            pseudo_yuv_mse_meter.update(sum(pseudo_yuv_mse_list) / len(pseudo_yuv_mse_list))
        
        # 释放中间变量，防止显存碎片化
        del total_loss, out_net, out_criterion, psnr_list, bpp_list, perceptual_list
        del latent_mse_list, pseudo_yuv_mse_list
        del ref, input_images, current_rgb
        
        # 强制同步CUDA操作以释放显存
        if i % 50 == 0:
            torch.cuda.synchronize()
        
        # 更频繁地清理显存碎片 (每100步而不是1000步)
        if i % 100 == 0:
            torch.cuda.empty_cache()
        
        # 打印日志
        if i % 500 == 0 and dist.get_rank() == 0:
            extra_str = ""
            if perceptual_meter.count > 0:
                extra_str += f' | LPIPS: {perceptual_meter.avg:.4f}'
            # ⭐ 打印 latent mode 专用指标
            if latent_mse_meter.count > 0:
                extra_str += f' | LatentMSE: {latent_mse_meter.avg:.4f}'
            if pseudo_yuv_mse_meter.count > 0:
                extra_str += f' | PseudoYUV_MSE: {pseudo_yuv_mse_meter.avg:.4f}'
            logging.info(
                f'[{i}/{len(train_dataloader)}] | '
                f'Loss: {loss_meter.avg:.3f} | '
                f'PSNR: {psnr_meter.avg:.3f} | '
                f'BPP: {bpp_meter.avg:.4f}'
                f'{extra_str}'
            )


def _pad_to_multiple(x, multiple=16):
    """将张量的空间维度 pad 到 multiple 的倍数，避免 DCVC 内部下采样时奇偶尺寸不匹配"""
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')


def test_epoch(epoch, i_frame_net, test_dataloader, model, criterion, test_num, mode="pixel"):
    """测试一个epoch（带异常保护，确保单个样本错误不会崩溃整个测试）"""
    model.eval()
    device = next(model.parameters()).device
    i_frame_net = i_frame_net.to(device)
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    psnr_meter = AverageMeter()
    
    skipped = 0  # 跳过的样本数
    
    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            try:
                ref, input_images = d[0].cuda(non_blocking=True), d[1].cuda(non_blocking=True)
                
                qs_global = 71   # 与训练 QP 一致，评测模型在训练目标上的实际性能
                lamada = qp_to_lambda(qs_global)
                
                # 记录原始空间尺寸（用于后续裁剪）
                _, _, H_orig, W_orig = ref.shape
                
                # Padding 到 16 的倍数，避免 I 帧重建与 P 帧特征尺寸不匹配
                ref_padded = _pad_to_multiple(ref, 16)
                
                # 压缩参考帧（在 padded 的参考帧上）
                ref_compressed = i_frame_net.compress_(ref_padded, min(qs_global, 63))
                
                # I帧重建可能进一步改变尺寸，裁剪回 padded 尺寸
                _, _, H_pad, W_pad = ref_padded.shape
                if ref_compressed.shape[2] != H_pad or ref_compressed.shape[3] != W_pad:
                    ref_compressed = ref_compressed[:, :, :H_pad, :W_pad]
                
                # 清空缓冲区
                model.module.clear_dpb()
                model.module.add_ref_frame(None, ref_compressed)
                
                total_loss = 0.0
                total_bpp = 0.0
                total_psnr = 0.0
                
                # 确保 test_num 不超过可用帧数
                available_frames = input_images.shape[1] - 1  # 减去参考帧
                actual_test_num = min(test_num, available_frames)
                if actual_test_num <= 0:
                    skipped += 1
                    continue
                
                # 处理每一帧
                for j in range(1, actual_test_num + 1):
                    current_rgb = input_images[:, j, :, :, :]
                    # Padding 当前帧到与参考帧相同的 padded 尺寸
                    current_rgb_padded = _pad_to_multiple(current_rgb, 16)
                    
                    # 前向传播（使用 padded 帧）
                    out_net = model(current_rgb_padded, qs_global)
                    
                    # 裁剪输出回原始尺寸，确保指标计算正确
                    if 'x_hat' in out_net:
                        out_net['x_hat'] = out_net['x_hat'][:, :, :H_orig, :W_orig]
                    if 'x_hat_rgb' in out_net:
                        out_net['x_hat_rgb'] = out_net['x_hat_rgb'][:, :, :H_orig, :W_orig]
                    
                    out_criterion = criterion(epoch, out_net, current_rgb, lamada)
                    
                    # 累积指标
                    loss_val = out_criterion["loss"]
                    if loss_val.dim() > 0:
                        loss_val = loss_val.mean()
                    total_loss += loss_val.item()
                    total_bpp += out_net["bpp"].mean().item()
                    total_psnr += calculate_psnr(current_rgb, out_net.get('x_hat_rgb', out_net['x_hat']))
                    
                    # 释放中间变量
                    del out_net, out_criterion, current_rgb, current_rgb_padded
                
                # 计算平均值
                loss_meter.update(total_loss / actual_test_num)
                bpp_meter.update(total_bpp / actual_test_num)
                psnr_meter.update(total_psnr / actual_test_num)
                
                # 释放batch数据
                del ref, input_images, ref_padded, ref_compressed
                
            except Exception as e:
                skipped += 1
                if dist.get_rank() == 0:
                    logging.warning(f"Test sample {i} failed: {e}")
                # 确保释放显存
                torch.cuda.empty_cache()
                continue
            
            # 定期清理显存
            if i % 10 == 0:
                torch.cuda.empty_cache()
    
    if dist.get_rank() == 0:
        log_msg = (
            f"Test epoch {epoch}: "
            f"Loss: {loss_meter.avg:.3f} | "
            f"PSNR: {psnr_meter.avg:.3f} | "
            f"BPP: {bpp_meter.avg:.4f}"
        )
        if skipped > 0:
            log_msg += f" | Skipped: {skipped}"
        logging.info(log_msg + "\n")
    
    return loss_meter.avg


def save_checkpoint(state, is_best, base_dir, stage=1, filename=None):
    """保存checkpoint，按阶段区分文件名
    
    Args:
        state: checkpoint内容
        is_best: 是否是最佳模型
        base_dir: 保存目录
        stage: 训练阶段 (1-5)
        filename: 自定义文件名（可选）
    
    文件命名规则:
        - Stage 1: checkpoint_stage1_wan.pth.tar, checkpoint_best_stage1_wan.pth.tar
        - Stage 2: checkpoint_stage2_wan.pth.tar, checkpoint_best_stage2_wan.pth.tar
        - ...以此类推
    """
    if filename is None:
        filename = f"checkpoint_stage{stage}_wan.pth.tar"
    
    torch.save(state, os.path.join(base_dir, filename))
    if is_best:
        best_filename = f"checkpoint_best_stage{stage}_wan.pth.tar"
        shutil.copyfile(
            os.path.join(base_dir, filename),
            os.path.join(base_dir, best_filename)
        )


def parse_args(argv):
    """解析参数"""
    parser = argparse.ArgumentParser(description="Training DMC_WAN with Generative Latent Coding")
    
    # Model parameters
    parser.add_argument("-m", "--model", default="DMC_WAN", help="Model name")
    parser.add_argument("--mode", default="pixel", choices=["pixel", "latent"], 
                       help="Compression mode: pixel or latent space")
    
    # DCVC parameters
    parser.add_argument("--pretrained_dcvc", type=str, 
                       default="pretrained/DMC_SLF/checkpoint_best_loss_vd.pth.tar",
                       help="Pretrained DCVC checkpoint")
    parser.add_argument("--freeze_dcvc", action="store_true", default=False,
                       help="Freeze DCVC parameters (use in latent mode only)")
    
    # Wan VAE parameters
    parser.add_argument("--wan_vae_checkpoint", type=str, default=None,
                       help="Wan VAE checkpoint path (only needed for latent mode)")
    parser.add_argument("--freeze_vae", action="store_true", default=False,
                       help="Freeze VAE parameters (recommended: True)")
    parser.add_argument("--freeze_vae_encoder_only", action="store_true", default=False,
                       help="Freeze only VAE encoder, train decoder (for stage 3)")
    
    # Perceptual loss (GLC LPIPS)
    parser.add_argument("--perceptual_weight", type=float, default=0.0,
                       help="Weight for LPIPS perceptual loss (GLC implementation)")
    parser.add_argument("--perceptual_net", type=str, default="vgg",
                       choices=["vgg", "alex", "squeeze"],
                       help="LPIPS trunk network")
    parser.add_argument("--glc_root", type=str, default="/home/serverdn/hdd-0/wjw/GLC",
                       help="Path to GLC repo (for LPIPS implementation)")
    
    # I-frame model
    parser.add_argument("--model_path_i", type=str, 
                       default="checkpoints/cvpr2025_image.pth.tar",
                       help="I-frame model checkpoint")
    
    # Training parameters
    parser.add_argument("-e", "--epochs", default=120, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-5, type=float)
    parser.add_argument("--clip_max_norm", default=0.5, type=float)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("-n", "--num-workers", type=int, default=4)
    parser.add_argument("-q", "--quality-level", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true", default=False,
                       help="Use automatic mixed precision (AMP) to reduce memory")
    
    # Dataset
    parser.add_argument("-td", "--test_dataset", type=str,
                       default="/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/")
    parser.add_argument("-td_l", "--test_filelist", type=str,
                       default="/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/test.txt")
    
    # Distributed training
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', 
                       default=-1, type=int)
    
    # Other
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Resume training from checkpoint (optional)")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help="Training stage (1-5), used for checkpoint naming")
    
    # ✨ 新增：收敛加速参数
    parser.add_argument("--lr_schedule", type=str, default="step", choices=["step", "cosine"],
                       help="LR schedule type: step (legacy) or cosine (recommended)")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                       help="Number of warmup epochs (for cosine schedule)")
    parser.add_argument("--reset_epoch", action="store_true", default=False,
                       help="Reset epoch counter to 0 (useful when restarting training with new schedule)")
    parser.add_argument("--rgb_weight", type=float, default=0.0,
                       help="Weight for RGB-domain MSE loss (0.0 = off, 0.3 recommended for convergence)")
    parser.add_argument("--ramp_start", type=float, default=0.0,
                       help="Lambda ramp start ratio (0.0=gradual from 0.1, 1.0=no ramp/full lambda from start). "
                            "Use 1.0 for fine-tuning from trained checkpoint!")
    parser.add_argument("--ramp_epochs", type=int, default=10,
                       help="Number of epochs for lambda ramp (default 10)")
    parser.add_argument("--dcvc_lr_scale", type=float, default=1.0,
                       help="LR scale for DCVC params relative to base LR (e.g. 0.1 = 10x lower)")
    parser.add_argument("--adaptation_lr_scale", type=float, default=1.0,
                       help="LR scale for Adaptation layers relative to base LR (e.g. 5.0 = 5x higher)")
    
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    
    # torchrun sets LOCAL_RANK env instead of --local_rank
    if args.local_rank == -1:
        env_rank = os.environ.get("LOCAL_RANK")
        if env_rank is not None:
            args.local_rank = int(env_rank)
    
    # Pixel mode warmup should always train DCVC_RT
    if args.mode == "pixel" and args.freeze_dcvc:
        logging.warning("Pixel mode requires trainable DCVC_RT. Overriding --freeze_dcvc to False.")
        args.freeze_dcvc = False
    
    # Pixel mode does NOT use VAE - force it to None
    if args.mode == "pixel":
        if args.wan_vae_checkpoint is not None:
            logging.warning("⚠️  Pixel mode does NOT require WAN VAE! Ignoring --wan_vae_checkpoint.")
        args.wan_vae_checkpoint = None
        args.freeze_vae = True
        # Pixel mode is memory-intensive; reduce batch size if too high
        if args.batch_size > 1:
            if dist.is_initialized() and dist.get_rank() == 0:
                logging.warning(f"Pixel mode is memory-intensive. Consider using batch_size=1 instead of {args.batch_size}")
            elif not dist.is_initialized():
                logging.warning(f"Pixel mode is memory-intensive. Consider using batch_size=1 instead of {args.batch_size}")
    
    # Latent mode REQUIRES VAE checkpoint
    if args.mode == "latent" and args.wan_vae_checkpoint is None:
        raise ValueError("❌ Latent mode requires --wan_vae_checkpoint! Please provide Wan VAE path.")
    
    # 初始化分布式训练
    dist.init_process_group(backend='nccl')
    if args.local_rank < 0:
        raise RuntimeError("local_rank is not set. Use torchrun or pass --local_rank.")
    torch.cuda.set_device(args.local_rank)
    
    base_dir = init(args)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    # 设置日志
    if dist.get_rank() == 0:
        setup_logger(os.path.join(base_dir, time.strftime('%Y%m%d_%H%M%S') + '_wan.log'))
        logging.info(f'======================= {args.name} =======================')
        logging.info(f'Mode: {args.mode}')
        for k, v in args.__dict__.items():
            logging.info(f'{k}: {v}')
        logging.info('=' * 40)
    
    # 加载I帧模型
    device = "cuda"
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net.eval()
    
    # 初始化DMC_WAN模型
    if dist.get_rank() == 0:
        logging.info("Initializing DMC_WAN model...")
    
    model = DMC_WAN(
        dcvc_checkpoint=args.pretrained_dcvc,
        freeze_dcvc=args.freeze_dcvc,
        wan_vae_checkpoint=args.wan_vae_checkpoint,
        freeze_vae=args.freeze_vae,
        freeze_vae_encoder_only=args.freeze_vae_encoder_only,
        mode=args.mode,
    )
    
    model = model.to(torch.device("cuda", args.local_rank))
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[args.local_rank],
        find_unused_parameters=True
    )
    
    if dist.get_rank() == 0:
        model.module.count_parameters()
    
    # 设置优化器 (支持差异化学习率)
    param_dict = model.module.get_trainable_parameters()
    
    # ✨ 差异化学习率: adaptation 用高LR, DCVC 用低LR, VAE 用更低LR
    param_groups = []
    for key, params in param_dict.items():
        if params:
            num_params = sum(p.numel() for p in params)
            if key == "adaptation":
                lr_scale = args.adaptation_lr_scale
            elif key == "dcvc":
                lr_scale = args.dcvc_lr_scale
            elif key == "vae":
                lr_scale = 0.1  # VAE 始终用很低的LR
            else:
                lr_scale = 1.0
            
            group_lr = args.learning_rate * lr_scale
            param_groups.append({
                'params': params,
                'lr': group_lr,
                'name': key,
            })
            if dist.get_rank() == 0:
                logging.info(f"Training {key} parameters: {num_params:,} | LR: {group_lr:.2e} (scale={lr_scale})")
    
    if not param_groups:
        if dist.get_rank() == 0:
            logging.warning("No trainable parameters! Check freeze settings.")
        param_groups = [{'params': list(model.parameters()), 'lr': args.learning_rate}]
    
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate)
    # 保存每组的base_lr用于差异化衰减
    for pg in optimizer.param_groups:
        pg['_base_lr'] = pg['lr']
    
    criterion = RateDistortionLoss(
        perceptual_weight=args.perceptual_weight,
        perceptual_net=args.perceptual_net,
        glc_root=args.glc_root,
        device=device,
        ramp_start=args.ramp_start,
        ramp_epochs=args.ramp_epochs,
    )
    # ✨ 设置 RGB 域损失权重
    criterion.rgb_weight = args.rgb_weight
    if dist.get_rank() == 0 and args.rgb_weight > 0:
        logging.info(f"RGB domain MSE weight: {args.rgb_weight}")
    if dist.get_rank() == 0:
        if args.ramp_start >= 1.0:
            logging.info(f"Lambda ramp: DISABLED (full lambda from epoch 0)")
        else:
            logging.info(f"Lambda ramp: start={args.ramp_start:.1f}, epochs={args.ramp_epochs}")
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    if dist.get_rank() == 0 and args.use_amp:
        logging.info("Using Automatic Mixed Precision (AMP) training")
    
    # 准备数据集
    train_dataset = DataSet()
    train_dataset.set_frame_count(3)  # 初始3帧训练
    
    test_transforms = TetsDataSet(
        root=args.test_dataset,
        filelist=args.test_filelist,
        gop=32,
        testfull=True
    )
    
    # 加载checkpoint
    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint:
        if dist.get_rank() == 0:
            logging.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if args.reset_epoch:
            last_epoch = 0
            if dist.get_rank() == 0:
                logging.info("⚡ Epoch counter reset to 0 (--reset_epoch)")
        else:
            last_epoch = checkpoint["epoch"] + 1
        
        # 检查checkpoint的模式
        ckpt_mode = checkpoint.get("mode", "unknown")
        if dist.get_rank() == 0:
            logging.info(f"Checkpoint mode: {ckpt_mode}, Current mode: {args.mode}")
        
        # 如果模式不匹配，使用strict=False只加载兼容的部分
        if ckpt_mode != args.mode:
            if dist.get_rank() == 0:
                logging.warning(f"⚠️  Mode mismatch! Checkpoint is {ckpt_mode} but training in {args.mode} mode.")
                logging.warning("Loading compatible parameters only (strict=False)...")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
            if dist.get_rank() == 0:
                if missing_keys:
                    logging.info(f"Missing keys (will be randomly initialized): {len(missing_keys)} keys")
                    if len(missing_keys) <= 10:
                        for key in missing_keys:
                            logging.info(f"  - {key}")
                if unexpected_keys:
                    logging.info(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
        else:
            # 模式匹配，正常加载
            model.load_state_dict(checkpoint["state_dict"])
            if dist.get_rank() == 0:
                logging.info("✅ Checkpoint loaded successfully (strict mode)")
        
        best_loss = checkpoint.get("loss", float("inf"))
    
    # 训练循环
    factors = [0.4, 0.1, 0.04, 0.01]
    
    for epoch in range(last_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.learning_rate, factors, 
                           total_epochs=args.epochs, schedule_type=args.lr_schedule,
                           warmup_epochs=args.warmup_epochs)
        
        if dist.get_rank() == 0:
            logging.info(f"\n====== Epoch {epoch} ======")
            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            # 显示当前 lambda ramp 比例
            ramp_start = args.ramp_start
            ramp_epochs = args.ramp_epochs
            if ramp_start >= 1.0 or epoch >= ramp_epochs:
                cur_ramp = 1.0
            else:
                cur_ramp = ramp_start + (1.0 - ramp_start) * float(epoch + 1) / float(ramp_epochs)
                cur_ramp = min(cur_ramp, 1.0)
            logging.info(f"Mode: {args.mode} | Lambda ramp: {cur_ramp:.2f}")
        
        # 动态调整帧数
        num_frames = 3 if epoch < 50 else min(7, 3 + (epoch - 50) // 10)
        # Pixel mode is memory-heavy; cap frames to reduce OOM risk
        if args.mode == "pixel":
            num_frames = min(num_frames, 2)
        train_dataset.set_frame_count(num_frames)
        test_num = num_frames - 1
        
        if dist.get_rank() == 0:
            logging.info(f"Training with {num_frames} frames")
        
        # 创建DataLoader
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_sampler
        )
        
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_transforms)
        test_dataloader = DataLoader(
            test_transforms,
            batch_size=1,
            num_workers=args.num_workers,
            sampler=test_sampler
        )
        
        # 训练和测试
        train_one_epoch(epoch, model, i_frame_net, criterion, train_dataloader,
                       optimizer, args.batch_size, args.clip_max_norm, args.mode, scaler)
        
        loss = test_epoch(epoch, i_frame_net, test_dataloader, model, criterion,
                         test_num, args.mode)
        
        # 保存checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        
        if args.save and dist.get_rank() == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "mode": args.mode,
                    "stage": args.stage,  # 记录训练阶段
                },
                is_best,
                base_dir,
                stage=args.stage
            )


if __name__ == "__main__":
    main(sys.argv[1:])
