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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.video_t import DMC
from src.models.image_model import DMCI
from compressai.datasets import ImageFolder  
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np

from src.dataload import DataSet,TetsDataSet
import torch.distributed as dist


def adjust_learning_rate(optimizer, epoch, initial_lr, factors):
    """
    改进的学习率调整策略 - 更保守的衰减
    """
    lr = initial_lr

    # 更温和的学习率衰减策略
    if epoch >= 175:
        lr *= factors[3]  # 0.01
    elif epoch >= 170:
        lr *= factors[2]  # 0.04
    elif epoch >= 160:
        lr *= factors[1]  # 0.1
    elif epoch >= 100:
        lr *= factors[0]  # 0.4
    
    # 对于早期训练，使用更小的学习率防止过拟合
    if epoch < 20:
        lr *= 0.5  # 前20个epoch使用一半学习率

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class RateDistortionLoss(nn.Module):
    """改进的率失真损失函数 - 添加正则化"""
    def __init__(self, lamada=3600):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, epoch, result, target, lamada):
        N, _, H, W = target.size()
        out = {}
        
        # BPP loss
        out["bpp_loss"] = result["bpp"]
        
        # MSE loss
        out["mse_loss"] = result["mse"]
        
        # 改进的损失权重策略
        if 0 <= epoch < 10:
            # 前10个epoch专注于重建质量
            out["loss"] = out["mse_loss"] * 100000
        elif 10 <= epoch < 30:
            # 逐渐引入码率约束
            alpha = (epoch - 10) / 20.0  # 从0到1线性增长
            out["loss"] = out["mse_loss"] * 100000 * (1 - alpha) + \
                         (lamada * out["mse_loss"] + out["bpp_loss"]) * alpha
        else:
            # 正常的率失真优化
            out["loss"] = lamada * out["mse_loss"] + out["bpp_loss"]
       
        return out


class AverageMeter:
    """计算运行过程中的平均值"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class CustomDataParallel(nn.DataParallel):
    """自定义 DataParallel 以便访问模型内的方法"""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'./pretrained/{args.model}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_dir):
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


def Var(x):
    return Variable(x.cuda())


def calculate_psnr(x, x_hat, max_val=1.0):
    mse = F.mse_loss(x, x_hat, reduction='mean')
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr


def psnr(x, x_hat, max_val=1.0):
    y_hat_420, uv_hat_420 = yuv_444_to_420(x_hat)
    y_420, uv_420 = yuv_444_to_420(x)
    u_420 = uv_420[:, 0:1, :, :]
    v_420 = uv_420[:, 1:2, :, :]
    u_hat_420 = uv_hat_420[:, 0:1, :, :]
    v_hat_420 = uv_hat_420[:, 1:2, :, :]
     
    psnr_y = calculate_psnr(y_420, y_hat_420, max_val)
    psnr_u = calculate_psnr(u_420, u_hat_420, max_val)
    psnr_v = calculate_psnr(v_420, v_hat_420, max_val)
    psnr = (6 * psnr_y + psnr_u + psnr_v) / 8.0
    return psnr


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


def get_sync_random_value(epoch, i):
    # 由主进程生成qs_global
    if epoch < 48:
        qs_global = 71
    else:
        qs_global = random.randint(63, 79)
    return qs_global


def sync_random_value(value):
    """同步随机值到所有进程"""
    if dist.is_initialized():
        value_tensor = torch.tensor(value, dtype=torch.int32).cuda()
        dist.broadcast(value_tensor, src=0)
        return value_tensor.item()
    return value


def qp_to_lambda(qp):
    """QP转Lambda"""
    return 0.85 * (2 ** ((qp - 12) / 3.0))


def train_one_epoch(epoch, model, i_frame_net, criterion, train_dataloader, optimizer, gpu_per_batch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    
    # 权重调整
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    for i, d in enumerate(train_dataloader):
        ref, input_images = Var(d[0]), Var(d[1])
        ref = ref.cuda(non_blocking=True)
        input_images = input_images.cuda(non_blocking=True)
        input_images = list(input_images.split(3, dim=1))
        
        optimizer.zero_grad()

        # 获取QP值
        qs_global = get_sync_random_value(epoch, i)
        qs_global = sync_random_value(qs_global)
        lamada_qs = qp_to_lambda(qs_global)

        idx = 1
        ssim_list = []
        psnr_list = []
        bpp_list = []
        total_loss = 0.0 
        
        # I帧压缩
        if qs_global > 63:
            ref = i_frame_net.compress_(ref, 63)
        else:
            ref = i_frame_net.compress_(ref, qs_global)
   
        model.module.clear_dpb()
        model.module.add_ref_frame(None, ref)

        for input_image in input_images:
            current = input_image
            out_net = model(current, qs_global)

            if idx == 1:
                lamada = 1.0 * lamada_qs
            else:
                lamada = lamada_qs
                
            ssim_list.append(out_net["ssim"])
            psnr_list.append(psnr(current, out_net["x_hat"]))
            bpp_list.append(out_net["bpp"])

            out_criterion = criterion(epoch, out_net, current, lamada * weights[idx % 8])
            loss_i = out_criterion["loss"].mean()
            total_loss += loss_i
            idx += 1

        total_loss.backward()
        
        # 梯度裁剪 - 使用更小的值防止梯度爆炸
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        
        optimizer.step()
        
        if i % 500 == 0:
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_bpp = sum(bpp_list) / len(bpp_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            if dist.get_rank() == 0:
                logging.info(
                    f'[{i}/{len(train_dataloader.dataset)//gpu_per_batch//4}] | '
                    f'Multi-frame Loss: {total_loss.item():.3f} | '
                    f'PSNR: {avg_psnr:.3f} | '
                    f'SSIM: {avg_ssim.mean():.3f} | '
                    f'BPP: {avg_bpp.mean():.3f}'
                )


def test_epoch(epoch, i_frame_net, test_dataloaders, model, criterion, test_num):
    """
    改进的测试函数 - 支持多个测试集
    
    Args:
        test_dataloaders: 字典，包含多个测试集 {'name': dataloader}
    """
    model.eval()
    device = next(model.parameters()).device
    i_frame_net = i_frame_net.to(device)
    
    all_results = {}
    
    for test_name, test_dataloader in test_dataloaders.items():
        loss_meter = AverageMeter()
        bpp_meter = AverageMeter()
        mse_meter = AverageMeter()
        psnr_meter = AverageMeter()

        with torch.no_grad():
            for i, d in enumerate(test_dataloader):
                ref, input_images = Var(d[0]), Var(d[1])
                ref = ref.cuda(non_blocking=True)
                input_images = input_images.cuda(non_blocking=True)
                
                lamada = 768
                qs_global = 71
                
                total_loss = 0.0
                total_bpp = 0.0
                total_psnr = 0.0
                
                if qs_global > 63:
                    ref = i_frame_net.compress_(ref, 63)
                else:
                    ref = i_frame_net.compress_(ref, qs_global)
                
                model.module.clear_dpb()
                model.module.add_ref_frame(None, ref)
                
                for j in range(1, test_num + 1):
                    current = input_images[:, j, :, :, :]
                    out_net = model(current, qs_global)
                    out_criterion = criterion(epoch, out_net, current, lamada)

                    total_loss += out_criterion["loss"].mean()
                    total_bpp += out_net["bpp"].mean()
                    total_psnr += psnr(current, out_net["x_hat"])

                avg_loss = total_loss / test_num
                avg_bpp = total_bpp / test_num
                avg_psnr = total_psnr / test_num

                loss_meter.update(avg_loss.item())
                psnr_meter.update(avg_psnr)
                bpp_meter.update(avg_bpp)
                mse_meter.update(out_criterion["mse_loss"].mean())
        
        all_results[test_name] = {
            'loss': loss_meter.avg,
            'psnr': psnr_meter.avg,
            'bpp': bpp_meter.avg,
            'mse': mse_meter.avg
        }
        
        if dist.get_rank() == 0:
            logging.info(
                f"Test epoch {epoch} - {test_name}: "
                f"Loss: {loss_meter.avg:.3f} | "
                f"PSNR: {psnr_meter.avg:.3f} | "
                f"MSE: {mse_meter.avg:.8f} | "
                f"Bpp: {bpp_meter.avg:.4f}"
            )
    
    # 返回平均loss用于early stopping
    avg_loss = sum([r['loss'] for r in all_results.values()]) / len(all_results)
    return avg_loss


def save_checkpoint(state, is_best, base_dir, filename="checkpoint_vd.pth.tar"):
    torch.save(state, os.path.join(base_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(base_dir, filename),
            os.path.join(base_dir, "checkpoint_best_loss_vd.pth.tar")
        )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Improved training script with overfitting prevention.")
    parser.add_argument("-m", "--model", default="DMC_slf_yuv420", choices=["DMC"], help="Model architecture")
    
    # 多个测试集配置
    parser.add_argument("--test_datasets", type=str, nargs='+',
                       default=[
                           "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/Johnny_1280x720_60",
                           "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/BasketballDrive_1920x1080_50",
                           "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_C/RaceHorses_832x480_30"
                       ],
                       help="Multiple test dataset paths")
    
    parser.add_argument("--test_filelists", type=str, nargs='+',
                       default=[
                           "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_E/test.txt",
                           "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_B/test.txt",
                           "/home/serverdn/hdd-0/slf_data/data/testdata/HEVC_C/test.txt"
                       ],
                       help="Multiple test filelist paths")
    
    parser.add_argument("-e", "--epochs", default=180, type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learning-rate", default=5e-5, type=float, help="Learning rate (降低初始学习率)")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Number of dataloader threads")
    parser.add_argument("-q", "--quality-level", type=int, default=2, help="Quality level")
    parser.add_argument("--lamada", type=float, default=1024, help="Rate-distortion parameter")
    parser.add_argument("--batch-size", type=int, default=4, help="Initial batch size")
    parser.add_argument("--test-batch-size", type=int, default=1, help="Test batch size")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Patch size")
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument("--cuda", default=True, help="Use cuda")
    parser.add_argument("--gpu-id", type=str, default=0, help="GPU id")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=float, help="Random seed for reproducibility")
    parser.add_argument("--clip_max_norm", default=0.5, type=float, help="Gradient clipping (降低到0.5)")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay for regularization")
    parser.add_argument("--early_stopping_patience", default=7, type=int, help="Early stopping patience")
    parser.add_argument("--name", default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str, help="Result dir name")
    parser.add_argument("--model_path_i", type=str, default="checkpoints/cvpr2025_image.pth.tar", help="Path to checkpoint")
    parser.add_argument("--checkpoint", type=str, default="pretrained/DMC_slf_yuv420/1/checkpoint_vd.pth.tar", help="Path to checkpoint")
    parser.add_argument("--manyframe_epoch", type=int, default=120, help="Epoch threshold to switch to 7-frame training")
    
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if dist.get_rank() == 0:
        setup_logger(os.path.join(base_dir, time.strftime('%Y%m%d_%H%M%S') + '_improved.log'))
        logging.info(f'======================= {args.name} (IMPROVED) =======================')
        for k, v in args.__dict__.items():
            logging.info(f'{k}: {v}')
        logging.info('=' * 40)

    # 构造多个测试数据集
    test_dataloaders = {}
    for test_dataset, test_filelist in zip(args.test_datasets, args.test_filelists):
        dataset_name = os.path.basename(test_dataset)
        test_transforms = TetsDataSet(
            root=test_dataset,
            filelist=test_filelist,
            gop=32,
            testfull=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_transforms)
        test_dataloader = DataLoader(
            test_transforms,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=(args.cuda and torch.cuda.is_available()),
            sampler=test_sampler
        )
        test_dataloaders[dataset_name] = test_dataloader
        if dist.get_rank() == 0:
            logging.info(f"Loaded test dataset: {dataset_name} with {len(test_transforms)} samples")

    # 训练数据集
    train_dataset = DataSet()
    global_step = 0
    gpu_per_batch = args.batch_size
    test_num = 1

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    # I帧网络
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net.eval()
    
    # 视频压缩模型
    model = DMC()
    model = model.to(torch.device("cuda", args.local_rank))
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[args.local_rank],
        find_unused_parameters=True
    )

    # 优化器 - 添加权重衰减
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay  # L2正则化
    )
    
    criterion = RateDistortionLoss(lamada=args.lamada)
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=0.001,
        verbose=True
    )

    last_epoch = 0
    if args.checkpoint:
        if dist.get_rank() == 0:
            logging.info("Loading checkpoint from %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])

    factors = [0.4, 0.1, 0.04, 0.01]
    best_loss = float("inf")

    for epoch in range(last_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.learning_rate, factors)
        
        if dist.get_rank() == 0:
            logging.info(f"\n====== Current epoch {epoch} ======")
            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # 动态调整帧数
        num = 2
        if epoch < args.manyframe_epoch:
            if (epoch - 50) > 0:
                num = (epoch - 50) / 10 + 3
            else:
                num = 3     
            num = int(num)
            if num > 7:
                num = 7
            if num <= 5:
                gpu_per_batch = args.batch_size
            else:
                gpu_per_batch = int(args.batch_size / 2)
        else:
            num = 8 + (epoch - 120) / 2
            num = int(num)
            if num > 32:
                num = 32
            gpu_per_batch = 1

        test_num = num - 1
        if test_num > 30:
            test_num = 30

        train_dataset.set_frame_count(num)
        if dist.get_rank() == 0:
            logging.info(f"Switched to {num} frame training dataset mode.")

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=gpu_per_batch,
            num_workers=args.num_workers,
            sampler=train_sampler
        )
        
        # 训练
        train_one_epoch(
            epoch, model, i_frame_net, criterion, 
            train_dataloader, optimizer, gpu_per_batch, args.clip_max_norm
        )
        
        # 测试（多个测试集）
        loss = test_epoch(epoch, i_frame_net, test_dataloaders, model, criterion, test_num)
        
        # 早停检查
        early_stopping(loss)
        if early_stopping.early_stop:
            if dist.get_rank() == 0:
                logging.info("Early stopping triggered!")
            break
        
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": None,
                },
                is_best,
                base_dir
            )

        global_step += len(train_dataloader)
        if dist.get_rank() == 0:
            logging.info(f"Global step updated to: {global_step}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
