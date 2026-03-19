# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
DCVC_RT + Wan VAE Integration for Generative Latent Coding (GLC)
Implements the GLC paradigm: compress in generative latent space instead of pixel space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .video_t import DMC, FeatureExtractor, Encoder, Decoder
from .wan_vae_wrapper import WanVAEWrapper
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb


class DMC_WAN(nn.Module):
    """
    DCVC_RT with Wan VAE for Generative Latent Coding
    
    Pipeline:
    1. Input RGB video -> Wan VAE Encoder -> Latent representations
    2. Latent -> DCVC_RT compression (in latent space) -> Compressed bitstream
    3. Compressed bitstream -> DCVC_RT decompression -> Decoded latent
    4. Decoded latent -> Wan VAE Decoder -> Reconstructed RGB video
    
    This follows the GLVC (Generative Latent Video Compression) paradigm:
    Reference: "Generative Latent Video Compression"
    (Integrating Wan VAE with DCVC_RT for ultra-low bitrate video coding)
    
    Key advantages:
    - Semantic-aware compression in generative latent space
    - Better perceptual quality at ultra-low bitrates
    - Alignment with human perception through VQ-VAE latent space
    """
    
    def __init__(
        self,
        # DCVC_RT parameters
        dcvc_checkpoint=None,
        freeze_dcvc=True,
        # Wan VAE parameters
        wan_vae_checkpoint=None,
        freeze_vae=True,
        freeze_vae_encoder_only=False,
        latent_channels=48,
        # Training parameters
        mode="pixel",  # "pixel" or "latent"
        use_vae_for_ref=True,  # Use VAE for reference frames
    ):
        super().__init__()
        
        self.mode = mode
        self.use_vae_for_ref = use_vae_for_ref
        self.freeze_dcvc = freeze_dcvc
        self.freeze_vae = freeze_vae
        self.freeze_vae_encoder_only = freeze_vae_encoder_only
        self.latent_channels = latent_channels
        
        # Initialize DCVC_RT base model
        print("Initializing DCVC_RT base model...")
        self.dcvc_model = DMC()
        
        if dcvc_checkpoint:
            print(f"Loading DCVC_RT checkpoint: {dcvc_checkpoint}")
            self._load_dcvc_checkpoint(dcvc_checkpoint)
        
        if freeze_dcvc:
            self._freeze_dcvc()
            print("✓ DCVC_RT frozen")
        
        # Initialize Wan VAE
        print("Initializing Wan VAE...")
        self.vae_wrapper = WanVAEWrapper(
            vae_ckpt_path=wan_vae_checkpoint,
            freeze_vae=freeze_vae,
            freeze_vae_encoder_only=freeze_vae_encoder_only,
            latent_channels=latent_channels,
        )
        
        # Adaptation layers to bridge VAE latent space and DCVC_RT
        # VAE latent: [B, 48, T', H', W'] where H'=H/16, W'=W/16
        # DCVC_RT expects: [B, 3, H, W] in YUV space
        # 
        # ✨ 增强型设计：减少信息瓶颈，增加网络容量
        if mode == "latent":
            # Encoder: latent (48) → compressed (16) with residual connections
            self.latent_to_pseudo_yuv = nn.ModuleList([
                # Block 1: 48 → 96 (扩展)
                nn.Sequential(
                    nn.Conv3d(latent_channels, 96, kernel_size=3, padding=1),
                    nn.GroupNorm(16, 96),
                    nn.SiLU(),
                ),
                # Block 2: 96 → 64 (压缩)
                nn.Sequential(
                    nn.Conv3d(96, 64, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.SiLU(),
                ),
                # Block 3: 64 → 32
                nn.Sequential(
                    nn.Conv3d(64, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.SiLU(),
                ),
                # Block 4: 32 → 16 (中间表示，保留更多信息)
                nn.Sequential(
                    nn.Conv3d(32, 16, kernel_size=3, padding=1),
                    nn.GroupNorm(4, 16),
                    nn.SiLU(),
                ),
                # Final: 16 → 3 (pseudo-YUV)
                nn.Conv3d(16, 3, kernel_size=1),
            ])
            
            # Decoder: compressed (16) → latent (48) with skip connections
            self.pseudo_yuv_to_latent = nn.ModuleList([
                # Block 1: 3 → 16
                nn.Sequential(
                    nn.Conv3d(3, 16, kernel_size=1),
                    nn.GroupNorm(4, 16),
                    nn.SiLU(),
                ),
                # Block 2: 16 → 32
                nn.Sequential(
                    nn.Conv3d(16, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.SiLU(),
                ),
                # Block 3: 32 → 64
                nn.Sequential(
                    nn.Conv3d(32, 64, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.SiLU(),
                ),
                # Block 4: 64 → 96
                nn.Sequential(
                    nn.Conv3d(64, 96, kernel_size=3, padding=1),
                    nn.GroupNorm(16, 96),
                    nn.SiLU(),
                ),
                # Final: 96 → 48 (恢复latent)
                nn.Conv3d(96, latent_channels, kernel_size=3, padding=1),
            ])
        
        print(f"✓ DMC_WAN initialized in '{mode}' mode")
    
    def _load_dcvc_checkpoint(self, checkpoint_path):
        """Load DCVC_RT checkpoint"""
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt['state_dict']
        if "net" in ckpt:
            ckpt = ckpt["net"]
        consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
        
        # Load with strict=False to allow for architectural differences
        missing_keys, unexpected_keys = self.dcvc_model.load_state_dict(ckpt, strict=False)
        
        if missing_keys:
            print(f"  Warning: Missing keys in DCVC checkpoint: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys in DCVC checkpoint: {len(unexpected_keys)}")
        
        print(f"✓ Loaded DCVC_RT checkpoint")
    
    def _freeze_dcvc(self):
        """Freeze DCVC_RT parameters"""
        for param in self.dcvc_model.parameters():
            param.requires_grad = False
    
    def clear_dpb(self):
        """Clear decoded picture buffer"""
        self.dcvc_model.clear_dpb()
    
    def add_ref_frame(self, ref_feature, ref_yuv):
        """Add reference frame to buffer"""
        self.dcvc_model.add_ref_frame(ref_feature, ref_yuv)
    
    def forward(self, current_frame, qp, ref_frame=None, return_intermediates=False):
        """
        Forward pass
        
        Args:
            current_frame: RGB frame [B, 3, H, W] in range [0, 1]
            qp: Quality parameter
            ref_frame: Reference RGB frame [B, 3, H, W] (optional)
            return_intermediates: Return intermediate representations
        
        Returns:
            dict with keys:
                - x_hat: Reconstructed frame [B, 3, H, W]
                - bpp: Bits per pixel
                - mse: MSE loss
                - ssim: SSIM metric
                - (optional) latent, latent_hat, pseudo_yuv
        """
        device = current_frame.device
        B, C, H, W = current_frame.shape
        
        result = {}
        
        if self.mode == "latent":
            # === Generative Latent Coding Mode ===
            # 
            # 修复版本：确保两个 adaptation layers 都能被训练
            # 1. latent_to_pseudo_yuv: 通过 pseudo_yuv_mse 损失训练
            # 2. pseudo_yuv_to_latent: 通过 latent_mse 和 LPIPS 损失训练
            
            # ⭐ 保存原始尺寸（用于最终裁剪，因为 VAE 内部可能填充到16的倍数）
            H_orig, W_orig = H, W
            
            # 1. Encode current frame to latent space (VAE encoder is frozen)
            with torch.no_grad():
                current_latent = self.vae_wrapper.encode_to_latent(current_frame)
                # latent shape: [B, 48, 1, H_lat, W_lat]
                # 其中 H_lat = ceil(H/16), W_lat = ceil(W/16)
                # 例如 1080 → pad到1088 → latent 68; 1920 → latent 120
            
            # ⭐ 从实际 latent 获取维度，而非用 H//16（H 不一定是16的倍数！）
            H_latent = current_latent.shape[3]
            W_latent = current_latent.shape[4]
            
            # 2. Convert latent to pseudo-YUV for DCVC_RT
            # 这里保留梯度，用于 pseudo_yuv_mse 损失
            current_latent_squeeze = current_latent.squeeze(2)  # [B, 48, H_lat, W_lat]
            current_latent_3d = current_latent_squeeze.unsqueeze(2)  # [B, 48, 1, H_lat, W_lat]
            
            # ✨ 使用增强型 encoder（多层级联）
            x = current_latent_3d
            for layer in self.latent_to_pseudo_yuv:
                x = layer(x)
            pseudo_yuv = x  # [B, 3, 1, H_lat, W_lat] ⭐ HAS GRADIENT
            pseudo_yuv = pseudo_yuv.squeeze(2)  # [B, 3, H_lat, W_lat]
            
            # 3. Upsample to full resolution for DCVC_RT（使用原始尺寸）
            pseudo_yuv_upsampled = F.interpolate(pseudo_yuv, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            
            # 4. Compress with DCVC_RT
            # ⚠️ 注意：如果DCVC冻结，用no_grad；如果DCVC训练，不用no_grad
            if self.freeze_dcvc:
                with torch.no_grad():
                    dcvc_result = self.dcvc_model(pseudo_yuv_upsampled, qp)
            else:
                # DCVC未冻结，需要梯度
                dcvc_result = self.dcvc_model(pseudo_yuv_upsampled, qp)
            
            pseudo_yuv_hat = dcvc_result['x_hat']  # [B, 3, H_orig, W_orig]
            
            # ⭐ 修复：让pseudo_yuv_mse有意义的梯度
            # 目标：让encoder学会生成"DCVC能准确重建"的pseudo-YUV
            # 计算encoder输出和DCVC重建之间的差异（不detach pseudo_yuv_upsampled，保留梯度）
            pseudo_yuv_mse = F.mse_loss(pseudo_yuv_upsampled, pseudo_yuv_hat.detach())
            
            # 5. Downsample decoded pseudo-YUV back to latent resolution
            # ⭐ 使用实际 latent 维度 (H_latent, W_latent)，而非 H//16
            pseudo_yuv_hat_downsampled = F.interpolate(
                pseudo_yuv_hat, 
                size=(H_latent, W_latent), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 6. Convert pseudo-YUV back to latent
            pseudo_yuv_hat_3d = pseudo_yuv_hat_downsampled.unsqueeze(2)  # [B, 3, 1, H_lat, W_lat]
            
            # ✨ 使用增强型 decoder（多层级联）
            x = pseudo_yuv_hat_3d
            for layer in self.pseudo_yuv_to_latent:
                x = layer(x)
            latent_hat_3d = x  # [B, 48, 1, H_lat, W_lat] ⭐ HAS GRADIENT
            
            # ⭐ 关键修复2：让 x_hat_rgb 有梯度，用于 LPIPS
            # VAE decoder 虽然是 frozen 的，但梯度可以通过它传回 pseudo_yuv_to_latent
            # 这样 LPIPS 损失就能有效训练 pseudo_yuv_to_latent
            x_hat_rgb = self.vae_wrapper.decode_from_latent(latent_hat_3d)  # [B, 3, 1, H_lat*16, W_lat*16] ⭐ HAS GRADIENT
            x_hat_rgb = x_hat_rgb.squeeze(2)  # [B, 3, H_lat*16, W_lat*16]
            
            # ⭐ 修复：VAE decode 输出可能比原始尺寸大（因为 encode 时填充过），裁剪回原始尺寸
            if x_hat_rgb.shape[2] != H_orig or x_hat_rgb.shape[3] != W_orig:
                x_hat_rgb = x_hat_rgb[:, :, :H_orig, :W_orig]
            
            # Convert to YUV for compatibility (但保留梯度)
            x_hat = self.vae_wrapper.rgb_to_yuv444(x_hat_rgb)
            
            current_yuv = self.vae_wrapper.rgb_to_yuv444(current_frame)
            
            # 7. Calculate MSE in latent space (训练 pseudo_yuv_to_latent)
            latent_hat_squeeze = latent_hat_3d.squeeze(2)  # [B, 48, H_lat, W_lat]
            latent_mse = F.mse_loss(latent_hat_squeeze, current_latent_squeeze.detach())
            
            # 8. ⭐ 组合损失：latent_mse + pseudo_yuv_mse
            # 这确保两个 adaptation layers 都能被训练
            # - latent_mse: 训练 pseudo_yuv_to_latent
            # - pseudo_yuv_mse: 训练 latent_to_pseudo_yuv
            # ⚠️ 修复：取平均而非相加，避免MSE值翻倍导致Loss爆炸
            combined_mse = (latent_mse + pseudo_yuv_mse) / 2.0
            
            result.update({
                'x_hat': x_hat,
                'x_hat_rgb': x_hat_rgb,  # ⭐ 现在有梯度，LPIPS 可用
                'bpp': dcvc_result['bpp'],
                'mse': combined_mse,  # ⭐ 组合损失，两个 adaptation layers 都能训练
                'latent_mse': latent_mse,  # 单独记录，方便监控
                'pseudo_yuv_mse': pseudo_yuv_mse,  # 单独记录，方便监控
                'ssim': dcvc_result['ssim'],
            })
            
            if return_intermediates:
                result.update({
                    'current_latent': current_latent,
                    'latent_hat': latent_hat_3d,
                    'pseudo_yuv': pseudo_yuv_upsampled,
                    'pseudo_yuv_hat': pseudo_yuv_hat,
                })
        
        else:
            # === Pixel Space Mode (Standard DCVC_RT) ===
            # Convert RGB to YUV for DCVC_RT
            current_yuv = self.vae_wrapper.rgb_to_yuv444(current_frame)
            
            # Standard DCVC_RT forward pass
            dcvc_result = self.dcvc_model(current_yuv, qp)
            
            # Convert YUV result back to RGB
            x_hat_yuv = dcvc_result['x_hat']
            x_hat_rgb = self.vae_wrapper.yuv444_to_rgb(x_hat_yuv)
            
            result.update({
                'x_hat': x_hat_yuv,
                'x_hat_rgb': x_hat_rgb,
                'bpp': dcvc_result['bpp'],
                'mse': dcvc_result['mse'],
                'ssim': dcvc_result['ssim'],
            })
        
        return result
    
    def compress_(self, x, qp):
        """
        Compress a frame (for I-frame compression)
        
        Args:
            x: RGB frame [B, 3, H, W]
            qp: Quality parameter
        
        Returns:
            Compressed representation (feature space)
        """
        if self.mode == "latent" and self.use_vae_for_ref:
            # Encode to latent space
            latent = self.vae_wrapper.encode_to_latent(x)
            # For reference, we keep in latent space
            return latent
        else:
            # Standard pixel space
            yuv = self.vae_wrapper.rgb_to_yuv444(x)
            return self.dcvc_model.compress_(yuv, qp)
    
    def switch_mode(self, mode):
        """
        Switch between pixel and latent mode
        
        Args:
            mode: "pixel" or "latent"
        """
        assert mode in ["pixel", "latent"], f"Invalid mode: {mode}"
        self.mode = mode
        print(f"✓ Switched to '{mode}' mode")
    
    def get_trainable_parameters(self):
        """Get trainable parameters for different components"""
        params = {
            'dcvc': [],
            'vae': [],
            'adaptation': [],
        }
        
        if not self.freeze_dcvc:
            params['dcvc'] = list(self.dcvc_model.parameters())
        
        if not self.freeze_vae:
            if hasattr(self.vae_wrapper, 'vae') and self.vae_wrapper.vae is not None:
                # Wan2_2_VAE is not nn.Module, access its internal model
                if hasattr(self.vae_wrapper.vae, 'model') and self.vae_wrapper.vae.model is not None:
                    params['vae'] = list(self.vae_wrapper.vae.model.parameters())
        
        if self.mode == "latent":
            params['adaptation'] = (
                list(self.latent_to_pseudo_yuv.parameters()) +
                list(self.pseudo_yuv_to_latent.parameters())
            )
        
        return params
    
    def count_parameters(self):
        """Count parameters in different components"""
        dcvc_params = sum(p.numel() for p in self.dcvc_model.parameters())
        dcvc_trainable = sum(p.numel() for p in self.dcvc_model.parameters() if p.requires_grad)
        
        vae_params = 0
        vae_trainable = 0
        if hasattr(self.vae_wrapper, 'vae') and self.vae_wrapper.vae is not None:
            # Wan2_2_VAE is not nn.Module, access its internal model
            vae_model = self.vae_wrapper.vae.model if hasattr(self.vae_wrapper.vae, 'model') else None
            if vae_model is not None:
                if hasattr(vae_model, 'encoder'):
                    vae_params += sum(p.numel() for p in vae_model.encoder.parameters())
                    vae_trainable += sum(p.numel() for p in vae_model.encoder.parameters() if p.requires_grad)
                if hasattr(vae_model, 'decoder'):
                    vae_params += sum(p.numel() for p in vae_model.decoder.parameters())
                    vae_trainable += sum(p.numel() for p in vae_model.decoder.parameters() if p.requires_grad)
        
        adapt_params = 0
        adapt_trainable = 0
        if self.mode == "latent":
            adapt_params = (
                sum(p.numel() for p in self.latent_to_pseudo_yuv.parameters()) +
                sum(p.numel() for p in self.pseudo_yuv_to_latent.parameters())
            )
            adapt_trainable = (
                sum(p.numel() for p in self.latent_to_pseudo_yuv.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.pseudo_yuv_to_latent.parameters() if p.requires_grad)
            )
        
        print(f"\n{'='*60}")
        print(f"Parameter Count:")
        print(f"  DCVC_RT: {dcvc_params:,} total, {dcvc_trainable:,} trainable")
        print(f"  Wan VAE: {vae_params:,} total, {vae_trainable:,} trainable")
        print(f"  Adaptation: {adapt_params:,} total, {adapt_trainable:,} trainable")
        print(f"  Total: {dcvc_params + vae_params + adapt_params:,} total, "
              f"{dcvc_trainable + vae_trainable + adapt_trainable:,} trainable")
        print(f"{'='*60}\n")
        
        return {
            'dcvc_total': dcvc_params,
            'dcvc_trainable': dcvc_trainable,
            'vae_total': vae_params,
            'vae_trainable': vae_trainable,
            'adaptation_total': adapt_params,
            'adaptation_trainable': adapt_trainable,
        }


if __name__ == "__main__":
    # Test DMC_WAN model
    print("Testing DMC_WAN model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test in pixel mode
    print("\n1. Testing pixel mode...")
    model_pixel = DMC_WAN(
        mode="pixel",
        freeze_dcvc=True,
        freeze_vae=True,
    ).to(device)
    
    model_pixel.count_parameters()
    
    dummy_frame = torch.randn(2, 3, 256, 256).to(device) * 0.5 + 0.5
    qp = 37
    
    result = model_pixel(dummy_frame, qp)
    print(f"   Input shape: {dummy_frame.shape}")
    print(f"   Output shape: {result['x_hat'].shape}")
    print(f"   BPP: {result['bpp'].mean().item():.4f}")
    
    # Test in latent mode
    print("\n2. Testing latent mode...")
    model_latent = DMC_WAN(
        mode="latent",
        freeze_dcvc=True,
        freeze_vae=True,
    ).to(device)
    
    model_latent.count_parameters()
    
    result = model_latent(dummy_frame, qp, return_intermediates=True)
    print(f"   Input shape: {dummy_frame.shape}")
    print(f"   Latent shape: {result['current_latent'].shape}")
    print(f"   Output shape: {result['x_hat'].shape}")
    print(f"   BPP: {result['bpp'].mean().item():.4f}")
    
    print("\n✅ All tests passed!")
