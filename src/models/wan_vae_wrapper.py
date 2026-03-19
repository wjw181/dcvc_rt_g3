# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Wan VAE Wrapper for DCVC_RT Integration
This module wraps the Wan2.2 VAE to work with DCVC_RT for latent space video compression.
Following GLC (Generative Latent Coding) paradigm.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Add Wan module path (directly import vae2_2 to avoid full package dependencies)
WAN_PATH = "./Wan2.2-main"
if WAN_PATH not in sys.path:
    sys.path.insert(0, WAN_PATH)

# Direct import to avoid loading full Wan package
import importlib.util
spec = importlib.util.spec_from_file_location("vae2_2", os.path.join(WAN_PATH, "wan/modules/vae2_2.py"))
vae2_2_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vae2_2_module)
Wan2_2_VAE = vae2_2_module.Wan2_2_VAE


class WanVAEWrapper(nn.Module):
    """
    Wrapper around Wan2.2 VAE for video compression in latent space.
    
    Architecture:
    - RGB Video -> VAE Encoder -> Latent (16x16x4 compression) -> DCVC_RT -> Compressed Latent
    - Compressed Latent -> DCVC_RT Decoder -> Latent -> VAE Decoder -> RGB Video
    
    Args:
        vae_ckpt_path: Path to Wan2.2 VAE checkpoint
        freeze_vae: Whether to freeze VAE parameters during training
        latent_channels: Number of latent channels (default: 48 for Wan2.2)
        dtype: Data type for VAE
        device: Device for VAE
    """
    
    def __init__(
        self,
        vae_ckpt_path=None,
        freeze_vae=True,
        freeze_vae_encoder_only=False,
        latent_channels=48,
        z_dim=48,
        c_dim=160,
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        dtype=torch.bfloat16,
        device="cuda",
    ):
        super().__init__()
        
        self.freeze_vae = freeze_vae
        self.freeze_vae_encoder_only = freeze_vae_encoder_only
        self.latent_channels = latent_channels
        self.z_dim = z_dim
        
        # Initialize Wan2.2 VAE
        print(f"Initializing Wan2.2 VAE with checkpoint: {vae_ckpt_path}")
        
        # Note: Wan2_2_VAE is actually a factory class, not nn.Module
        # For testing without checkpoint, we create a placeholder
        if vae_ckpt_path is None:
            print("  Warning: No VAE checkpoint provided. Using placeholder VAE.")
            print("  VAE encode/decode will not work. For testing structure only.")
            self.vae = None  # Placeholder
        else:
            self.vae = Wan2_2_VAE(
                z_dim=z_dim,
                c_dim=c_dim,
                vae_pth=vae_ckpt_path,
                dim_mult=dim_mult,
                temperal_downsample=temperal_downsample,
                dtype=dtype,
                device=device,
            )
        
        # Freeze VAE if specified
        if self.freeze_vae:
            self._freeze_vae()
            print("✓ Wan VAE frozen")
        elif self.freeze_vae_encoder_only:
            self._freeze_vae_encoder_only()
            print("✓ Wan VAE encoder frozen, decoder trainable")
        else:
            print("✓ Wan VAE trainable")
    
    def _freeze_vae(self):
        """Freeze all VAE parameters"""
        if self.vae is None:
            return
        if hasattr(self.vae, 'encoder'):
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
        if hasattr(self.vae, 'decoder'):
            for param in self.vae.decoder.parameters():
                param.requires_grad = False
    
    def _freeze_vae_encoder_only(self):
        """Freeze only VAE encoder, keep decoder trainable"""
        if self.vae is None:
            return
        if hasattr(self.vae, 'encoder'):
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
        if hasattr(self.vae, 'decoder'):
            for param in self.vae.decoder.parameters():
                param.requires_grad = True
    
    def encode_to_latent(self, video_frames):
        """
        Encode RGB video frames to latent space
        
        Args:
            video_frames: Tensor of shape [B, T, C, H, W] or [B, C, H, W]
                         Values in range [0, 1]
        
        Returns:
            latent: Tensor of shape [B, z_dim, T', H', W']
                   where T' = T/4, H' = H/16, W' = W/16 for Wan2.2
        """
        if self.vae is None:
            raise RuntimeError("VAE not initialized. Please provide vae_ckpt_path.")
        
        if len(video_frames.shape) == 4:
            # Single frame: [B, C, H, W] -> [B, C, 1, H, W]
            video_frames = video_frames.unsqueeze(2)
        
        # Rearrange to [B, C, T, H, W] if needed
        if video_frames.shape[1] == 3:  # [B, C, T, H, W]
            pass
        else:  # [B, T, C, H, W]
            video_frames = rearrange(video_frames, 'b t c h w -> b c t h w')
        
        # ⭐ 修复：确保高度和宽度是 16 的倍数
        # Wan VAE 的处理流程：
        # 1. patchify(patch_size=2) → H/2, W/2
        # 2. 3 次下采样，每次 ×2 → H/16, W/16
        # 所以输入必须是 16 的倍数
        B, C, T, H, W = video_frames.shape
        pad_h = (16 - H % 16) % 16  # 计算需要填充的高度
        pad_w = (16 - W % 16) % 16  # 计算需要填充的宽度
        
        if pad_h > 0 or pad_w > 0:
            # 注意：reflect 模式不支持 5D 张量，先 squeeze 时间维度再填充
            # video_frames: [B, C, T, H, W]
            orig_shape = video_frames.shape
            # 合并 B 和 T 维度: [B, C, T, H, W] -> [B*T, C, H, W]
            video_frames = video_frames.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            video_frames = F.pad(video_frames, (0, pad_w, 0, pad_h), mode='reflect')
            # 恢复: [B*T, C, H+pad_h, W+pad_w] -> [B, C, T, H+pad_h, W+pad_w]
            video_frames = video_frames.reshape(B, T, C, H + pad_h, W + pad_w).permute(0, 2, 1, 3, 4)
            H_padded = H + pad_h
            W_padded = W + pad_w
        else:
            H_padded = H
            W_padded = W
        
        # Scale from [0, 1] to [-1, 1] for VAE
        video_frames = video_frames * 2.0 - 1.0
        
        with torch.set_grad_enabled(not self.freeze_vae):
            # Wan VAE encode expects a list of videos, not a tensor
            # Convert tensor to list: [B, C, T, H, W] -> list of [C, T, H, W]
            video_list = [video_frames[i] for i in range(B)]
            latent_list = self.vae.encode(video_list)  # Returns list of [z_dim, T', H', W']
            
            if latent_list is None:
                raise RuntimeError("VAE encode returned None. Check input format and VAE initialization.")
            
            # Convert list back to tensor: [B, z_dim, T', H', W']
            latent = torch.stack(latent_list, dim=0)
            
            # ⭐ 注意：不裁剪 latent！保持填充后的尺寸 (H_padded//16, W_padded//16)
            # 在最终输出（decode 后）再裁剪回原始分辨率
            # 如果裁剪 latent 会导致 decode 后尺寸不够（比如 1080→pad1088→latent68→crop67→decode1072 < 1080）
        
        return latent
    
    def decode_from_latent(self, latent, allow_grad=True):
        """
        Decode latent representation back to RGB video frames
        
        Args:
            latent: Tensor of shape [B, z_dim, T', H', W']
            allow_grad: Whether to allow gradient flow through decoder.
                       Even if VAE is frozen (params not updated), gradients can still flow
                       through the computation graph for training upstream modules.
        
        Returns:
            video_frames: Tensor of shape [B, C, T, H, W]
                         Values in range [0, 1]
        """
        if self.vae is None:
            raise RuntimeError("VAE not initialized. Please provide vae_ckpt_path.")
        
        # ⭐ 关键修复：即使 VAE 被冻结，梯度仍然可以通过计算图传递
        # 这对于训练 upstream modules（如 adaptation layers）是必要的
        # 只有当 allow_grad=False 时才禁用梯度
        grad_enabled = allow_grad and latent.requires_grad
        
        with torch.set_grad_enabled(grad_enabled):
            # Wan VAE decode expects a list of latents, not a tensor
            # Convert tensor to list: [B, z_dim, T', H', W'] -> list of [z_dim, T', H', W']
            B = latent.shape[0]
            latent_list = [latent[i] for i in range(B)]
            video_list = self.vae.decode(latent_list)  # Returns list of [C, T, H, W]
            
            if video_list is None:
                raise RuntimeError("VAE decode returned None. Check latent format and VAE initialization.")
            
            # Convert list back to tensor: [B, C, T, H, W]
            video_frames = torch.stack(video_list, dim=0)
        
        # Scale from [-1, 1] back to [0, 1]
        video_frames = (video_frames + 1.0) / 2.0
        video_frames = torch.clamp(video_frames, 0.0, 1.0)
        
        return video_frames
    
    def forward(self, video_frames, return_latent=False):
        """
        Forward pass: encode and decode
        
        Args:
            video_frames: Input video [B, T, C, H, W] or [B, C, H, W]
            return_latent: If True, also return latent representation
        
        Returns:
            reconstructed: Reconstructed video
            latent (optional): Latent representation
        """
        latent = self.encode_to_latent(video_frames)
        reconstructed = self.decode_from_latent(latent)
        
        if return_latent:
            return reconstructed, latent
        return reconstructed
    
    def get_latent_shape(self, input_shape):
        """
        Calculate output latent shape given input video shape
        
        Args:
            input_shape: Tuple (B, C, T, H, W) or (B, T, C, H, W)
        
        Returns:
            latent_shape: Tuple (B, z_dim, T', H', W')
        """
        if len(input_shape) == 5:
            if input_shape[1] == 3:  # [B, C, T, H, W]
                B, C, T, H, W = input_shape
            else:  # [B, T, C, H, W]
                B, T, C, H, W = input_shape
        else:  # [B, C, H, W]
            B, C, H, W = input_shape
            T = 1
        
        # Wan2.2 compression ratios:
        # Temporal: 4x (when T > 1)
        # Spatial: 16x16
        T_latent = max(1, T // 4)
        H_latent = H // 16
        W_latent = W // 16
        
        return (B, self.z_dim, T_latent, H_latent, W_latent)
    
    def rgb_to_yuv444(self, rgb):
        """
        Convert RGB to YUV444 for compatibility with DCVC_RT
        Uses ITU-R BT.709 standard (HD)
        
        Args:
            rgb: Tensor [B, 3, H, W] or [B, 3, T, H, W] in range [0, 1]
        
        Returns:
            yuv: Tensor [B, 3, ...] in range [0, 1]
        """
        # ITU-R BT.709 (HD) RGB to YCbCr conversion
        # Y  =  0.2126*R + 0.7152*G + 0.0722*B
        # Cb = -0.1146*R - 0.3854*G + 0.5000*B + 0.5
        # Cr =  0.5000*R - 0.4542*G - 0.0458*B + 0.5
        
        rgb_shape = rgb.shape
        if len(rgb_shape) == 5:  # [B, C, T, H, W]
            B, C, T, H, W = rgb_shape
            rgb = rearrange(rgb, 'b c t h w -> (b t) c h w')
        
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        u = -0.1146 * r - 0.3854 * g + 0.5000 * b + 0.5
        v = 0.5000 * r - 0.4542 * g - 0.0458 * b + 0.5
        
        # Clamp to [0, 1] range
        y = torch.clamp(y, 0.0, 1.0)
        u = torch.clamp(u, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)
        
        yuv = torch.cat([y, u, v], dim=1)
        
        if len(rgb_shape) == 5:
            yuv = rearrange(yuv, '(b t) c h w -> b c t h w', b=B, t=T)
        
        return yuv
    
    def yuv444_to_rgb(self, yuv):
        """
        Convert YUV444 back to RGB
        Uses ITU-R BT.709 standard (HD)
        
        Args:
            yuv: Tensor [B, 3, ...] in range [0, 1]
        
        Returns:
            rgb: Tensor [B, 3, ...] in range [0, 1]
        """
        yuv_shape = yuv.shape
        if len(yuv_shape) == 5:  # [B, C, T, H, W]
            B, C, T, H, W = yuv_shape
            yuv = rearrange(yuv, 'b c t h w -> (b t) c h w')
        
        y, cb, cr = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
        
        # Center Cb and Cr around 0
        cb = cb - 0.5
        cr = cr - 0.5
        
        # ITU-R BT.709 YCbCr to RGB
        r = y + 1.5748 * cr
        g = y - 0.1873 * cb - 0.4681 * cr
        b = y + 1.8556 * cb
        
        rgb = torch.cat([r, g, b], dim=1)
        rgb = torch.clamp(rgb, 0.0, 1.0)
        
        if len(yuv_shape) == 5:
            rgb = rearrange(rgb, '(b t) c h w -> b c t h w', b=B, t=T)
        
        return rgb


if __name__ == "__main__":
    # Test the wrapper
    print("Testing WanVAEWrapper...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize wrapper (without checkpoint for testing)
    wrapper = WanVAEWrapper(
        vae_ckpt_path=None,
        freeze_vae=True,
        device=device
    )
    
    # Test with single frame
    print("\n1. Testing single frame encoding/decoding...")
    dummy_frame = torch.randn(2, 3, 256, 256).to(device) * 0.5 + 0.5  # [B, C, H, W]
    latent = wrapper.encode_to_latent(dummy_frame)
    print(f"   Input shape: {dummy_frame.shape}")
    print(f"   Latent shape: {latent.shape}")
    
    reconstructed = wrapper.decode_from_latent(latent)
    print(f"   Reconstructed shape: {reconstructed.shape}")
    
    # Test with video sequence
    print("\n2. Testing video sequence encoding/decoding...")
    dummy_video = torch.randn(1, 3, 8, 256, 256).to(device) * 0.5 + 0.5  # [B, C, T, H, W]
    latent_video = wrapper.encode_to_latent(dummy_video)
    print(f"   Input shape: {dummy_video.shape}")
    print(f"   Latent shape: {latent_video.shape}")
    
    reconstructed_video = wrapper.decode_from_latent(latent_video)
    print(f"   Reconstructed shape: {reconstructed_video.shape}")
    
    # Test latent shape calculation
    print("\n3. Testing latent shape calculation...")
    input_shapes = [
        (2, 3, 256, 256),      # Single frame
        (1, 3, 8, 512, 512),   # Video sequence
        (2, 16, 3, 1024, 1024) # Batch of videos
    ]
    for shape in input_shapes:
        latent_shape = wrapper.get_latent_shape(shape)
        print(f"   Input: {shape} -> Latent: {latent_shape}")
    
    print("\n✅ All tests passed!")
