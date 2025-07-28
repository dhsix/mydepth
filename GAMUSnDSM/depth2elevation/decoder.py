import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class ProjectionBlock(nn.Module):
    """Projection Block - 基于论文Figure 4右上"""
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.GELU()  # 论文明确使用GELU
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 从1D latent space映射到2D space的准备
        return self.projection(x)

class RefineBlock(nn.Module):
    """Refine Block - 基于论文Figure 4右下"""
    def __init__(self, in_channels: int,deeper_channels:int=None):
        super().__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(True)
        )
        # 如果deeper特征的通道数与当前不同，需要1x1卷积对齐
        if deeper_channels is not None and deeper_channels != in_channels:
            self.channel_proj = nn.Conv2d(deeper_channels, in_channels, 1, 1, 0)
        else:
            self.channel_proj = None
    def forward(self, fn: torch.Tensor, fn_plus_1: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            fn: 当前层特征
            fn_plus_1: 更深层特征 (可选)
        """
        if fn_plus_1 is not None:
            # 论文描述：features of current layer combined with deeper features
            # after residual connection
            if self.channel_proj is not None:
                fn_plus_1 = self.channel_proj(fn_plus_1)
            combined = fn + fn_plus_1
            return self.conv_relu(combined)
        else:
            return self.conv_relu(fn)

class ResolutionAgnosticDecoder(nn.Module):
    """Resolution-Agnostic Decoder - 基于论文Figure 4"""
    def __init__(self, 
                 embed_dim: int, 
                 num_register_tokens: int = 0):
        super().__init__()
        self.num_register_tokens = num_register_tokens
        
        # 论文Figure 4: 4个不同尺度的处理通道
        self.out_channels = [256, 512, 1024, 1024]  # 论文中的配置
        
        # Projection Blocks for each scale
        self.projection_blocks = nn.ModuleList([
            ProjectionBlock(embed_dim, out_channel) 
            for out_channel in self.out_channels
        ])
        
        # Resize operations (论文Figure 4左侧)
        self.resize_ops = nn.ModuleList([
            nn.ConvTranspose2d(self.out_channels[0], self.out_channels[0], 4, 4, 0),  # 4x upsample
            nn.ConvTranspose2d(self.out_channels[1], self.out_channels[1], 2, 2, 0),  # 2x upsample  
            nn.Identity(),  # No change
            nn.Conv2d(self.out_channels[3], self.out_channels[3], 3, 2, 1)  # 0.5x downsample
        ])
        
        # Refine Blocks for multi-scale fusion
        # self.refine_blocks = nn.ModuleList([
        #     RefineBlock(out_channel) for out_channel in self.out_channels
        # ])
        self.refine_blocks = nn.ModuleList([
            RefineBlock(self.out_channels[0]),  # 最深层，无deeper输入
            RefineBlock(self.out_channels[1], self.out_channels[0]),  # 512, deeper=256
            RefineBlock(self.out_channels[2], self.out_channels[1]),  # 1024, deeper=512
            RefineBlock(self.out_channels[3], self.out_channels[2])   # 1024, deeper=1024
        ])
        
        # Final prediction heads for each scale
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel//2, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channel//2, 32, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, 1, 1, 0),
                nn.ReLU(True)
                # nn.Softplus()
            ) for out_channel in self.out_channels
        ])
        
    def forward(self, 
               scale_features: List[torch.Tensor], 
               patch_h: int, 
               patch_w: int) -> Dict[str, torch.Tensor]:
        """
        Args:
            scale_features: 来自Scale Modulator的4个尺度特征
            patch_h, patch_w: patch的高度和宽度
        Returns:
            dict: 包含4个尺度预测结果的字典
        """
        processed_features = []
        
        # 1. Projection: 1D to 2D conversion
        for i, (feat, proj_block) in enumerate(zip(scale_features, self.projection_blocks)):
            # 移除cls token和register tokens，只保留patch tokens
            if self.num_register_tokens > 0:
                patch_tokens = feat[:, 1 + self.num_register_tokens:, :]
            else:
                patch_tokens = feat[:, 1:, :]  # 移除cls token
            
            # Projection
            projected = proj_block(patch_tokens)  # [B, N, out_channel]
            
            # Reshape to 2D: [B, N, C] -> [B, C, H, W]
            B = projected.shape[0]
            projected_2d = projected.permute(0, 2, 1).reshape(
                B, self.out_channels[i], patch_h, patch_w
            )
            
            # Resize operations
            resized = self.resize_ops[i](projected_2d)
            processed_features.append(resized)
        
        # 2. Multi-scale refinement (论文的核心融合策略)
        refined_features = []
        for i in range(len(processed_features)):
            if i == 0:
                # 最深层特征，无需融合
                refined = self.refine_blocks[i](processed_features[i])
            else:
                # 与更深层特征融合
                # 需要调整尺寸匹配
                deeper_feat = F.interpolate(
                    refined_features[i-1], 
                    size=processed_features[i].shape[-2:],
                    mode='bilinear', 
                    align_corners=True
                )
                refined = self.refine_blocks[i](processed_features[i], deeper_feat)
            
            refined_features.append(refined)
        
        # 3. Generate multi-scale predictions
        predictions = {}
        for i, (feat, pred_head) in enumerate(zip(refined_features, self.prediction_heads)):
            height_map = pred_head(feat)  # [B, 1, H, W]
            
            # 论文描述：upsample to match input image resolution before calculating loss
            target_size = (patch_h * 14, patch_w * 14)  # 恢复到原始尺寸
            height_map = F.interpolate(
                height_map, 
                size=target_size,
                mode='bilinear', 
                align_corners=True
            )
            
            predictions[f'scale_{i+1}'] = height_map.squeeze(1)
            
        return predictions