import torch
import torch.nn as nn
from functools import partial
from typing import List

class ScaleAdapter(nn.Module):
    """Scale Adapter - 基于论文Figure 3(c)的精确实现"""
    def __init__(self, embed_dim: int):
        super().__init__()
        # 基于论文：Down-projection maps to 128 dimensions
        self.down_projection = nn.Linear(embed_dim, 128)
        self.activation = nn.ReLU(True)  # 论文明确使用ReLU
        self.up_projection = nn.Linear(128, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 论文描述：residual connection with input feature
        identity = x
        x = self.down_projection(x)
        x = self.activation(x)
        x = self.up_projection(x)
        return x + identity  # Residual connection

class HeightBlock(nn.Module):
    """Height Block - 基于DINOv2 Transformer Block的修改版本"""
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 ffn_bias: bool = True,
                 drop_path: float = 0.0):
        super().__init__()
        
        # 导入DINOv2的组件（假设已经实现）
        try:
            from .dinov2_layers import MemEffAttention, Mlp
        except ImportError:
            # 如果没有DINOv2，使用标准实现
            from torch.nn import MultiheadAttention
            MemEffAttention = MultiheadAttention
            
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.norm1 = norm_layer(embed_dim)
        self.attn = MemEffAttention(
            embed_dim, 
            num_heads=num_heads,
            batch_first=True
        ) if hasattr(MemEffAttention, 'embed_dim') else nn.MultiheadAttention(
            embed_dim, 
            num_heads,
            batch_first=True
        )
        
        self.norm2 = norm_layer(embed_dim)
        
        # 标准MLP
        if 'Mlp' in locals():
            self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                nn.Dropout(0.1)
            )
        
        # 论文关键创新：集成trainable MLP after attention residual connection
        self.additional_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
        # Drop path (Stochastic Depth)
        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard transformer with additional MLP
        if hasattr(self.attn, 'forward'):
            attn_out = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        else:
            attn_out = x + self.drop_path(self.attn(self.norm1(x)))
            
        attn_out = attn_out + self.additional_mlp(attn_out)  # 论文创新点
        x = attn_out + self.drop_path(self.mlp(self.norm2(attn_out)))
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ScaleModulator(nn.Module):
    """Scale Modulator - 论文核心创新，基于Figure 3(a)"""
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        
        # 论文明确：12个Height Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 12)]  # stochastic depth decay rule
        
        self.height_blocks = nn.ModuleList([
            HeightBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i]
            ) for i in range(12)
        ])
        
        # 论文明确：4个Scale Adapters，插入在最后4个blocks
        self.scale_adapters = nn.ModuleList([
            ScaleAdapter(embed_dim) for _ in range(4)
        ])
        
        # 论文Figure 3(a)显示：从最后4个blocks提取特征
        self.adapter_positions = [8, 9, 10, 11]  # 最后4个位置
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: 来自patch embedding的特征 [B, N, C]
        Returns:
            list: 4个scale features [fs1, fs2, fs3, fs4]
        """
        scale_features = []
        
        # 通过12个Height Blocks
        for i, height_block in enumerate(self.height_blocks):
            x = height_block(x)
            
            # 在最后4个blocks处提取并处理特征
            if i in self.adapter_positions:
                adapter_idx = self.adapter_positions.index(i)
                adapted_feature = self.scale_adapters[adapter_idx](x)
                scale_features.append(adapted_feature)
                
        return scale_features  # [fs1, fs2, fs3, fs4]