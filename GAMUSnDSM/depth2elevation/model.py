import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

from .base_model import BaseDepthModel
from .multi_scale_loss import get_loss_function, SingleScaleLoss
from .scale_modulator import ScaleModulator
from .decoder import ResolutionAgnosticDecoder

class Depth2ElevationEncoder(nn.Module):
    """基于DINOv2的高程编码器，集成Scale Modulator"""
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 img_size: int = 448,
                 patch_size: int = 14,
                 in_chans: int = 3,
                 num_register_tokens: int = 0,
                 interpolate_antialias: bool = False,
                 interpolate_offset: float = 0.1):
        super().__init__()
        
        # Patch embedding (基于DINOv2)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        self.num_tokens = 1
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        
        # Position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        else:
            self.register_tokens = None
        
        # 用Scale Modulator替换原来的blocks
        self.scale_modulator = ScaleModulator(embed_dim, num_heads)
        
        # 输出norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # 初始化
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            torch.nn.init.normal_(self.register_tokens, std=1e-6)
        
    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """位置编码插值，保持原DINOv2的逻辑"""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
            
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
        
    def prepare_tokens_with_masks(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """准备输入tokens，保持原DINOv2的逻辑"""
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        
        if masks is not None:
            # 如果有mask token，应用掩码
            mask_token = getattr(self, 'mask_token', None)
            if mask_token is not None:
                x = torch.where(masks.unsqueeze(-1), mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat((
                x[:, :1],
                self.register_tokens.expand(x.shape[0], -1, -1),
                x[:, 1:],
            ), dim=1)

        return x
        
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, 3, H, W] 
            masks: 可选掩码
        Returns:
            list: 4个尺度的特征 [fs1, fs2, fs3, fs4]
        """
        x = self.prepare_tokens_with_masks(x, masks)
        
        # 通过Scale Modulator获取多尺度特征
        scale_features = self.scale_modulator(x)
        
        # 对每个尺度特征进行norm
        scale_features = [self.norm(feat) for feat in scale_features]
        
        return scale_features

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x

class Depth2Elevation(BaseDepthModel):
    """Depth2Elevation主模型 - 基于论文完整实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        model_config = config.get('model_config', {})
        
        # 模型参数
        self.encoder_type = model_config.get('encoder', 'vitb')
        self.img_size = model_config.get('img_size', 448)
        self.patch_size = model_config.get('patch_size', 14)
        self.use_multi_scale_output = config.get('use_multi_scale_output', True)
        
        # 确定模型配置
        model_configs = {
            'vits': {'embed_dim': 384, 'num_heads': 6},
            'vitb': {'embed_dim': 768, 'num_heads': 12},
            'vitl': {'embed_dim': 1024, 'num_heads': 16},
            'vitg': {'embed_dim': 1536, 'num_heads': 24}
        }
        
        if self.encoder_type not in model_configs:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
        
        encoder_config = model_configs[self.encoder_type]
        self.embed_dim = encoder_config['embed_dim']
        self.num_heads = encoder_config['num_heads']
        
        # 创建编码器
        self.height_encoder = Depth2ElevationEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_register_tokens=0  # 可配置
        )
        
        # 创建解码器
        self.decoder = ResolutionAgnosticDecoder(
            embed_dim=self.embed_dim,
            num_register_tokens=0
        )
        
        # 损失函数
        loss_config = config.get('loss_config', {})
        self.loss_fn = get_loss_function(loss_config)
        
        # 加载预训练权重
        pretrained_path = model_config.get('pretrained_path')
        if pretrained_path and Path(pretrained_path).exists():
            self.load_pretrained_weights(pretrained_path)
        
        # # 是否冻结编码器
        # if model_config.get('freeze_encoder', False):
        #     self.freeze_encoder()
        # 获取冻结策略配置
        freezing_config = config.get('freezing_config', {})
        freezing_strategy = freezing_config.get('strategy', 'none')
        
        # 向后兼容：如果model_config中设置了freeze_encoder=True，且没有明确设置strategy
        if (model_config.get('freeze_encoder', False) and 
            freezing_strategy == 'none' and 
            'strategy' not in freezing_config):
            freezing_strategy = 'simple'
            print("📌 Detected legacy freeze_encoder=True, using 'simple' strategy")
        
        # 应用冻结策略
        if freezing_strategy != 'none':
            self.apply_freezing_strategy(freezing_strategy)
    
    def load_pretrained_weights(self, pretrained_path: str):
        """加载DAM预训练权重"""
        print(f"Loading pretrained weights from {pretrained_path}")
        
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 处理不同的checkpoint格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 加载编码器权重（忽略不匹配的键）
            encoder_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('patch_embed') or key.startswith('cls_token') or key.startswith('pos_embed'):
                    encoder_state_dict[key] = value
                elif key.startswith('blocks'):
                    # 将原来的blocks权重映射到height blocks
                    new_key = key.replace('blocks', 'scale_modulator.height_blocks')
                    encoder_state_dict[new_key] = value
            
            # 载入权重，允许部分匹配
            missing_keys, unexpected_keys = self.height_encoder.load_state_dict(encoder_state_dict, strict=False)
            
            print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
    
    def freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.height_encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")
    
    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.height_encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen")
    def freeze_encoder_selectively(self):
        """精细化冻结策略 - 按论文要求冻结DAM原有组件，解冻新增组件"""
        print("Applying selective encoder freezing strategy...")
        
        # 1. 冻结基础组件（保留DAM的预训练知识）
        for param in self.height_encoder.patch_embed.parameters():
            param.requires_grad = False
        print("  ✓ Frozen patch_embed")
        
        # pos_embed和cls_token可以微调（遥感图像可能需要不同的空间关系）
        self.height_encoder.pos_embed.requires_grad = True
        self.height_encoder.cls_token.requires_grad = True
        if hasattr(self.height_encoder, 'register_tokens') and self.height_encoder.register_tokens is not None:
            self.height_encoder.register_tokens.requires_grad = True
        print("  ✓ Unfrozen pos_embed, cls_token")
        
        # 2. Height Blocks的精细化处理
        frozen_components = []
        trainable_components = []
        
        for i, height_block in enumerate(self.height_encoder.scale_modulator.height_blocks):
            # 冻结DAM原有组件
            for param in height_block.attn.parameters():
                param.requires_grad = False
            for param in height_block.norm1.parameters():
                param.requires_grad = False
            for param in height_block.norm2.parameters():
                param.requires_grad = False
            for param in height_block.mlp.parameters():
                param.requires_grad = False
            
            frozen_components.extend(['attn', 'norm1', 'norm2', 'mlp'])
            
            # 解冻新增的additional_mlp（论文明确说明是trainable）
            for param in height_block.additional_mlp.parameters():
                param.requires_grad = True
            
            trainable_components.append(f'height_block_{i}.additional_mlp')
        
        print(f"  ✓ Frozen DAM components in {len(self.height_encoder.scale_modulator.height_blocks)} height blocks")
        print(f"  ✓ Unfrozen additional_mlp in {len(self.height_encoder.scale_modulator.height_blocks)} height blocks")
        
        # 3. 解冻所有scale_adapters（论文说明需要learn）
        for i, scale_adapter in enumerate(self.height_encoder.scale_modulator.scale_adapters):
            for param in scale_adapter.parameters():
                param.requires_grad = True
            trainable_components.append(f'scale_adapter_{i}')
        
        print(f"  ✓ Unfrozen {len(self.height_encoder.scale_modulator.scale_adapters)} scale adapters")
        
        # 4. 冻结最后的norm层
        for param in self.height_encoder.norm.parameters():
            param.requires_grad = False
        print("  ✓ Frozen final norm layer")
        
        # 5. 确保decoder完全可训练
        for param in self.decoder.parameters():
            param.requires_grad = True
        print("  ✓ Decoder remains fully trainable")
        
        # 6. 打印参数统计
        self.print_trainable_parameters()
    def print_trainable_parameters(self):
        """打印可训练参数的统计信息，用于验证冻结策略"""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        component_stats = {}
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
            
            # 统计各组件的参数情况
            component = name.split('.')[0] if '.' in name else name
            if component not in component_stats:
                component_stats[component] = {'total': 0, 'trainable': 0}
            component_stats[component]['total'] += param.numel()
            if param.requires_grad:
                component_stats[component]['trainable'] += param.numel()
        
        print("\n" + "="*60)
        print("📊 PARAMETER STATISTICS")
        print("="*60)
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Frozen Parameters:    {frozen_params:,}")
        print(f"Trainable Ratio:      {trainable_params/total_params*100:.2f}%")
        
        print("\n📋 Component Breakdown:")
        print("-"*40)
        for component, stats in component_stats.items():
            ratio = stats['trainable']/stats['total']*100
            status = "🟢" if ratio > 90 else "🟡" if ratio > 10 else "🔴"
            print(f"{status} {component:<20}: {stats['trainable']:>8,}/{stats['total']:>8,} ({ratio:>5.1f}%)")
        print("="*60 + "\n")

    def apply_freezing_strategy(self, strategy: str = "none"):
        """根据配置应用不同的冻结策略"""
        strategy = strategy.lower()
        
        if strategy == "none":
            print("🔓 No freezing applied - all parameters trainable")
            return
        elif strategy == "simple":
            self.freeze_encoder()
        elif strategy == "selective":
            self.freeze_encoder_selectively()
        else:
            raise ValueError(f"Unknown freezing strategy: {strategy}. "
                            f"Available: ['none', 'simple', 'selective']")
    def forward(self, 
               x: torch.Tensor, 
               return_multi_scale: bool = None,
               masks: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            return_multi_scale: 是否返回多尺度结果
            masks: 可选掩码
        Returns:
            如果return_multi_scale=True: dict with multiple scales
            否则: 最高分辨率的高程图
        """
        if return_multi_scale is None:
            return_multi_scale = self.use_multi_scale_output
        
        # 计算patch数量
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        
        # Height Encoder: 获取多尺度特征
        scale_features = self.height_encoder(x, masks)
        
        # Decoder: 生成多尺度高程预测
        predictions = self.decoder(scale_features, patch_h, patch_w)
        
        if return_multi_scale:
            return predictions
        else:
            # 返回最高分辨率的结果
            return predictions.get('scale_4', list(predictions.values())[-1])
    
    def compute_loss(self, 
                    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                    targets: torch.Tensor,
                    masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            # 检查损失函数类型
        if isinstance(self.loss_fn, SingleScaleLoss):
            # 单尺度损失：需要提取一个tensor
            if isinstance(predictions, dict):
                # 使用最高分辨率的预测（通常是scale_4或最后一个）
                main_prediction = predictions.get('scale_4', list(predictions.values())[-1])
            else:
                main_prediction = predictions
            
            return self.loss_fn(main_prediction, targets, masks)
        
        else:
            # 多尺度损失：需要字典格式
            if isinstance(predictions, torch.Tensor):
                # 单尺度预测，转换为字典格式
                predictions = {'scale_1': predictions}
            
            return self.loss_fn(predictions, targets, masks)

    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'encoder_type': self.encoder_type,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'img_size': self.img_size,
            'patch_size': self.patch_size,
        })
        return info


def create_depth2elevation_model(config: Dict[str, Any]) -> Depth2Elevation:
    """创建Depth2Elevation模型的工厂函数"""
    return Depth2Elevation(config)