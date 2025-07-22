import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import logging
import os
from typing import Tuple, Optional, Dict, Any
from torchvision.transforms import Compose

# 导入必要模块
try:
    from dinov2 import DINOv2
    from util.blocks import FeatureFusionBlock, _make_scratch
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False
    class DINOv2: pass
    class FeatureFusionBlock: pass
    def _make_scratch(*args, **kwargs): pass

def _make_fusion_block(features, use_bn, size=None):
    """创建特征融合块"""
    if DINOV2_AVAILABLE:
        return FeatureFusionBlock(
            features, nn.ReLU(False), deconv=False, bn=use_bn,
            expand=False, align_corners=True, size=size
        )
    else:
        return SimpleFusionBlock(features, use_bn)

class SimpleFusionBlock(nn.Module):
    """简化的特征融合块"""
    def __init__(self, features, use_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(features) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x1, x2=None, size=None):
        if x2 is not None:
            if x1.shape != x2.shape:
                x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
            x = x1 + x2
        else:
            x = x1
            
        if size is not None:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
            
        return self.relu(self.bn(self.conv(x)))

class SimpleScratch(nn.Module):
    """简化的scratch模块"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.layer1_rn = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=3, padding=1)
        self.layer2_rn = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=3, padding=1)  
        self.layer3_rn = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=3, padding=1)
        self.layer4_rn = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=3, padding=1)
        
        self.refinenet1 = SimpleFusionBlock(out_channels)
        self.refinenet2 = SimpleFusionBlock(out_channels)
        self.refinenet3 = SimpleFusionBlock(out_channels)
        self.refinenet4 = SimpleFusionBlock(out_channels)

class ScaleAdapter(nn.Module):
    """尺度适配器"""
    def __init__(self, embed_dim, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # 处理为 (B, N, C) 格式
        if x.dim() == 4:  # Conv特征 (B, C, H, W)
            B, C, H, W = x.shape
            x_reshaped = x.view(B, C, -1).permute(0, 2, 1)
        else:  # ViT特征 (B, N, C)
            x_reshaped = x
        
        # 计算和应用scale
        pooled = x_reshaped.permute(0, 2, 1)  # (B, C, N)
        scale = self.adapter(pooled).unsqueeze(1)  # (B, 1, C)
        scaled = self.norm(x_reshaped * scale * self.scale_factor)
        
        # 恢复原始形状
        if x.dim() == 4:
            scaled = scaled.permute(0, 2, 1).view(B, C, H, W)
        
        return scaled

class ScaleModulator(nn.Module):
    """尺度调制器"""
    def __init__(self, embed_dim, base_scales=[0.5, 0.75, 1.0, 1.25]):
        super().__init__()
        self.scale_adapters = nn.ModuleList([
            ScaleAdapter(embed_dim, scale) for scale in base_scales
        ])
        
    def forward(self, features_list):
        modulated_features = []
        for i, feature in enumerate(features_list):
            adapter_idx = min(i, len(self.scale_adapters) - 1)
            try:
                adapted_feature = self.scale_adapters[adapter_idx](feature)
                modulated_features.append(adapted_feature)
            except:
                modulated_features.append(feature)
        return modulated_features

class EnhancedDPTHead(nn.Module):
    """增强的DPT头部"""
    def __init__(self, in_channels, features=256, out_channels=[256, 512, 1024, 1024], enable_scale_modulator=True):
        super().__init__()
        
        self.enable_scale_modulator = enable_scale_modulator
        
        if enable_scale_modulator:
            self.scale_modulator = ScaleModulator(in_channels)
        
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, kernel_size=1)
            for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        if DINOV2_AVAILABLE:
            self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
            self.scratch.refinenet1 = _make_fusion_block(features, False)
            self.scratch.refinenet2 = _make_fusion_block(features, False)
            self.scratch.refinenet3 = _make_fusion_block(features, False)
            self.scratch.refinenet4 = _make_fusion_block(features, False)
        else:
            self.scratch = SimpleScratch(out_channels, features)

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, out_features):
        # 处理DINOv2特征：(patch_features, class_token) -> conv格式
        processed_features = []
        for feature_tuple in out_features:
            if isinstance(feature_tuple, tuple):
                patch_features, _ = feature_tuple  # 忽略class token
                B, N, C = patch_features.shape
                # 1024 patches -> 32x32 grid
                x_conv = patch_features.permute(0, 2, 1).reshape(B, C, 32, 32)
            else:
                x_conv = feature_tuple
            processed_features.append(x_conv)
        
        # 应用Scale Modulator
        try:
            modulated_features = self.scale_modulator(processed_features)
        except:
            modulated_features = processed_features
        
        # 投影和调整尺寸
        out = []
        for i, x in enumerate(modulated_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = F.interpolate(path_1, size=(448, 448), mode="bilinear", align_corners=True)
        return self.output_conv(out)

class SimpleNDSMHead(nn.Module):
    """简单的nDSM预测头"""
    def __init__(self, input_channels=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x).squeeze(1)

class FixedGAMUSNDSMPredictor(nn.Module):
    """修复的GAMUS nDSM预测模型"""
    
    def __init__(self, encoder='vits', features=256, pretrained_path=None, enable_scale_modulator=True):
        super().__init__()
        
        self.encoder = encoder
        self.enable_scale_modulator = enable_scale_modulator
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
        }
        
        if not DINOV2_AVAILABLE:
            raise ImportError("DINOv2模块不可用")
        
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_head = EnhancedDPTHead(
            self.pretrained.embed_dim, 
            features,
            enable_scale_modulator=enable_scale_modulator
        )
        self.ndsm_head = SimpleNDSMHead(input_channels=1)
        
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"加载了 {len(pretrained_dict)} 个预训练参数")
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )
        
        # 深度预测
        depth = self.depth_head(features)
        depth = F.relu(depth)
        
        # nDSM预测
        ndsm_pred = self.ndsm_head(depth)
        return ndsm_pred
    
    def freeze_encoder(self, freeze=True):
        """冻结/解冻编码器"""
        for param in self.pretrained.parameters():
            param.requires_grad = not freeze

def create_gamus_model(encoder='vits', pretrained_path=None, freeze_encoder=True, enable_scale_modulator=True):
    """创建GAMUS模型"""
    model = FixedGAMUSNDSMPredictor(
        encoder=encoder, 
        pretrained_path=pretrained_path,
        enable_scale_modulator=enable_scale_modulator
    )
    if freeze_encoder:
        model.freeze_encoder(True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计{total_params:,}, 可训练{trainable_params:,}")
    
    return model

# 测试
if __name__ == '__main__':
    try:
        model = create_gamus_model(encoder='vitb')
        test_input = torch.randn(1, 3, 448, 448)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"输入: {test_input.shape}")
        print(f"输出: {output.shape}")
        print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
        print("✓ 测试通过!")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")