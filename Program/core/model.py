#!/usr/bin/env python3
"""
统一的模型模块
整合所有模型架构定义，消除重复代码，提供统一的模型接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import logging
import os
from typing import Tuple, Optional, Dict, Any, List, Union
from torchvision.transforms import Compose
from abc import ABC, abstractmethod

from ..utils.common import setup_logger, get_device_info
from ..utils.model_utils import count_parameters, log_model_info

# 尝试导入DINOv2模块
try:
    from ..models.dinov2 import DINOv2
    from ..models.components import FeatureFusionBlock, _make_scratch
    DINOV2_AVAILABLE = True
    logging.info("DINOv2模块导入成功")
except ImportError as e:
    logging.debug(f"DINOv2模块导入失败: {e}")
    logging.debug("将使用简化的预训练编码器")
    DINOV2_AVAILABLE = False
    
    # 提供占位符以避免NameError
    class DINOv2:
        pass
    
    class FeatureFusionBlock:
        pass
    
    def _make_scratch(*args, **kwargs):
        pass


class BaseModel(ABC, nn.Module):
    """模型基类"""
    
    def __init__(self, model_name: str = "BaseModel", 
                 logger: Optional[logging.Logger] = None):
        super().__init__()
        self.model_name = model_name
        self.logger = logger or setup_logger('model')
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        param_info = count_parameters(self)
        return {
            'model_name': self.model_name,
            'parameters': param_info,
            'device': next(self.parameters()).device
        }
    
    def log_model_info(self):
        """记录模型信息"""
        log_model_info(self, self.model_name, self.logger)
    
    def freeze_layers(self, layer_patterns: List[str]):
        """冻结指定层"""
        frozen_count = 0
        for name, param in self.named_parameters():
            for pattern in layer_patterns:
                if pattern in name:
                    param.requires_grad = False
                    frozen_count += 1
                    break
        
        self.logger.info(f"冻结了 {frozen_count} 个参数")
    
    def unfreeze_layers(self, layer_patterns: List[str]):
        """解冻指定层"""
        unfrozen_count = 0
        for name, param in self.named_parameters():
            for pattern in layer_patterns:
                if pattern in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    break
        
        self.logger.info(f"解冻了 {unfrozen_count} 个参数")


class SimpleFusionBlock(nn.Module):
    """简化的特征融合块"""
    
    def __init__(self, features: int, use_bn: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(features) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, 
                size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if x2 is not None:
            # 融合两个特征
            if x1.shape != x2.shape:
                x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
            x = x1 + x2
        else:
            x = x1
            
        if size is not None:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
            
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SimpleScratch(nn.Module):
    """简化的scratch模块"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int, groups: int = 1):
        super().__init__()
        self.layer1_rn = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=3, padding=1)
        self.layer2_rn = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=3, padding=1)  
        self.layer3_rn = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=3, padding=1)
        self.layer4_rn = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=3, padding=1)
        
        self.refinenet1 = SimpleFusionBlock(out_channels)
        self.refinenet2 = SimpleFusionBlock(out_channels)
        self.refinenet3 = SimpleFusionBlock(out_channels)
        self.refinenet4 = SimpleFusionBlock(out_channels)


class DPTHead(nn.Module):
    """统一的DPT头部实现"""
    
    def __init__(self, in_channels: int, features: int = 256, use_bn: bool = False, 
                 out_channels: List[int] = None, target_size: Tuple[int, int] = (448, 448)):
        super().__init__()
        
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]
        
        self.target_size = target_size
        
        # 投影层
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, kernel_size=1)
            for out_channel in out_channels
        ])
        
        # 调整尺寸层
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])
        
        # Scratch模块
        if DINOV2_AVAILABLE:
            self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
            self.scratch.stem_transpose = None
            self.scratch.refinenet1 = FeatureFusionBlock(features, nn.ReLU(False), 
                                                        deconv=False, bn=use_bn, expand=False, align_corners=True)
            self.scratch.refinenet2 = FeatureFusionBlock(features, nn.ReLU(False), 
                                                        deconv=False, bn=use_bn, expand=False, align_corners=True)
            self.scratch.refinenet3 = FeatureFusionBlock(features, nn.ReLU(False), 
                                                        deconv=False, bn=use_bn, expand=False, align_corners=True)
            self.scratch.refinenet4 = FeatureFusionBlock(features, nn.ReLU(False), 
                                                        deconv=False, bn=use_bn, expand=False, align_corners=True)
        else:
            self.scratch = SimpleScratch(out_channels, features)
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, features: List[torch.Tensor], patch_h: int, patch_w: int) -> torch.Tensor:
        out = []
        for i, x in enumerate(features):
            if isinstance(x, tuple):
                x = x[0]  # 移除class token（如果存在）
            
            # 确保正确的维度处理
            if x.dim() == 3:  # (B, N, C)
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            elif x.dim() == 4:  # 已经是(B, C, H, W)
                pass
            
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
        
        # 直接输出到目标尺寸
        out = F.interpolate(path_1, size=self.target_size, mode="bilinear", align_corners=True)
        out = self.output_conv(out)
        
        return out


class NDSMHead(nn.Module):
    """nDSM预测头"""
    
    def __init__(self, input_channels: int = 1, target_size: Tuple[int, int] = (448, 448),
                 enable_zero_output: bool = True):
        super().__init__()
        self.target_size = target_size
        self.enable_zero_output = enable_zero_output
        
        self.layers = nn.Sequential(
            # 保持目标分辨率的卷积
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入尺寸正确
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        x = self.layers(x)
        
        if self.enable_zero_output:
            # 使用ReLU确保非负，然后clamp到[0,1]
            x = F.relu(x)
            x = torch.clamp(x, 0, 1)
        else:
            # 使用Sigmoid方式
            x = torch.sigmoid(x)
        
        return x.squeeze(1)  # 移除通道维度


class BasicCNNEncoder(nn.Module):
    """基础CNN编码器（当DINOv2不可用时的备选方案）"""
    
    def __init__(self, input_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            # Stage 1
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Stage 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 4
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # 中间层索引（模拟ViT的中间层）
        self.intermediate_layer_idx = [0, 1, 2, 3]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def get_intermediate_layers(self, x: torch.Tensor, 
                               layer_indices: List[int],
                               return_class_token: bool = True) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """获取中间层特征（模拟DINOv2接口）"""
        features = []
        current_x = x
        
        # 分阶段提取特征
        stages = [
            self.encoder[:4],   # Stage 1
            self.encoder[4:7],  # Stage 2  
            self.encoder[7:10], # Stage 3
            self.encoder[10:]   # Stage 4
        ]
        
        for i, stage in enumerate(stages):
            current_x = stage(current_x)
            if i in layer_indices:
                # 返回格式：(特征, class_token)
                # 对于CNN，没有class token，所以返回None
                features.append((current_x, None))
        
        return features


class GAMUSNDSMPredictor(BaseModel):
    """GAMUS nDSM预测模型"""
    
    def __init__(self, 
                 encoder: str = 'vitb',
                 features: int = 256,
                 use_pretrained_dpt: bool = True,
                 pretrained_path: Optional[str] = None,
                 target_size: Tuple[int, int] = (448, 448),
                 logger: Optional[logging.Logger] = None):
        """
        初始化GAMUS nDSM预测模型
        
        Args:
            encoder: 编码器类型 ('vits', 'vitb', 'vitl', 'basic_cnn')
            features: 特征维度
            use_pretrained_dpt: 是否使用预训练DPT
            pretrained_path: 预训练模型路径
            target_size: 目标输出尺寸
            logger: 日志记录器
        """
        super().__init__("GAMUS_nDSM_Predictor", logger)
        
        self.encoder = encoder
        self.use_pretrained_dpt = use_pretrained_dpt
        self.target_size = target_size
        
        # ViT编码器层索引
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'basic_cnn': [0, 1, 2, 3]
        }
        
        # 选择编码器
        self._initialize_encoder()
        
        # 初始化头部
        self._initialize_heads(features)
        
        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_weights(pretrained_path)
        elif pretrained_path:
            self.logger.warning(f"预训练文件不存在: {pretrained_path}")
        
        # 记录模型信息
        self.log_model_info()
    
    def _initialize_encoder(self):
        """初始化编码器"""
        if DINOV2_AVAILABLE and self.encoder in ['vits', 'vitb', 'vitl']:
            # 使用DINOv2编码器
            self.logger.info(f"使用DINOv2编码器: {self.encoder}")
            self.pretrained = DINOv2(model_name=self.encoder)
            self.embed_dim = self.pretrained.embed_dim
        elif self.encoder == 'basic_cnn':
            # 使用基础CNN编码器
            self.logger.info("使用基础CNN编码器")
            self.pretrained = BasicCNNEncoder()
            self.embed_dim = self.pretrained.embed_dim
        else:
            # 如果DINOv2不可用，回退到基础CNN
            if not DINOV2_AVAILABLE:
                self.logger.warning("DINOv2不可用，回退到基础CNN编码器")
                self.pretrained = BasicCNNEncoder()
                self.embed_dim = self.pretrained.embed_dim
                self.encoder = 'basic_cnn'
            else:
                raise ValueError(f"不支持的编码器: {self.encoder}")
    
    def _initialize_heads(self, features: int):
        """初始化预测头"""
        if self.use_pretrained_dpt and self.encoder != 'basic_cnn':
            # 使用DPT头
            self.depth_head = DPTHead(
                self.embed_dim, 
                features, 
                use_bn=False,
                target_size=self.target_size
            )
        else:
            # 使用简单的nDSM头
            self.depth_head = None
        
        # nDSM预测头
        input_channels = 1 if self.use_pretrained_dpt else self.embed_dim
        self.ndsm_head = NDSMHead(
            input_channels=input_channels,
            target_size=self.target_size
        )
    
    def load_pretrained_weights(self, pretrained_path: str):
        """加载预训练权重"""
        try:
            self.logger.info(f"正在加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint)
            else:
                state_dict = checkpoint
            
            # 只加载匹配的权重
            model_dict = self.state_dict()
            pretrained_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                    self.logger.debug(f"匹配权重: {k} - {v.shape}")
                else:
                    self.logger.debug(f"跳过权重: {k} - 形状不匹配或不存在")
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            self.logger.info(f"成功加载预训练权重: {len(pretrained_dict)}/{len(model_dict)} 个参数")
                
        except Exception as e:
            self.logger.error(f"加载预训练权重失败: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = x.shape[0]
        
        # 计算patch尺寸
        if self.encoder in ['vits', 'vitb', 'vitl']:
            # DINOv2: 448x448输入，patch大小14x14，所以patch_h=patch_w=32
            patch_h = patch_w = self.target_size[0] // 14
        else:
            # 基础CNN编码器
            patch_h = patch_w = self.target_size[0] // 16
        
        # 特征提取
        if hasattr(self.pretrained, 'get_intermediate_layers'):
            features = self.pretrained.get_intermediate_layers(
                x, 
                self.intermediate_layer_idx[self.encoder],
                return_class_token=True
            )
        else:
            # 基础CNN编码器的情况
            features = self.pretrained.get_intermediate_layers(
                x, 
                self.intermediate_layer_idx[self.encoder],
                return_class_token=True
            )
        
        if self.use_pretrained_dpt and self.depth_head is not None:
            # 使用DPT头
            depth = self.depth_head(features, patch_h, patch_w)
            depth = F.relu(depth)  # 确保非负
        else:
            # 简化版本：直接处理最后一层特征
            if self.encoder == 'basic_cnn':
                last_feature = features[-1][0]  # CNN特征
            else:
                last_feature = features[-1][0]  # 移除class token
            
            if last_feature.dim() == 3:  # (B, N, C)
                last_feature = last_feature.permute(0, 2, 1).reshape(
                    batch_size, -1, patch_h, patch_w
                )
            
            depth = F.interpolate(
                last_feature, 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # 简单投影到单通道
            if depth.shape[1] > 1:
                depth = F.conv2d(depth, 
                               torch.ones(1, depth.shape[1], 1, 1, device=depth.device) / depth.shape[1],
                               bias=None)
            depth = F.relu(depth)
        
        # nDSM预测头
        ndsm_pred = self.ndsm_head(depth)
        
        return ndsm_pred
    
    def freeze_encoder(self, freeze: bool = True):
        """冻结/解冻编码器"""
        for param in self.pretrained.parameters():
            param.requires_grad = not freeze
        
        if self.use_pretrained_dpt and hasattr(self, 'depth_head'):
            for param in self.depth_head.parameters():
                param.requires_grad = not freeze
        
        status = '冻结' if freeze else '解冻'
        self.logger.info(f"编码器参数已{status}")
    
    @torch.no_grad()
    def predict_single_image(self, image_path: Union[str, np.ndarray], 
                           output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """预测单张图像的nDSM"""
        self.eval()
        
        # 加载图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # 预处理
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        
        # 标准化 (ImageNet标准)
        normalize = Compose([
            lambda x: (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / 
                     torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        ])
        image_tensor = normalize(image_tensor)
        
        # 推理
        device = next(self.parameters()).device
        image_tensor = image_tensor.to(device)
        
        ndsm_pred = self.forward(image_tensor)
        ndsm_pred = ndsm_pred.cpu().numpy()[0]
        
        # 调整输出尺寸
        if output_size and output_size != self.target_size:
            ndsm_pred = cv2.resize(ndsm_pred, output_size, interpolation=cv2.INTER_LINEAR)
        
        return ndsm_pred


class ModelConfig:
    """模型配置类"""
    
    def __init__(self, 
                 encoder: str = 'vitb',
                 features: int = 256,
                 use_pretrained_dpt: bool = True,
                 pretrained_path: Optional[str] = None,
                 target_size: Tuple[int, int] = (448, 448),
                 freeze_encoder: bool = True):
        self.encoder = encoder
        self.features = features
        self.use_pretrained_dpt = use_pretrained_dpt
        self.pretrained_path = pretrained_path
        self.target_size = target_size
        self.freeze_encoder = freeze_encoder
    
    def create_model(self, logger: Optional[logging.Logger] = None) -> GAMUSNDSMPredictor:
        """创建模型"""
        model = GAMUSNDSMPredictor(
            encoder=self.encoder,
            features=self.features,
            use_pretrained_dpt=self.use_pretrained_dpt,
            pretrained_path=self.pretrained_path,
            target_size=self.target_size,
            logger=logger
        )
        
        if self.freeze_encoder:
            model.freeze_encoder(True)
        
        return model


# 便利函数
def create_gamus_model(encoder: str = 'vitb', 
                      pretrained_path: Optional[str] = None, 
                      freeze_encoder: bool = True,
                      target_size: Tuple[int, int] = (448, 448),
                      logger: Optional[logging.Logger] = None) -> GAMUSNDSMPredictor:
    """创建GAMUS nDSM预测模型的便利函数"""
    config = ModelConfig(
        encoder=encoder,
        pretrained_path=pretrained_path,
        freeze_encoder=freeze_encoder,
        target_size=target_size
    )
    
    return config.create_model(logger)


def create_model_from_config(config: Dict[str, Any], 
                           logger: Optional[logging.Logger] = None) -> GAMUSNDSMPredictor:
    """从配置字典创建模型"""
    model_config = ModelConfig(**config)
    return model_config.create_model(logger)


# 兼容性别名
SimplifiedHeightGaoFen = GAMUSNDSMPredictor
create_simple_model = create_gamus_model


# 使用示例和测试代码
if __name__ == '__main__':
    # 设置日志
    logger = setup_logger('model_test')
    
    # 测试模型创建
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 测试不同编码器
        encoders = ['basic_cnn']
        if DINOV2_AVAILABLE:
            encoders.extend(['vits', 'vitb'])
        
        for encoder in encoders:
            logger.info(f"测试编码器: {encoder}")
            
            try:
                model = create_gamus_model(
                    encoder=encoder,
                    freeze_encoder=True,
                    logger=logger
                ).to(device)
                
                # 测试前向传播
                test_input = torch.randn(2, 3, 448, 448).to(device)
                with torch.no_grad():
                    output = model(test_input)
                
                logger.info(f"  输入形状: {test_input.shape}")
                logger.info(f"  输出形状: {output.shape}")
                logger.info(f"  输出值范围: [{output.min():.3f}, {output.max():.3f}]")
                
                # 测试模型信息
                model_info = model.get_model_info()
                logger.info(f"  模型参数: {model_info['parameters']}")
                
                del model  # 清理内存
                
            except Exception as e:
                logger.error(f"编码器 {encoder} 测试失败: {e}")
        
        logger.info("所有模型测试完成")
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
