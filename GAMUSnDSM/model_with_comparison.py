import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import logging
import os
from typing import Tuple, Optional, Dict, Any, Union
from torchvision.transforms import Compose

# ==================== 原有的DINOv2模型代码 ====================
# 保持原有的DINOv2相关导入和类定义
try:
    from dinov2 import DINOv2
    from util.blocks import FeatureFusionBlock, _make_scratch
    from util.transform import Resize, NormalizeImage, PrepareForNet
    DINOV2_AVAILABLE = True
    logging.info("DINOv2模块导入成功")
except ImportError as e:
    logging.debug(f"DINOv2模块导入失败: {e}")
    logging.debug("将使用简化的预训练编码器")
    DINOV2_AVAILABLE = False
    
    class DINOv2:
        pass
    class FeatureFusionBlock:
        pass
    def _make_scratch(*args, **kwargs):
        pass

# ==================== 新增：Depth2Elevation模型支持 ====================
try:
    # 导入新模型的组件（假设它们在相应的模块中）
    from depth2elevation import create_depth2elevation_model, Depth2Elevation
    DEPTH2ELEVATION_AVAILABLE = True
    logging.info("Depth2Elevation模块导入成功")
except ImportError as e:
    logging.debug(f"Depth2Elevation模块导入失败: {e}")
    DEPTH2ELEVATION_AVAILABLE = False
    
    # 提供占位符
    def create_depth2elevation_model(*args, **kwargs):
        raise ImportError("Depth2Elevation模块不可用")
    
    class Depth2Elevation:
        pass

# ==================== 新增：多尺度自适应特征聚合模块 ====================
class AdaptiveFeatureAggregation(nn.Module):
    """
    多尺度自适应特征聚合网络
    参考: FPN++ (CVPR 2019), PANet (CVPR 2018), EfficientDet (CVPR 2020)
    创新: 结合高度估测任务的特点，引入高度敏感的自适应权重学习
    """
    
    def __init__(self, feature_channels=[256, 512, 1024, 1024], 
                 out_channels=256, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        
        # 特征对齐卷积
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in feature_channels
        ])
        
        # 自适应权重学习网络
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])
        
        # 高度敏感的特征增强模块
        self.height_sensitive_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=max(1, out_channels//8)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            ) for _ in range(num_scales)
        ])
        
        # 跨尺度特征交互模块
        self.cross_scale_fusion = nn.ModuleList([
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
            for _ in range(num_scales - 1)
        ])
        
        # 最终输出卷积
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * num_scales, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1)
        )
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from different scales
        Returns:
            Aggregated feature map
        """
        # Step 1: 特征对齐
        aligned_features = []
        for i, feat in enumerate(features):
            aligned = self.lateral_convs[i](feat)
            aligned_features.append(aligned)
        
        # Step 2: 自适应权重计算
        weighted_features = []
        for i, feat in enumerate(aligned_features):
            weight = self.attention_weights[i](feat)
            weighted_feat = feat * weight
            weighted_features.append(weighted_feat)
        
        # Step 3: 高度敏感特征增强
        enhanced_features = []
        for i, feat in enumerate(weighted_features):
            enhanced = self.height_sensitive_conv[i](feat)
            enhanced = feat + enhanced  # 残差连接
            enhanced_features.append(enhanced)
        
        # Step 4: 跨尺度特征交互 (自顶向下)
        refined_features = [enhanced_features[-1]]  # 最高层特征
        
        for i in range(len(enhanced_features) - 2, -1, -1):
            # 上采样高层特征
            upsampled = F.interpolate(
                refined_features[0], 
                size=enhanced_features[i].shape[2:],
                mode='bilinear', 
                align_corners=False
            )
            
            # 特征融合
            fused = torch.cat([enhanced_features[i], upsampled], dim=1)
            refined = self.cross_scale_fusion[i](fused)
            refined_features.insert(0, refined)
        
        # Step 5: 统一尺寸并聚合
        target_size = refined_features[0].shape[2:]
        resized_features = []
        
        for feat in refined_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, 
                                   mode='bilinear', align_corners=False)
            resized_features.append(feat)
        
        # 最终聚合
        aggregated = torch.cat(resized_features, dim=1)
        output = self.output_conv(aggregated)
        
        return output

# ==================== 新增：高度感知注意力模块 ====================
class HeightAwareAttention(nn.Module):
    """高度感知的注意力机制"""
    
    def __init__(self, depth_channels=1, feature_channels=256):
        super().__init__()
        
        # 高度分析网络
        self.height_analyzer = nn.Sequential(
            nn.Conv2d(depth_channels, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 3, padding=1),  # 4个高度区间
            nn.Softmax(dim=1)
        )
        
        # 多高度区间的注意力权重
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(depth_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            ) for _ in range(4)
        ])
        
        # 高度自适应卷积
        self.height_adaptive_conv = nn.Sequential(
            nn.Conv2d(depth_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, depth_channels, 1)
        )
    
    def forward(self, depth_features):
        """
        Args:
            depth_features: 深度特征 [B, 1, H, W]
        Returns:
            enhanced_depth: 增强的深度特征
        """
        # 分析高度分布
        height_distribution = self.height_analyzer(depth_features)  # [B, 4, H, W]
        
        # 为每个高度区间计算注意力
        attention_maps = []
        for i, attention_layer in enumerate(self.attention_weights):
            attention = attention_layer(depth_features)
            # 加权当前高度区间的重要性
            weighted_attention = attention * height_distribution[:, i:i+1]
            attention_maps.append(weighted_attention)
        
        # 综合所有高度区间的注意力
        combined_attention = sum(attention_maps)
        
        # 应用注意力增强
        attended_depth = depth_features * (1 + combined_attention)
        
        # 高度自适应处理
        adaptive_residual = self.height_adaptive_conv(attended_depth)
        enhanced_depth = attended_depth + 0.1 * adaptive_residual
        
        return enhanced_depth

# ==================== 原有代码保持不变 ====================
def _make_fusion_block(features, use_bn, size=None):
    """创建特征融合块"""
    if DINOV2_AVAILABLE:
        return FeatureFusionBlock(
            features,
            nn.ReLU(False),
            deconv=False,
            bn=use_bn,
            expand=False,
            align_corners=True,
            size=size,
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
            
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SimpleScratch(nn.Module):
    """简化的scratch模块"""
    def __init__(self, in_channels_list, out_channels, groups=1):
        super().__init__()
        self.layer1_rn = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=3, padding=1)
        self.layer2_rn = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=3, padding=1)  
        self.layer3_rn = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=3, padding=1)
        self.layer4_rn = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=3, padding=1)
        
        self.refinenet1 = SimpleFusionBlock(out_channels)
        self.refinenet2 = SimpleFusionBlock(out_channels)
        self.refinenet3 = SimpleFusionBlock(out_channels)
        self.refinenet4 = SimpleFusionBlock(out_channels)

class SimplifiedDPTHead(nn.Module):
    """简化的DPT头部 - 适配448x448输入和nDSM预测"""
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_adaptive_aggregation=False):
        super().__init__()
        self.use_adaptive_aggregation = use_adaptive_aggregation  # 新增属性
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
                # 新增：条件性创建不同的特征聚合模块
        if use_adaptive_aggregation:
            logging.info("使用自适应特征聚合模块")
            self.feature_aggregation = AdaptiveFeatureAggregation(
                feature_channels=out_channels,
                out_channels=features,
                num_scales=4
            )
        else:
            logging.info("使用传统特征融合模块")
            # 保持原有的逻辑
            if DINOV2_AVAILABLE:
                self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
                self.scratch.stem_transpose = None
                self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
                self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
                self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
                self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
            else:
                self.scratch = SimpleScratch(out_channels, features)
        # if DINOV2_AVAILABLE:
        #     self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        #     self.scratch.stem_transpose = None
        #     self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        #     self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        #     self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        #     self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        # else:
        #     self.scratch = SimpleScratch(out_channels, features)

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

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if isinstance(x, tuple):
                x = x[0]
            
            if x.dim() == 3:
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            elif x.dim() == 4:
                pass
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
                # 修改：根据配置选择不同的特征聚合方式
        if self.use_adaptive_aggregation:
            # 使用新的自适应特征聚合
            path_1 = self.feature_aggregation(out)
        else:
            # 保持原有的逻辑
            layer_1, layer_2, layer_3, layer_4 = out

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)

            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # layer_1, layer_2, layer_3, layer_4 = out

        # layer_1_rn = self.scratch.layer1_rn(layer_1)
        # layer_2_rn = self.scratch.layer2_rn(layer_2)
        # layer_3_rn = self.scratch.layer3_rn(layer_3)
        # layer_4_rn = self.scratch.layer4_rn(layer_4)

        # path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        # path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        # path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        # path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = F.interpolate(path_1, size=(448, 448), mode="bilinear", align_corners=True)
        out = self.output_conv(out)

        return out

class SimpleNDSMHead(nn.Module):
    """简单的nDSM预测头"""
    def __init__(self, input_channels=1, enable_zero_output=True):
        super().__init__()
        self.enable_zero_output = enable_zero_output
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
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, kernel_size=1),
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
    
    def forward(self, x):
        x = self.layers(x)
        
        if self.enable_zero_output:
            x = F.relu(x)
            x = torch.clamp(x, 0, 1)
        else:
            x = torch.sigmoid(x)
        
        return x.squeeze(1)

class GAMUSNDSMPredictor(nn.Module):
    """GAMUS nDSM预测模型 - 针对448x448输入优化"""
    
    def __init__(self, 
                 encoder='vits',
                 features=256,
                 use_pretrained_dpt=True,
                 use_adaptive_aggregation=False,
                 use_height_attention=False,  # 新增参数
                 pretrained_path=None):
        super().__init__()
        
        self.encoder = encoder
        self.use_pretrained_dpt = use_pretrained_dpt
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
        }
        
        if DINOV2_AVAILABLE and encoder in ['vits', 'vitb', 'vitl']:
            logging.info(f"使用DINOv2编码器: {encoder}")
            self.pretrained = DINOv2(model_name=encoder)
            self.embed_dim = self.pretrained.embed_dim
        else:
            if not DINOV2_AVAILABLE:
                raise ImportError("DINOv2模块不可用，请检查安装")
            else:
                raise ValueError(f"不支持的编码器: {encoder}, 支持的编码器: vits, vitb, vitl")
        
        if use_pretrained_dpt:
            self.depth_head = SimplifiedDPTHead(
                self.embed_dim, 
                features, 
                use_bn=False,
                use_adaptive_aggregation=use_adaptive_aggregation
            )
        # 新增：高度感知注意力模块
        self.use_height_attention = use_height_attention
        if use_height_attention:
            self.height_attention = HeightAwareAttention(
                depth_channels=1,
                feature_channels=features
            )
            logging.info("启用高度感知注意力模块")
        self.ndsm_head = SimpleNDSMHead(input_channels=1)
        
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_weights(pretrained_path)
        elif pretrained_path:
            logging.warning(f"预训练文件不存在: {pretrained_path}")
    
    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重"""
        try:
            logging.info(f"正在加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint)
            else:
                state_dict = checkpoint
            
            model_dict = self.state_dict()
            pretrained_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                    logging.debug(f"匹配权重: {k} - {v.shape}")
                else:
                    logging.debug(f"跳过权重: {k} - 形状不匹配或不存在")
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            logging.info(f"成功加载预训练权重: {len(pretrained_dict)}/{len(model_dict)} 个参数")
                
        except Exception as e:
            logging.error(f"加载预训练权重失败: {e}")
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.shape[0]
        patch_h = patch_w = 32
        
        features = self.pretrained.get_intermediate_layers(
            x, 
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True
        )
        
        if self.use_pretrained_dpt:
            depth = self.depth_head(features, patch_h, patch_w)
            depth = F.relu(depth)
        else:
            last_feature = features[-1][0]
            if last_feature.dim() == 3:
                last_feature = last_feature.permute(0, 2, 1).reshape(
                    batch_size, -1, patch_h, patch_w
                )
            
            depth = F.interpolate(
                last_feature, 
                size=(448, 448), 
                mode='bilinear', 
                align_corners=False
            )
            if depth.shape[1] > 1:
                depth = F.conv2d(depth, 
                               torch.ones(1, depth.shape[1], 1, 1, device=depth.device) / depth.shape[1],
                               bias=None)
            depth = F.relu(depth)
        # 新增：应用高度感知注意力
        if self.use_height_attention:
            depth = self.height_attention(depth)
        ndsm_pred = self.ndsm_head(depth)
        
        return ndsm_pred
    
    def freeze_encoder(self, freeze=True):
        """冻结/解冻编码器"""
        for param in self.pretrained.parameters():
            param.requires_grad = not freeze
        
        if self.use_pretrained_dpt:
            for param in self.depth_head.parameters():
                param.requires_grad = not freeze
        
        logging.info(f"编码器参数已{'冻结' if freeze else '解冻'}")
    
    @torch.no_grad()
    def predict_single_image(self, image_path, output_size=None):
        """预测单张图像的nDSM"""
        self.eval()
        
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        if image.shape[:2] != (448, 448):
            image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_CUBIC)
        
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        
        normalize = Compose([
            lambda x: (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / 
                     torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        ])
        image_tensor = normalize(image_tensor)
        
        device = next(self.parameters()).device
        image_tensor = image_tensor.to(device)
        
        ndsm_pred = self.forward(image_tensor)
        ndsm_pred = ndsm_pred.cpu().numpy()[0]
        
        if output_size and output_size != (448, 448):
            ndsm_pred = cv2.resize(ndsm_pred, output_size, interpolation=cv2.INTER_LINEAR)
        
        return ndsm_pred

# ==================== 新增：Depth2Elevation适配器 ====================
class Depth2ElevationAdapter(nn.Module):
    """Depth2Elevation模型的适配器，使其与现有训练代码兼容"""
    
    def __init__(self, model_config):
        super().__init__()
        
        # 创建Depth2Elevation模型的配置
        self.d2e_config = {
            'model_config': {
                'encoder': model_config.get('encoder', 'vitb'),
                'img_size': model_config.get('img_size', 448),
                'patch_size': model_config.get('patch_size', 14),
                'pretrained_path': model_config.get('pretrained_path'),
                'freeze_encoder': model_config.get('freeze_encoder', True)
            },
            'use_multi_scale_output': model_config.get('use_multi_scale_output', False),
            'loss_config': model_config.get('loss_config', {}),
            'freezing_config': model_config.get('freezing_config', {})
        }
        
        # 创建实际的Depth2Elevation模型
        self.model = create_depth2elevation_model(self.d2e_config)
        
        # 记录是否使用多尺度输出
        self.use_multi_scale = self.d2e_config['use_multi_scale_output']
        
        logging.info(f"Depth2Elevation适配器创建完成:")
        logging.info(f"  编码器: {self.d2e_config['model_config']['encoder']}")
        logging.info(f"  多尺度输出: {self.use_multi_scale}")
    
    def forward(self, x):
        """前向传播 - 适配现有训练代码的期望输出格式"""
        # 调用Depth2Elevation模型
        output = self.model(x, return_multi_scale=self.use_multi_scale)
        
        if isinstance(output, dict):
            # 多尺度输出，返回最高分辨率的结果
            return output.get('scale_4', list(output.values())[-1])
        else:
            # 单尺度输出，直接返回
            return output
    
    def freeze_encoder(self, freeze=True):
        """冻结/解冻编码器 - 兼容现有接口"""
        if hasattr(self.model, 'freeze_encoder'):
            self.model.freeze_encoder()
        elif hasattr(self.model, 'apply_freezing_strategy'):
            strategy = 'simple' if freeze else 'none'
            self.model.apply_freezing_strategy(strategy)
        
        logging.info(f"Depth2Elevation编码器参数已{'冻结' if freeze else '解冻'}")
    
    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重 - 兼容现有接口"""
        if hasattr(self.model, 'load_pretrained_weights'):
            self.model.load_pretrained_weights(pretrained_path)
        else:
            logging.warning("Depth2Elevation模型不支持load_pretrained_weights方法")

# ==================== 统一的模型创建函数 ====================
def create_gamus_model(encoder='vits', pretrained_path=None, freeze_encoder=True, 
                      model_type='gamus',use_adaptive_aggregation=False, **kwargs):
    """
    创建模型的统一接口 - 支持多种模型类型
    
    参数:
        encoder: 编码器类型 ('vits', 'vitb', 'vitl')
        pretrained_path: 预训练权重路径
        freeze_encoder: 是否冻结编码器
        model_type: 模型类型 ('gamus', 'depth2elevation')
        **kwargs: 其他模型特定参数
    """
    
    if model_type.lower() == 'gamus':
        # 创建原有的GAMUS模型
        logging.info("创建GAMUS nDSM预测模型")
        model = GAMUSNDSMPredictor(
            encoder=encoder,
            use_pretrained_dpt=True,
            pretrained_path=pretrained_path,
            use_adaptive_aggregation=use_adaptive_aggregation,
            use_height_attention=kwargs.get('use_height_attention', False),  # 新增参数
        )
        
        if freeze_encoder:
            model.freeze_encoder(True)
            
    elif model_type.lower() == 'depth2elevation':
        # 创建Depth2Elevation模型适配器
        if not DEPTH2ELEVATION_AVAILABLE:
            raise ImportError("Depth2Elevation模块不可用，请检查安装")
        
        logging.info("创建Depth2Elevation模型")
        
        # 构建模型配置
        model_config = {
            'encoder': encoder,
            'img_size': kwargs.get('img_size', 448),
            'patch_size': kwargs.get('patch_size', 14),
            'pretrained_path': pretrained_path,
            'freeze_encoder': freeze_encoder,
            'use_multi_scale_output': kwargs.get('use_multi_scale_output', False),
            'loss_config': kwargs.get('loss_config', {}),
            'freezing_config': kwargs.get('freezing_config', {})
        }
        
        model = Depth2ElevationAdapter(model_config)
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: 'gamus', 'depth2elevation'")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"模型创建完成:")
    logging.info(f"  模型类型: {model_type}")
    logging.info(f"  编码器: {encoder}")
    logging.info(f"  总参数: {total_params:,}")
    logging.info(f"  可训练参数: {trainable_params:,}")
    logging.info(f"  参数比例: {trainable_params/total_params*100:.2f}%")
    
    return model

# 兼容性别名
SimplifiedHeightGaoFen = GAMUSNDSMPredictor
create_simple_model = create_gamus_model

# 测试代码
if __name__ == '__main__':
    # 测试GAMUS模型
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=== 测试GAMUS模型 ===")
        gamus_model = create_gamus_model(
            encoder='vits',
            model_type='gamus',
            pretrained_path='/home/hudong26/hudong26/Height/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth',
            freeze_encoder=True
        ).to(device)
        
        test_input = torch.randn(2, 3, 448, 448).to(device)
        with torch.no_grad():
            gamus_output = gamus_model(test_input)
        
        print(f"GAMUS - 输入形状: {test_input.shape}")
        print(f"GAMUS - 输出形状: {gamus_output.shape}")
        print(f"GAMUS - 输出值范围: [{gamus_output.min():.3f}, {gamus_output.max():.3f}]")
        
        # 测试Depth2Elevation模型（如果可用）
        if DEPTH2ELEVATION_AVAILABLE:
            print("\n=== 测试Depth2Elevation模型 ===")
            d2e_model = create_gamus_model(
                encoder='vitb',
                model_type='depth2elevation',
                freeze_encoder=True
            ).to(device)
            
            with torch.no_grad():
                d2e_output = d2e_model(test_input)
            
            print(f"Depth2Elevation - 输入形状: {test_input.shape}")
            print(f"Depth2Elevation - 输出形状: {d2e_output.shape}")
            print(f"Depth2Elevation - 输出值范围: [{d2e_output.min():.3f}, {d2e_output.max():.3f}]")
        else:
            print("\n=== Depth2Elevation模块不可用 ===")
        
    except Exception as e:
        print(f"模型测试失败: {e}")
        import traceback
        traceback.print_exc()