import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import logging
import os
from typing import Tuple, Optional, Dict, Any
from torchvision.transforms import Compose

# 简化版本的必要导入 - 适配GAMUS数据集
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
    
    # 提供占位符以避免NameError
    class DINOv2:
        pass
    
    class FeatureFusionBlock:
        pass
    
    def _make_scratch(*args, **kwargs):
        pass

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
        # 简化版本的融合块
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
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024]):
        super().__init__()

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
            self.scratch.stem_transpose = None
            self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        else:
            self.scratch = SimpleScratch(out_channels, features)

        # 输出层 - 针对nDSM预测优化
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
            nn.Sigmoid()  # 输出[0,1]范围，适合归一化的nDSM
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
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

        # 直接输出到448x448
        out = F.interpolate(path_1, size=(448, 448), mode="bilinear", align_corners=True)
        out = self.output_conv(out)

        return out

class SimpleNDSMHead(nn.Module):
    """简单的nDSM预测头"""
    def __init__(self, input_channels=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            # 保持448x448分辨率的卷积
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
            nn.Sigmoid()  # 输出[0,1]范围，适合归一化的nDSM
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
        return self.layers(x).squeeze(1)  # 移除通道维度

class GAMUSNDSMPredictor(nn.Module):
    """GAMUS nDSM预测模型 - 针对448x448输入优化"""
    
    def __init__(self, 
                 encoder='vits',
                 features=256,
                 use_pretrained_dpt=True,
                 pretrained_path=None):
        super().__init__()
        
        self.encoder = encoder
        self.use_pretrained_dpt = use_pretrained_dpt
        
        # ViT编码器层索引
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
        }
        
        # 选择编码器 - 只使用DINOv2
        if DINOV2_AVAILABLE and encoder in ['vits', 'vitb', 'vitl']:
            # 使用DINOv2编码器
            logging.info(f"使用DINOv2编码器: {encoder}")
            self.pretrained = DINOv2(model_name=encoder)
            self.embed_dim = self.pretrained.embed_dim
        else:
            # 如果DINOv2不可用或编码器不支持，抛出错误
            if not DINOV2_AVAILABLE:
                raise ImportError("DINOv2模块不可用，请检查安装")
            else:
                raise ValueError(f"不支持的编码器: {encoder}, 支持的编码器: vits, vitb, vitl")
        
        if use_pretrained_dpt:
            # 使用预训练的DPT头
            self.depth_head = SimplifiedDPTHead(
                self.embed_dim, 
                features, 
                use_bn=False
            )
        
        # 简单的nDSM预测头
        self.ndsm_head = SimpleNDSMHead(input_channels=1)
        
        # 加载预训练权重
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
            
            # 只加载匹配的权重
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
        
        # 448x448输入，patch大小14x14，所以patch_h=patch_w=32
        patch_h = patch_w = 32
        
        # 特征提取
        features = self.pretrained.get_intermediate_layers(
            x, 
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True
        )
        
        if self.use_pretrained_dpt:
            # 使用DPT头
            depth = self.depth_head(features, patch_h, patch_w)
            depth = F.relu(depth)  # 确保非负
        else:
            # 简化版本：直接处理最后一层特征
            last_feature = features[-1][0]  # 移除class token
            if last_feature.dim() == 3:  # (B, N, C)
                last_feature = last_feature.permute(0, 2, 1).reshape(
                    batch_size, -1, patch_h, patch_w
                )
            
            depth = F.interpolate(
                last_feature, 
                size=(448, 448), 
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
        
        # 加载图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # 预处理
        if image.shape[:2] != (448, 448):
            image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_CUBIC)
        
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
        if output_size and output_size != (448, 448):
            ndsm_pred = cv2.resize(ndsm_pred, output_size, interpolation=cv2.INTER_LINEAR)
        
        return ndsm_pred

# 便利函数
def create_gamus_model(encoder='vits', pretrained_path=None, freeze_encoder=True):
    """创建GAMUS nDSM预测模型"""
    model = GAMUSNDSMPredictor(
        encoder=encoder,
        use_pretrained_dpt=True,
        pretrained_path=pretrained_path
    )
    
    if freeze_encoder:
        model.freeze_encoder(True)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"GAMUS模型创建完成:")
    logging.info(f"  编码器: {model.encoder}")
    logging.info(f"  总参数: {total_params:,}")
    logging.info(f"  可训练参数: {trainable_params:,}")
    logging.info(f"  参数比例: {trainable_params/total_params*100:.2f}%")
    
    return model

# 兼容性别名
SimplifiedHeightGaoFen = GAMUSNDSMPredictor
create_simple_model = create_gamus_model

# 测试代码
if __name__ == '__main__':
    # 测试模型
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = create_gamus_model(
            encoder='vits',
            pretrained_path='/home/hudong26/hudong26/Height/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth',
            freeze_encoder=True
        ).to(device)
        
        # 测试前向传播
        test_input = torch.randn(2, 3, 448, 448).to(device)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出值范围: [{output.min():.3f}, {output.max():.3f}]")
        
    except Exception as e:
        print(f"模型测试失败: {e}")
        import traceback
        traceback.print_exc()