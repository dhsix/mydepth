import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from ..base_model import BaseDepthModel
from . import util as imele_util
from .modules import E_resnet, E_densenet, E_senet, D2, MFF, R
from . import resnet, densenet, senet

class IMELEModel(BaseDepthModel):
    """IMELE模型适配BaseDepthModel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 配置参数
        self.backbone = config.get('backbone', 'resnet50')
        self.pretrained = config.get('pretrained', True)
        self.loss_type = config.get('loss_type', 'l1')
        
        # 根据backbone设置参数
        backbone_configs = {
            'resnet50': {
                'num_features': 2048,
                'block_channel': [256, 512, 1024, 2048]
            },
            'densenet161': {
                'num_features': 2208, 
                'block_channel': [96, 384, 1056, 2208]
            },
            'senet154': {
                'num_features': 2048,
                'block_channel': [256, 512, 1024, 2048] 
            }
        }
        
        if self.backbone not in backbone_configs:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
            
        backbone_config = backbone_configs[self.backbone]
        self.num_features = backbone_config['num_features']
        self.block_channel = backbone_config['block_channel']
        
        # 构建模型
        self._build_model()
        
    def _build_model(self):
        """构建IMELE模型组件"""
        # 创建encoder
        if self.backbone == 'resnet50':
            original_model = resnet.resnet50(pretrained=self.pretrained)
            self.encoder = E_resnet(original_model, self.num_features)
        elif self.backbone == 'densenet161':
            original_model = densenet.densenet161(pretrained=self.pretrained)
            self.encoder = E_densenet(original_model, self.num_features)
        elif self.backbone == 'senet154':
            original_model = senet.senet154(pretrained='imagenet' if self.pretrained else None)
            self.encoder = E_senet(original_model, self.num_features)
        
        # 创建decoder等组件
        self.decoder = D2(num_features=self.num_features)
        self.mff = MFF(self.block_channel)
        self.refinement = R(self.block_channel)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        # Encoder
        if self.backbone == 'senet154':
            x_block0, x_block1, x_block2, x_block3, x_block4 = self.encoder(x)
        else:
            x_block1, x_block2, x_block3, x_block4 = self.encoder(x)
            # 对于resnet和densenet，使用x_block1作为x_block0
            x_block0 = x_block1
        
        # Decoder
        x_decoder = self.decoder(x_block0, x_block1, x_block2, x_block3, x_block4)
        
        # Multi-scale Feature Fusion
        x_mff = self.mff(x_block0, x_block1, x_block2, x_block3, x_block4, 
                        [x_decoder.size(2), x_decoder.size(3)])
        
        # Refinement
        out = self.refinement(torch.cat((x_decoder, x_mff), 1))
        
        return out
    
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor,
                    masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算损失函数"""
        
        # 处理NaN值
        pred_clean, target_clean, nanMask, nValidElement = imele_util.setNanToZero(predictions, targets)
        
        losses = {}
        
        if self.loss_type == 'l1':
            losses['l1_loss'] = nn.L1Loss()(pred_clean, target_clean)
            losses['total_loss'] = losses['l1_loss']
        elif self.loss_type == 'mse':
            losses['mse_loss'] = nn.MSELoss()(pred_clean, target_clean)
            losses['total_loss'] = losses['mse_loss']
        elif self.loss_type == 'combined':
            # 使用IMELE原始的组合损失
            l1_loss = nn.L1Loss()(pred_clean, target_clean)
            mse_loss = nn.MSELoss()(pred_clean, target_clean)
            losses['l1_loss'] = l1_loss
            losses['mse_loss'] = mse_loss
            losses['total_loss'] = l1_loss + 0.1 * mse_loss  # 可调权重
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
            
        return losses
    
    def freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def evaluate_with_imele_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                  idx: int = 0, batch_size: int = 1) -> Dict[str, float]:
        """使用IMELE原始的评估指标"""
        return imele_util.evaluateError(predictions, targets, idx, batch_size)