import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

# 导入你现有的IMELE模型文件
from models.imele.imele_model import IMELEModel

class IMELEAdapter(nn.Module):
    """
    IMELE模型适配器，用于集成到对比实验框架中
    使其接口与GAMUS和Depth2Elevation保持一致
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # 提取配置参数
        self.backbone = config.get('backbone', 'resnet50')
        self.pretrained = config.get('pretrained', True)
        self.freeze_encoder = config.get('freeze_encoder', True)
        self.loss_type = config.get('loss_type', 'l1')
        
        # 创建IMELE模型配置
        imele_config = {
            'backbone': self.backbone,
            'pretrained': self.pretrained,
            'loss_type': self.loss_type
        }
        
        # 创建IMELE模型
        self.model = IMELEModel(imele_config)
        
        # 应用编码器冻结策略
        if self.freeze_encoder:
            self.model.freeze_encoder()
        
        logging.info(f"IMELE适配器创建完成:")
        logging.info(f"  Backbone: {self.backbone}")
        logging.info(f"  预训练: {self.pretrained}")
        logging.info(f"  编码器冻结: {self.freeze_encoder}")
        logging.info(f"  损失类型: {self.loss_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 统一接口"""
        return self.model(x)
    
    def freeze_encoder(self, freeze: bool = True):
        """冻结/解冻编码器"""
        if freeze:
            self.model.freeze_encoder()
        else:
            self.model.unfreeze_encoder()
        
        logging.info(f"IMELE编码器已{'冻结' if freeze else '解冻'}")
    
    def load_pretrained_weights(self, pretrained_path: str):
        """加载预训练权重 - 兼容现有接口"""
        if pretrained_path and pretrained_path.lower() != 'none':
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # 只加载匹配的权重
                self.model.load_state_dict(state_dict, strict=False)
                logging.info(f"成功加载IMELE预训练权重: {pretrained_path}")
            except Exception as e:
                logging.warning(f"加载IMELE预训练权重失败: {e}")
        else:
            logging.info("IMELE使用随机初始化权重")
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算损失 - 使用IMELE原始损失函数"""
        return self.model.compute_loss(predictions, targets, masks)
    
    def evaluate_with_imele_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                                  idx: int = 0, batch_size: int = 1) -> Dict[str, float]:
        """使用IMELE原始评估指标"""
        return self.model.evaluate_with_imele_metrics(predictions, targets, idx, batch_size)
