import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseDepthModel(nn.Module, ABC):
    """
    深度估计模型的基础类
    所有深度/高度估计模型都应该继承这个类
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 子类必须实现"""
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算损失 - 子类必须实现"""
        pass
    
    def freeze_encoder(self):
        """冻结编码器 - 子类可以重写"""
        pass
    
    def unfreeze_encoder(self):
        """解冻编码器 - 子类可以重写"""  
        pass