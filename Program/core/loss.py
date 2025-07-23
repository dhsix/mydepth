#!/usr/bin/env python3
"""
统一的损失函数模块
从归一化模块中分离出来，专注于损失函数的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod

from .normalizer import HeightNormalizer
from ..utils.common import setup_logger


class BaseLoss(ABC, nn.Module):
    """损失函数基类"""
    
    def __init__(self, reduction: str = 'mean', 
                 logger: Optional[logging.Logger] = None):
        super().__init__()
        self.reduction = reduction
        self.logger = logger or setup_logger('loss')
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"不支持的reduction方式: {reduction}")
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算损失的核心方法"""
        pass
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        return self.compute_loss(predictions, targets, mask)
    
    def _apply_reduction(self, loss: torch.Tensor, 
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """应用reduction"""
        if mask is not None:
            loss = loss * mask
            valid_pixels = mask.sum()
            
            if self.reduction == 'mean':
                return loss.sum() / (valid_pixels + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
            else:  # none
                return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:  # none
                return loss


class MSELoss(BaseLoss):
    """均方误差损失"""
    
    def __init__(self, reduction: str = 'mean', 
                 logger: Optional[logging.Logger] = None):
        super().__init__(reduction, logger)
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = self.mse_loss(predictions, targets)
        return self._apply_reduction(loss, mask)


class MAELoss(BaseLoss):
    """平均绝对误差损失"""
    
    def __init__(self, reduction: str = 'mean', 
                 logger: Optional[logging.Logger] = None):
        super().__init__(reduction, logger)
        self.mae_loss = nn.L1Loss(reduction='none')
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = self.mae_loss(predictions, targets)
        return self._apply_reduction(loss, mask)


class HuberLoss(BaseLoss):
    """Huber损失"""
    
    def __init__(self, delta: float = 0.1, reduction: str = 'mean', 
                 logger: Optional[logging.Logger] = None):
        super().__init__(reduction, logger)
        self.huber_loss = nn.HuberLoss(delta=delta, reduction='none')
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = self.huber_loss(predictions, targets)
        return self._apply_reduction(loss, mask)


class FocalLoss(BaseLoss):
    """Focal损失（用于回归任务）"""
    
    def __init__(self, alpha: float = 2.0, reduction: str = 'mean', 
                 logger: Optional[logging.Logger] = None):
        super().__init__(reduction, logger)
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mse = self.mse_loss(predictions, targets)
        pt = torch.exp(-mse)
        focal_weight = (1 - pt) ** self.alpha
        loss = focal_weight * mse
        return self._apply_reduction(loss, mask)


class GradientLoss(BaseLoss):
    """梯度损失"""
    
    def __init__(self, reduction: str = 'mean', 
                 logger: Optional[logging.Logger] = None):
        super().__init__(reduction, logger)
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 确保输入是4D张量
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # 计算梯度
        pred_grad_x = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]
        target_grad_x = targets[:, :, :, 1:] - targets[:, :, :, :-1]
        
        pred_grad_y = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
        target_grad_y = targets[:, :, 1:, :] - targets[:, :, :-1, :]
        
        # 计算梯度差异
        grad_loss_x = torch.abs(pred_grad_x - target_grad_x)
        grad_loss_y = torch.abs(pred_grad_y - target_grad_y)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            mask_x = mask[:, :, :, 1:]
            mask_y = mask[:, :, 1:, :]
            
            grad_loss_x = self._apply_reduction(grad_loss_x, mask_x)
            grad_loss_y = self._apply_reduction(grad_loss_y, mask_y)
            
            return (grad_loss_x + grad_loss_y) / 2
        else:
            return (grad_loss_x.mean() + grad_loss_y.mean()) / 2


class HeightAwareLoss(BaseLoss):
    """高度感知损失"""
    
    def __init__(self, base_loss: BaseLoss, 
                 height_normalizer: Optional[HeightNormalizer] = None,
                 min_height: float = -5.0, max_height: float = 200.0,
                 reduction: str = 'mean', logger: Optional[logging.Logger] = None):
        super().__init__(reduction, logger)
        self.base_loss = base_loss
        self.height_normalizer = height_normalizer
        self.min_height = min_height
        self.max_height = max_height
        
        # 从归一化器获取高度范围信息
        if height_normalizer is not None:
            self.min_height = getattr(height_normalizer, 'global_min_h', 
                                    getattr(height_normalizer, 'min_val', min_height))
            self.max_height = getattr(height_normalizer, 'global_max_h',
                                    getattr(height_normalizer, 'max_val', max_height))
        
        self.height_range = self.max_height - self.min_height
        
        # 计算归一化阈值
        self._compute_normalized_thresholds()
    
    def _compute_normalized_thresholds(self):
        """计算归一化后的高度阈值"""
        self.ground_norm_threshold = 5.0 / self.height_range
        self.low_norm_threshold = 20.0 / self.height_range  
        self.mid_norm_threshold = 50.0 / self.height_range
        self.high_norm_threshold = min(0.8, 80.0 / self.height_range)
    
    def get_height_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """基于高度的权重计算"""
        weights = torch.ones_like(targets)
        
        # 使用预计算的归一化阈值
        ground_mask = targets <= self.ground_norm_threshold
        weights[ground_mask] *= 1.5
        
        low_mask = (targets > self.ground_norm_threshold) & (targets <= self.low_norm_threshold)
        weights[low_mask] *= 2.0
        
        mid_mask = (targets > self.low_norm_threshold) & (targets <= self.mid_norm_threshold)
        weights[mid_mask] *= 4.0
        
        high_mask = (targets > self.mid_norm_threshold) & (targets <= self.high_norm_threshold)
        weights[high_mask] *= 8.0
        
        very_high_mask = targets > self.high_norm_threshold
        weights[very_high_mask] *= 15.0
        
        return weights
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 计算高度权重
        height_weights = self.get_height_weights(targets)
        
        # 合并掩码
        if mask is not None:
            final_mask = mask * height_weights
        else:
            final_mask = height_weights
        
        # 使用基础损失函数计算
        return self.base_loss.compute_loss(predictions, targets, final_mask)


class CombinedLoss(BaseLoss):
    """组合损失函数"""
    
    def __init__(self, losses: Dict[str, BaseLoss], 
                 weights: Dict[str, float],
                 reduction: str = 'mean', 
                 logger: Optional[logging.Logger] = None):
        super().__init__(reduction, logger)
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        
        # 验证权重
        for loss_name in losses.keys():
            if loss_name not in weights:
                self.logger.warning(f"损失 {loss_name} 没有指定权重，使用默认权重1.0")
                self.weights[loss_name] = 1.0
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        total_loss = 0.0
        
        for loss_name, loss_fn in self.losses.items():
            weight = self.weights[loss_name]
            loss_value = loss_fn.compute_loss(predictions, targets, mask)
            total_loss += weight * loss_value
        
        return total_loss


class ImprovedHeightLoss(nn.Module):
    """改进的nDSM高度损失函数（兼容性包装）"""
    
    def __init__(self, 
                 loss_type: str = 'mse',
                 height_aware: bool = True,
                 huber_delta: float = 0.1,
                 focal_alpha: float = 2.0,
                 weights: Optional[Dict[str, float]] = None,
                 height_normalizer: Optional[HeightNormalizer] = None,
                 min_height: Optional[float] = None,
                 max_height: Optional[float] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化改进的高度损失函数
        
        Args:
            loss_type: 损失类型 ('mse', 'mae', 'huber', 'focal', 'combined')
            height_aware: 是否使用高度感知权重
            huber_delta: Huber损失的参数
            focal_alpha: Focal损失的参数
            weights: 组合损失的权重字典
            height_normalizer: 高度归一化器
            min_height: 最小高度
            max_height: 最大高度
            logger: 日志记录器
        """
        super().__init__()
        self.loss_type = loss_type
        self.height_aware = height_aware
        self.logger = logger or setup_logger('height_loss')
        
        # 默认权重
        if weights is None:
            weights = {
                'mse': 1.0,
                'mae': 0.3,
                'huber': 0.5,
                'focal': 0.2,
                'gradient': 0.1
            }
        self.weights = weights
        
        # 创建基础损失函数
        if loss_type == 'mse':
            base_loss = MSELoss()
        elif loss_type == 'mae':
            base_loss = MAELoss()
        elif loss_type == 'huber':
            base_loss = HuberLoss(delta=huber_delta)
        elif loss_type == 'focal':
            base_loss = FocalLoss(alpha=focal_alpha)
        elif loss_type == 'combined':
            losses = {
                'mse': MSELoss(),
                'mae': MAELoss(),
                'huber': HuberLoss(delta=huber_delta),
                'focal': FocalLoss(alpha=focal_alpha),
                'gradient': GradientLoss()
            }
            base_loss = CombinedLoss(losses, weights)
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
        
        # 如果启用高度感知，包装基础损失
        if height_aware:
            self.loss_fn = HeightAwareLoss(
                base_loss=base_loss,
                height_normalizer=height_normalizer,
                min_height=min_height or -5.0,
                max_height=max_height or 200.0,
                logger=logger
            )
        else:
            self.loss_fn = base_loss
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            predictions: 预测值 [B, H, W] 或 [B, C, H, W]
            targets: 真实值 [B, H, W] 或 [B, C, H, W]
            mask: 有效像素掩码 [B, H, W] 或 [B, C, H, W]
        """
        # 处理多尺度输入
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]  # 只使用第一个（最高分辨率）
        
        # 确保尺寸匹配
        if predictions.shape != targets.shape:
            min_h = min(predictions.shape[-2], targets.shape[-2])
            min_w = min(predictions.shape[-1], targets.shape[-1])
            
            if predictions.dim() == 3:
                predictions = predictions[:, :min_h, :min_w]
                targets = targets[:, :min_h, :min_w]
                if mask is not None:
                    mask = mask[:, :min_h, :min_w]
            else:
                predictions = predictions[:, :, :min_h, :min_w]
                targets = targets[:, :, :min_h, :min_w]
                if mask is not None:
                    mask = mask[:, :, :min_h, :min_w]
        
        # 确保在[0,1]范围内
        predictions = torch.clamp(predictions, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        
        # 计算损失
        loss = self.loss_fn.compute_loss(predictions, targets, mask)
        
        # 安全检查
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning("检测到异常损失值，返回默认值")
            return torch.tensor(0.1, device=predictions.device, requires_grad=True)
        
        return loss


# 便利函数
def create_height_loss(loss_type: str = 'mse', 
                      height_aware: bool = True, 
                      height_normalizer: Optional[HeightNormalizer] = None, 
                      min_height: Optional[float] = None, 
                      max_height: Optional[float] = None, 
                      **kwargs) -> ImprovedHeightLoss:
    """
    创建高度损失函数的便利函数
    
    Args:
        loss_type: 损失类型
        height_aware: 是否使用高度感知权重
        height_normalizer: 高度归一化器
        min_height: 最小高度值（米）
        max_height: 最大高度值（米）
        **kwargs: 其他参数
    
    Returns:
        高度损失函数实例
    """
    return ImprovedHeightLoss(
        loss_type=loss_type,
        height_aware=height_aware,
        height_normalizer=height_normalizer,
        min_height=min_height,
        max_height=max_height,
        **kwargs
    )


def create_loss_from_config(config: Dict[str, Any], 
                           height_normalizer: Optional[HeightNormalizer] = None) -> ImprovedHeightLoss:
    """
    从配置创建损失函数
    
    Args:
        config: 损失函数配置
        height_normalizer: 高度归一化器
    
    Returns:
        损失函数实例
    """
    return ImprovedHeightLoss(
        height_normalizer=height_normalizer,
        **config
    )


# 使用示例
if __name__ == '__main__':
    # 设置日志
    logger = setup_logger('loss_test')
    
    # 创建测试数据
    batch_size, height, width = 2, 448, 448
    predictions = torch.rand(batch_size, height, width)
    targets = torch.rand(batch_size, height, width)
    mask = torch.ones(batch_size, height, width)
    
    # 测试不同损失函数
    loss_configs = [
        {'loss_type': 'mse', 'height_aware': False},
        {'loss_type': 'mae', 'height_aware': False},
        {'loss_type': 'huber', 'height_aware': False},
        {'loss_type': 'focal', 'height_aware': False},
        {'loss_type': 'mse', 'height_aware': True},
        {'loss_type': 'combined', 'height_aware': True}
    ]
    
    logger.info("测试不同损失函数:")
    
    for config in loss_configs:
        try:
            loss_fn = create_height_loss(**config)
            loss_value = loss_fn(predictions, targets, mask)
            
            config_str = f"{config['loss_type']}" + ("+height_aware" if config['height_aware'] else "")
            logger.info(f"  {config_str}: {loss_value.item():.6f}")
            
        except Exception as e:
            logger.error(f"测试配置 {config} 失败: {e}")
    
    logger.info("损失函数测试完成")