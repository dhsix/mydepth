import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .base_losses import MSELoss, SILoss, GradientLoss

class MultiScaleLoss(nn.Module):
    """多尺度损失 - 基于论文Equations (1)-(6)"""
    
    def __init__(self, 
                 gamma: float = 1.0,      # L_ai权重
                 delta: float = 1.0,      # L_si权重  
                 mu: float = 0.05,        # L_grad权重
                 beta: float = 0.15,      # SI loss中的beta
                 epsilon: float = 1e-7,   # log计算中的epsilon
                 lambda_grad: float = 1e-3):  # 梯度损失权重
        super().__init__()
        
        self.gamma = gamma
        self.delta = delta
        self.mu = mu
        
        # 各个损失组件
        self.mse_loss = MSELoss()
        self.si_loss = SILoss(beta=beta, epsilon=epsilon)
        self.grad_loss = GradientLoss(lambda_grad=lambda_grad)
    
    def compute_lai(self, 
                   predictions: Dict[str, torch.Tensor], 
                   targets: torch.Tensor, 
                   masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """L_ai: MSE Loss - Equation (1)(2)"""
        total_loss = 0
        N = len(predictions)
        
        for pred_key, pred_map in predictions.items():
            # 将target调整到当前尺度
            target_scaled = F.interpolate(
                targets.unsqueeze(1), 
                size=pred_map.shape[-2:],
                mode='bilinear', 
                align_corners=True
            ).squeeze(1)
            
            # 处理mask
            mask_scaled = None
            if masks is not None:
                mask_scaled = F.interpolate(
                    masks.unsqueeze(1).float(), 
                    size=pred_map.shape[-2:],
                    mode='nearest'
                ).squeeze(1)
            
            # 计算MSE损失
            mse = self.mse_loss(pred_map, target_scaled, mask_scaled)
            total_loss += mse
            
        return total_loss / N
    
    def compute_lsi(self, 
                   predictions: Dict[str, torch.Tensor], 
                   targets: torch.Tensor, 
                   masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """L_si: Scale-Invariant Loss - Equations (3)(4)"""
        total_loss = 0
        N = len(predictions)
        
        for pred_key, pred_map in predictions.items():
            # 调整target尺度
            target_scaled = F.interpolate(
                targets.unsqueeze(1), 
                size=pred_map.shape[-2:],
                mode='bilinear', 
                align_corners=True
            ).squeeze(1)
            
            # 处理mask
            mask_scaled = None
            if masks is not None:
                mask_scaled = F.interpolate(
                    masks.unsqueeze(1).float(), 
                    size=pred_map.shape[-2:],
                    mode='nearest'
                ).squeeze(1)
            
            # 计算SI损失
            si_loss = self.si_loss(pred_map, target_scaled, mask_scaled)
            total_loss += si_loss
            
        return total_loss / N
    
    def compute_lgrad(self, 
                     predictions: Dict[str, torch.Tensor], 
                     targets: torch.Tensor, 
                     masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """L_grad: Gradient Loss - Equation (5)"""
        total_loss = 0
        N = len(predictions)
        
        for pred_key, pred_map in predictions.items():
            # 调整target尺度
            target_scaled = F.interpolate(
                targets.unsqueeze(1), 
                size=pred_map.shape[-2:],
                mode='bilinear', 
                align_corners=True
            ).squeeze(1)
            
            # 计算梯度损失
            grad_loss = self.grad_loss(pred_map, target_scaled, masks)
            total_loss += grad_loss
            
        return total_loss / N
    
    def forward(self, 
               predictions: Dict[str, torch.Tensor], 
               targets: torch.Tensor,
               masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict, 多尺度预测结果 {'scale_1': tensor, 'scale_2': tensor, ...}
            targets: 真实高程图 [B, H, W]
            masks: outlier filtering mask [B, H, W] (可选)
        Returns:
            loss_dict: 包含各项损失的字典
        """
        lai = self.compute_lai(predictions, targets, masks)
        lsi = self.compute_lsi(predictions, targets, masks)
        lgrad = self.compute_lgrad(predictions, targets, masks)
        
        # 论文公式(6): L = γL_ai + δL_si + μL_grad
        total_loss = self.gamma * lai + self.delta * lsi + self.mu * lgrad
        
        loss_dict = {
            'total_loss': total_loss,
            'lai': lai,
            'lsi': lsi, 
            'lgrad': lgrad
        }
        
        return loss_dict

class SingleScaleLoss(nn.Module):
    """单尺度损失，用于对比模型"""
    
    def __init__(self, loss_type: str = 'mse',delta: float = 1.0):
        super().__init__()
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss(delta=delta)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, 
               predictions: torch.Tensor, 
               targets: torch.Tensor,
               masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 模型预测 [B, H, W] 
            targets: 真实标签 [B, H, W]
            masks: 可选掩码 [B, H, W]
        """
        if masks is not None:
            # 应用掩码
            valid_mask = masks.bool()
            if valid_mask.sum() > 0:
                loss = self.criterion(predictions[valid_mask], targets[valid_mask])
            else:
                loss = self.criterion(predictions, targets)
        else:
            loss = self.criterion(predictions, targets)
        
        return {
            'total_loss': loss,
            'main_loss': loss
        }

def get_loss_function(loss_config: Dict[str, any]) -> nn.Module:
    """损失函数工厂函数"""
    loss_type = loss_config.get('type', 'single_scale_loss')
    
    if loss_type == 'multi_scale_loss':
        return MultiScaleLoss(
            gamma=loss_config.get('gamma', 1.0),
            delta=loss_config.get('delta', 1.0),
            mu=loss_config.get('mu', 0.05),
            beta=loss_config.get('beta', 0.15),
            epsilon=loss_config.get('epsilon', 1e-7),
            lambda_grad=loss_config.get('lambda_grad', 1e-3)
        )
    elif loss_type == 'single_scale_loss':
        return SingleScaleLoss(
            loss_type=loss_config.get('criterion', 'mse')
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")