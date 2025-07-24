import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class MSELoss(nn.Module):
    """MSE损失"""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = (pred - target) ** 2
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class SILoss(nn.Module):
    """Scale-Invariant Loss"""
    def __init__(self, beta: float = 0.15, epsilon: float = 1e-7):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 计算log差值
        log_pred = torch.log(pred + self.epsilon)
        log_target = torch.log(target + self.epsilon)
        g = log_pred - log_target
        
        if mask is not None:
            g_masked = g[mask.bool()]
            if len(g_masked) > 0:
                g_mean = g_masked.mean()
                g_var = g_masked.var()
            else:
                g_mean = g.mean()
                g_var = g.var()
        else:
            g_mean = g.mean()
            g_var = g.var()
        
        # 论文公式
        si_loss = 10 * torch.sqrt(g_var + self.beta * (g_mean ** 2))
        return si_loss

class GradientLoss(nn.Module):
    """梯度损失"""
    def __init__(self, lambda_grad: float = 1e-3):
        super().__init__()
        self.lambda_grad = lambda_grad
    
    def compute_gradients(self, img: torch.Tensor) -> tuple:
        """计算水平和垂直梯度"""
        grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
        return grad_x, grad_y
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 确保输入有4个维度
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # 计算梯度
        pred_grad_x, pred_grad_y = self.compute_gradients(pred)
        target_grad_x, target_grad_y = self.compute_gradients(target)
        
        # 梯度差值
        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)
        
        # 平均梯度损失
        grad_loss = self.lambda_grad * (grad_diff_x.mean() + grad_diff_y.mean())
        return grad_loss