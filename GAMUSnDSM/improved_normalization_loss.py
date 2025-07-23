import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Union

class HeightNormalizer:
    """简化版nDSM数据归一化器"""
    
    def __init__(self, method='minmax', height_filter=None):
        """
        初始化归一化器
        
        Args:
            method: 归一化方法 ('minmax', 'percentile', 'zscore')
            height_filter: 高度过滤器 {'min_height': -5.0, 'max_height': 100.0}
        """
        self.method = method
        self.height_filter = height_filter or {'min_height': -5.0, 'max_height': 100.0}
        self.fitted = False
        
        # 统计信息
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None
        self.range_val = None
        
        # 兼容旧代码的属性
        self.global_min_h = None
        self.global_max_h = None
        self.height_range = None
        
    def fit_from_json_stats(self, stats_data: Dict[str, Any]):
        """从JSON统计信息拟合归一化器"""
        try:
            global_stats = stats_data['global_statistics']
            
            # 获取原始统计信息
            original_min = float(global_stats['min'])
            original_max = float(global_stats['max'])
            
            # 应用高度过滤
            self.min_val = max(original_min, self.height_filter['min_height'])
            self.max_val = min(original_max, self.height_filter['max_height'])
            self.mean_val = float(global_stats['mean'])
            self.std_val = float(global_stats['std'])
            self.range_val = self.max_val - self.min_val
            
            # 设置兼容属性
            self.global_min_h = self.min_val
            self.global_max_h = self.max_val
            self.height_range = self.range_val
            
            self.fitted = True
            
            if self.min_val != original_min or self.max_val != original_max:
                logging.info(f"应用高度过滤: [{original_min:.1f}, {original_max:.1f}] -> [{self.min_val:.1f}, {self.max_val:.1f}]")
            
            logging.info(f"归一化器拟合完成: 方法={self.method}, 范围=[{self.min_val:.1f}, {self.max_val:.1f}]")
            
        except Exception as e:
            raise ValueError(f"从JSON拟合归一化器失败: {e}")
    
    def fit(self, height_data):
        """从数据拟合归一化器"""
        if isinstance(height_data, torch.Tensor):
            height_data = height_data.detach().cpu().numpy()
        
        # 过滤有效数据
        valid_data = height_data[~(np.isnan(height_data) | np.isinf(height_data))]
        
        if len(valid_data) == 0:
            raise ValueError("没有有效的高度数据")
        
        # 应用高度过滤
        filtered_data = valid_data[
            (valid_data >= self.height_filter['min_height']) & 
            (valid_data <= self.height_filter['max_height'])
        ]
        
        if len(filtered_data) == 0:
            logging.warning("高度过滤后没有有效数据，使用原始数据")
            filtered_data = valid_data
        
        # 计算统计信息
        self.min_val = float(filtered_data.min())
        self.max_val = float(filtered_data.max())
        self.mean_val = float(filtered_data.mean())
        self.std_val = float(filtered_data.std())
        self.range_val = self.max_val - self.min_val
        
        # 设置兼容属性
        self.global_min_h = self.min_val
        self.global_max_h = self.max_val
        self.height_range = self.range_val
        
        self.fitted = True
        
        logging.info(f"归一化器拟合完成: {len(filtered_data)} 个数据点")
        logging.info(f"数据范围: [{self.min_val:.1f}, {self.max_val:.1f}] 米")
    
    def normalize(self, height_data):
        """归一化高度数据"""
        if not self.fitted:
            raise ValueError("归一化器未拟合")
        
        # 处理张量输入
        if isinstance(height_data, torch.Tensor):
            height_data_np = height_data.detach().cpu().numpy()
            return_tensor = True
            device = height_data.device
        else:
            height_data_np = np.array(height_data)
            return_tensor = False
            device = None
        
        # 处理无效值
        valid_mask = ~(np.isnan(height_data_np) | np.isinf(height_data_np))
        normalized = np.zeros_like(height_data_np)
        
        if not np.any(valid_mask):
            if return_tensor:
                return torch.zeros_like(height_data)
            return normalized.astype(np.float32)
        
        valid_data = height_data_np[valid_mask]
        
        # 应用高度过滤
        height_filtered = np.clip(valid_data, self.min_val, self.max_val)
        
        # 根据方法归一化
        if self.method == 'minmax':
            normalized_valid = (height_filtered - self.min_val) / (self.range_val + 1e-8)
            
        elif self.method == 'percentile':
            # 使用5%和95%分位数
            p5 = self.min_val + 0.05 * self.range_val
            p95 = self.min_val + 0.95 * self.range_val
            clipped_data = np.clip(height_filtered, p5, p95)
            normalized_valid = (clipped_data - p5) / (p95 - p5 + 1e-8)
            
        elif self.method == 'zscore':
            zscore = (height_filtered - self.mean_val) / (self.std_val + 1e-8)
            normalized_valid = np.clip(zscore, -3, 3)
            normalized_valid = (normalized_valid + 3) / 6.0
            
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")
        
        # 确保在[0,1]范围内
        normalized_valid = np.clip(normalized_valid, 0, 1)
        normalized[valid_mask] = normalized_valid
        
        if return_tensor:
            return torch.from_numpy(normalized).to(device).float()
        return normalized.astype(np.float32)
    
    def denormalize(self, normalized_data):
        """反归一化到真实高度值"""
        if not self.fitted:
            raise ValueError("归一化器未拟合")
        
        if isinstance(normalized_data, torch.Tensor):
            norm_data_np = normalized_data.detach().cpu().numpy()
            return_tensor = True
            device = normalized_data.device
        else:
            norm_data_np = np.array(normalized_data)
            return_tensor = False
            device = None
        
        norm_data_np = np.clip(norm_data_np, 0, 1)
        
        if self.method == 'minmax':
            height = norm_data_np * self.range_val + self.min_val
            
        elif self.method == 'percentile':
            p5 = self.min_val + 0.05 * self.range_val
            p95 = self.min_val + 0.95 * self.range_val
            height = norm_data_np * (p95 - p5) + p5
            
        elif self.method == 'zscore':
            zscore = (norm_data_np * 6.0) - 3.0
            height = zscore * self.std_val + self.mean_val
            
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")
        
        if return_tensor:
            return torch.from_numpy(height).to(device).float()
        return height.astype(np.float32)


class ImprovedHeightLoss(nn.Module):
    """简化版高度损失函数"""
    
    def __init__(self, 
                 loss_type='mse',
                 height_aware=True,
                 huber_delta=0.1,
                 focal_alpha=2.0,
                 weights=None,
                 ground_constraint_weight=0.2,
                 ground_threshold=0.1,
                 height_normalizer=None,  # 新增：传入归一化器
                 min_height=None,         # 新增：最小高度
                 max_height=None):        # 新增：最大高度
        """
        初始化损失函数
        
        Args:
            loss_type: 损失类型 ('mse', 'mae', 'huber', 'focal', 'combined')
            height_aware: 是否使用高度感知权重
            huber_delta: Huber损失的参数
            focal_alpha: Focal损失的参数
            weights: 组合损失的权重字典
        """
        super().__init__()
        self.loss_type = loss_type
        self.height_aware = height_aware
        self.huber_delta = huber_delta
        self.focal_alpha = focal_alpha
        self.ground_constraint_weight = ground_constraint_weight
        self.ground_threshold = ground_threshold       
        # 高度范围信息
        self.height_normalizer = height_normalizer
        if height_normalizer is not None:
            self.min_height = getattr(height_normalizer, 'global_min_h', 
                                    getattr(height_normalizer, 'min_val', min_height or -5.0))
            self.max_height = getattr(height_normalizer, 'global_max_h',
                                    getattr(height_normalizer, 'max_val', max_height or 200.0))
        else:
            self.min_height = min_height or -5.0
            self.max_height = max_height or 200.0
        
        self.height_range = self.max_height - self.min_height
        
        # 计算归一化阈值
        self.ground_norm_threshold = 5.0 / self.height_range
        self.low_norm_threshold = 20.0 / self.height_range  
        self.mid_norm_threshold = 50.0 / self.height_range
        self.high_norm_threshold = min(0.8, 80.0 / self.height_range)
        
        # 默认权重
        self.weights = weights or {
            'mse': 1.0,
            'mae': 0.3,
            'huber': 0.5,
            'focal': 0.2,
            'gradient': 0.1
        }
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
        self.huber_loss = nn.HuberLoss(delta=huber_delta, reduction='none')
        
    def get_height_weights(self, targets):
        """
        基于高度的权重计算
        
        Args:
            targets: 归一化后的目标值 [0, 1]
        """
        weights = torch.ones_like(targets)
        
        if self.height_aware:
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
    
    def compute_mse_loss(self, pred, target, mask=None):
        """计算MSE损失"""
        loss = self.mse_loss(pred, target)
        
        if mask is not None:
            loss = loss * mask
            valid_pixels = mask.sum() + 1e-8
            loss = loss.sum() / valid_pixels
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_mae_loss(self, pred, target, mask=None):
        """计算MAE损失"""
        loss = self.mae_loss(pred, target)
        
        if mask is not None:
            loss = loss * mask
            valid_pixels = mask.sum() + 1e-8
            loss = loss.sum() / valid_pixels
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_huber_loss(self, pred, target, mask=None):
        """计算Huber损失"""
        loss = self.huber_loss(pred, target)
        
        if mask is not None:
            loss = loss * mask
            valid_pixels = mask.sum() + 1e-8
            loss = loss.sum() / valid_pixels
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_focal_loss(self, pred, target, mask=None):
        """计算Focal损失"""
        mse = self.mse_loss(pred, target)
        pt = torch.exp(-mse)
        focal_weight = (1 - pt) ** self.focal_alpha
        loss = focal_weight * mse
        
        if mask is not None:
            loss = loss * mask
            valid_pixels = mask.sum() + 1e-8
            loss = loss.sum() / valid_pixels
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_gradient_loss(self, pred, target, mask=None):
        """计算梯度损失"""
        # 确保输入是4D张量
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # 计算梯度
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # 计算梯度差异
        grad_loss_x = torch.abs(pred_grad_x - target_grad_x)
        grad_loss_y = torch.abs(pred_grad_y - target_grad_y)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            mask_x = mask[:, :, :, 1:]
            mask_y = mask[:, :, 1:, :]
            
            grad_loss_x = grad_loss_x * mask_x
            grad_loss_y = grad_loss_y * mask_y
            
            valid_x = mask_x.sum() + 1e-8
            valid_y = mask_y.sum() + 1e-8
            
            grad_loss = (grad_loss_x.sum() / valid_x + grad_loss_y.sum() / valid_y) / 2
        else:
            grad_loss = (grad_loss_x.mean() + grad_loss_y.mean()) / 2
        
        return grad_loss
    
    def forward(self, pred, target, mask=None):
        """
        前向传播
        
        Args:
            pred: 预测值 [B, H, W] 或 [B, C, H, W]
            target: 真实值 [B, H, W] 或 [B, C, H, W]
            mask: 有效像素掩码 [B, H, W] 或 [B, C, H, W]
        """
        # 处理多尺度输入
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # 只使用第一个（最高分辨率）
        
        # 确保尺寸匹配
        if pred.shape != target.shape:
            min_h = min(pred.shape[-2], target.shape[-2])
            min_w = min(pred.shape[-1], target.shape[-1])
            
            if pred.dim() == 3:
                pred = pred[:, :min_h, :min_w]
                target = target[:, :min_h, :min_w]
                if mask is not None:
                    mask = mask[:, :min_h, :min_w]
            else:
                pred = pred[:, :, :min_h, :min_w]
                target = target[:, :, :min_h, :min_w]
                if mask is not None:
                    mask = mask[:, :, :min_h, :min_w]
        
        # 确保在[0,1]范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # 计算高度权重
        height_weights = self.get_height_weights(target)
        
        # 合并掩码
        if mask is not None:
            final_mask = mask * height_weights
        else:
            final_mask = height_weights
        
        # 根据损失类型计算
        if self.loss_type == 'mse':
            loss = self.compute_mse_loss(pred, target, final_mask)
            
        elif self.loss_type == 'mae':
            loss = self.compute_mae_loss(pred, target, final_mask)
            
        elif self.loss_type == 'huber':
            loss = self.compute_huber_loss(pred, target, final_mask)
            
        elif self.loss_type == 'focal':
            loss = self.compute_focal_loss(pred, target, final_mask)
            
        elif self.loss_type == 'combined':
            mse_loss = self.compute_mse_loss(pred, target, final_mask)
            mae_loss = self.compute_mae_loss(pred, target, final_mask)
            huber_loss = self.compute_huber_loss(pred, target, final_mask)
            focal_loss = self.compute_focal_loss(pred, target, final_mask)
            gradient_loss = self.compute_gradient_loss(pred, target, mask)
            
            loss = (self.weights['mse'] * mse_loss +
                   self.weights['mae'] * mae_loss +
                   self.weights['huber'] * huber_loss +
                   self.weights['focal'] * focal_loss +
                   self.weights['gradient'] * gradient_loss)
        
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
        
        # 安全检查
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning("检测到异常损失值，返回默认值")
            return torch.tensor(0.1, device=pred.device, requires_grad=True)
        
        return loss


# 便利函数
def create_height_normalizer(method='minmax', height_filter=None):
    """
    创建高度归一化器
    
    Args:
        method: 归一化方法 ('minmax', 'percentile', 'zscore')
        height_filter: 高度过滤器 {'min_height': -5.0, 'max_height': 100.0}
    """
    return HeightNormalizer(method=method, height_filter=height_filter)


def create_height_loss(loss_type='mse', height_aware=True, height_normalizer=None, 
                      min_height=None, max_height=None, **kwargs):
    """
    创建高度损失函数
    
    Args:
        loss_type: 损失类型 ('mse', 'mae', 'huber', 'focal', 'combined')
        height_aware: 是否使用高度感知权重
        height_normalizer: 高度归一化器
        min_height: 最小高度值（米）
        max_height: 最大高度值（米）
        **kwargs: 其他参数
    """
    return ImprovedHeightLoss(
        loss_type=loss_type,
        height_aware=height_aware,
        height_normalizer=height_normalizer,
        min_height=min_height,
        max_height=max_height,
        **kwargs
    )


# 使用示例
if __name__ == '__main__':
    # 创建测试数据
    pred = torch.rand(2, 448, 448)
    target = torch.rand(2, 448, 448)
    mask = torch.ones(2, 448, 448)
    
    # 测试归一化器
    print("=== 测试归一化器 ===")
    
    # 创建归一化器
    normalizer = create_height_normalizer(
        method='minmax',
        height_filter={'min_height': -5.0, 'max_height': 100.0}
    )
    
    # 模拟高度数据
    height_data = np.random.uniform(-10, 150, 1000)
    
    # 拟合归一化器
    normalizer.fit(height_data)
    
    # 测试归一化
    normalized = normalizer.normalize(height_data[:10])
    denormalized = normalizer.denormalize(normalized)
    
    print(f"原始数据: {height_data[:5]}")
    print(f"归一化后: {normalized[:5]}")
    print(f"反归一化: {denormalized[:5]}")
    
    # 测试损失函数
    print("\n=== 测试损失函数 ===")
    
    loss_functions = {
        'MSE': create_height_loss('mse'),
        'MAE': create_height_loss('mae'),
        'Huber': create_height_loss('huber'),
        'Focal': create_height_loss('focal'),
        'Combined': create_height_loss('combined')
    }
    
    for name, loss_fn in loss_functions.items():
        loss_value = loss_fn(pred, target, mask)
        print(f"{name} Loss: {loss_value.item():.6f}")
    
    print("\n=== 测试完成 ===")