import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class HybridSpaceLoss(nn.Module):
    """
    混合空间损失函数：结合归一化空间和真实空间的损失计算
    
    设计思路：
    1. 归一化空间损失：保证训练稳定性和快速收敛
    2. 真实空间损失：保证尺度感知和实际精度
    3. 梯度损失：保持边缘锐度和细节
    4. 动态权重：训练过程中调整各损失组件的重要性
    """
    
    def __init__(self, 
                 height_normalizer,
                 # 基础损失配置
                 base_loss_type='huber',
                 huber_delta_norm=0.05,      # 归一化空间的huber delta
                 huber_delta_real=2.0,       # 真实空间的huber delta (米)
                 
                 # 混合权重配置
                 weight_normalized=0.5,      # 归一化空间损失权重
                 weight_real=0.3,           # 真实空间损失权重  
                 weight_gradient=0.2,       # 梯度损失权重
                 
                 # 动态权重配置
                 use_dynamic_weights=True,   # 是否使用动态权重
                 warmup_epochs=20,          # 热身轮数
                 
                 # 高度感知配置
                 height_aware=True,
                 **kwargs):
        super().__init__()
        
        self.height_normalizer = height_normalizer
        self.base_loss_type = base_loss_type
        self.huber_delta_norm = huber_delta_norm
        self.huber_delta_real = huber_delta_real
        
        # 静态权重
        self.weight_normalized = weight_normalized
        self.weight_real = weight_real
        self.weight_gradient = weight_gradient
        
        # 动态权重配置
        self.use_dynamic_weights = use_dynamic_weights
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # 高度感知
        self.height_aware = height_aware
        
        # 基础损失函数
        self.huber_loss_norm = nn.HuberLoss(delta=huber_delta_norm, reduction='none')
        self.huber_loss_real = nn.HuberLoss(delta=huber_delta_real, reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
        
        # 高度范围信息
        if hasattr(height_normalizer, 'min_height'):
            self.min_height = height_normalizer.min_height
            self.max_height = height_normalizer.max_height
        else:
            self.min_height = -5.0
            self.max_height = 200.0
        
        self.height_range = self.max_height - self.min_height
        
        # 计算归一化阈值（用于高度感知权重）
        self.ground_norm_threshold = 5.0 / self.height_range
        self.low_norm_threshold = 20.0 / self.height_range  
        self.mid_norm_threshold = 50.0 / self.height_range
        self.high_norm_threshold = min(0.8, 80.0 / self.height_range)
        
        logging.info(f"混合损失初始化完成:")
        logging.info(f"  - 归一化空间权重: {weight_normalized}")
        logging.info(f"  - 真实空间权重: {weight_real}")
        logging.info(f"  - 梯度损失权重: {weight_gradient}")
        logging.info(f"  - 动态权重: {use_dynamic_weights}")
        
    def update_epoch(self, epoch):
        """更新当前epoch，用于动态权重计算"""
        self.current_epoch = epoch
        
    def get_dynamic_weights(self):
        """
        计算动态权重
        
        策略：
        - 前期（warmup阶段）：主要依赖归一化空间损失
        - 中期：逐渐增加真实空间损失权重
        - 后期：保持平衡，但略微倾向真实空间
        """
        if not self.use_dynamic_weights:
            return self.weight_normalized, self.weight_real, self.weight_gradient
            
        # 计算训练进度 [0, 1]
        progress = min(1.0, self.current_epoch / (self.warmup_epochs * 2))
        
        # 归一化空间权重：从高到中等
        w_norm = self.weight_normalized * (1.2 - 0.4 * progress)
        
        # 真实空间权重：从低到高
        w_real = self.weight_real * (0.5 + 1.0 * progress)
        
        # 梯度权重：中后期增加
        w_grad = self.weight_gradient * (0.3 + 0.7 * progress)
        
        # 归一化权重
        total = w_norm + w_real + w_grad
        return w_norm/total, w_real/total, w_grad/total
        
    def get_height_weights(self, targets_norm):
        """
        基于高度的权重计算（在归一化空间）
        
        Args:
            targets_norm: 归一化后的目标值 [0, 1]
        """
        if not self.height_aware:
            return torch.ones_like(targets_norm)
            
        weights = torch.ones_like(targets_norm)
        
        # 使用预计算的归一化阈值
        ground_mask = targets_norm <= self.ground_norm_threshold
        weights[ground_mask] *= 1.5
        
        low_mask = (targets_norm > self.ground_norm_threshold) & (targets_norm <= self.low_norm_threshold)
        weights[low_mask] *= 2.0
        
        mid_mask = (targets_norm > self.low_norm_threshold) & (targets_norm <= self.mid_norm_threshold)
        weights[mid_mask] *= 4.0
        
        high_mask = (targets_norm > self.mid_norm_threshold) & (targets_norm <= self.high_norm_threshold)
        weights[high_mask] *= 8.0
        
        very_high_mask = targets_norm > self.high_norm_threshold
        weights[very_high_mask] *= 15.0
        
        return weights
        
    def compute_normalized_loss(self, pred_norm, target_norm, mask=None):
        """计算归一化空间的损失"""
        if self.base_loss_type == 'huber':
            loss = self.huber_loss_norm(pred_norm, target_norm)
        elif self.base_loss_type == 'mse':
            loss = self.mse_loss(pred_norm, target_norm)
        elif self.base_loss_type == 'mae':
            loss = self.mae_loss(pred_norm, target_norm)
        else:
            loss = self.huber_loss_norm(pred_norm, target_norm)
            
        if mask is not None:
            loss = loss * mask
            valid_pixels = mask.sum() + 1e-8
            loss = loss.sum() / valid_pixels
        else:
            loss = loss.mean()
            
        return loss
        
    def compute_real_space_loss(self, pred_norm, target_norm, mask=None):
        """
        计算真实空间的损失
        
        关键：先反归一化到真实高度，再计算损失
        """
        # 反归一化到真实空间
        pred_real = self.denormalize_to_real(pred_norm)
        target_real = self.denormalize_to_real(target_norm)
        
        # 在真实空间计算损失
        if self.base_loss_type == 'huber':
            loss = self.huber_loss_real(pred_real, target_real)
        elif self.base_loss_type == 'mse':
            loss = self.mse_loss(pred_real, target_real)
        elif self.base_loss_type == 'mae':
            loss = self.mae_loss(pred_real, target_real)
        else:
            loss = self.huber_loss_real(pred_real, target_real)
            
        if mask is not None:
            loss = loss * mask
            valid_pixels = mask.sum() + 1e-8
            loss = loss.sum() / valid_pixels
        else:
            loss = loss.mean()
            
        return loss
        
    def compute_gradient_loss(self, pred_norm, target_norm, mask=None):
        """
        计算梯度损失，保持边缘锐度
        
        计算水平和垂直方向的梯度差异
        """
        # 计算预测值的梯度
        pred_grad_x = torch.abs(pred_norm[:, :, 1:] - pred_norm[:, :, :-1])
        pred_grad_y = torch.abs(pred_norm[:, 1:, :] - pred_norm[:, :-1, :])
        
        # 计算目标值的梯度  
        target_grad_x = torch.abs(target_norm[:, :, 1:] - target_norm[:, :, :-1])
        target_grad_y = torch.abs(target_norm[:, 1:, :] - target_norm[:, :-1, :])
        
        # 计算梯度损失
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        gradient_loss = (loss_x + loss_y) / 2
        
        return gradient_loss
        
    def denormalize_to_real(self, normalized_data):
        """将归一化数据转换回真实高度"""
        if hasattr(self.height_normalizer, 'denormalize'):
            # 如果是tensor，需要转换处理
            if torch.is_tensor(normalized_data):
                device = normalized_data.device
                shape = normalized_data.shape
                
                # 转为numpy进行反归一化
                norm_np = normalized_data.detach().cpu().numpy()
                real_np = self.height_normalizer.denormalize(norm_np)
                
                # 转回tensor
                real_tensor = torch.from_numpy(real_np).to(device).reshape(shape)
                return real_tensor
            else:
                return self.height_normalizer.denormalize(normalized_data)
        else:
            # 简单的线性反归一化
            return normalized_data * self.height_range + self.min_height
            
    def forward(self, pred, target, mask=None):
        """
        混合损失的前向计算
        
        Args:
            pred: 预测值 (归一化空间 [0,1])
            target: 目标值 (归一化空间 [0,1])  
            mask: 可选的掩码
            
        Returns:
            total_loss: 总损失
            loss_dict: 各组件损失的详细信息
        """
        # 确保输入在正确范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # 获取动态权重
        w_norm, w_real, w_grad = self.get_dynamic_weights()
        
        # 计算高度感知权重
        height_weights = self.get_height_weights(target)
        
        # 合并掩码
        if mask is not None:
            final_mask = mask * height_weights
        else:
            final_mask = height_weights
            
        # 1. 归一化空间损失
        loss_normalized = self.compute_normalized_loss(pred, target, final_mask)
        
        # 2. 真实空间损失
        loss_real = self.compute_real_space_loss(pred, target, mask)  # 这里不用height_weights
        
        # 3. 梯度损失
        loss_gradient = self.compute_gradient_loss(pred, target, mask)
        
        # 4. 总损失
        total_loss = (w_norm * loss_normalized + 
                     w_real * loss_real + 
                     w_grad * loss_gradient)
        
        # 安全检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.warning("检测到异常损失值，使用归一化损失作为备选")
            total_loss = loss_normalized
            
        # 返回详细信息用于监控
        loss_dict = {
            'total': total_loss.item(),
            'normalized': loss_normalized.item(),
            'real_space': loss_real.item(),
            'gradient': loss_gradient.item(),
            'weights': {
                'normalized': w_norm,
                'real': w_real,
                'gradient': w_grad
            },
            'epoch': self.current_epoch
        }
        
        return total_loss, loss_dict


def create_hybrid_loss(height_normalizer, **kwargs):
    """
    便利函数：创建混合空间损失
    
    Args:
        height_normalizer: 高度归一化器
        **kwargs: 其他配置参数
        
    Returns:
        HybridSpaceLoss实例
    """
    return HybridSpaceLoss(height_normalizer, **kwargs)


# 使用示例和测试
if __name__ == '__main__':
    # 创建测试数据
    batch_size, height, width = 2, 448, 448
    pred = torch.rand(batch_size, height, width)
    target = torch.rand(batch_size, height, width)
    mask = torch.ones(batch_size, height, width)
    
    # 模拟归一化器
    class MockNormalizer:
        def __init__(self):
            self.min_height = -5.0
            self.max_height = 200.0
            
        def denormalize(self, data):
            return data * (self.max_height - self.min_height) + self.min_height
    
    normalizer = MockNormalizer()
    
    # 创建混合损失
    hybrid_loss = create_hybrid_loss(
        height_normalizer=normalizer,
        base_loss_type='huber',
        weight_normalized=0.5,
        weight_real=0.3,
        weight_gradient=0.2,
        use_dynamic_weights=True,
        warmup_epochs=20
    )
    
    # 模拟训练过程
    print("=== 混合损失测试 ===")
    for epoch in [0, 10, 20, 40]:
        hybrid_loss.update_epoch(epoch)
        total_loss, loss_dict = hybrid_loss(pred, target, mask)
        
        print(f"\nEpoch {epoch}:")
        print(f"  总损失: {loss_dict['total']:.6f}")
        print(f"  归一化空间: {loss_dict['normalized']:.6f}")
        print(f"  真实空间: {loss_dict['real_space']:.6f}")
        print(f"  梯度损失: {loss_dict['gradient']:.6f}")
        print(f"  权重 - 归一化: {loss_dict['weights']['normalized']:.3f}")
        print(f"  权重 - 真实: {loss_dict['weights']['real']:.3f}")
        print(f"  权重 - 梯度: {loss_dict['weights']['gradient']:.3f}")
    
    print("\n=== 测试完成 ===")