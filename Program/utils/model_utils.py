#!/usr/bin/env python3
"""
模型相关工具模块
统一管理模型创建、加载、保存、检查点管理等功能
"""

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple, List

from .common import setup_logger, ensure_dir, format_time


def create_model_config(encoder: str = 'vitb', 
                       pretrained_path: Optional[str] = None,
                       freeze_encoder: bool = True,
                       use_pretrained_dpt: bool = True,
                       **kwargs) -> Dict[str, Any]:
    """
    创建模型配置
    
    Args:
        encoder: 编码器类型
        pretrained_path: 预训练模型路径
        freeze_encoder: 是否冻结编码器
        use_pretrained_dpt: 是否使用预训练DPT
        **kwargs: 其他配置参数
    
    Returns:
        模型配置字典
    """
    config = {
        'encoder': encoder,
        'pretrained_path': pretrained_path,
        'freeze_encoder': freeze_encoder,
        'use_pretrained_dpt': use_pretrained_dpt,
        **kwargs
    }
    
    return config


def count_parameters(model: nn.Module, only_trainable: bool = False) -> Dict[str, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        only_trainable: 是否只统计可训练参数
    
    Returns:
        参数统计字典
    """
    if only_trainable:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params = total_params
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
    }


def log_model_info(model: nn.Module, model_name: str = 'Model',
                  logger: Optional[logging.Logger] = None) -> None:
    """
    记录模型信息
    
    Args:
        model: PyTorch模型
        model_name: 模型名称
        logger: 日志记录器
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    param_info = count_parameters(model)
    
    logger.info(f"{model_name} 信息:")
    logger.info(f"  总参数: {param_info['total_params']:,}")
    logger.info(f"  可训练参数: {param_info['trainable_params']:,}")
    logger.info(f"  冻结参数: {param_info['frozen_params']:,}")
    logger.info(f"  可训练比例: {param_info['trainable_ratio']:.1%}")


def load_checkpoint(checkpoint_path: str, device: torch.device,
                   logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    统一的检查点加载函数
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 目标设备
        logger: 日志记录器
    
    Returns:
        检查点字典
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    logger.info(f"正在加载检查点: {checkpoint_path}")
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 记录检查点信息
        if isinstance(checkpoint, dict):
            logger.info(f"检查点信息:")
            if 'epoch' in checkpoint:
                logger.info(f"  训练轮次: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                logger.info(f"  验证损失: {checkpoint['val_loss']:.6f}")
            if 'loss' in checkpoint:
                logger.info(f"  损失: {checkpoint['loss']:.6f}")
            if 'timestamp' in checkpoint:
                logger.info(f"  保存时间: {checkpoint['timestamp']}")
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                logger.info(f"  最佳指标:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"    {key}: {value:.4f}")
        
        logger.info("✓ 检查点加载成功")
        return checkpoint
        
    except Exception as e:
        logger.error(f"检查点加载失败: {e}")
        raise


def load_model_weights(model: nn.Module, checkpoint: Union[str, Dict[str, Any]],
                      device: torch.device, strict: bool = False,
                      logger: Optional[logging.Logger] = None) -> nn.Module:
    """
    加载模型权重
    
    Args:
        model: PyTorch模型
        checkpoint: 检查点路径或字典
        device: 目标设备
        strict: 是否严格匹配权重
        logger: 日志记录器
    
    Returns:
        加载权重后的模型
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    # 加载检查点
    if isinstance(checkpoint, str):
        checkpoint = load_checkpoint(checkpoint, device, logger)
    
    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 加载权重
    try:
        # 统计匹配的权重
        model_dict = model.state_dict()
        matched_keys = []
        unmatched_keys = []
        
        for key, value in state_dict.items():
            if key in model_dict and model_dict[key].shape == value.shape:
                matched_keys.append(key)
            else:
                unmatched_keys.append(key)
        
        # 加载匹配的权重
        if not strict:
            # 过滤不匹配的权重
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in matched_keys}
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)
        
        logger.info(f"权重加载结果:")
        logger.info(f"  匹配权重: {len(matched_keys)}/{len(model_dict)}")
        if unmatched_keys and not strict:
            logger.info(f"  跳过权重: {len(unmatched_keys)} 个")
            if len(unmatched_keys) <= 5:
                logger.debug(f"  跳过的权重: {unmatched_keys}")
        
        logger.info("✓ 模型权重加载成功")
        return model
        
    except Exception as e:
        logger.error(f"模型权重加载失败: {e}")
        raise


def save_checkpoint(epoch: int, model: nn.Module, optimizer: Optimizer,
                   loss: float, save_dir: str, scheduler: _LRScheduler = None,
                   metrics: Dict[str, float] = None, is_best: bool = False,
                   checkpoint_interval: int = 10, keep_latest: int = 5,
                   logger: Optional[logging.Logger] = None) -> str:
    """
    统一的检查点保存函数
    
    Args:
        epoch: 训练轮次
        model: PyTorch模型
        optimizer: 优化器
        loss: 当前损失
        save_dir: 保存目录
        scheduler: 学习率调度器
        metrics: 指标字典
        is_best: 是否为最佳模型
        checkpoint_interval: 检查点保存间隔
        keep_latest: 保留最新检查点数量
        logger: 日志记录器
    
    Returns:
        保存的检查点路径
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    ensure_dir(save_dir)
    
    # 创建检查点
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'dataset_type': 'GAMUS_nDSM'
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
        # 添加最佳指标信息
        checkpoint['best_metrics'] = {
            'mae': metrics.get('mae', float('inf')),
            'rmse': metrics.get('rmse', float('inf')),
            'r2': metrics.get('r2', -float('inf'))
        }
    
    # 保存最新检查点
    latest_path = os.path.join(save_dir, 'latest_gamus_model.pth')
    torch.save(checkpoint, latest_path)
    logger.info(f"保存最新检查点: {latest_path}")
    
    # 保存最佳模型
    saved_path = latest_path
    if is_best:
        best_path = os.path.join(save_dir, 'best_gamus_model.pth')
        torch.save(checkpoint, best_path)
        logger.info(f"★ 保存最佳模型: {best_path} (Loss: {loss:.4f})")
        saved_path = best_path
    
    # 定期保存检查点
    if epoch % checkpoint_interval == 0:
        epoch_path = os.path.join(save_dir, f'gamus_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        logger.info(f"保存周期性检查点: {epoch_path}")
    
    # 清理旧检查点
    _cleanup_checkpoints(save_dir, keep_latest, logger)
    
    return saved_path


def _cleanup_checkpoints(save_dir: str, keep_latest: int,
                        logger: Optional[logging.Logger] = None) -> None:
    """
    清理旧的检查点文件
    
    Args:
        save_dir: 保存目录
        keep_latest: 保留数量
        logger: 日志记录器
    """
    try:
        # 查找周期性检查点
        checkpoint_files = []
        for file in os.listdir(save_dir):
            if file.startswith('gamus_epoch_') and file.endswith('.pth'):
                checkpoint_files.append(file)
        
        if len(checkpoint_files) > keep_latest:
            # 按修改时间排序
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
            
            # 删除旧文件
            for old_file in checkpoint_files[:-keep_latest]:
                old_path = os.path.join(save_dir, old_file)
                os.remove(old_path)
                if logger:
                    logger.debug(f"清理旧检查点: {old_file}")
                    
    except Exception as e:
        if logger:
            logger.warning(f"清理检查点失败: {e}")


def freeze_model_layers(model: nn.Module, freeze_patterns: List[str] = None,
                       unfreeze_patterns: List[str] = None,
                       logger: Optional[logging.Logger] = None) -> None:
    """
    冻结模型层
    
    Args:
        model: PyTorch模型
        freeze_patterns: 要冻结的层名称模式列表
        unfreeze_patterns: 要解冻的层名称模式列表
        logger: 日志记录器
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    frozen_count = 0
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        should_freeze = False
        
        # 检查冻结模式
        if freeze_patterns:
            for pattern in freeze_patterns:
                if pattern in name:
                    should_freeze = True
                    break
        
        # 检查解冻模式
        if unfreeze_patterns:
            for pattern in unfreeze_patterns:
                if pattern in name:
                    should_freeze = False
                    break
        
        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            unfrozen_count += 1
    
    logger.info(f"模型层冻结结果:")
    logger.info(f"  冻结参数: {frozen_count}")
    logger.info(f"  可训练参数: {unfrozen_count}")


def validate_model_output(model: nn.Module, input_shape: Tuple[int, ...],
                         device: torch.device, logger: Optional[logging.Logger] = None) -> bool:
    """
    验证模型输出
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状 (batch_size, channels, height, width)
        device: 设备
        logger: 日志记录器
    
    Returns:
        是否验证通过
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    try:
        model.eval()
        
        # 创建测试输入
        test_input = torch.randn(*input_shape).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(test_input)
        
        # 检查输出
        if output is None:
            logger.error("模型输出为None")
            return False
        
        if torch.isnan(output).any():
            logger.error("模型输出包含NaN")
            return False
        
        if torch.isinf(output).any():
            logger.error("模型输出包含Inf")
            return False
        
        logger.info(f"模型验证通过:")
        logger.info(f"  输入形状: {input_shape}")
        logger.info(f"  输出形状: {output.shape}")
        logger.info(f"  输出范围: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"模型验证失败: {e}")
        return False


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...] = None) -> Dict[str, Any]:
    """
    获取模型摘要信息
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状 (不包括batch_size)
    
    Returns:
        模型摘要字典
    """
    summary = {
        'model_name': model.__class__.__name__,
        'parameters': count_parameters(model)
    }
    
    # 统计层数
    total_layers = 0
    layer_types = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            total_layers += 1
            layer_type = module.__class__.__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    summary['total_layers'] = total_layers
    summary['layer_types'] = layer_types
    
    # 尝试计算模型大小
    if input_shape:
        try:
            dummy_input = torch.randn(1, *input_shape)
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            summary['model_size_mb'] = model_size / (1024 * 1024)
        except:
            pass
    
    return summary


def compare_model_weights(model1: nn.Module, model2: nn.Module,
                         logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    比较两个模型的权重差异
    
    Args:
        model1: 模型1
        model2: 模型2
        logger: 日志记录器
    
    Returns:
        比较结果字典
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    # 找到共同的权重
    common_keys = set(state_dict1.keys()) & set(state_dict2.keys())
    unique_keys1 = set(state_dict1.keys()) - common_keys
    unique_keys2 = set(state_dict2.keys()) - common_keys
    
    # 计算权重差异
    differences = {}
    total_diff = 0.0
    
    for key in common_keys:
        w1 = state_dict1[key]
        w2 = state_dict2[key]
        
        if w1.shape == w2.shape:
            diff = torch.norm(w1 - w2).item()
            differences[key] = diff
            total_diff += diff
        else:
            logger.warning(f"权重 {key} 形状不匹配: {w1.shape} vs {w2.shape}")
    
    result = {
        'common_weights': len(common_keys),
        'unique_weights_model1': len(unique_keys1),
        'unique_weights_model2': len(unique_keys2),
        'total_difference': total_diff,
        'max_difference': max(differences.values()) if differences else 0.0,
        'min_difference': min(differences.values()) if differences else 0.0,
        'avg_difference': total_diff / len(differences) if differences else 0.0
    }
    
    logger.info(f"模型权重比较结果:")
    logger.info(f"  共同权重: {result['common_weights']}")
    logger.info(f"  模型1独有: {result['unique_weights_model1']}")
    logger.info(f"  模型2独有: {result['unique_weights_model2']}")
    logger.info(f"  平均差异: {result['avg_difference']:.6f}")
    logger.info(f"  最大差异: {result['max_difference']:.6f}")
    
    return result


def estimate_training_time(model: nn.Module, dataset_size: int, batch_size: int,
                          num_epochs: int, device: torch.device,
                          logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """
    估算训练时间
    
    Args:
        model: PyTorch模型
        dataset_size: 数据集大小
        batch_size: 批次大小
        num_epochs: 训练轮数
        device: 设备
        logger: 日志记录器
    
    Returns:
        时间估算结果
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    logger.info("正在估算训练时间...")
    
    model.train()
    model.to(device)
    
    # 创建测试输入
    test_input = torch.randn(batch_size, 3, 448, 448).to(device)
    test_target = torch.randn(batch_size, 448, 448).to(device)
    
    # 测试前向传播时间
    forward_times = []
    for _ in range(10):
        start_time = time.time()
        with torch.no_grad():
            _ = model(test_input)
        forward_times.append(time.time() - start_time)
    
    avg_forward_time = sum(forward_times) / len(forward_times)
    
    # 估算完整批次时间（包括反向传播）
    estimated_batch_time = avg_forward_time * 2.5  # 经验值
    
    # 计算总时间
    batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
    total_batches = batches_per_epoch * num_epochs
    estimated_total_time = total_batches * estimated_batch_time
    
    result = {
        'avg_forward_time': avg_forward_time,
        'estimated_batch_time': estimated_batch_time,
        'batches_per_epoch': batches_per_epoch,
        'total_batches': total_batches,
        'estimated_total_time_seconds': estimated_total_time,
        'estimated_total_time_hours': estimated_total_time / 3600
    }
    
    logger.info(f"训练时间估算:")
    logger.info(f"  每批次时间: {estimated_batch_time:.3f} 秒")
    logger.info(f"  每轮批次数: {batches_per_epoch}")
    logger.info(f"  总批次数: {total_batches}")
    logger.info(f"  估算总时间: {format_time(estimated_total_time)}")
    
    return result


def safe_load_model_for_inference(model_class, checkpoint_path: str, device: torch.device,
                                 model_kwargs: Dict[str, Any] = None,
                                 logger: Optional[logging.Logger] = None):
    """
    安全加载模型用于推理
    
    Args:
        model_class: 模型类
        checkpoint_path: 检查点路径
        device: 设备
        model_kwargs: 模型初始化参数
        logger: 日志记录器
    
    Returns:
        加载好的模型
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    model_kwargs = model_kwargs or {}
    
    try:
        # 加载检查点
        checkpoint = load_checkpoint(checkpoint_path, device, logger)
        
        # 创建模型
        model = model_class(**model_kwargs)
        
        # 加载权重
        model = load_model_weights(model, checkpoint, device, strict=False, logger=logger)
        
        # 移动到设备并设置为评估模式
        model = model.to(device)
        model.eval()
        
        # 验证模型
        if validate_model_output(model, (1, 3, 448, 448), device, logger):
            logger.info("✓ 模型加载并验证成功")
            return model
        else:
            raise ValueError("模型验证失败")
            
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise


def create_optimizer_and_scheduler(model: nn.Module, learning_rate: float = 1e-5,
                                  weight_decay: float = 1e-4,
                                  scheduler_type: str = 'plateau',
                                  scheduler_kwargs: Dict[str, Any] = None,
                                  logger: Optional[logging.Logger] = None):
    """
    创建优化器和学习率调度器
    
    Args:
        model: PyTorch模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        scheduler_type: 调度器类型
        scheduler_kwargs: 调度器参数
        logger: 日志记录器
    
    Returns:
        优化器和调度器
    """
    if logger is None:
        logger = setup_logger('model_utils')
    
    # 创建优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        raise ValueError("模型没有可训练参数")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 创建学习率调度器
    scheduler_kwargs = scheduler_kwargs or {}
    
    if scheduler_type == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True,
            **scheduler_kwargs
        )
    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6,
            **scheduler_kwargs
        )
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=20,
            gamma=0.1,
            **scheduler_kwargs
        )
    else:
        scheduler = None
    
    param_count = sum(p.numel() for p in trainable_params)
    logger.info(f"优化器和调度器创建完成:")
    logger.info(f"  优化器: AdamW (lr={learning_rate}, wd={weight_decay})")
    logger.info(f"  调度器: {scheduler_type}")
    logger.info(f"  可训练参数: {param_count:,}")
    
    return optimizer, scheduler