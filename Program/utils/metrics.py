#!/usr/bin/env python3
"""
统一的指标计算模块
整合所有评估指标的计算功能，避免重复代码
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import gc
import time

from .common import setup_logger, Timer


@dataclass
class MetricsConfig:
    """指标计算配置"""
    calculate_correlation: bool = True
    calculate_accuracy: bool = True
    calculate_layered_error: bool = True
    max_samples_for_metrics: int = 50000
    sample_for_correlation: int = 10000
    accuracy_thresholds: List[float] = None
    height_layers: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.accuracy_thresholds is None:
            self.accuracy_thresholds = [1.0, 2.0, 5.0, 10.0]
        
        if self.height_layers is None:
            self.height_layers = {
                'ground': (-5.0, 5.0),
                'low_buildings': (5.0, 20.0),
                'mid_buildings': (20.0, 50.0),
                'high_buildings': (50.0, float('inf'))
            }


class BaseMetricsCalculator:
    """基础指标计算器"""
    
    def __init__(self, config: MetricsConfig = None, 
                 logger: Optional[logging.Logger] = None):
        self.config = config or MetricsConfig()
        self.logger = logger or setup_logger('metrics')
        
    def calculate_basic_metrics(self, predictions: np.ndarray, 
                              targets: np.ndarray) -> Dict[str, float]:
        """
        计算基础回归指标
        
        Args:
            predictions: 预测值数组
            targets: 真实值数组
        
        Returns:
            基础指标字典
        """
        # 移除无效值
        valid_mask = self._get_valid_mask(predictions, targets)
        if np.sum(valid_mask) == 0:
            return self._get_default_metrics()
        
        valid_preds = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        # 基础指标
        mae = mean_absolute_error(valid_targets, valid_preds)
        mse = mean_squared_error(valid_targets, valid_preds)
        rmse = np.sqrt(mse)
        
        # R²
        r2 = r2_score(valid_targets, valid_preds)
        r2 = max(-10, min(1, r2))  # 限制R²范围
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'valid_samples': int(np.sum(valid_mask))
        }
    
    def calculate_correlation_metrics(self, predictions: np.ndarray,
                                    targets: np.ndarray) -> Dict[str, float]:
        """
        计算相关性指标
        
        Args:
            predictions: 预测值数组
            targets: 真实值数组
        
        Returns:
            相关性指标字典
        """
        if not self.config.calculate_correlation:
            return {}
        
        # 移除无效值并采样
        valid_mask = self._get_valid_mask(predictions, targets)
        if np.sum(valid_mask) < 10:
            return {'pearson_r': 0.0, 'pearson_p': 1.0, 
                   'spearman_r': 0.0, 'spearman_p': 1.0}
        
        valid_preds = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        # 采样以提高计算效率
        if len(valid_preds) > self.config.sample_for_correlation:
            indices = np.random.choice(len(valid_preds), 
                                     self.config.sample_for_correlation, 
                                     replace=False)
            valid_preds = valid_preds[indices]
            valid_targets = valid_targets[indices]
        
        try:
            # Pearson相关性
            if len(np.unique(valid_targets)) > 1 and len(np.unique(valid_preds)) > 1:
                pearson_r, pearson_p = pearsonr(valid_targets, valid_preds)
                spearman_r, spearman_p = spearmanr(valid_targets, valid_preds)
            else:
                pearson_r = pearson_p = spearman_r = spearman_p = 0.0
            
            # 处理NaN值
            if np.isnan(pearson_r):
                pearson_r = 0.0
            if np.isnan(pearson_p):
                pearson_p = 1.0
            if np.isnan(spearman_r):
                spearman_r = 0.0
            if np.isnan(spearman_p):
                spearman_p = 1.0
                
            return {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p)
            }
            
        except Exception as e:
            self.logger.warning(f"相关性计算失败: {e}")
            return {'pearson_r': 0.0, 'pearson_p': 1.0,
                   'spearman_r': 0.0, 'spearman_p': 1.0}
    
    def calculate_accuracy_metrics(self, predictions: np.ndarray,
                                 targets: np.ndarray) -> Dict[str, float]:
        """
        计算精度指标
        
        Args:
            predictions: 预测值数组
            targets: 真实值数组
        
        Returns:
            精度指标字典
        """
        if not self.config.calculate_accuracy:
            return {}
        
        valid_mask = self._get_valid_mask(predictions, targets)
        if np.sum(valid_mask) == 0:
            return {f'height_accuracy_{t}m': 0.0 for t in self.config.accuracy_thresholds}
        
        valid_preds = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        errors = np.abs(valid_preds - valid_targets)
        
        accuracy_metrics = {}
        for threshold in self.config.accuracy_thresholds:
            accuracy = np.mean(errors <= threshold)
            accuracy_metrics[f'height_accuracy_{threshold}m'] = float(accuracy)
        
        return accuracy_metrics
    
    def calculate_layered_error(self, predictions: np.ndarray,
                              targets: np.ndarray) -> Dict[str, float]:
        """
        计算分层误差
        
        Args:
            predictions: 预测值数组
            targets: 真实值数组
        
        Returns:
            分层误差字典
        """
        if not self.config.calculate_layered_error:
            return {}
        
        valid_mask = self._get_valid_mask(predictions, targets)
        if np.sum(valid_mask) == 0:
            return {f'mae_{layer}': 0.0 for layer in self.config.height_layers}
        
        valid_preds = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        errors = np.abs(valid_preds - valid_targets)
        
        layered_metrics = {}
        layer_counts = {}
        
        for layer_name, (min_height, max_height) in self.config.height_layers.items():
            if max_height == float('inf'):
                layer_mask = valid_targets >= min_height
            else:
                layer_mask = (valid_targets >= min_height) & (valid_targets < max_height)
            
            layer_count = np.sum(layer_mask)
            layer_counts[f'{layer_name}_count'] = int(layer_count)
            
            if layer_count > 0:
                layer_mae = np.mean(errors[layer_mask])
                layered_metrics[f'mae_{layer_name}'] = float(layer_mae)
            else:
                layered_metrics[f'mae_{layer_name}'] = 0.0
        
        # 添加计数信息
        layered_metrics.update(layer_counts)
        
        return layered_metrics
    
    def calculate_relative_error(self, predictions: np.ndarray,
                               targets: np.ndarray) -> Dict[str, float]:
        """
        计算相对误差
        
        Args:
            predictions: 预测值数组
            targets: 真实值数组
        
        Returns:
            相对误差字典
        """
        valid_mask = self._get_valid_mask(predictions, targets)
        if np.sum(valid_mask) == 0:
            return {'relative_error': 100.0}
        
        valid_preds = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        # 计算相对误差（避免除零）
        denominator = np.maximum(np.abs(valid_targets), 0.1)
        relative_errors = np.abs(valid_preds - valid_targets) / denominator
        relative_errors = np.minimum(relative_errors, 10.0)  # 限制最大相对误差
        
        mean_relative_error = np.mean(relative_errors) * 100  # 转换为百分比
        
        return {'relative_error': float(mean_relative_error)}
    
    def calculate_data_range(self, predictions: np.ndarray,
                           targets: np.ndarray) -> Dict[str, float]:
        """
        计算数据范围信息
        
        Args:
            predictions: 预测值数组
            targets: 真实值数组
        
        Returns:
            数据范围字典
        """
        valid_mask = self._get_valid_mask(predictions, targets)
        if np.sum(valid_mask) == 0:
            return {}
        
        valid_preds = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        return {
            'data_range_min': float(valid_targets.min()),
            'data_range_max': float(valid_targets.max()),
            'data_range_mean': float(valid_targets.mean()),
            'data_range_std': float(valid_targets.std()),
            'prediction_range_min': float(valid_preds.min()),
            'prediction_range_max': float(valid_preds.max()),
            'prediction_range_mean': float(valid_preds.mean()),
            'prediction_range_std': float(valid_preds.std())
        }
    
    def _get_valid_mask(self, predictions: np.ndarray, 
                       targets: np.ndarray) -> np.ndarray:
        """获取有效数据掩码"""
        return (~np.isnan(predictions) & ~np.isnan(targets) & 
                ~np.isinf(predictions) & ~np.isinf(targets))
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """获取默认指标值"""
        return {
            'mae': 999.0,
            'mse': 999.0,
            'rmse': 999.0,
            'r2': -1.0,
            'valid_samples': 0
        }


class OnlineMetricsCalculator(BaseMetricsCalculator):
    """在线指标计算器，适用于大数据集"""
    
    def __init__(self, height_normalizer, config: MetricsConfig = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.height_normalizer = height_normalizer
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.count = 0
        self.sum_se = 0.0  # 平方误差和
        self.sum_ae = 0.0  # 绝对误差和
        self.sum_targets = 0.0
        self.sum_targets_sq = 0.0
        self.sum_preds = 0.0
        self.sum_cross = 0.0  # 交叉项
        
        # 精度计数器
        self.accuracy_counts = {f'height_accuracy_{t}m': 0 for t in self.config.accuracy_thresholds}
        
        # 分层误差
        self.layer_errors = {layer: [] for layer in self.config.height_layers}
        
        # 相关性计算的采样数据
        self.sampled_preds = []
        self.sampled_targets = []
        
        # 数据范围
        self.min_target = float('inf')
        self.max_target = float('-inf')
        self.min_pred = float('inf')
        self.max_pred = float('-inf')
    
    def update(self, predictions: Union[torch.Tensor, np.ndarray], 
               targets: Union[torch.Tensor, np.ndarray]):
        """
        更新在线统计
        
        Args:
            predictions: 预测值
            targets: 真实值
        """
        # 转换为numpy并反归一化
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # 反归一化到真实高度值
        pred_heights = self.height_normalizer.denormalize(predictions.flatten())
        target_heights = self.height_normalizer.denormalize(targets.flatten())
        
        # 移除无效值
        valid_mask = self._get_valid_mask(pred_heights, target_heights)
        if np.sum(valid_mask) == 0:
            return
        
        valid_preds = pred_heights[valid_mask]
        valid_targets = target_heights[valid_mask]
        
        n = len(valid_preds)
        self.count += n
        
        # 基础统计
        errors = valid_preds - valid_targets
        abs_errors = np.abs(errors)
        
        self.sum_se += np.sum(errors ** 2)
        self.sum_ae += np.sum(abs_errors)
        self.sum_targets += np.sum(valid_targets)
        self.sum_targets_sq += np.sum(valid_targets ** 2)
        self.sum_preds += np.sum(valid_preds)
        self.sum_cross += np.sum(valid_preds * valid_targets)
        
        # 精度统计
        for threshold in self.config.accuracy_thresholds:
            self.accuracy_counts[f'height_accuracy_{threshold}m'] += np.sum(abs_errors <= threshold)
        
        # 分层误差（只存储限量样本）
        max_layer_samples = 2000
        for layer_name, (min_height, max_height) in self.config.height_layers.items():
            if max_height == float('inf'):
                layer_mask = valid_targets >= min_height
            else:
                layer_mask = (valid_targets >= min_height) & (valid_targets < max_height)
            
            if np.sum(layer_mask) > 0:
                layer_errors = abs_errors[layer_mask]
                self.layer_errors[layer_name].extend(layer_errors.tolist())
                
                # 限制存储大小
                if len(self.layer_errors[layer_name]) > max_layer_samples:
                    self.layer_errors[layer_name] = self.layer_errors[layer_name][-max_layer_samples:]
        
        # 数据范围
        self.min_target = min(self.min_target, valid_targets.min())
        self.max_target = max(self.max_target, valid_targets.max())
        self.min_pred = min(self.min_pred, valid_preds.min())
        self.max_pred = max(self.max_pred, valid_preds.max())
        
        # 相关性计算的采样
        if len(self.sampled_preds) < self.config.sample_for_correlation:
            sample_size = min(len(valid_preds), 
                            self.config.sample_for_correlation - len(self.sampled_preds))
            indices = np.random.choice(len(valid_preds), sample_size, replace=False)
            self.sampled_preds.extend(valid_preds[indices].tolist())
            self.sampled_targets.extend(valid_targets[indices].tolist())
    
    def compute_metrics(self) -> Dict[str, float]:
        """计算最终指标"""
        if self.count == 0:
            return self._get_default_metrics()
        
        # 基础指标
        mse = self.sum_se / self.count
        mae = self.sum_ae / self.count
        rmse = np.sqrt(mse)
        
        # R²计算
        mean_target = self.sum_targets / self.count
        ss_tot = self.sum_targets_sq - 2 * mean_target * self.sum_targets + self.count * (mean_target ** 2)
        ss_res = self.sum_se
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        r2 = max(-10, min(1, r2))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'valid_samples': self.count
        }
        
        # 相关性指标
        if self.config.calculate_correlation and len(self.sampled_preds) > 10:
            correlation_metrics = self.calculate_correlation_metrics(
                np.array(self.sampled_preds), np.array(self.sampled_targets)
            )
            metrics.update(correlation_metrics)
        
        # 精度指标
        if self.config.calculate_accuracy:
            for key, count in self.accuracy_counts.items():
                metrics[key] = float(count / self.count)
        
        # 分层误差
        if self.config.calculate_layered_error:
            for layer_name, errors in self.layer_errors.items():
                if errors:
                    metrics[f'mae_{layer_name}'] = float(np.mean(errors))
                else:
                    metrics[f'mae_{layer_name}'] = 0.0
        
        # 相对误差
        mean_target_abs = abs(mean_target) if abs(mean_target) > 0.1 else 0.1
        metrics['relative_error'] = float((mae / mean_target_abs) * 100)
        
        # 数据范围
        metrics.update({
            'data_range_min': float(self.min_target),
            'data_range_max': float(self.max_target),
            'prediction_range_min': float(self.min_pred),
            'prediction_range_max': float(self.max_pred)
        })
        
        return metrics


class BatchMetricsCalculator(BaseMetricsCalculator):
    """批量指标计算器，适用于小到中等数据集"""
    
    def __init__(self, height_normalizer, config: MetricsConfig = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.height_normalizer = height_normalizer
    
    def calculate_all_metrics(self, predictions: Union[torch.Tensor, np.ndarray],
                            targets: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            predictions: 预测值
            targets: 真实值
        
        Returns:
            所有指标字典
        """
        # 转换为numpy并反归一化
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        pred_heights = self.height_normalizer.denormalize(predictions.flatten())
        target_heights = self.height_normalizer.denormalize(targets.flatten())
        
        # 采样以控制内存使用
        if len(pred_heights) > self.config.max_samples_for_metrics:
            indices = np.random.choice(len(pred_heights), 
                                     self.config.max_samples_for_metrics, 
                                     replace=False)
            pred_heights = pred_heights[indices]
            target_heights = target_heights[indices]
        
        # 计算各类指标
        metrics = {}
        
        # 基础指标
        basic_metrics = self.calculate_basic_metrics(pred_heights, target_heights)
        metrics.update(basic_metrics)
        
        # 相关性指标
        correlation_metrics = self.calculate_correlation_metrics(pred_heights, target_heights)
        metrics.update(correlation_metrics)
        
        # 精度指标
        accuracy_metrics = self.calculate_accuracy_metrics(pred_heights, target_heights)
        metrics.update(accuracy_metrics)
        
        # 分层误差
        layered_metrics = self.calculate_layered_error(pred_heights, target_heights)
        metrics.update(layered_metrics)
        
        # 相对误差
        relative_metrics = self.calculate_relative_error(pred_heights, target_heights)
        metrics.update(relative_metrics)
        
        # 数据范围
        range_metrics = self.calculate_data_range(pred_heights, target_heights)
        metrics.update(range_metrics)
        
        return metrics


class ValidationMetricsCalculator:
    """验证指标计算器"""
    
    def __init__(self, height_normalizer, save_dir: str = None,
                 config: MetricsConfig = None, logger: Optional[logging.Logger] = None):
        self.height_normalizer = height_normalizer
        self.save_dir = save_dir
        self.config = config or MetricsConfig()
        self.logger = logger or setup_logger('validation_metrics')
        
        # 历史指标记录
        self.metrics_history = {}
        self.max_history_length = 100
    
    def validate_model(self, model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
                      criterion: torch.nn.Module, device: torch.device,
                      epoch: int = None) -> Dict[str, float]:
        """
        执行模型验证
        
        Args:
            model: PyTorch模型
            val_loader: 验证数据加载器
            criterion: 损失函数
            device: 设备
            epoch: 当前轮次
        
        Returns:
            验证指标字典
        """
        model.eval()
        
        # 使用在线计算器以节省内存
        metrics_calculator = OnlineMetricsCalculator(
            self.height_normalizer, self.config, self.logger
        )
        
        batch_losses = []
        start_time = time.time()
        
        self.logger.info(f"开始验证 (Epoch {epoch})...")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                try:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    # 数据质量检查
                    if torch.isnan(images).any() or torch.isinf(images).any():
                        self.logger.warning(f"验证批次{batch_idx}: 输入图像包含NaN或Inf")
                        continue
                        
                    if torch.isnan(targets).any() or torch.isinf(targets).any():
                        self.logger.warning(f"验证批次{batch_idx}: 目标包含NaN或Inf")
                        continue
                    
                    # 前向传播
                    predictions = model(images)
                    
                    # 预测结果检查
                    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                        self.logger.warning(f"验证批次{batch_idx}: 预测包含NaN或Inf")
                        continue
                    
                    # 确保维度一致性
                    if predictions.shape != targets.shape:
                        if predictions.dim() == 3 and targets.dim() == 2:
                            targets = targets.unsqueeze(0) if targets.shape[0] != predictions.shape[0] else targets
                        elif predictions.shape[-2:] != targets.shape[-2:]:
                            predictions = F.interpolate(
                                predictions.unsqueeze(1) if predictions.dim() == 3 else predictions,
                                size=targets.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            )
                            if predictions.dim() == 4:
                                predictions = predictions.squeeze(1)
                    
                    # 数值范围检查
                    predictions = torch.clamp(predictions, 0, 1)
                    targets = torch.clamp(targets, 0, 1)
                    
                    # 计算损失
                    if criterion is not None:
                        try:
                            loss = criterion(predictions, targets)
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                batch_losses.append(loss.item())
                        except Exception as e:
                            self.logger.warning(f"验证批次{batch_idx}损失计算错误: {e}")
                    
                    # 更新指标
                    metrics_calculator.update(predictions, targets)
                    
                    # 定期清理内存
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                except Exception as e:
                    self.logger.error(f"验证批次{batch_idx}处理错误: {e}")
                    continue
        
        # 计算最终指标
        metrics = metrics_calculator.compute_metrics()
        
        # 添加损失信息
        if batch_losses:
            metrics['loss'] = float(np.mean(batch_losses))
        else:
            metrics['loss'] = float('inf')
        
        # 添加时间信息
        metrics['validation_time'] = time.time() - start_time
        
        # 更新历史记录
        self._update_metrics_history(metrics, epoch)
        
        # 记录指标
        self.log_validation_metrics(metrics, epoch)
        
        return metrics
    
    def _update_metrics_history(self, metrics: Dict[str, float], epoch: int):
        """更新指标历史记录"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                
                self.metrics_history[key].append((epoch, value))
                
                # 限制历史记录长度
                if len(self.metrics_history[key]) > self.max_history_length:
                    self.metrics_history[key] = self.metrics_history[key][-self.max_history_length:]
    
    def log_validation_metrics(self, metrics: Dict[str, float], 
                             epoch: int = None, is_best: bool = False):
        """记录验证指标"""
        epoch_str = f"Epoch {epoch}" if epoch is not None else "验证"
        
        self.logger.info(f'{epoch_str} 验证结果:')
        self.logger.info(f'  损失: {metrics.get("loss", "N/A"):.6f}')
        self.logger.info(f'  MAE: {metrics.get("mae", "N/A"):.4f} m')
        self.logger.info(f'  RMSE: {metrics.get("rmse", "N/A"):.4f} m')
        self.logger.info(f'  R²: {metrics.get("r2", "N/A"):.4f}')
        
        if 'pearson_r' in metrics:
            self.logger.info(f'  Pearson r: {metrics.get("pearson_r", "N/A"):.4f}')
        
        # 精度指标
        accuracy_keys = [k for k in metrics.keys() if k.startswith('height_accuracy_')]
        if accuracy_keys:
            self.logger.info('  nDSM高度精度:')
            for key in sorted(accuracy_keys):
                threshold = key.replace('height_accuracy_', '').replace('m', '')
                self.logger.info(f'    ±{threshold}m: {metrics[key]:.1%}')
        
        # 分层误差
        layer_keys = [k for k in metrics.keys() if k.startswith('mae_') and not k.endswith('_count')]
        if layer_keys:
            self.logger.info('  分层MAE:')
            for key in sorted(layer_keys):
                layer_name = key.replace('mae_', '')
                layer_display_names = {
                    'ground': '地面层(-5~5m)',
                    'low_buildings': '低建筑(5~20m)', 
                    'mid_buildings': '中建筑(20~50m)',
                    'high_buildings': '高建筑(>50m)'
                }
                display_name = layer_display_names.get(layer_name, layer_name)
                self.logger.info(f'    {display_name}: {metrics[key]:.2f}m')
        
        if 'relative_error' in metrics:
            self.logger.info(f'  相对误差: {metrics.get("relative_error", "N/A"):.2f}%')
        
        if 'valid_samples' in metrics:
            self.logger.info(f'  有效样本: {metrics["valid_samples"]:,}')
        
        if 'validation_time' in metrics:
            self.logger.info(f'  验证耗时: {metrics["validation_time"]:.2f}秒')
        
        if is_best:
            self.logger.info('  ★ 最佳验证性能 ★')
    
    def get_metrics_history(self) -> Dict[str, List[Tuple[int, float]]]:
        """获取指标历史记录"""
        return self.metrics_history.copy()
    
    def get_best_metrics(self) -> Dict[str, Tuple[int, float]]:
        """获取历史最佳指标"""
        best_metrics = {}
        
        # 对于损失和误差指标，取最小值
        minimize_keys = ['loss', 'mae', 'mse', 'rmse', 'relative_error']
        
        # 对于其他指标，取最大值
        maximize_keys = ['r2', 'pearson_r', 'spearman_r'] + \
                      [k for k in self.metrics_history.keys() if k.startswith('height_accuracy_')]
        
        for key, history in self.metrics_history.items():
            if not history:
                continue
            
            if any(minimize_key in key for minimize_key in minimize_keys):
                best_epoch, best_value = min(history, key=lambda x: x[1])
            elif any(maximize_key in key for maximize_key in maximize_keys):
                best_epoch, best_value = max(history, key=lambda x: x[1])
            else:
                # 默认取最新值
                best_epoch, best_value = history[-1]
            
            best_metrics[key] = (best_epoch, best_value)
        
        return best_metrics


def create_metrics_calculator(calculator_type: str = 'online', 
                             height_normalizer=None,
                             config: MetricsConfig = None,
                             logger: Optional[logging.Logger] = None):
    """
    创建指标计算器的工厂函数
    
    Args:
        calculator_type: 计算器类型 ('online', 'batch', 'validation')
        height_normalizer: 高度归一化器
        config: 指标配置
        logger: 日志记录器
    
    Returns:
        指标计算器实例
    """
    if calculator_type == 'online':
        return OnlineMetricsCalculator(height_normalizer, config, logger)
    elif calculator_type == 'batch':
        return BatchMetricsCalculator(height_normalizer, config, logger)
    elif calculator_type == 'validation':
        return ValidationMetricsCalculator(height_normalizer, config=config, logger=logger)
    else:
        raise ValueError(f"不支持的计算器类型: {calculator_type}")


# 便利函数
def calculate_ndsm_metrics(predictions: Union[torch.Tensor, np.ndarray],
                          targets: Union[torch.Tensor, np.ndarray],
                          height_normalizer,
                          config: MetricsConfig = None) -> Dict[str, float]:
    """
    计算nDSM指标的便利函数
    
    Args:
        predictions: 预测值
        targets: 真实值
        height_normalizer: 高度归一化器
        config: 指标配置
    
    Returns:
        指标字典
    """
    calculator = BatchMetricsCalculator(height_normalizer, config)
    return calculator.calculate_all_metrics(predictions, targets)