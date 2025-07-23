#!/usr/bin/env python3
"""
统一的归一化器模块
从损失函数中分离出归一化功能，统一管理所有数据归一化逻辑
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from ..utils.common import setup_logger


class BaseNormalizer(ABC):
    """归一化器基类"""
    
    def __init__(self, method: str = 'minmax', 
                 height_filter: Optional[Dict[str, float]] = None,
                 logger: Optional[logging.Logger] = None):
        self.method = method
        self.height_filter = height_filter or {'min_height': -5.0, 'max_height': 200.0}
        self.logger = logger or setup_logger('normalizer')
        self.fitted = False
        
    @abstractmethod
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """拟合归一化器"""
        pass
    
    @abstractmethod
    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """归一化数据"""
        pass
    
    @abstractmethod
    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """反归一化数据"""
        pass
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> tuple[np.ndarray, bool, Optional[torch.device]]:
        """将输入转换为numpy数组，并记录原始类型信息"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy(), True, data.device
        else:
            return np.array(data), False, None
    
    def _to_original_type(self, data: np.ndarray, was_tensor: bool, 
                         device: Optional[torch.device]) -> Union[np.ndarray, torch.Tensor]:
        """将数据转换回原始类型"""
        if was_tensor:
            result = torch.from_numpy(data)
            if device is not None:
                result = result.to(device)
            return result.float()
        return data.astype(np.float32)


class HeightNormalizer(BaseNormalizer):
    """nDSM高度数据归一化器"""
    
    def __init__(self, method: str = 'minmax', 
                 height_filter: Optional[Dict[str, float]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化高度归一化器
        
        Args:
            method: 归一化方法 ('minmax', 'percentile', 'zscore', 'robust')
            height_filter: 高度过滤器
            logger: 日志记录器
        """
        super().__init__(method, height_filter, logger)
        
        # 统计信息
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None
        self.range_val = None
        self.percentiles = None
        
        # 兼容旧代码的属性
        self.global_min_h = None
        self.global_max_h = None
        self.height_range = None
        
        # 支持的归一化方法
        self.supported_methods = ['minmax', 'percentile', 'zscore', 'robust', 'log_minmax', 'sqrt_minmax']
        
        if self.method not in self.supported_methods:
            raise ValueError(f"不支持的归一化方法: {self.method}，支持的方法: {self.supported_methods}")
    
    def fit_from_json_stats(self, stats_data: Dict[str, Any]) -> None:
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
            
            # 获取分位数信息
            if 'percentiles' in stats_data:
                self.percentiles = {
                    k: float(v) for k, v in stats_data['percentiles'].items()
                }
            
            # 设置兼容属性
            self._set_compatibility_attributes()
            
            self.fitted = True
            
            if self.min_val != original_min or self.max_val != original_max:
                self.logger.info(f"应用高度过滤: [{original_min:.1f}, {original_max:.1f}] -> [{self.min_val:.1f}, {self.max_val:.1f}]")
            
            self.logger.info(f"归一化器拟合完成:")
            self.logger.info(f"  方法: {self.method}")
            self.logger.info(f"  范围: [{self.min_val:.1f}, {self.max_val:.1f}] 米")
            self.logger.info(f"  均值±标准差: {self.mean_val:.2f} ± {self.std_val:.2f} 米")
            
        except Exception as e:
            raise ValueError(f"从JSON拟合归一化器失败: {e}")
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """从数据拟合归一化器"""
        data_np, was_tensor, device = self._to_numpy(data)
        
        # 过滤有效数据
        valid_data = data_np[~(np.isnan(data_np) | np.isinf(data_np))]
        
        if len(valid_data) == 0:
            raise ValueError("没有有效的高度数据")
        
        # 应用高度过滤
        filtered_data = valid_data[
            (valid_data >= self.height_filter['min_height']) & 
            (valid_data <= self.height_filter['max_height'])
        ]
        
        if len(filtered_data) == 0:
            self.logger.warning("高度过滤后没有有效数据，使用原始数据")
            filtered_data = valid_data
        
        # 计算统计信息
        self.min_val = float(filtered_data.min())
        self.max_val = float(filtered_data.max())
        self.mean_val = float(filtered_data.mean())
        self.std_val = float(filtered_data.std())
        self.range_val = self.max_val - self.min_val
        
        # 计算分位数
        self.percentiles = {
            'p5': float(np.percentile(filtered_data, 5)),
            'p25': float(np.percentile(filtered_data, 25)),
            'p50': float(np.percentile(filtered_data, 50)),
            'p75': float(np.percentile(filtered_data, 75)),
            'p95': float(np.percentile(filtered_data, 95))
        }
        
        # 设置兼容属性
        self._set_compatibility_attributes()
        
        self.fitted = True
        
        self.logger.info(f"归一化器拟合完成: {len(filtered_data)} 个数据点")
        self.logger.info(f"数据范围: [{self.min_val:.1f}, {self.max_val:.1f}] 米")
    
    def _set_compatibility_attributes(self):
        """设置兼容旧代码的属性"""
        self.global_min_h = self.min_val
        self.global_max_h = self.max_val
        self.height_range = self.range_val
    
    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """归一化高度数据"""
        if not self.fitted:
            raise ValueError("归一化器未拟合")
        
        data_np, was_tensor, device = self._to_numpy(data)
        
        # 处理无效值
        valid_mask = ~(np.isnan(data_np) | np.isinf(data_np))
        normalized = np.zeros_like(data_np)
        
        if not np.any(valid_mask):
            return self._to_original_type(normalized, was_tensor, device)
        
        valid_data = data_np[valid_mask]
        
        # 应用高度过滤
        height_filtered = np.clip(valid_data, self.min_val, self.max_val)
        
        # 根据方法归一化
        if self.method == 'minmax':
            normalized_valid = self._minmax_normalize(height_filtered)
            
        elif self.method == 'percentile':
            normalized_valid = self._percentile_normalize(height_filtered)
            
        elif self.method == 'zscore':
            normalized_valid = self._zscore_normalize(height_filtered)
            
        elif self.method == 'robust':
            normalized_valid = self._robust_normalize(height_filtered)
            
        elif self.method == 'log_minmax':
            normalized_valid = self._log_minmax_normalize(height_filtered)
            
        elif self.method == 'sqrt_minmax':
            normalized_valid = self._sqrt_minmax_normalize(height_filtered)
            
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")
        
        # 确保在[0,1]范围内
        normalized_valid = np.clip(normalized_valid, 0, 1)
        normalized[valid_mask] = normalized_valid
        
        return self._to_original_type(normalized, was_tensor, device)
    
    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """反归一化到真实高度值"""
        if not self.fitted:
            raise ValueError("归一化器未拟合")
        
        data_np, was_tensor, device = self._to_numpy(data)
        data_np = np.clip(data_np, 0, 1)
        
        # 根据方法反归一化
        if self.method == 'minmax':
            height = self._minmax_denormalize(data_np)
            
        elif self.method == 'percentile':
            height = self._percentile_denormalize(data_np)
            
        elif self.method == 'zscore':
            height = self._zscore_denormalize(data_np)
            
        elif self.method == 'robust':
            height = self._robust_denormalize(data_np)
            
        elif self.method == 'log_minmax':
            height = self._log_minmax_denormalize(data_np)
            
        elif self.method == 'sqrt_minmax':
            height = self._sqrt_minmax_denormalize(data_np)
            
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")
        
        return self._to_original_type(height, was_tensor, device)
    
    def _minmax_normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-Max归一化"""
        return (data - self.min_val) / (self.range_val + 1e-8)
    
    def _minmax_denormalize(self, data: np.ndarray) -> np.ndarray:
        """Min-Max反归一化"""
        return data * self.range_val + self.min_val
    
    def _percentile_normalize(self, data: np.ndarray) -> np.ndarray:
        """分位数归一化"""
        p5 = self.percentiles.get('p5', self.min_val + 0.05 * self.range_val)
        p95 = self.percentiles.get('p95', self.min_val + 0.95 * self.range_val)
        clipped_data = np.clip(data, p5, p95)
        return (clipped_data - p5) / (p95 - p5 + 1e-8)
    
    def _percentile_denormalize(self, data: np.ndarray) -> np.ndarray:
        """分位数反归一化"""
        p5 = self.percentiles.get('p5', self.min_val + 0.05 * self.range_val)
        p95 = self.percentiles.get('p95', self.min_val + 0.95 * self.range_val)
        return data * (p95 - p5) + p5
    
    def _zscore_normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score归一化"""
        zscore = (data - self.mean_val) / (self.std_val + 1e-8)
        zscore = np.clip(zscore, -3, 3)
        return (zscore + 3) / 6.0
    
    def _zscore_denormalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score反归一化"""
        zscore = (data * 6.0) - 3.0
        return zscore * self.std_val + self.mean_val
    
    def _robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """鲁棒归一化（使用中位数和MAD）"""
        if self.percentiles is None:
            return self._minmax_normalize(data)
        
        median = self.percentiles.get('p50', self.mean_val)
        q25 = self.percentiles.get('p25', self.min_val + 0.25 * self.range_val)
        q75 = self.percentiles.get('p75', self.min_val + 0.75 * self.range_val)
        iqr = q75 - q25
        
        normalized = (data - median) / (iqr + 1e-8)
        normalized = np.clip(normalized, -2, 2)
        return (normalized + 2) / 4.0
    
    def _robust_denormalize(self, data: np.ndarray) -> np.ndarray:
        """鲁棒反归一化"""
        if self.percentiles is None:
            return self._minmax_denormalize(data)
        
        median = self.percentiles.get('p50', self.mean_val)
        q25 = self.percentiles.get('p25', self.min_val + 0.25 * self.range_val)
        q75 = self.percentiles.get('p75', self.min_val + 0.75 * self.range_val)
        iqr = q75 - q25
        
        normalized = (data * 4.0) - 2.0
        return normalized * iqr + median
    
    def _log_minmax_normalize(self, data: np.ndarray) -> np.ndarray:
        """对数Min-Max归一化"""
        # 确保数据为正数
        shifted_data = data - self.min_val + 1.0
        log_data = np.log(shifted_data)
        log_min = np.log(1.0)
        log_max = np.log(self.range_val + 1.0)
        return (log_data - log_min) / (log_max - log_min + 1e-8)
    
    def _log_minmax_denormalize(self, data: np.ndarray) -> np.ndarray:
        """对数Min-Max反归一化"""
        log_min = np.log(1.0)
        log_max = np.log(self.range_val + 1.0)
        log_data = data * (log_max - log_min) + log_min
        shifted_data = np.exp(log_data)
        return shifted_data + self.min_val - 1.0
    
    def _sqrt_minmax_normalize(self, data: np.ndarray) -> np.ndarray:
        """平方根Min-Max归一化"""
        # 确保数据为非负数
        shifted_data = data - self.min_val
        sqrt_data = np.sqrt(np.maximum(shifted_data, 0))
        sqrt_max = np.sqrt(self.range_val)
        return sqrt_data / (sqrt_max + 1e-8)
    
    def _sqrt_minmax_denormalize(self, data: np.ndarray) -> np.ndarray:
        """平方根Min-Max反归一化"""
        sqrt_max = np.sqrt(self.range_val)
        sqrt_data = data * sqrt_max
        shifted_data = sqrt_data ** 2
        return shifted_data + self.min_val
    
    def get_normalization_info(self) -> Dict[str, Any]:
        """获取归一化信息"""
        if not self.fitted:
            return {'fitted': False}
        
        info = {
            'fitted': True,
            'method': self.method,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'mean_val': self.mean_val,
            'std_val': self.std_val,
            'range_val': self.range_val,
            'height_filter': self.height_filter
        }
        
        if self.percentiles:
            info['percentiles'] = self.percentiles
        
        return info
    
    def save_normalizer_config(self, save_path: str) -> None:
        """保存归一化器配置"""
        import json
        
        config = self.get_normalization_info()
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"归一化器配置已保存: {save_path}")
            
        except Exception as e:
            self.logger.error(f"保存归一化器配置失败: {e}")
            raise
    
    @classmethod
    def load_normalizer_config(cls, config_path: str, 
                              logger: Optional[logging.Logger] = None) -> 'HeightNormalizer':
        """从配置文件加载归一化器"""
        import json
        
        if logger is None:
            logger = setup_logger('normalizer')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if not config.get('fitted', False):
                raise ValueError("配置文件中的归一化器未拟合")
            
            # 创建归一化器实例
            normalizer = cls(
                method=config['method'],
                height_filter=config['height_filter'],
                logger=logger
            )
            
            # 设置统计信息
            normalizer.min_val = config['min_val']
            normalizer.max_val = config['max_val']
            normalizer.mean_val = config['mean_val']
            normalizer.std_val = config['std_val']
            normalizer.range_val = config['range_val']
            normalizer.percentiles = config.get('percentiles')
            normalizer._set_compatibility_attributes()
            normalizer.fitted = True
            
            logger.info(f"归一化器配置已加载: {config_path}")
            
            return normalizer
            
        except Exception as e:
            logger.error(f"加载归一化器配置失败: {e}")
            raise


class MultiModalNormalizer:
    """多模态数据归一化器"""
    
    def __init__(self, normalizers: Dict[str, BaseNormalizer],
                 logger: Optional[logging.Logger] = None):
        self.normalizers = normalizers
        self.logger = logger or setup_logger('multi_normalizer')
    
    def normalize(self, data_dict: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """归一化多模态数据"""
        normalized_data = {}
        
        for key, data in data_dict.items():
            if key in self.normalizers:
                normalized_data[key] = self.normalizers[key].normalize(data)
            else:
                self.logger.warning(f"没有为 {key} 配置归一化器，跳过归一化")
                normalized_data[key] = data
        
        return normalized_data
    
    def denormalize(self, data_dict: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """反归一化多模态数据"""
        denormalized_data = {}
        
        for key, data in data_dict.items():
            if key in self.normalizers:
                denormalized_data[key] = self.normalizers[key].denormalize(data)
            else:
                self.logger.warning(f"没有为 {key} 配置归一化器，跳过反归一化")
                denormalized_data[key] = data
        
        return denormalized_data


# 便利函数
def create_height_normalizer(method: str = 'minmax', 
                           height_filter: Optional[Dict[str, float]] = None,
                           logger: Optional[logging.Logger] = None) -> HeightNormalizer:
    """
    创建高度归一化器的便利函数
    
    Args:
        method: 归一化方法
        height_filter: 高度过滤器
        logger: 日志记录器
    
    Returns:
        高度归一化器实例
    """
    return HeightNormalizer(method=method, height_filter=height_filter, logger=logger)


def create_normalizer_from_stats(stats_json_path: str, 
                                method: str = 'minmax',
                                height_filter: Optional[Dict[str, float]] = None,
                                logger: Optional[logging.Logger] = None) -> HeightNormalizer:
    """
    从统计信息文件创建归一化器的便利函数
    
    Args:
        stats_json_path: 统计信息JSON文件路径
        method: 归一化方法
        height_filter: 高度过滤器
        logger: 日志记录器
    
    Returns:
        已拟合的高度归一化器
    """
    from ..utils.data_utils import load_statistics_config
    
    # 加载统计信息
    stats_data = load_statistics_config(stats_json_path, logger)
    
    # 创建归一化器
    normalizer = create_height_normalizer(method, height_filter, logger)
    
    # 从统计信息拟合
    normalizer.fit_from_json_stats(stats_data)
    
    return normalizer


# 使用示例
if __name__ == '__main__':
    # 设置日志
    logger = setup_logger('normalizer_test')
    
    try:
        # 创建测试数据
        test_data = np.random.uniform(-10, 150, 1000)
        
        # 测试不同的归一化方法
        methods = ['minmax', 'percentile', 'zscore', 'robust', 'log_minmax', 'sqrt_minmax']
        
        for method in methods:
            logger.info(f"测试 {method} 归一化方法:")
            
            # 创建归一化器
            normalizer = create_height_normalizer(
                method=method,
                height_filter={'min_height': -5.0, 'max_height': 100.0},
                logger=logger
            )
            
            # 拟合归一化器
            normalizer.fit(test_data)
            
            # 测试归一化
            normalized = normalizer.normalize(test_data[:10])
            denormalized = normalizer.denormalize(normalized)
            
            logger.info(f"  原始数据范围: [{test_data.min():.2f}, {test_data.max():.2f}]")
            logger.info(f"  归一化后范围: [{normalized.min():.3f}, {normalized.max():.3f}]")
            logger.info(f"  反归一化误差: {np.abs(test_data[:10] - denormalized).max():.6f}")
            
            # 获取归一化信息
            info = normalizer.get_normalization_info()
            logger.info(f"  归一化信息: {info}")
        
        logger.info("所有归一化方法测试完成")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()