#!/usr/bin/env python3
"""
数据配置模块
管理GAMUS nDSM数据处理相关的所有配置参数
"""

import os
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging

from .base_config import BaseConfig, ConfigError


class DataConfig(BaseConfig):
    """数据配置类"""
    
    def _set_defaults(self):
        """设置默认值"""
        # 基础数据路径
        self.data_dir: str = ""
        self.stats_json_path: Optional[str] = None
        
        # 数据加载参数
        self.batch_size: int = 8
        self.num_workers: int = 4
        self.shuffle: bool = True
        self.pin_memory: bool = True
        self.drop_last: bool = True
        
        # 数据预处理参数
        self.target_size: Tuple[int, int] = (448, 448)
        self.normalization_method: str = 'minmax'
        self.enable_augmentation: bool = False
        
        # 高度过滤器
        self.height_filter: Dict[str, float] = {
            'min_height': -5.0,
            'max_height': 200.0
        }
        
        # 数据集分割参数
        self.train_split: float = 0.8
        self.val_split: float = 0.2
        self.test_split: float = 0.0
        
        # 文件扩展名
        self.image_extensions: List[str] = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        self.label_extensions: List[str] = ['.tif', '.tiff']
        
        # 数据验证参数
        self.validate_data: bool = True
        self.max_validation_samples: int = 100
        self.validation_ratio: float = 0.05
        
        # 缓存参数
        self.enable_cache: bool = False
        self.cache_dir: Optional[str] = None
        
        # 内存管理
        self.prefetch_factor: int = 2
        self.persistent_workers: bool = False
        
        # 数据增强参数（当enable_augmentation=True时使用）
        self.augmentation_config: Dict[str, Any] = {
            'horizontal_flip_prob': 0.5,
            'vertical_flip_prob': 0.5,
            'rotation_prob': 0.3,
            'rotation_degrees': 15,
            'brightness_prob': 0.2,
            'brightness_factor': 0.1,
            'contrast_prob': 0.2,
            'contrast_factor': 0.1
        }
    
    def validate(self) -> bool:
        """验证数据配置"""
        # 验证数据目录
        if not self.data_dir:
            raise ConfigError("data_dir 不能为空")
        
        if not os.path.exists(self.data_dir):
            raise ConfigError(f"数据目录不存在: {self.data_dir}")
        
        if not os.path.isdir(self.data_dir):
            raise ConfigError(f"data_dir 不是目录: {self.data_dir}")
        
        # 验证统计信息文件
        if self.stats_json_path:
            if not os.path.exists(self.stats_json_path):
                self.logger.warning(f"统计信息文件不存在: {self.stats_json_path}")
        
        # 验证批次大小
        self._validator.validate_positive_number(self.batch_size, "batch_size")
        self._validator.validate_range(self.batch_size, min_val=1, max_val=1024, name="batch_size")
        
        # 验证工作线程数
        self._validator.validate_range(self.num_workers, min_val=0, max_val=32, name="num_workers")
        
        # 验证目标尺寸
        self._validator.validate_tuple_length(self.target_size, 2, "target_size")
        for dim in self.target_size:
            self._validator.validate_positive_number(dim, "target_size维度")
            self._validator.validate_range(dim, min_val=32, max_val=2048, name="target_size维度")
        
        # 验证归一化方法
        valid_methods = ['minmax', 'log_minmax', 'sqrt_minmax', 'percentile', 'zscore_clip']
        self._validator.validate_in_choices(
            self.normalization_method, valid_methods, "normalization_method"
        )
        
        # 验证高度过滤器
        if not isinstance(self.height_filter, dict):
            raise ConfigError("height_filter 必须是字典")
        
        required_keys = ['min_height', 'max_height']
        for key in required_keys:
            if key not in self.height_filter:
                raise ConfigError(f"height_filter 缺少必需键: {key}")
        
        min_h = self.height_filter['min_height']
        max_h = self.height_filter['max_height']
        
        if min_h >= max_h:
            raise ConfigError(f"min_height ({min_h}) 必须小于 max_height ({max_h})")
        
        # 验证数据集分割比例
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ConfigError(f"数据集分割比例总和必须为1.0，当前为: {total_split}")
        
        for split_name, split_ratio in [
            ('train_split', self.train_split),
            ('val_split', self.val_split),
            ('test_split', self.test_split)
        ]:
            self._validator.validate_range(split_ratio, min_val=0.0, max_val=1.0, name=split_name)
        
        # 验证文件扩展名
        if not self.image_extensions:
            raise ConfigError("image_extensions 不能为空")
        
        if not self.label_extensions:
            raise ConfigError("label_extensions 不能为空")
        
        # 验证验证参数
        self._validator.validate_range(
            self.validation_ratio, min_val=0.0, max_val=1.0, name="validation_ratio"
        )
        self._validator.validate_positive_number(
            self.max_validation_samples, "max_validation_samples"
        )
        
        # 验证缓存目录
        if self.enable_cache and self.cache_dir:
            cache_path = Path(self.cache_dir)
            if cache_path.exists() and not cache_path.is_dir():
                raise ConfigError(f"cache_dir 不是目录: {self.cache_dir}")
        
        # 验证数据增强配置
        if self.enable_augmentation:
            self._validate_augmentation_config()
        
        self.logger.info("数据配置验证通过")
        return True
    
    def _validate_augmentation_config(self):
        """验证数据增强配置"""
        aug_config = self.augmentation_config
        
        # 验证概率值
        prob_keys = [
            'horizontal_flip_prob', 'vertical_flip_prob', 'rotation_prob',
            'brightness_prob', 'contrast_prob'
        ]
        
        for key in prob_keys:
            if key in aug_config:
                self._validator.validate_range(
                    aug_config[key], min_val=0.0, max_val=1.0, name=key
                )
        
        # 验证旋转角度
        if 'rotation_degrees' in aug_config:
            self._validator.validate_range(
                aug_config['rotation_degrees'], min_val=0, max_val=180, name="rotation_degrees"
            )
        
        # 验证亮度和对比度因子
        for factor_key in ['brightness_factor', 'contrast_factor']:
            if factor_key in aug_config:
                self._validator.validate_range(
                    aug_config[factor_key], min_val=0.0, max_val=1.0, name=factor_key
                )
    
    def get_dataloader_config(self) -> Dict[str, Any]:
        """
        获取DataLoader配置
        
        Returns:
            DataLoader配置字典
        """
        return {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': self.drop_last,
            'prefetch_factor': self.prefetch_factor,
            'persistent_workers': self.persistent_workers if self.num_workers > 0 else False
        }
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """
        获取Dataset配置
        
        Returns:
            Dataset配置字典
        """
        return {
            'image_dir': self.data_dir,
            'label_dir': self.data_dir,
            'normalization_method': self.normalization_method,
            'enable_augmentation': self.enable_augmentation,
            'stats_json_path': self.stats_json_path,
            'height_filter': self.height_filter,
            'target_size': self.target_size
        }
    
    def get_splits_config(self) -> Dict[str, float]:
        """
        获取数据集分割配置
        
        Returns:
            分割配置字典
        """
        return {
            'train': self.train_split,
            'val': self.val_split,
            'test': self.test_split
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """
        获取数据验证配置
        
        Returns:
            验证配置字典
        """
        return {
            'validate_data': self.validate_data,
            'max_validation_samples': self.max_validation_samples,
            'validation_ratio': self.validation_ratio,
            'image_extensions': self.image_extensions,
            'label_extensions': self.label_extensions
        }
    
    def setup_cache_dir(self):
        """设置缓存目录"""
        if self.enable_cache and self.cache_dir:
            cache_path = Path(self.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"缓存目录已设置: {self.cache_dir}")
    
    def get_expected_structure(self) -> Dict[str, List[str]]:
        """
        获取期望的数据目录结构
        
        Returns:
            期望的目录结构
        """
        return {
            'required_dirs': ['train', 'val'],
            'optional_dirs': ['test'],
            'required_subdirs': ['images', 'depths']
        }
    
    def check_data_structure(self) -> Dict[str, Any]:
        """
        检查数据目录结构
        
        Returns:
            检查结果
        """
        if not self.data_dir:
            return {'valid': False, 'error': 'data_dir 未设置'}
        
        data_path = Path(self.data_dir)
        if not data_path.exists():
            return {'valid': False, 'error': f'数据目录不存在: {self.data_dir}'}
        
        structure = self.get_expected_structure()
        result = {
            'valid': True,
            'splits': {},
            'warnings': [],
            'errors': []
        }
        
        # 检查必需的分割目录
        for split in structure['required_dirs']:
            split_path = data_path / split
            split_result = {'exists': split_path.exists(), 'subdirs': {}}
            
            if split_result['exists']:
                # 检查子目录
                for subdir in structure['required_subdirs']:
                    subdir_path = split_path / subdir
                    split_result['subdirs'][subdir] = {
                        'exists': subdir_path.exists(),
                        'file_count': len(list(subdir_path.glob('*'))) if subdir_path.exists() else 0
                    }
                    
                    if not subdir_path.exists():
                        result['errors'].append(f'缺少子目录: {split}/{subdir}')
                        result['valid'] = False
            else:
                result['errors'].append(f'缺少必需目录: {split}')
                result['valid'] = False
            
            result['splits'][split] = split_result
        
        # 检查可选的分割目录
        for split in structure['optional_dirs']:
            split_path = data_path / split
            if split_path.exists():
                split_result = {'exists': True, 'subdirs': {}}
                for subdir in structure['required_subdirs']:
                    subdir_path = split_path / subdir
                    split_result['subdirs'][subdir] = {
                        'exists': subdir_path.exists(),
                        'file_count': len(list(subdir_path.glob('*'))) if subdir_path.exists() else 0
                    }
                result['splits'][split] = split_result
        
        return result
    
    def _get_critical_fields(self) -> List[str]:
        """获取关键字段列表"""
        return [
            'data_dir', 'target_size', 'normalization_method',
            'height_filter', 'batch_size'
        ]
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        估算内存使用量
        
        Returns:
            内存使用估算（MB）
        """
        # 单个样本的估算内存使用（MB）
        image_size = self.target_size[0] * self.target_size[1] * 3 * 4 / (1024 * 1024)  # 假设float32
        label_size = self.target_size[0] * self.target_size[1] * 4 / (1024 * 1024)  # 假设float32
        sample_size = image_size + label_size
        
        # 批次内存使用
        batch_memory = sample_size * self.batch_size
        
        # 考虑多个worker的内存使用
        total_memory = batch_memory * (self.num_workers + 1)  # +1 for main process
        
        # 如果启用pin_memory，增加额外内存
        if self.pin_memory:
            total_memory *= 1.5
        
        return {
            'per_sample_mb': sample_size,
            'per_batch_mb': batch_memory,
            'total_estimated_mb': total_memory,
            'image_mb': image_size,
            'label_mb': label_size
        }
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """
        获取推荐设置
        
        Returns:
            推荐的配置设置
        """
        import psutil
        
        # 获取系统信息
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        recommendations = {}
        
        # 推荐的worker数量
        recommended_workers = min(cpu_count, 8)  # 不超过8个worker
        if self.num_workers != recommended_workers:
            recommendations['num_workers'] = {
                'current': self.num_workers,
                'recommended': recommended_workers,
                'reason': f'基于CPU核心数({cpu_count})的推荐值'
            }
        
        # 推荐的批次大小
        memory_per_batch = self.estimate_memory_usage()['per_batch_mb'] / 1024  # GB
        max_safe_batch_size = int(memory_gb * 0.3 / memory_per_batch)  # 使用30%内存
        max_safe_batch_size = max(1, min(max_safe_batch_size, 64))  # 限制在1-64之间
        
        if self.batch_size > max_safe_batch_size:
            recommendations['batch_size'] = {
                'current': self.batch_size,
                'recommended': max_safe_batch_size,
                'reason': f'基于可用内存({memory_gb:.1f}GB)的推荐值'
            }
        
        # pin_memory推荐
        if memory_gb < 8 and self.pin_memory:
            recommendations['pin_memory'] = {
                'current': self.pin_memory,
                'recommended': False,
                'reason': '内存较少时建议禁用pin_memory'
            }
        
        return recommendations


def create_data_config(data_dir: str = "",
                      batch_size: int = 8,
                      normalization_method: str = 'minmax',
                      stats_json_path: Optional[str] = None,
                      enable_augmentation: bool = False,
                      num_workers: int = 4,
                      target_size: Tuple[int, int] = (448, 448),
                      **kwargs) -> DataConfig:
    """
    创建数据配置的便利函数
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        normalization_method: 归一化方法
        stats_json_path: 统计信息文件路径
        enable_augmentation: 是否启用数据增强
        num_workers: 工作线程数
        target_size: 目标尺寸
        **kwargs: 其他配置参数
    
    Returns:
        数据配置实例
    """
    config = DataConfig()
    
    # 设置主要参数
    config.data_dir = data_dir
    config.batch_size = batch_size
    config.normalization_method = normalization_method
    config.stats_json_path = stats_json_path
    config.enable_augmentation = enable_augmentation
    config.num_workers = num_workers
    config.target_size = target_size
    
    # 设置其他参数
    config.update(**kwargs)
    
    return config


def create_quick_data_config(data_dir: str,
                           stats_json_path: str,
                           mode: str = 'production') -> DataConfig:
    """
    创建快速数据配置的便利函数
    
    Args:
        data_dir: 数据目录
        stats_json_path: 统计信息文件路径
        mode: 模式 ('debug', 'development', 'production')
    
    Returns:
        数据配置实例
    """
    mode_configs = {
        'debug': {
            'batch_size': 1,
            'num_workers': 0,
            'enable_augmentation': False,
            'shuffle': False,
            'pin_memory': False
        },
        'development': {
            'batch_size': 4,
            'num_workers': 2,
            'enable_augmentation': True,
            'shuffle': True,
            'pin_memory': True
        },
        'production': {
            'batch_size': 8,
            'num_workers': 4,
            'enable_augmentation': True,
            'shuffle': True,
            'pin_memory': True
        }
    }
    
    if mode not in mode_configs:
        raise ValueError(f"不支持的模式: {mode}. 支持: {list(mode_configs.keys())}")
    
    config = create_data_config(
        data_dir=data_dir,
        stats_json_path=stats_json_path,
        **mode_configs[mode]
    )
    
    return config


# 使用示例
if __name__ == '__main__':
    # 创建数据配置
    config = create_data_config(
        data_dir="/path/to/data",
        batch_size=16,
        stats_json_path="stats.json"
    )
    
    print("数据配置摘要:")
    print(config.get_summary())
    
    # 验证配置
    try:
        config.validate()
        print("✓ 配置验证通过")
    except ConfigError as e:
        print(f"✗ 配置验证失败: {e}")
    
    # 检查数据结构
    structure_check = config.check_data_structure()
    print(f"\n数据结构检查: {structure_check}")
    
    # 获取内存使用估算
    memory_usage = config.estimate_memory_usage()
    print(f"\n内存使用估算: {memory_usage}")
    
    # 获取推荐设置
    recommendations = config.get_recommended_settings()
    if recommendations:
        print(f"\n推荐设置: {recommendations}")
    else:
        print("\n当前设置已是推荐配置")