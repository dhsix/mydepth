#!/usr/bin/env python3
"""
核心数据集模块
统一管理GAMUS nDSM数据集的创建、加载和处理功能，消除重复代码
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import cv2
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings
import random

# 导入重构后的模块
from .normalizer import HeightNormalizer
from ..utils.common import setup_logger, validate_paths, count_files, clear_memory
from ..utils.data_utils import (
    extract_base_name, match_file_pairs, validate_data_structure,
    load_statistics_config, get_valid_mask
)

warnings.filterwarnings('ignore')


class GAMUSDataset(Dataset):
    """
    统一的GAMUS nDSM数据集类
    整合所有数据集功能，消除重复代码
    """
    
    def __init__(self, 
                 image_dir: str, 
                 label_dir: str, 
                 normalization_method: str = 'minmax',
                 enable_augmentation: bool = False,
                 stats_json_path: Optional[str] = None,
                 height_filter: Optional[Dict[str, float]] = None,
                 target_size: Tuple[int, int] = (448, 448),
                 force_recompute: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        初始化GAMUS数据集
        
        Args:
            image_dir: 影像切片目录
            label_dir: nDSM标签切片目录  
            normalization_method: 归一化方法
            enable_augmentation: 是否启用数据增强
            stats_json_path: 统计信息JSON文件路径
            height_filter: 高度过滤配置
            target_size: 目标图像尺寸
            force_recompute: 是否强制重新计算统计信息
            logger: 日志记录器
        """
        self.logger = logger or setup_logger(self.__class__.__name__)
        
        # 参数验证
        if not stats_json_path:
            raise ValueError("必须指定stats_json_path参数，请先运行预计算脚本")
        
        if not os.path.exists(stats_json_path):
            raise FileNotFoundError(f"统计信息文件不存在: {stats_json_path}，请先运行预计算脚本")
        
        # 基本参数
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.normalization_method = normalization_method
        self.enable_augmentation = enable_augmentation
        self.stats_json_path = stats_json_path
        self.target_size = target_size
        self.force_recompute = force_recompute
        
        # 设置高度过滤器（默认合理范围）
        self.height_filter = height_filter or {'min_height': -5.0, 'max_height': 200.0}
        
        # 验证目录
        self._validate_directories()
        
        # 获取文件对
        self.file_pairs = self._get_file_pairs()
        
        # 初始化统计信息和归一化器
        self._initialize_normalizer()
        
        # 设置数据变换
        self.transform = self._setup_transforms()
        
        self.logger.info(f"GAMUS数据集初始化完成:")
        self.logger.info(f"  样本数量: {len(self.file_pairs)}")
        self.logger.info(f"  图像尺寸: {self.target_size}")
        self.logger.info(f"  归一化方法: {self.normalization_method}")
        self.logger.info(f"  数据增强: {self.enable_augmentation}")
        self.logger.info(f"  高度范围: [{self.height_filter['min_height']:.1f}, {self.height_filter['max_height']:.1f}] 米")
     
    def _validate_directories(self):
        """验证目录存在性"""
        if not validate_paths(self.image_dir, self.label_dir, logger=self.logger):
            raise FileNotFoundError("数据目录验证失败")
        
        # 统计文件数量
        image_count = count_files(self.image_dir, ('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
        label_count = count_files(self.label_dir, ('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
        
        self.logger.info(f"目录验证通过:")
        self.logger.info(f"  图像文件: {image_count} 个")
        self.logger.info(f"  标签文件: {label_count} 个")
    
    def _get_file_pairs(self) -> List[Tuple[str, str]]:
        """获取匹配的文件对"""
        # 获取文件列表
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        label_files = [f for f in os.listdir(self.label_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        
        # 使用统一的文件匹配函数
        matched_pairs = match_file_pairs(image_files, label_files, self.logger)
        
        if not matched_pairs:
            raise ValueError("未找到任何匹配的图像-nDSM标签对")
        
        return matched_pairs
    
    def _initialize_normalizer(self):
        """初始化归一化器"""
        # 加载统计信息
        stats_data = load_statistics_config(self.stats_json_path, self.logger)
        
        # 创建归一化器
        self.height_normalizer = HeightNormalizer(
            method=self.normalization_method,
            height_filter=self.height_filter
        )
        
        # 从JSON统计信息拟合归一化器
        self.height_normalizer.fit_from_json_stats(stats_data)
        
        # 兼容性属性
        self.global_min = self.height_normalizer.min_val
        self.global_max = self.height_normalizer.max_val
        self.global_mean = self.height_normalizer.mean_val
        self.global_std = self.height_normalizer.std_val
        
        self.logger.info(f"归一化器初始化完成:")
        self.logger.info(f"  方法: {self.normalization_method}")
        self.logger.info(f"  数据范围: [{self.global_min:.2f}, {self.global_max:.2f}] 米")
        self.logger.info(f"  均值±标准差: {self.global_mean:.2f} ± {self.global_std:.2f} 米")
    
    def _setup_transforms(self) -> transforms.Compose:
        """设置数据变换"""
        transform_list = []
        
        if self.enable_augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.01),
                transforms.RandomRotation(degrees=5, fill=0),
            ])
        
        # ImageNet标准化
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        try:
            image_file, label_file = self.file_pairs[idx]
            
            # 加载影像
            image = self._load_image(os.path.join(self.image_dir, image_file))
            
            # 加载nDSM标签
            label = self._load_label(os.path.join(self.label_dir, label_file))
            
            # 转换为tensor
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
            label_tensor = torch.from_numpy(label).float()
            
            # 应用变换
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            return image_tensor, label_tensor
            
        except Exception as e:
            self.logger.error(f"加载样本 {idx} 失败: {e}")
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self.file_pairs))
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载影像文件"""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法加载影像: {image_path}")
        
        # 颜色空间转换和归一化
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # 调整大小
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        # 归一化到[0,1]
        if image.max() > 1:
            image = image / 255.0
        
        return image
    
    def _load_label(self, label_path: str) -> np.ndarray:
        """加载nDSM标签文件"""
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            raise ValueError(f"无法加载nDSM标签: {label_path}")
        
        label = label.astype(np.float32)
        
        # 调整大小
        if label.shape != self.target_size:
            label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # 处理无效值
        INVALID_MARKER = -999.0
        valid_mask = get_valid_mask(label)
        
        # 标记无效值
        label[~valid_mask] = INVALID_MARKER
        
        # 单位转换：厘米转米
        label[valid_mask] = label[valid_mask] / 100.0
        
        # 应用高度过滤
        height_valid_mask = (
            (label >= self.height_filter['min_height']) & 
            (label <= self.height_filter['max_height'])
        )
        
        # 创建最终的有效掩码
        final_valid_mask = valid_mask & height_valid_mask
        
        # 归一化
        normalized_label = np.full_like(label, -1.0)  # 无效值标记为-1
        
        if np.any(final_valid_mask):
            valid_heights = label[final_valid_mask]
            normalized_heights = self.height_normalizer.normalize(valid_heights)
            normalized_label[final_valid_mask] = np.clip(normalized_heights, 0, 1)
        
        return normalized_label
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'min_height': self.global_min,
            'max_height': self.global_max,
            'mean_height': self.global_mean,
            'std_height': self.global_std,
            'height_filter': self.height_filter,
            'normalization_method': self.normalization_method,
            'total_samples': len(self.file_pairs),
            'target_size': self.target_size,
            'augmentation_enabled': self.enable_augmentation
        }
    
    def get_normalizer(self) -> HeightNormalizer:
        """获取归一化器"""
        return self.height_normalizer
    
    def sample_subset(self, max_samples: int, random_seed: int = 42) -> 'GAMUSDataset':
        """创建数据集的子集"""
        if len(self.file_pairs) <= max_samples:
            return self
        
        # 设置随机种子确保可重复性
        random.seed(random_seed)
        sampled_pairs = random.sample(self.file_pairs, max_samples)
        
        # 创建新的数据集实例
        subset_dataset = GAMUSDataset(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            normalization_method=self.normalization_method,
            enable_augmentation=self.enable_augmentation,
            stats_json_path=self.stats_json_path,
            height_filter=self.height_filter,
            target_size=self.target_size,
            force_recompute=False,
            logger=self.logger
        )
        
        # 替换文件对
        subset_dataset.file_pairs = sampled_pairs
        
        self.logger.info(f"创建数据集子集: {len(sampled_pairs)}/{len(self.file_pairs)}")
        
        return subset_dataset
    
    def verify_integrity(self, check_ratio: float = 0.1) -> Dict[str, Any]:
        """验证数据集完整性"""
        check_count = max(1, int(len(self.file_pairs) * check_ratio))
        check_indices = random.sample(range(len(self.file_pairs)), check_count)
        
        self.logger.info(f"验证数据集完整性 (检查 {check_count}/{len(self.file_pairs)} 个样本)")
        
        results = {
            'total_checked': check_count,
            'valid_samples': 0,
            'invalid_samples': 0,
            'error_types': {},
            'is_valid': True
        }
        
        for idx in check_indices:
            try:
                image, label = self[idx]
                
                # 基本检查
                if image is None or label is None:
                    raise ValueError("数据为None")
                
                # 形状检查
                if len(image.shape) != 3:  # C, H, W
                    raise ValueError(f"图像形状异常: {image.shape}")
                
                if len(label.shape) != 2:  # H, W
                    raise ValueError(f"标签形状异常: {label.shape}")
                
                # 数值检查
                if torch.isnan(image).any() or torch.isinf(image).any():
                    raise ValueError("图像包含NaN或Inf")
                
                if torch.isnan(label).any() or torch.isinf(label).any():
                    raise ValueError("标签包含NaN或Inf")
                
                results['valid_samples'] += 1
                
            except Exception as e:
                results['invalid_samples'] += 1
                error_type = type(e).__name__
                results['error_types'][error_type] = results['error_types'].get(error_type, 0) + 1
                
                if results['invalid_samples'] <= 5:  # 只记录前5个错误
                    self.logger.warning(f"样本 {idx} 验证失败: {e}")
        
        # 判断整体有效性
        invalid_ratio = results['invalid_samples'] / results['total_checked']
        results['is_valid'] = invalid_ratio < 0.1  # 无效样本小于10%认为有效
        results['invalid_ratio'] = invalid_ratio
        
        self.logger.info(f"数据集完整性验证结果:")
        self.logger.info(f"  有效样本: {results['valid_samples']}/{results['total_checked']}")
        self.logger.info(f"  无效样本: {results['invalid_samples']}/{results['total_checked']}")
        self.logger.info(f"  无效比例: {invalid_ratio:.2%}")
        self.logger.info(f"  整体状态: {'✓ 有效' if results['is_valid'] else '✗ 无效'}")
        
        return results


class GAMUSDatasetConfig:
    """GAMUS数据集配置类"""
    
    def __init__(self,
                 image_dir: str,
                 label_dir: str,
                 normalization_method: str = 'minmax',
                 enable_augmentation: bool = False,
                 stats_json_path: Optional[str] = None,
                 height_filter: Optional[Dict[str, float]] = None,
                 target_size: Tuple[int, int] = (448, 448),
                 batch_size: int = 8,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last: bool = True):
        
        self.dataset_config = {
            'image_dir': image_dir,
            'label_dir': label_dir,
            'normalization_method': normalization_method,
            'enable_augmentation': enable_augmentation,
            'stats_json_path': stats_json_path,
            'height_filter': height_filter or {'min_height': -5.0, 'max_height': 200.0},
            'target_size': target_size
        }
        
        self.dataloader_config = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last
        }
    
    def create_dataset(self, logger: Optional[logging.Logger] = None) -> GAMUSDataset:
        """创建数据集"""
        return GAMUSDataset(logger=logger, **self.dataset_config)
    
    def create_dataloader(self, dataset: GAMUSDataset = None,
                         logger: Optional[logging.Logger] = None) -> Tuple[DataLoader, GAMUSDataset]:
        """创建数据加载器"""
        if dataset is None:
            dataset = self.create_dataset(logger)
        
        dataloader = DataLoader(dataset, **self.dataloader_config)
        
        # 添加归一化器属性以便访问
        dataloader.height_normalizer = dataset.get_normalizer()
        
        return dataloader, dataset


def create_gamus_dataloader(image_dir: str, label_dir: str, 
                           batch_size: int = 8, shuffle: bool = True, 
                           normalization_method: str = 'minmax', 
                           enable_augmentation: bool = False,
                           stats_json_path: Optional[str] = None, 
                           height_filter: Optional[Dict[str, float]] = None, 
                           target_size: Tuple[int, int] = (448, 448),
                           force_recompute: bool = False, 
                           num_workers: int = 4,
                           pin_memory: bool = True,
                           drop_last: bool = True,
                           logger: Optional[logging.Logger] = None) -> Tuple[DataLoader, GAMUSDataset]:
    """
    创建GAMUS数据加载器的便利函数
    
    Args:
        image_dir: 影像目录
        label_dir: 标签目录
        batch_size: 批次大小
        shuffle: 是否打乱数据
        normalization_method: 归一化方法
        enable_augmentation: 是否启用数据增强
        stats_json_path: 统计信息JSON文件路径
        height_filter: 高度过滤器
        target_size: 目标图像尺寸
        force_recompute: 是否强制重新计算统计信息
        num_workers: 数据加载线程数
        pin_memory: 是否使用pin_memory
        drop_last: 是否丢弃最后一个不完整批次
        logger: 日志记录器
    
    Returns:
        数据加载器和数据集
    """
    config = GAMUSDatasetConfig(
        image_dir=image_dir,
        label_dir=label_dir,
        normalization_method=normalization_method,
        enable_augmentation=enable_augmentation,
        stats_json_path=stats_json_path,
        height_filter=height_filter,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return config.create_dataloader(logger=logger)


def create_multi_split_dataloaders(data_dir: str, 
                                  splits: List[str] = None,
                                  configs: Dict[str, Dict] = None,
                                  common_config: Dict = None,
                                  logger: Optional[logging.Logger] = None) -> Dict[str, Tuple[DataLoader, GAMUSDataset]]:
    """
    创建多个数据分割的数据加载器
    
    Args:
        data_dir: 数据根目录
        splits: 数据分割列表
        configs: 每个分割的特定配置
        common_config: 通用配置
        logger: 日志记录器
    
    Returns:
        数据加载器字典
    """
    if logger is None:
        logger = setup_logger('multi_split_dataloaders')
    
    if splits is None:
        splits = ['train', 'val']
    
    configs = configs or {}
    common_config = common_config or {}
    
    # 验证数据结构
    data_structure = validate_data_structure(data_dir, splits, logger)
    
    dataloaders = {}
    
    for split in splits:
        if not data_structure[split]['image_dir']['exists'] or not data_structure[split]['depth_dir']['exists']:
            logger.warning(f"跳过 {split} 数据集：目录不存在")
            continue
        
        if not data_structure[split]['matched_pairs']:
            logger.warning(f"跳过 {split} 数据集：没有匹配的文件对")
            continue
        
        # 合并配置
        split_config = {**common_config, **configs.get(split, {})}
        
        # 设置分割特定的默认值
        if split == 'train':
            split_config.setdefault('shuffle', True)
            split_config.setdefault('enable_augmentation', True)
            split_config.setdefault('drop_last', True)
        else:
            split_config.setdefault('shuffle', False)
            split_config.setdefault('enable_augmentation', False)
            split_config.setdefault('drop_last', False)
        
        # 创建数据加载器
        try:
            dataloader, dataset = create_gamus_dataloader(
                image_dir=data_structure[split]['image_dir']['path'],
                label_dir=data_structure[split]['depth_dir']['path'],
                logger=logger,
                **split_config
            )
            
            dataloaders[split] = (dataloader, dataset)
            logger.info(f"✓ {split} 数据集创建成功: {len(dataset)} 个样本")
            
        except Exception as e:
            logger.error(f"创建 {split} 数据集失败: {e}")
            continue
    
    return dataloaders


# 使用示例
if __name__ == '__main__':
    # 设置日志
    logger = setup_logger('gamus_dataset_test')
    
    # 配置参数
    config = {
        'normalization_method': 'minmax',
        'stats_json_path': './gamus_full_stats.json',
        'height_filter': {'min_height': -5.0, 'max_height': 200.0},
        'batch_size': 8,
        'num_workers': 2
    }
    
    try:
        # 创建多分割数据加载器
        dataloaders = create_multi_split_dataloaders(
            data_dir='/path/to/data',
            splits=['train', 'val'],
            common_config=config,
            logger=logger
        )
        
        # 测试数据加载
        for split, (dataloader, dataset) in dataloaders.items():
            logger.info(f"测试 {split} 数据集:")
            
            # 验证数据集完整性
            integrity_result = dataset.verify_integrity(check_ratio=0.05)
            
            # 打印统计信息
            stats = dataset.get_stats()
            logger.info(f"  统计信息: {stats}")
            
            # 测试数据加载
            for batch_idx, (images, labels) in enumerate(dataloader):
                logger.info(f"  批次 {batch_idx}: 图像 {images.shape}, 标签 {labels.shape}")
                if batch_idx >= 2:  # 只测试前几个批次
                    break
            
            # 清理内存
            clear_memory()
    
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()