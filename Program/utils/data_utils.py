#!/usr/bin/env python3
"""
数据处理工具模块
统一管理数据集创建、文件匹配、统计信息处理等功能
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import random

from .common import setup_logger, validate_paths, count_files


def extract_base_name(filename: str, is_image: bool = True) -> Optional[str]:
    """
    提取文件的基础名称用于匹配
    
    Args:
        filename: 文件名
        is_image: 是否为图像文件
    
    Returns:
        基础名称
    """
    try:
        # 处理不同的命名模式
        if '_RGB_' in filename and is_image:
            base_name = filename.replace('_RGB_', '_').rsplit('.', 1)[0]
        elif '_AGL_' in filename and not is_image:
            base_name = filename.replace('_AGL_', '_').rsplit('.', 1)[0]
        elif '_DSM_' in filename and not is_image:
            base_name = filename.replace('_DSM_', '_').rsplit('.', 1)[0]
        elif '_depth_' in filename and not is_image:
            base_name = filename.replace('_depth_', '_').rsplit('.', 1)[0]
        else:
            # 通用处理：去掉扩展名
            base_name = Path(filename).stem
        
        return base_name
    except Exception:
        return None


def match_file_pairs(image_files: List[str], depth_files: List[str], 
                    logger: Optional[logging.Logger] = None) -> List[Tuple[str, str]]:
    """
    匹配图像和深度文件对
    
    Args:
        image_files: 图像文件列表
        depth_files: 深度文件列表
        logger: 日志记录器
    
    Returns:
        匹配的文件对列表
    """
    if logger is None:
        logger = setup_logger('data_utils')
    
    # 创建映射字典
    image_dict = {}
    depth_dict = {}
    
    # 处理图像文件
    for img_file in image_files:
        base_name = extract_base_name(img_file, is_image=True)
        if base_name:
            image_dict[base_name] = img_file
    
    # 处理深度文件
    for depth_file in depth_files:
        base_name = extract_base_name(depth_file, is_image=False)
        if base_name:
            depth_dict[base_name] = depth_file
    
    # 匹配文件对
    matched_pairs = []
    for base_name in image_dict:
        if base_name in depth_dict:
            matched_pairs.append((image_dict[base_name], depth_dict[base_name]))
    
    logger.info(f"文件匹配结果:")
    logger.info(f"  图像文件: {len(image_files)} 个")
    logger.info(f"  深度文件: {len(depth_files)} 个")
    logger.info(f"  成功匹配: {len(matched_pairs)} 对")
    logger.info(f"  未匹配图像: {len(image_files) - len(matched_pairs)} 个")
    logger.info(f"  未匹配深度: {len(depth_files) - len(matched_pairs)} 个")
    
    return matched_pairs


def validate_data_structure(data_dir: str, splits: List[str] = None, 
                          logger: Optional[logging.Logger] = None) -> Dict[str, Dict]:
    """
    验证数据目录结构
    
    Args:
        data_dir: 数据根目录
        splits: 数据分割列表，默认为 ['train', 'val']
        logger: 日志记录器
    
    Returns:
        验证结果字典
    """
    if logger is None:
        logger = setup_logger('data_utils')
    
    if splits is None:
        splits = ['train', 'val']
    
    logger.info(f"验证数据目录结构: {data_dir}")
    
    results = {}
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    for split in splits:
        logger.info(f"检查 {split} 数据集...")
        
        # 图像目录
        image_dir = os.path.join(data_dir, split, 'images')
        depth_dir = os.path.join(data_dir, split, 'depths')
        
        split_result = {
            'image_dir': {
                'path': image_dir,
                'exists': os.path.exists(image_dir),
                'files': [],
                'count': 0
            },
            'depth_dir': {
                'path': depth_dir,
                'exists': os.path.exists(depth_dir),
                'files': [],
                'count': 0
            },
            'matched_pairs': []
        }
        
        # 检查图像目录
        if split_result['image_dir']['exists']:
            image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(image_extensions)]
            split_result['image_dir']['files'] = image_files
            split_result['image_dir']['count'] = len(image_files)
            
            logger.info(f"  ✓ {split}/images: {len(image_files)} 个文件")
            if image_files:
                logger.info(f"    示例: {image_files[0]}")
        else:
            logger.warning(f"  ⚠️  {split}/images: 目录不存在")
        
        # 检查深度目录
        if split_result['depth_dir']['exists']:
            depth_files = [f for f in os.listdir(depth_dir) 
                          if f.lower().endswith(image_extensions)]
            split_result['depth_dir']['files'] = depth_files
            split_result['depth_dir']['count'] = len(depth_files)
            
            logger.info(f"  ✓ {split}/depths: {len(depth_files)} 个文件")
            if depth_files:
                logger.info(f"    示例: {depth_files[0]}")
        else:
            logger.warning(f"  ⚠️  {split}/depths: 目录不存在")
        
        # 匹配文件对
        if (split_result['image_dir']['exists'] and split_result['depth_dir']['exists'] and
            split_result['image_dir']['files'] and split_result['depth_dir']['files']):
            
            matched_pairs = match_file_pairs(
                split_result['image_dir']['files'],
                split_result['depth_dir']['files'],
                logger
            )
            split_result['matched_pairs'] = matched_pairs
        
        results[split] = split_result
    
    return results


def load_statistics_config(stats_json_path: str, 
                         logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    加载统计信息配置文件
    
    Args:
        stats_json_path: 统计信息JSON文件路径
        logger: 日志记录器
    
    Returns:
        统计信息字典
    """
    if logger is None:
        logger = setup_logger('data_utils')
    
    if not os.path.exists(stats_json_path):
        raise FileNotFoundError(f"统计信息文件不存在: {stats_json_path}")
    
    try:
        with open(stats_json_path, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        
        # 验证必要字段
        required_fields = ['global_statistics', 'processing_info']
        for field in required_fields:
            if field not in stats_data:
                raise ValueError(f"统计信息文件缺少必要字段: {field}")
        
        logger.info(f"成功加载统计信息: {stats_json_path}")
        logger.info(f"  处理文件数: {stats_data['processing_info'].get('processed_files', 'N/A')}")
        logger.info(f"  数据点数: {stats_data['global_statistics'].get('total_samples', 'N/A'):,}")
        
        return stats_data
        
    except Exception as e:
        raise ValueError(f"加载统计信息文件失败: {e}")


def create_dataloader_config(image_dir: str, label_dir: str, batch_size: int = 8,
                           shuffle: bool = True, num_workers: int = 4,
                           pin_memory: bool = True, drop_last: bool = True,
                           normalization_method: str = 'minmax',
                           enable_augmentation: bool = False,
                           stats_json_path: Optional[str] = None,
                           height_filter: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    创建数据加载器配置
    
    Args:
        image_dir: 图像目录
        label_dir: 标签目录
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作线程数
        pin_memory: 是否使用pin_memory
        drop_last: 是否丢弃最后一个不完整批次
        normalization_method: 归一化方法
        enable_augmentation: 是否启用数据增强
        stats_json_path: 统计信息文件路径
        height_filter: 高度过滤器
    
    Returns:
        数据加载器配置字典
    """
    config = {
        'dataset_config': {
            'image_dir': image_dir,
            'label_dir': label_dir,
            'normalization_method': normalization_method,
            'enable_augmentation': enable_augmentation,
            'stats_json_path': stats_json_path,
            'height_filter': height_filter or {'min_height': -5.0, 'max_height': 200.0}
        },
        'dataloader_config': {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last
        }
    }
    
    return config


def sample_dataset(dataset, max_samples: int, logger: Optional[logging.Logger] = None):
    """
    对数据集进行采样
    
    Args:
        dataset: 原始数据集
        max_samples: 最大样本数
        logger: 日志记录器
    
    Returns:
        采样后的数据集
    """
    if logger is None:
        logger = setup_logger('data_utils')
    
    if len(dataset) <= max_samples:
        return dataset
    
    # 随机采样
    indices = random.sample(range(len(dataset)), max_samples)
    sampled_dataset = Subset(dataset, indices)
    
    logger.info(f"数据集采样: {len(dataset)} -> {len(sampled_dataset)}")
    
    return sampled_dataset


def get_dataset_statistics(dataset, sample_size: int = 1000,
                         logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    获取数据集统计信息
    
    Args:
        dataset: 数据集对象
        sample_size: 采样大小用于统计
        logger: 日志记录器
    
    Returns:
        统计信息字典
    """
    if logger is None:
        logger = setup_logger('data_utils')
    
    logger.info(f"计算数据集统计信息 (采样: {sample_size})")
    
    # 采样数据
    sample_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    
    images_stats = []
    labels_stats = []
    
    for idx in sample_indices:
        try:
            image, label = dataset[idx]
            
            # 转换为numpy
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            
            images_stats.append({
                'shape': image.shape if hasattr(image, 'shape') else None,
                'min': float(np.min(image)) if hasattr(image, 'min') else None,
                'max': float(np.max(image)) if hasattr(image, 'max') else None,
                'mean': float(np.mean(image)) if hasattr(image, 'mean') else None
            })
            
            labels_stats.append({
                'shape': label.shape if hasattr(label, 'shape') else None,
                'min': float(np.min(label)) if hasattr(label, 'min') else None,
                'max': float(np.max(label)) if hasattr(label, 'max') else None,
                'mean': float(np.mean(label)) if hasattr(label, 'mean') else None
            })
            
        except Exception as e:
            logger.warning(f"采样索引 {idx} 处理失败: {e}")
            continue
    
    # 汇总统计
    stats = {
        'total_samples': len(dataset),
        'sampled_samples': len(images_stats),
        'image_stats': {
            'shape': images_stats[0]['shape'] if images_stats else None,
            'min': min([s['min'] for s in images_stats if s['min'] is not None], default=None),
            'max': max([s['max'] for s in images_stats if s['max'] is not None], default=None),
            'mean': np.mean([s['mean'] for s in images_stats if s['mean'] is not None]) if images_stats else None
        },
        'label_stats': {
            'shape': labels_stats[0]['shape'] if labels_stats else None,
            'min': min([s['min'] for s in labels_stats if s['min'] is not None], default=None),
            'max': max([s['max'] for s in labels_stats if s['max'] is not None], default=None),
            'mean': np.mean([s['mean'] for s in labels_stats if s['mean'] is not None]) if labels_stats else None
        }
    }
    
    logger.info(f"数据集统计完成:")
    logger.info(f"  总样本数: {stats['total_samples']}")
    logger.info(f"  图像形状: {stats['image_stats']['shape']}")
    logger.info(f"  标签形状: {stats['label_stats']['shape']}")
    
    return stats


def verify_dataset_integrity(dataset, check_ratio: float = 0.1,
                           logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    验证数据集完整性
    
    Args:
        dataset: 数据集对象
        check_ratio: 检查比例
        logger: 日志记录器
    
    Returns:
        验证结果字典
    """
    if logger is None:
        logger = setup_logger('data_utils')
    
    check_count = max(1, int(len(dataset) * check_ratio))
    check_indices = random.sample(range(len(dataset)), check_count)
    
    logger.info(f"验证数据集完整性 (检查 {check_count}/{len(dataset)} 个样本)")
    
    results = {
        'total_checked': check_count,
        'valid_samples': 0,
        'invalid_samples': 0,
        'error_types': {},
        'is_valid': True
    }
    
    for idx in check_indices:
        try:
            image, label = dataset[idx]
            
            # 基本检查
            if image is None or label is None:
                raise ValueError("数据为None")
            
            # 形状检查
            if isinstance(image, torch.Tensor):
                if len(image.shape) != 3:  # C, H, W
                    raise ValueError(f"图像形状异常: {image.shape}")
            
            if isinstance(label, torch.Tensor):
                if len(label.shape) != 2:  # H, W
                    raise ValueError(f"标签形状异常: {label.shape}")
            
            # 数值检查
            if isinstance(image, torch.Tensor):
                if torch.isnan(image).any() or torch.isinf(image).any():
                    raise ValueError("图像包含NaN或Inf")
            
            if isinstance(label, torch.Tensor):
                if torch.isnan(label).any() or torch.isinf(label).any():
                    raise ValueError("标签包含NaN或Inf")
            
            results['valid_samples'] += 1
            
        except Exception as e:
            results['invalid_samples'] += 1
            error_type = type(e).__name__
            results['error_types'][error_type] = results['error_types'].get(error_type, 0) + 1
            
            if results['invalid_samples'] <= 5:  # 只记录前5个错误
                logger.warning(f"样本 {idx} 验证失败: {e}")
    
    # 判断整体有效性
    invalid_ratio = results['invalid_samples'] / results['total_checked']
    results['is_valid'] = invalid_ratio < 0.1  # 无效样本小于10%认为有效
    results['invalid_ratio'] = invalid_ratio
    
    logger.info(f"数据集完整性验证结果:")
    logger.info(f"  有效样本: {results['valid_samples']}/{results['total_checked']}")
    logger.info(f"  无效样本: {results['invalid_samples']}/{results['total_checked']}")
    logger.info(f"  无效比例: {invalid_ratio:.2%}")
    logger.info(f"  整体状态: {'✓ 有效' if results['is_valid'] else '✗ 无效'}")
    
    if results['error_types']:
        logger.info(f"  错误类型统计: {results['error_types']}")
    
    return results


def create_data_split_config(data_dir: str, splits: List[str] = None,
                           logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """
    创建数据分割配置
    
    Args:
        data_dir: 数据根目录
        splits: 数据分割列表
        logger: 日志记录器
    
    Returns:
        数据分割配置字典
    """
    if logger is None:
        logger = setup_logger('data_utils')
    
    if splits is None:
        splits = ['train', 'val', 'test']
    
    config = {}
    
    for split in splits:
        image_dir = os.path.join(data_dir, split, 'images')
        depth_dir = os.path.join(data_dir, split, 'depths')
        
        if os.path.exists(image_dir) and os.path.exists(depth_dir):
            config[split] = {
                'image_dir': image_dir,
                'depth_dir': depth_dir
            }
            logger.info(f"找到 {split} 数据集: {image_dir}")
        else:
            logger.warning(f"未找到 {split} 数据集")
    
    return config


def get_valid_mask(data: np.ndarray) -> np.ndarray:
    """
    获取有效数据掩码
    
    Args:
        data: 输入数据数组
    
    Returns:
        有效数据掩码
    """
    invalid_values = [-9999, -32768, 9999, 32767]
    valid_mask = ~(np.isnan(data) | np.isinf(data))
    
    for invalid_val in invalid_values:
        valid_mask = valid_mask & (data != invalid_val)
    
    return valid_mask


def compute_file_statistics(file_pairs: List[Tuple[str, str]], depth_dir: str,
                          height_filter: Dict[str, float],
                          logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    计算文件统计信息
    
    Args:
        file_pairs: 文件对列表
        depth_dir: 深度文件目录
        height_filter: 高度过滤器
        logger: 日志记录器
    
    Returns:
        统计信息字典
    """
    if logger is None:
        logger = setup_logger('data_utils')
    
    logger.info(f"开始计算文件统计信息...")
    logger.info(f"  总文件数: {len(file_pairs)}")
    logger.info(f"  高度过滤范围: [{height_filter['min_height']:.1f}, {height_filter['max_height']:.1f}] 米")
    
    all_heights = []
    processed_count = 0
    error_count = 0
    total_pixels = 0
    valid_pixels = 0
    
    for idx, (image_file, depth_file) in enumerate(file_pairs):
        if idx % 1000 == 0 and idx > 0:
            logger.info(f"  处理进度: {idx}/{len(file_pairs)} ({idx/len(file_pairs)*100:.1f}%)")
        
        try:
            depth_path = os.path.join(depth_dir, depth_file)
            
            # 加载深度数据
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            if depth_data is None:
                error_count += 1
                continue
            
            # 转换数据类型
            depth_data = depth_data.astype(np.float32)
            total_pixels += depth_data.size
            
            # 过滤无效值
            valid_mask = get_valid_mask(depth_data)
            valid_heights = depth_data[valid_mask]
            valid_pixels += len(valid_heights)
            
            if len(valid_heights) > 0:
                # 单位转换：厘米转米
                valid_heights = valid_heights / 100.0
                
                # 应用高度过滤
                filtered_heights = valid_heights[
                    (valid_heights >= height_filter['min_height']) & 
                    (valid_heights <= height_filter['max_height'])
                ]
                
                if len(filtered_heights) > 0:
                    all_heights.append(filtered_heights)
                    processed_count += 1
            
        except Exception as e:
            error_count += 1
            logger.debug(f"处理文件 {depth_file} 失败: {e}")
            continue
    
    if not all_heights:
        raise ValueError("无法提取有效的高度数据")
    
    # 合并所有数据
    logger.info("合并并计算最终统计信息...")
    combined_heights = np.concatenate(all_heights)
    
    # 计算统计信息
    stats = {
        'processing_info': {
            'total_files': len(file_pairs),
            'processed_files': processed_count,
            'error_files': error_count,
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixels,
            'height_filter': height_filter
        },
        'global_statistics': {
            'min': float(np.min(combined_heights)),
            'max': float(np.max(combined_heights)),
            'mean': float(np.mean(combined_heights)),
            'std': float(np.std(combined_heights)),
            'median': float(np.median(combined_heights)),
            'total_samples': len(combined_heights)
        },
        'percentiles': {
            f'p{p}': float(np.percentile(combined_heights, p))
            for p in [1, 5, 25, 50, 75, 95, 99]
        }
    }
    
    # 计算直方图
    hist, bin_edges = np.histogram(combined_heights, bins=100)
    stats['histogram'] = {
        'counts': hist.tolist(),
        'bin_edges': bin_edges.tolist()
    }
    
    logger.info(f"统计信息计算完成:")
    logger.info(f"  成功处理: {processed_count}/{len(file_pairs)} 个文件")
    logger.info(f"  数据范围: [{stats['global_statistics']['min']:.2f}, {stats['global_statistics']['max']:.2f}] 米")
    logger.info(f"  有效数据点: {stats['global_statistics']['total_samples']:,} 个")
    
    return stats