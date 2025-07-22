import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings
from improved_normalization_loss import HeightNormalizer
import gc
warnings.filterwarnings('ignore')

class GAMUSDataset(Dataset):
    """内存优化版GAMUS nDSM数据集加载器"""
    
    def __init__(self, 
                 image_dir: str, 
                 label_dir: str, 
                 normalization_method: str = 'percentile',
                 enable_augmentation: bool = False,
                 stats_json_path: Optional[str] = None,
                 height_filter: Optional[Dict[str, float]] = None,
                 force_recompute: bool = False,
                 max_memory_samples: int = 1000):  # 新增：限制内存中的样本数量
        """
        初始化GAMUS数据集 - 内存优化版
        
        参数:
            max_memory_samples: 用于统计计算的最大样本数，减少内存占用
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.normalization_method = normalization_method
        self.enable_augmentation = enable_augmentation
        self.stats_json_path = stats_json_path
        self.force_recompute = force_recompute
        self.max_memory_samples = max_memory_samples
        
        # 设置高度过滤器（默认合理范围）
        self.height_filter = height_filter or {'min_height': -5.0, 'max_height': 100.0}
        
        # 验证目录
        self._validate_directories()
        
        # 获取文件对
        self.file_pairs = self._get_file_pairs()
        
        # 初始化统计信息和归一化器
        self._initialize_normalizer()
        
        # 设置数据变换
        self.transform = self._setup_transforms()
        
        logging.info(f"GAMUS数据集初始化完成: {len(self.file_pairs)} 个样本")
        logging.info(f"高度过滤范围: [{self.height_filter['min_height']:.1f}, {self.height_filter['max_height']:.1f}] 米")
    
    def _validate_directories(self):
        """验证目录存在性"""
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"影像目录不存在: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"nDSM标签目录不存在: {self.label_dir}")
    
    def _get_file_pairs(self):
        """获取匹配的文件对"""
        # 获取文件列表
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        label_files = [f for f in os.listdir(self.label_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        
        # 创建匹配映射
        image_dict = {}
        label_dict = {}
        
        # 处理图像文件
        for img_file in image_files:
            if '_RGB_' in img_file:
                base_name = img_file.replace('_RGB_', '_').rsplit('.', 1)[0]
            else:
                base_name = Path(img_file).stem
            image_dict[base_name] = img_file
        
        # 处理标签文件  
        for label_file in label_files:
            if '_AGL_' in label_file:
                base_name = label_file.replace('_AGL_', '_').rsplit('.', 1)[0]
            else:
                base_name = Path(label_file).stem
            label_dict[base_name] = label_file
        
        # 找到匹配的文件对
        matched_pairs = []
        for base_name in image_dict:
            if base_name in label_dict:
                matched_pairs.append((image_dict[base_name], label_dict[base_name]))
        
        if not matched_pairs:
            raise ValueError("未找到任何匹配的图像-nDSM标签对")
        
        logging.info(f"成功匹配 {len(matched_pairs)} 个文件对")
        return matched_pairs
    
    def _initialize_normalizer(self):
        """初始化归一化器 - 内存优化版"""
        # 尝试从JSON加载统计信息
        if not self.force_recompute and self.stats_json_path and os.path.exists(self.stats_json_path):
            if self._load_from_json():
                logging.info(f"从JSON文件加载统计信息: {self.stats_json_path}")
                return
        
        # 重新计算统计信息（内存优化版）
        logging.info("开始计算统计信息（内存优化模式）...")
        self._compute_statistics_optimized()
        
        # 保存统计信息
        if self.stats_json_path:
            self._save_to_json()
    
    def _load_from_json(self) -> bool:
        """从JSON文件加载统计信息"""
        try:
            with open(self.stats_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'global_statistics' not in data:
                logging.warning("JSON文件格式不正确")
                return False
            
            stats = data['global_statistics']
            
            # 加载基本统计信息
            self.global_min = float(stats['min'])
            self.global_max = float(stats['max'])
            self.global_mean = float(stats['mean'])
            self.global_std = float(stats['std'])
            
            # 应用高度过滤
            original_min, original_max = self.global_min, self.global_max
            self.global_min = max(self.global_min, self.height_filter['min_height'])
            self.global_max = min(self.global_max, self.height_filter['max_height'])
            
            if self.global_min != original_min or self.global_max != original_max:
                logging.info(f"应用高度过滤: [{original_min:.1f}, {original_max:.1f}] -> [{self.global_min:.1f}, {self.global_max:.1f}]")
            
            # 创建归一化器
            self.height_normalizer = HeightNormalizer(self.normalization_method)
            
            # 使用过滤后的范围重新拟合归一化器
            dummy_data = np.linspace(self.global_min, self.global_max, 1000)
            self.height_normalizer.fit(dummy_data)
            
            logging.info(f"数据范围: [{self.global_min:.1f}, {self.global_max:.1f}] 米")
            return True
            
        except Exception as e:
            logging.error(f"加载JSON文件失败: {e}")
            return False
    
    def _compute_statistics_optimized(self):
        """计算统计信息 - 内存优化版本"""
        # 随机采样文件，避免加载所有数据到内存
        import random
        sample_files = self.file_pairs.copy()
        if len(sample_files) > self.max_memory_samples:
            sample_files = random.sample(sample_files, self.max_memory_samples)
            logging.info(f"随机采样 {len(sample_files)} 个文件用于统计计算")
        
        # 使用在线算法计算统计信息，避免存储所有数据
        n_total = 0
        sum_values = 0.0
        sum_squares = 0.0
        min_value = float('inf')
        max_value = float('-inf')
        
        logging.info(f"处理 {len(sample_files)} 个采样文件...")
        
        for idx, (_, label_file) in enumerate(sample_files):
            if idx % 100 == 0:
                logging.info(f"进度: {idx+1}/{len(sample_files)}")
            
            try:
                label_path = os.path.join(self.label_dir, label_file)
                label_data = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                
                if label_data is None:
                    continue
                
                # 转换数据类型
                label_data = label_data.astype(np.float32)
                
                # 过滤无效值
                valid_mask = self._get_valid_mask(label_data)
                valid_heights = label_data[valid_mask]
                
                if len(valid_heights) > 0:
                    # 单位转换：厘米转米
                    valid_heights = valid_heights / 100.0
                    
                    # 应用高度过滤
                    filtered_heights = valid_heights[
                        (valid_heights >= self.height_filter['min_height']) & 
                        (valid_heights <= self.height_filter['max_height'])
                    ]
                    
                    if len(filtered_heights) > 0:
                        # 在线更新统计信息
                        n_current = len(filtered_heights)
                        n_total += n_current
                        sum_values += np.sum(filtered_heights)
                        sum_squares += np.sum(filtered_heights ** 2)
                        min_value = min(min_value, np.min(filtered_heights))
                        max_value = max(max_value, np.max(filtered_heights))
                        
                        # 立即删除数据释放内存
                        del filtered_heights
                        del valid_heights
                
                # 清理内存
                del label_data
                
                # 每处理10个文件强制垃圾回收
                if idx % 10 == 0:
                    gc.collect()
                        
            except Exception as e:
                logging.warning(f"处理文件 {label_file} 失败: {e}")
                continue
        
        if n_total == 0:
            raise ValueError("无法提取有效的高度数据")
        
        # 计算最终统计信息
        self.global_min = float(min_value)
        self.global_max = float(max_value)
        self.global_mean = float(sum_values / n_total)
        variance = (sum_squares / n_total) - (self.global_mean ** 2)
        self.global_std = float(np.sqrt(max(variance, 0)))
        
        # 创建并训练归一化器（使用理论分布而不是实际数据）
        self.height_normalizer = HeightNormalizer(self.normalization_method)
        
        # 使用统计信息创建虚拟数据来拟合归一化器
        if self.normalization_method == 'percentile':
            # 对于百分位数方法，创建足够的样本点
            dummy_data = np.random.normal(
                self.global_mean, 
                self.global_std, 
                size=10000
            )
            # 限制在合理范围内
            dummy_data = np.clip(dummy_data, self.global_min, self.global_max)
        else:
            # 对于其他方法，简单的线性分布即可
            dummy_data = np.linspace(self.global_min, self.global_max, 1000)
        
        self.height_normalizer.fit(dummy_data)
        
        # 强制垃圾回收
        del dummy_data
        gc.collect()
        
        logging.info(f"统计信息计算完成:")
        logging.info(f"  数据范围: [{self.global_min:.1f}, {self.global_max:.1f}] 米")
        logging.info(f"  均值±标准差: {self.global_mean:.1f} ± {self.global_std:.1f} 米")
        logging.info(f"  处理数据点: {n_total:,} 个")
    
    def _get_valid_mask(self, data):
        """获取有效数据掩码"""
        invalid_values = [-9999, -32768, 9999, 32767]
        valid_mask = ~(np.isnan(data) | np.isinf(data))
        
        for invalid_val in invalid_values:
            valid_mask = valid_mask & (data != invalid_val)
        
        return valid_mask
    
    def _save_to_json(self):
        """保存统计信息到JSON文件"""
        try:
            data = {
                "timestamp": self._get_timestamp(),
                "processing_info": {
                    "total_files": len(self.file_pairs),
                    "sampled_files": min(len(self.file_pairs), self.max_memory_samples),
                    "height_filter": self.height_filter,
                    "normalization_method": self.normalization_method
                },
                "global_statistics": {
                    "min": float(self.global_min),
                    "max": float(self.global_max),
                    "mean": float(self.global_mean),
                    "std": float(self.global_std)
                }
            }
            
            os.makedirs(os.path.dirname(self.stats_json_path), exist_ok=True)
            
            with open(self.stats_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"统计信息已保存到: {self.stats_json_path}")
            
        except Exception as e:
            logging.error(f"保存JSON文件失败: {e}")
    
    def _get_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _setup_transforms(self):
        """设置数据变换"""
        transform_list = []
        
        if self.enable_augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.01),
                transforms.RandomRotation(degrees=5, fill=0),
            ])
        
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        """获取单个样本 - 内存优化版"""
        try:
            image_file, label_file = self.file_pairs[idx]
            
            # 加载影像
            image = self._load_image(os.path.join(self.image_dir, image_file))
            
            # 加载nDSM标签
            label = self._load_label(os.path.join(self.label_dir, label_file))
            
            # 转换为tensor
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
            label_tensor = torch.from_numpy(label).float()
            
            # 立即清理numpy数组
            del image, label
            
            # 应用变换
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            return image_tensor, label_tensor
            
        except Exception as e:
            logging.error(f"加载样本 {idx} 失败: {e}")
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self.file_pairs))
    
    def _load_image(self, image_path):
        """加载影像文件 - 内存优化版"""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法加载影像: {image_path}")
        
        # 颜色空间转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        if image.shape[:2] != (448, 448):
            image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_CUBIC)
        
        # 转换为float32并归一化
        image = image.astype(np.float32)
        if image.max() > 1:
            image = image / 255.0
        
        return image
    
    def _load_label(self, label_path):
        """加载nDSM标签文件 - 内存优化版"""
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            raise ValueError(f"无法加载nDSM标签: {label_path}")
        
        label = label.astype(np.float32)
        
        # 调整大小
        if label.shape != (448, 448):
            label = cv2.resize(label, (448, 448), interpolation=cv2.INTER_LINEAR)
        
        # 处理无效值
        INVALID_MARKER = -999.0
        valid_mask = self._get_valid_mask(label)
        
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
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'min_height': self.global_min,
            'max_height': self.global_max,
            'mean_height': self.global_mean,
            'std_height': self.global_std,
            'height_filter': self.height_filter,
            'normalization_method': self.normalization_method,
            'total_samples': len(self.file_pairs)
        }
    
    def get_normalizer(self):
        """获取归一化器"""
        return self.height_normalizer


# 内存友好的便利函数
def create_gamus_dataloader(image_dir, label_dir, batch_size=8, shuffle=True, 
                           normalization_method='percentile', enable_augmentation=False,
                           stats_json_path=None, height_filter=None, 
                           force_recompute=False, num_workers=2,  # 减少默认worker数量
                           max_memory_samples=1000):  # 新增参数
    """
    创建内存优化的GAMUS数据加载器
    
    参数:
        max_memory_samples: 用于统计计算的最大样本数，减少内存占用
        其他参数同原函数
    """
    dataset = GAMUSDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        normalization_method=normalization_method,
        enable_augmentation=enable_augmentation,
        stats_json_path=stats_json_path,
        height_filter=height_filter,
        force_recompute=force_recompute,
        max_memory_samples=max_memory_samples
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # 改为False减少内存占用
        drop_last=True if shuffle else False,
        persistent_workers=False  # 避免worker进程常驻内存
    )
    dataloader.height_normalizer = dataset.height_normalizer
    return dataloader, dataset