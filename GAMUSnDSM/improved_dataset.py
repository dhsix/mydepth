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

warnings.filterwarnings('ignore')

class GAMUSDataset(Dataset):
    """简化版GAMUS nDSM数据集加载器"""
    
    def __init__(self, 
                 image_dir: str, 
                 label_dir: str, 
                 normalization_method: str = 'percentile',
                 enable_augmentation: bool = False,
                 stats_json_path: Optional[str] = None,
                 height_filter: Optional[Dict[str, float]] = None,
                 force_recompute: bool = False):
        """
        初始化GAMUS数据集
        
        参数:
            image_dir: 影像切片目录
            label_dir: nDSM标签切片目录  
            normalization_method: 归一化方法 ('percentile', 'minmax', 'zscore')
            enable_augmentation: 是否启用数据增强
            stats_json_path: 统计信息JSON文件路径
            height_filter: 高度过滤配置 {'min_height': -5.0, 'max_height': 100.0}
            force_recompute: 是否强制重新计算统计信息
        """
        # ✅ 添加logger初始化
        self.logger = logging.getLogger(self.__class__.__name__)
        # 在__init__最开始添加
        if not stats_json_path:
            raise ValueError("必须指定stats_json_path参数，请先运行预计算脚本")

        if not os.path.exists(stats_json_path):
            raise FileNotFoundError(f"统计信息文件不存在: {stats_json_path}，请先运行: python precompute_stats.py {image_dir}/../..")
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.normalization_method = normalization_method
        self.enable_augmentation = enable_augmentation
        self.stats_json_path = stats_json_path
        self.force_recompute = force_recompute

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
        self.logger.info(f"GAMUS数据集初始化完成: {len(self.file_pairs)} 个样本")
     
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
            # 处理 DC_02_26_RGB_r0c0.jpg -> DC_02_26_r0c0
            if '_RGB_' in img_file:
                base_name = img_file.replace('_RGB_', '_').rsplit('.', 1)[0]
            else:
                base_name = Path(img_file).stem
            image_dict[base_name] = img_file
        
        # 处理标签文件  
        for label_file in label_files:
            # 处理 DC_02_26_AGL_r0c0.png -> DC_02_26_r0c0
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
        """初始化归一化器"""

        # 尝试从JSON加载统计信息
        if self.stats_json_path and os.path.exists(self.stats_json_path):
            self.logger.info(f"🔄 加载预计算统计信息: {self.stats_json_path}")
            if self._load_from_json():
                self.logger.info("✅ 成功加载预计算统计信息，跳过重新计算")
                return
            else:
                self.logger.warning("⚠️ 预计算统计信息加载失败")
        
        # 如果强制重新计算或没有预计算文件
        if self.force_recompute or not self.stats_json_path or not os.path.exists(self.stats_json_path):
            self.logger.error("❌ 没有预计算统计信息且禁用实时计算")
            self.logger.info("💡 请先运行: python precompute_stats.py <data_dir>")
            raise ValueError("需要先运行预计算脚本生成统计信息")
    
    def _load_from_json(self) -> bool:
        """从JSON文件加载统计信息"""
        try:
            with open(self.stats_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证JSON格式
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
            # original_min, original_max = self.global_min, self.global_max
            precomputed_filter = data.get('processing_info', {}).get('height_filter', {})
            if (precomputed_filter.get('min_height') != self.height_filter['min_height'] or 
                precomputed_filter.get('max_height') != self.height_filter['max_height']):
                self.logger.warning(f"⚠️ 高度过滤设置与预计算不一致")
                self.logger.warning(f"   预计算: {precomputed_filter}")
                self.logger.warning(f"   当前设置: {self.height_filter}")

            original_min, original_max = self.global_min, self.global_max
            self.global_min = max(self.global_min, self.height_filter['min_height'])
            self.global_max = min(self.global_max, self.height_filter['max_height'])
            
            if self.global_min != original_min or self.global_max != original_max:
                self.logger.info(f"应用高度过滤: [{original_min:.1f}, {original_max:.1f}] -> [{self.global_min:.1f}, {self.global_max:.1f}]")
            
            # 创建归一化器
            self.height_normalizer = HeightNormalizer(self.normalization_method)
            
            # 使用过滤后的范围重新拟合归一化器
            dummy_data = np.linspace(self.global_min, self.global_max, 1000)
            self.height_normalizer.fit(dummy_data)
            
            # logging.info(f"数据范围: [{self.global_min:.1f}, {self.global_max:.1f}] 米")
            # 输出加载的统计信息
            processing_info = data.get('processing_info', {})
            self.logger.info(f"📊 加载统计信息:")
            self.logger.info(f"   数据范围: [{self.global_min:.2f}, {self.global_max:.2f}] 米")
            self.logger.info(f"   处理文件: {processing_info.get('processed_files', '?')} 个")
            self.logger.info(f"   数据点数: {stats.get('total_samples', '?'):,}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载JSON统计信息失败: {e}")
            return False
    
    def _compute_statistics(self):
        """计算统计信息"""
        all_heights = []
        
        logging.info(f"处理 {len(self.file_pairs)} 个文件...")
        
        for idx, (_, label_file) in enumerate(self.file_pairs):
            if idx % 500 == 0:
                logging.info(f"进度: {idx+1}/{len(self.file_pairs)}")
            
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
                        all_heights.append(filtered_heights)
                        
            except Exception as e:
                logging.warning(f"处理文件 {label_file} 失败: {e}")
                continue
        
        if not all_heights:
            raise ValueError("无法提取有效的高度数据")
        
        # 合并所有数据
        combined_heights = np.concatenate(all_heights)
        
        # 计算统计信息
        self.global_min = float(np.min(combined_heights))
        self.global_max = float(np.max(combined_heights))
        self.global_mean = float(np.mean(combined_heights))
        self.global_std = float(np.std(combined_heights))
        
        # 创建并训练归一化器
        self.height_normalizer = HeightNormalizer(self.normalization_method)
        self.height_normalizer.fit(combined_heights)
        
        logging.info(f"统计信息计算完成:")
        logging.info(f"  数据范围: [{self.global_min:.1f}, {self.global_max:.1f}] 米")
        logging.info(f"  均值±标准差: {self.global_mean:.1f} ± {self.global_std:.1f} 米")
        logging.info(f"  有效数据点: {len(combined_heights):,} 个")
    
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
            logging.error(f"加载样本 {idx} 失败: {e}")
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self.file_pairs))
    
    def _load_image(self, image_path):
        """加载影像文件"""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法加载影像: {image_path}")
        
        # 颜色空间转换和归一化
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # 调整大小
        if image.shape[:2] != (448, 448):
            image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_CUBIC)
        
        # 归一化到[0,1]
        if image.max() > 1:
            image = image / 255.0
        
        return image
    
    def _load_label(self, label_path):
        """加载nDSM标签文件"""
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


# 简化的便利函数
def create_gamus_dataloader(image_dir, label_dir, batch_size=8, shuffle=True, 
                           normalization_method='percentile', enable_augmentation=False,
                           stats_json_path=None, height_filter=None, 
                           force_recompute=False, num_workers=4):
    """
    创建GAMUS数据加载器
    
    参数:
        image_dir: 影像目录
        label_dir: 标签目录
        batch_size: 批次大小
        shuffle: 是否打乱数据
        normalization_method: 归一化方法
        enable_augmentation: 是否启用数据增强
        stats_json_path: 统计信息JSON文件路径
        height_filter: 高度过滤器，例如 {'min_height': -5.0, 'max_height': 100.0}
        force_recompute: 是否强制重新计算统计信息
        num_workers: 数据加载线程数
    """
    dataset = GAMUSDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        normalization_method=normalization_method,
        enable_augmentation=enable_augmentation,
        stats_json_path=stats_json_path,
        height_filter=height_filter,
        force_recompute=force_recompute
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if shuffle else False
    )
    dataloader.height_normalizer = dataset.height_normalizer
    return dataloader, dataset


# 使用示例
if __name__ == '__main__':
    # 设置高度过滤器，限制在合理范围内
    height_filter = {
        'min_height': -5.0,   # 最小高度：-5米
        'max_height': 100.0   # 最大高度：100米（而不是390米）
    }
    
    # 创建数据加载器
    dataloader, dataset = create_gamus_dataloader(
        image_dir='/path/to/images',
        label_dir='/path/to/labels',
        batch_size=8,
        shuffle=True,
        normalization_method='percentile',
        enable_augmentation=True,
        stats_json_path='./gamus_stats.json',
        height_filter=height_filter,  # 应用高度过滤
        force_recompute=False
    )
    
    # 打印统计信息
    print("数据集统计信息:")
    stats = dataset.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试数据加载
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"批次 {batch_idx}: 影像 {images.shape}, 标签 {labels.shape}")
        if batch_idx >= 2:  # 只测试前几个批次
            break