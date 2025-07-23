#!/usr/bin/env python3
"""
统一的可视化工具模块
整合所有可视化功能，消除重复代码
"""

import os
import json
import numpy as np
import torch
import cv2
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from tqdm import tqdm
import gc

from .common import setup_logger, clear_memory

warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 全局可视化配置
VISUALIZATION_CONFIG = {
    'dpi': 150,
    'bbox_inches': 'tight',
    'facecolor': 'white',
    'style': 'seaborn-v0_8',
    'figure_size': {
        'single_sample': (20, 5),
        'detailed_sample': (20, 16),
        'grid_summary': (16, 12),
        'error_analysis': (15, 12),
        'statistics': (12, 10)
    }
}

# nDSM专用配色方案
NDSM_COLORMAPS = {
    'terrain': plt.cm.terrain,        # 地形图，适合高程数据
    'viridis': plt.cm.viridis,       # 现代科学可视化
    'plasma': plt.cm.plasma,         # 高对比度
    'inferno': plt.cm.inferno,       # 暖色调
    'custom_terrain': None           # 自定义地形配色（延迟初始化）
}

ERROR_COLORMAPS = {
    'hot': plt.cm.hot,               # 热力图，适合误差可视化
    'reds': plt.cm.Reds,            # 红色渐变
    'oranges': plt.cm.Oranges,      # 橙色渐变
    'custom_error': None            # 自定义误差配色（延迟初始化）
}


class GAMUSVisualizer:
    """GAMUS nDSM可视化器主类"""
    
    def __init__(self, logger: Optional[logging.Logger] = None,
                 style: str = 'seaborn-v0_8',
                 dpi: int = 150):
        """
        初始化可视化器
        
        Args:
            logger: 日志记录器
            style: matplotlib样式
            dpi: 图像分辨率
        """
        self.logger = logger or setup_logger('visualizer')
        self.dpi = dpi
        
        # 设置matplotlib样式
        try:
            plt.style.use(style)
        except OSError:
            self.logger.warning(f"样式 {style} 不可用，使用默认样式")
            plt.style.use('default')
        
        # 初始化自定义配色
        self._init_custom_colormaps()
        
        self.logger.info("GAMUS可视化器初始化完成")
    
    def _init_custom_colormaps(self):
        """初始化自定义配色方案"""
        # 自定义地形配色
        terrain_colors = ['#0066cc', '#00ccff', '#66ff66', '#ffff00', '#ff6600', '#cc0000']
        NDSM_COLORMAPS['custom_terrain'] = LinearSegmentedColormap.from_list(
            'custom_terrain', terrain_colors, N=256
        )
        
        # 自定义误差配色
        error_colors = ['#ffffff', '#ffff99', '#ff9999', '#ff6666', '#ff0000', '#990000']
        ERROR_COLORMAPS['custom_error'] = LinearSegmentedColormap.from_list(
            'custom_error', error_colors, N=256
        )
    
    def denormalize_image(self, image: np.ndarray, 
                         method: str = 'imagenet') -> np.ndarray:
        """
        图像反归一化
        
        Args:
            image: 归一化后的图像 (C, H, W) 或 (H, W, C)
            method: 反归一化方法
            
        Returns:
            反归一化后的图像
        """
        image = image.copy()
        
        # 确保是 HWC 格式
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)  # CHW -> HWC
        
        if method == 'imagenet':
            # ImageNet标准化参数
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            if image.min() < 0:  # 已标准化
                image = image * std + mean
                
        elif method == 'zero_one':
            # 0-1标准化
            if image.min() < 0 or image.max() > 1:
                image = (image - image.min()) / (image.max() - image.min())
        
        # 确保值在[0,1]范围内
        image = np.clip(image, 0, 1)
        return image
    
    def convert_to_real_height(self, normalized_data: np.ndarray,
                              height_normalizer) -> np.ndarray:
        """
        将归一化的高度数据转换为真实高度（米）
        
        Args:
            normalized_data: 归一化后的高度数据
            height_normalizer: 高度归一化器
            
        Returns:
            真实高度数据（米）
        """
        if hasattr(height_normalizer, 'denormalize'):
            return height_normalizer.denormalize(normalized_data)
        elif hasattr(height_normalizer, 'inverse_transform'):
            return height_normalizer.inverse_transform(normalized_data)
        else:
            # 备用方法：使用min-max反归一化
            min_h = getattr(height_normalizer, 'global_min_h', -5.0)
            max_h = getattr(height_normalizer, 'global_max_h', 200.0)
            return normalized_data * (max_h - min_h) + min_h
    
    def create_single_sample_visualization(self, result: Dict[str, Any],
                                         save_path: Optional[str] = None,
                                         colormap: str = 'terrain',
                                         show_statistics: bool = True) -> str:
        """
        创建单样本详细可视化
        
        Args:
            result: 预测结果字典
            save_path: 保存路径
            colormap: 配色方案
            show_statistics: 是否显示统计信息
            
        Returns:
            保存的文件路径
        """
        try:
            fig, axes = plt.subplots(1, 5 if show_statistics else 4, 
                                   figsize=VISUALIZATION_CONFIG['figure_size']['single_sample'])
            
            # 提取数据
            image = result['image']
            target_real = result['target_real']
            prediction_real = result['prediction_real']
            error = result['error']
            mae = result['mae']
            rmse = result['rmse']
            index = result.get('index', 0)
            
            # 确定显示范围
            global_min = min(target_real.min(), prediction_real.min())
            global_max = max(target_real.max(), prediction_real.max())
            max_error = min(20, np.percentile(error, 95))
            
            # 选择配色方案
            ndsm_cmap = NDSM_COLORMAPS.get(colormap, plt.cm.terrain)
            error_cmap = ERROR_COLORMAPS['hot']
            
            # 1. 输入图像
            ax = axes[0]
            display_image = self.denormalize_image(image)
            ax.imshow(display_image)
            ax.set_title(f'样本 {index+1}\n输入图像', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # 2. 真实nDSM
            ax = axes[1]
            im1 = ax.imshow(target_real, cmap=ndsm_cmap, vmin=global_min, vmax=global_max)
            ax.set_title(f'真实nDSM\n范围: [{target_real.min():.1f}, {target_real.max():.1f}]m', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='高度 (m)')
            
            # 3. 预测nDSM
            ax = axes[2]
            im2 = ax.imshow(prediction_real, cmap=ndsm_cmap, vmin=global_min, vmax=global_max)
            ax.set_title(f'预测nDSM\n范围: [{prediction_real.min():.1f}, {prediction_real.max():.1f}]m', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='高度 (m)')
            
            # 4. 误差图
            ax = axes[3]
            im3 = ax.imshow(error, cmap=error_cmap, vmin=0, vmax=max_error)
            ax.set_title(f'绝对误差\nMAE: {mae:.2f}m, RMSE: {rmse:.2f}m', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label='误差 (m)')
            
            # 5. 统计信息（可选）
            if show_statistics:
                ax = axes[4]
                ax.axis('off')
                
                stats_text = f"""样本 {index+1} 统计信息:

精度指标:
  MAE: {mae:.2f} m
  RMSE: {rmse:.2f} m

真实值统计:
  最小值: {target_real.min():.2f} m
  最大值: {target_real.max():.2f} m
  均值: {target_real.mean():.2f} m
  标准差: {target_real.std():.2f} m

预测值统计:
  最小值: {prediction_real.min():.2f} m
  最大值: {prediction_real.max():.2f} m
  均值: {prediction_real.mean():.2f} m
  标准差: {prediction_real.std():.2f} m

误差统计:
  最大误差: {error.max():.2f} m
  90%分位数: {np.percentile(error, 90):.2f} m
  95%分位数: {np.percentile(error, 95):.2f} m
  99%分位数: {np.percentile(error, 99):.2f} m"""
                
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图像
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'sample_{index+1}_visualization_{timestamp}.png'
            
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, **{k: v for k, v in VISUALIZATION_CONFIG.items() 
                                                   if k not in ['dpi', 'figure_size']})
            plt.close()
            
            self.logger.info(f"单样本可视化已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"创建单样本可视化失败: {e}")
            plt.close('all')
            raise
    
    def create_grid_summary_visualization(self, results: List[Dict[str, Any]],
                                        save_path: Optional[str] = None,
                                        max_samples: int = 25,
                                        colormap: str = 'terrain') -> str:
        """
        创建网格摘要可视化
        
        Args:
            results: 预测结果列表
            save_path: 保存路径
            max_samples: 最大显示样本数
            colormap: 配色方案
            
        Returns:
            保存的文件路径
        """
        try:
            n_samples = min(len(results), max_samples)
            if n_samples == 0:
                raise ValueError("没有有效的预测结果")
            
            # 计算网格布局
            cols = min(5, n_samples)
            rows = (n_samples + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, 
                                   figsize=(4 * cols, 3 * rows))
            
            # 处理单行或单列情况
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            # 计算全局范围
            all_targets = np.concatenate([r['target_real'] for r in results[:n_samples]])
            all_predictions = np.concatenate([r['prediction_real'] for r in results[:n_samples]])
            global_min = min(all_targets.min(), all_predictions.min())
            global_max = max(all_targets.max(), all_predictions.max())
            
            # 选择配色方案
            ndsm_cmap = NDSM_COLORMAPS.get(colormap, plt.cm.terrain)
            
            for i, result in enumerate(results[:n_samples]):
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                
                # 显示误差图（更直观）
                error = result['error']
                max_error = min(10, np.percentile(error, 95))
                
                im = ax.imshow(error, cmap=ERROR_COLORMAPS['hot'], vmin=0, vmax=max_error)
                ax.set_title(f'样本{result.get("index", i)+1}\nMAE:{result["mae"]:.1f}m', 
                           fontsize=10)
                ax.axis('off')
            
            # 隐藏多余的子图
            for i in range(n_samples, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'grid_summary_{n_samples}_samples_{timestamp}.png'
            
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, **{k: v for k, v in VISUALIZATION_CONFIG.items() 
                                                   if k not in ['dpi', 'figure_size']})
            plt.close()
            
            self.logger.info(f"网格摘要可视化已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"创建网格摘要可视化失败: {e}")
            plt.close('all')
            raise
    
    def create_error_analysis_visualization(self, results: List[Dict[str, Any]],
                                          save_path: Optional[str] = None) -> str:
        """
        创建误差分析可视化
        
        Args:
            results: 预测结果列表
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=VISUALIZATION_CONFIG['figure_size']['error_analysis'])
            
            # 提取数据
            maes = [r['mae'] for r in results]
            rmses = [r['rmse'] for r in results]
            sample_indices = [r.get('index', i) + 1 for i, r in enumerate(results)]
            all_errors = np.concatenate([r['error'].flatten() for r in results])
            all_targets = np.concatenate([r['target_real'].flatten() for r in results])
            all_predictions = np.concatenate([r['prediction_real'].flatten() for r in results])
            
            # 1. MAE和RMSE对比
            ax = axes[0, 0]
            x = np.arange(len(results))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, maes, width, label='MAE', alpha=0.8, color='skyblue')
            bars2 = ax.bar(x + width/2, rmses, width, label='RMSE', alpha=0.8, color='lightcoral')
            
            ax.set_xlabel('样本索引')
            ax.set_ylabel('误差 (m)')
            ax.set_title('MAE和RMSE对比')
            ax.set_xticks(x[::max(1, len(x)//10)])  # 避免标签过密
            ax.set_xticklabels([f'#{i}' for i in sample_indices[::max(1, len(x)//10)]], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. 误差分布直方图
            ax = axes[0, 1]
            ax.hist(all_errors, bins=50, alpha=0.7, density=True, color='orange', edgecolor='black')
            ax.axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, 
                      label=f'均值: {np.mean(all_errors):.2f}m')
            ax.axvline(np.median(all_errors), color='blue', linestyle='--', linewidth=2,
                      label=f'中位数: {np.median(all_errors):.2f}m')
            
            ax.set_xlabel('绝对误差 (m)')
            ax.set_ylabel('密度')
            ax.set_title('误差分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. 累积分布函数
            ax = axes[1, 0]
            sorted_errors = np.sort(all_errors)
            y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            
            ax.plot(sorted_errors, y, linewidth=2, color='green')
            
            # 添加精度线
            thresholds = [1.0, 2.0, 5.0]
            colors = ['red', 'orange', 'purple']
            for threshold, color in zip(thresholds, colors):
                if threshold <= sorted_errors.max():
                    ax.axvline(threshold, color=color, linestyle='--', alpha=0.7, 
                              label=f'{threshold}m误差线')
            
            # 计算精度指标
            acc_1m = np.mean(all_errors <= 1.0) * 100
            acc_2m = np.mean(all_errors <= 2.0) * 100
            acc_5m = np.mean(all_errors <= 5.0) * 100
            
            ax.set_xlabel('绝对误差 (m)')
            ax.set_ylabel('累积概率')
            ax.set_title(f'误差累积分布\n±1m: {acc_1m:.1f}%, ±2m: {acc_2m:.1f}%, ±5m: {acc_5m:.1f}%')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. 散点图和相关性
            ax = axes[1, 1]
            
            # 采样以避免过多点
            n_points = min(10000, len(all_targets))
            indices = np.random.choice(len(all_targets), n_points, replace=False)
            sample_targets = all_targets[indices]
            sample_predictions = all_predictions[indices]
            
            ax.scatter(sample_targets, sample_predictions, alpha=0.5, s=1, color='blue')
            
            # 添加对角线
            min_val = min(sample_targets.min(), sample_predictions.min())
            max_val = max(sample_targets.max(), sample_predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线')
            
            # 计算相关系数
            r2 = np.corrcoef(sample_targets, sample_predictions)[0, 1] ** 2
            
            ax.set_xlabel('真实值 (m)')
            ax.set_ylabel('预测值 (m)')
            ax.set_title(f'真实值 vs 预测值\nR² = {r2:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图像
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'error_analysis_{len(results)}_samples_{timestamp}.png'
            
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, **{k: v for k, v in VISUALIZATION_CONFIG.items() 
                                                   if k not in ['dpi', 'figure_size']})
            plt.close()
            
            self.logger.info(f"误差分析可视化已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"创建误差分析可视化失败: {e}")
            plt.close('all')
            raise
    
    def create_statistics_summary_visualization(self, metrics: Dict[str, float],
                                              save_path: Optional[str] = None,
                                              additional_info: Optional[Dict] = None) -> str:
        """
        创建统计摘要可视化
        
        Args:
            metrics: 评估指标字典
            save_path: 保存路径
            additional_info: 额外信息
            
        Returns:
            保存的文件路径
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=VISUALIZATION_CONFIG['figure_size']['statistics'])
            
            # 1. 基础指标条形图
            ax = axes[0, 0]
            basic_metrics = ['mae', 'rmse', 'r2']
            basic_values = [metrics.get(m, 0) for m in basic_metrics]
            basic_labels = ['MAE (m)', 'RMSE (m)', 'R²']
            
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            bars = ax.bar(basic_labels, basic_values, color=colors, alpha=0.8)
            ax.set_title('基础评估指标', fontsize=14, fontweight='bold')
            ax.set_ylabel('值')
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, basic_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(basic_values)*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. 精度等级分布
            ax = axes[0, 1]
            if 'accuracy_by_threshold' in metrics:
                thresholds = ['≤0.5m', '≤1.0m', '≤2.0m', '≤5.0m', '>5.0m']
                accuracies = [
                    metrics.get('accuracy_0_5m', 0),
                    metrics.get('accuracy_1m', 0),
                    metrics.get('accuracy_2m', 0),
                    metrics.get('accuracy_5m', 0),
                    metrics.get('accuracy_above_5m', 0)
                ]
                
                wedges, texts, autotexts = ax.pie(accuracies, labels=thresholds, autopct='%1.1f%%',
                                                colors=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(thresholds))))
                ax.set_title('按误差阈值的精度分布', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, '精度分布数据\n不可用', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title('精度分布', fontsize=14, fontweight='bold')
            
            # 3. 高度范围性能
            ax = axes[1, 0]
            if 'height_range_performance' in metrics:
                height_ranges = list(metrics['height_range_performance'].keys())
                height_maes = [metrics['height_range_performance'][hr]['mae'] 
                              for hr in height_ranges]
                
                ax.bar(height_ranges, height_maes, color='orange', alpha=0.8)
                ax.set_title('不同高度范围的MAE', fontsize=14, fontweight='bold')
                ax.set_xlabel('高度范围 (m)')
                ax.set_ylabel('MAE (m)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '高度范围性能\n数据不可用', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title('高度范围性能', fontsize=14, fontweight='bold')
            
            # 4. 综合信息面板
            ax = axes[1, 1]
            ax.axis('off')
            
            # 创建信息文本
            info_text = f"""模型性能摘要

基础指标:
  MAE: {metrics.get('mae', 0):.3f} m
  RMSE: {metrics.get('rmse', 0):.3f} m
  R²: {metrics.get('r2', 0):.3f}
  
高级指标:
  MAPE: {metrics.get('mape', 0):.2f}%
  平均梯度误差: {metrics.get('mean_gradient_error', 0):.3f}
  
数据统计:
  样本总数: {metrics.get('total_samples', 0):,}
  有效像素数: {metrics.get('total_pixels', 0):,}
  
性能指标:
  处理时间: {metrics.get('processing_time', 0):.2f}s
  平均推理时间: {metrics.get('avg_inference_time', 0):.4f}s/样本"""
            
            if additional_info:
                info_text += f"\n\n额外信息:"
                for key, value in additional_info.items():
                    info_text += f"\n  {key}: {value}"
            
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图像
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'statistics_summary_{timestamp}.png'
            
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, **{k: v for k, v in VISUALIZATION_CONFIG.items() 
                                                   if k not in ['dpi', 'figure_size']})
            plt.close()
            
            self.logger.info(f"统计摘要可视化已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"创建统计摘要可视化失败: {e}")
            plt.close('all')
            raise
    
    def create_batch_visualizations(self, results: List[Dict[str, Any]],
                                  save_dir: str,
                                  create_individual: bool = True,
                                  create_summary: bool = True,
                                  create_analysis: bool = True,
                                  max_individual: int = 20,
                                  cleanup_interval: int = 10) -> Dict[str, List[str]]:
        """
        批量创建可视化
        
        Args:
            results: 预测结果列表
            save_dir: 保存目录
            create_individual: 是否创建单样本可视化
            create_summary: 是否创建摘要可视化
            create_analysis: 是否创建分析可视化
            max_individual: 最大单样本可视化数量
            cleanup_interval: 内存清理间隔
            
        Returns:
            保存的文件路径字典
        """
        saved_files = {
            'individual': [],
            'summary': [],
            'analysis': []
        }
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            self.logger.info(f"开始批量创建可视化，保存到: {save_dir}")
            
            # 1. 创建单样本可视化
            if create_individual and results:
                self.logger.info(f"创建单样本可视化（最多{max_individual}个）...")
                for i, result in enumerate(tqdm(results[:max_individual], desc="单样本可视化")):
                    try:
                        save_path = os.path.join(save_dir, f'individual_sample_{result.get("index", i)+1}.png')
                        path = self.create_single_sample_visualization(result, save_path)
                        saved_files['individual'].append(path)
                        
                        # 定期清理内存
                        if (i + 1) % cleanup_interval == 0:
                            clear_memory()
                            
                    except Exception as e:
                        self.logger.warning(f"创建第{i+1}个单样本可视化失败: {e}")
                        continue
            
            # 2. 创建摘要可视化
            if create_summary and results:
                self.logger.info("创建摘要可视化...")
                try:
                    save_path = os.path.join(save_dir, 'grid_summary.png')
                    path = self.create_grid_summary_visualization(results, save_path)
                    saved_files['summary'].append(path)
                except Exception as e:
                    self.logger.warning(f"创建摘要可视化失败: {e}")
            
            # 3. 创建分析可视化
            if create_analysis and results:
                self.logger.info("创建分析可视化...")
                try:
                    save_path = os.path.join(save_dir, 'error_analysis.png')
                    path = self.create_error_analysis_visualization(results, save_path)
                    saved_files['analysis'].append(path)
                except Exception as e:
                    self.logger.warning(f"创建分析可视化失败: {e}")
            
            # 最终清理
            clear_memory()
            
            total_files = sum(len(files) for files in saved_files.values())
            self.logger.info(f"批量可视化完成，共创建 {total_files} 个文件")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"批量创建可视化失败: {e}")
            raise
    
    def save_visualization_summary(self, saved_files: Dict[str, List[str]],
                                 save_dir: str,
                                 additional_info: Optional[Dict] = None):
        """
        保存可视化摘要信息
        
        Args:
            saved_files: 保存的文件路径字典
            save_dir: 保存目录
            additional_info: 额外信息
        """
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_files': sum(len(files) for files in saved_files.values()),
                'files_by_type': {k: len(v) for k, v in saved_files.items()},
                'file_paths': saved_files,
                'visualization_config': VISUALIZATION_CONFIG
            }
            
            if additional_info:
                summary['additional_info'] = additional_info
            
            summary_file = os.path.join(save_dir, 'visualization_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"可视化摘要已保存: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"保存可视化摘要失败: {e}")


# 便利函数
def create_quick_visualization(result: Dict[str, Any], 
                             save_path: Optional[str] = None,
                             logger: Optional[logging.Logger] = None) -> str:
    """
    快速创建单样本可视化的便利函数
    
    Args:
        result: 预测结果
        save_path: 保存路径
        logger: 日志记录器
        
    Returns:
        保存的文件路径
    """
    visualizer = GAMUSVisualizer(logger=logger)
    return visualizer.create_single_sample_visualization(result, save_path)


def create_batch_quick_visualization(results: List[Dict[str, Any]],
                                   save_dir: str,
                                   logger: Optional[logging.Logger] = None) -> Dict[str, List[str]]:
    """
    快速批量创建可视化的便利函数
    
    Args:
        results: 预测结果列表
        save_dir: 保存目录
        logger: 日志记录器
        
    Returns:
        保存的文件路径字典
    """
    visualizer = GAMUSVisualizer(logger=logger)
    return visualizer.create_batch_visualizations(results, save_dir)


# 使用示例
if __name__ == '__main__':
    # 设置日志
    logger = setup_logger('visualization_test')
    
    # 创建测试数据
    test_result = {
        'image': np.random.rand(3, 448, 448),
        'target_real': np.random.rand(448, 448) * 50,
        'prediction_real': np.random.rand(448, 448) * 50,
        'error': np.random.rand(448, 448) * 5,
        'mae': 2.5,
        'rmse': 3.2,
        'index': 0
    }
    
    # 测试单样本可视化
    logger.info("测试单样本可视化...")
    visualizer = GAMUSVisualizer(logger=logger)
    
    try:
        path = visualizer.create_single_sample_visualization(
            test_result, 
            save_path='test_single_sample.png'
        )
        logger.info(f"单样本可视化测试成功: {path}")
    except Exception as e:
        logger.error(f"单样本可视化测试失败: {e}")
    
    # 测试批量可视化
    logger.info("测试批量可视化...")
    test_results = [test_result] * 5  # 创建5个测试样本
    
    try:
        saved_files = visualizer.create_batch_visualizations(
            test_results,
            save_dir='./test_visualizations'
        )
        logger.info(f"批量可视化测试成功: {saved_files}")
    except Exception as e:
        logger.error(f"批量可视化测试失败: {e}")
    
    logger.info("可视化模块测试完成")