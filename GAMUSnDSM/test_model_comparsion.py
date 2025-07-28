#!/usr/bin/env python3
"""
多模型对比测试脚本（大数据集优化版本）
支持GAMUS和Depth2Elevation模型，支持mask功能
用于评估已训练模型的精度，针对12000+样本进行内存和速度优化
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import gc

# 导入更新后的模块
from improved_dataset_with_mask import create_gamus_dataloader
from improved_normalization_loss import create_height_loss
from model_with_comparison import create_gamus_model

def setup_logger(log_path):
    """设置日志记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def load_trained_model(checkpoint_path, device, logger):
    """加载已训练的模型"""
    logger.info(f"正在加载模型检查点: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 提取模型参数和类型信息
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            model_type = checkpoint.get('model_type', 'gamus')  # 获取模型类型
            logger.info(f"检查点信息:")
            logger.info(f"  模型类型: {model_type}")
            logger.info(f"  训练轮次: {checkpoint.get('epoch', 'N/A')}")
            if isinstance(checkpoint.get('loss'), (int, float)):
                logger.info(f"  验证损失: {checkpoint.get('loss'):.6f}")
        else:
            model_state_dict = checkpoint
            model_type = 'gamus'  # 默认为gamus
            logger.info("检查点为直接的模型状态字典，假设为GAMUS模型")
        
        return model_state_dict, model_type
        
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        raise

def create_test_dataset(data_dir, args, logger):
    """创建测试数据集（支持mask）"""
    logger.info("创建测试数据集...")
    
    # 测试集路径
    test_image_dir = os.path.join(data_dir, 'test','images')
    test_label_dir = os.path.join(data_dir, 'test','depths')
    test_mask_dir = None
    
    if args.mask_dir:
        test_mask_dir = os.path.join(args.mask_dir, 'test', 'classes')
    
    # 如果没有专门的测试集，使用验证集
    if not os.path.exists(test_image_dir):
        test_image_dir = os.path.join(data_dir, 'val','images')
        test_label_dir = os.path.join(data_dir, 'val','depths')
        if args.mask_dir:
            test_mask_dir = os.path.join(args.mask_dir, 'val', 'classes')
        logger.info("未找到测试集，使用验证集进行测试")
    
    # 如果还是没有，使用训练集的一部分
    if not os.path.exists(test_image_dir):
        test_image_dir = os.path.join(data_dir, 'train','images')
        test_label_dir = os.path.join(data_dir, 'train','depths')
        if args.mask_dir:
            test_mask_dir = os.path.join(args.mask_dir, 'train', 'classes')
        logger.info("未找到验证集，使用训练集进行测试（注意：这可能导致过拟合的结果）")
    
    if not os.path.exists(test_image_dir):
        raise FileNotFoundError(f"测试图像目录不存在: {test_image_dir}")
    if not os.path.exists(test_label_dir):
        raise FileNotFoundError(f"测试标签目录不存在: {test_label_dir}")
    
    # 检查mask目录
    if args.mask_dir and test_mask_dir and not os.path.exists(test_mask_dir):
        logger.warning(f"Mask目录不存在: {test_mask_dir}，将不使用mask")
        test_mask_dir = None
    
    logger.info(f"测试图像目录: {test_image_dir}")
    logger.info(f"测试标签目录: {test_label_dir}")
    if test_mask_dir:
        logger.info(f"测试mask目录: {test_mask_dir}")
    
    # 设置高度过滤器
    height_filter = {
        'min_height': args.min_height,
        'max_height': args.max_height
    }
    
    # 创建数据加载器（支持mask）
    try:
        test_loader, test_dataset = create_gamus_dataloader(
            image_dir=test_image_dir,
            label_dir=test_label_dir,
            mask_dir=test_mask_dir,
            building_class_id=args.building_class_id,
            tree_class_id=args.tree_class_id,
            # use_all_classes=args.use_all_classes,
            batch_size=args.batch_size,
            shuffle=False,
            normalization_method=args.normalization_method,
            enable_augmentation=False,
            stats_json_path=args.stats_json_path,
            height_filter=height_filter,
            force_recompute=False,
            num_workers=args.num_workers
        )
        
        logger.info(f"测试集大小: {len(test_dataset)}")
        
        # 如果测试集太大，可以选择采样
        if args.max_test_samples > 0 and len(test_dataset) > args.max_test_samples:
            logger.info(f"测试集样本数({len(test_dataset)})超过限制({args.max_test_samples})，将进行随机采样")
            # 创建子集
            indices = np.random.choice(len(test_dataset), args.max_test_samples, replace=False)
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
            
            # 重新创建数据加载器
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available() and not args.disable_pin_memory,
                drop_last=False,
                prefetch_factor=2 if args.num_workers > 0 else 2
            )
            logger.info(f"采样后测试集大小: {len(test_dataset)}")
    
    except Exception as e:
        logger.error(f"创建测试数据集失败: {e}")
        raise
    
    return test_loader, test_dataset

class OnlineMetricsCalculator:
    """在线指标计算器，避免存储所有数据"""
    
    def __init__(self, height_normalizer, sample_for_correlation=5000):
        self.height_normalizer = height_normalizer
        self.sample_for_correlation = sample_for_correlation
        self.reset()
        
    def reset(self):
        """重置统计"""
        self.count = 0
        self.sum_se = 0.0  # 平方误差和
        self.sum_ae = 0.0  # 绝对误差和
        self.sum_targets = 0.0
        self.sum_targets_sq = 0.0
        self.sum_preds = 0.0
        self.sum_cross = 0.0  # 交叉项用于相关性计算
        
        # 精度计数器
        self.accuracy_1m = 0
        self.accuracy_2m = 0
        self.accuracy_5m = 0
        self.accuracy_10m = 0
        
        # 分层误差
        self.ground_errors = []
        self.low_errors = []
        self.mid_errors = []
        self.high_errors = []
        
        # 相关性计算的采样数据
        self.sampled_preds = []
        self.sampled_targets = []
        
        # 数据范围
        self.min_target = float('inf')
        self.max_target = float('-inf')
        self.min_pred = float('inf')
        self.max_pred = float('-inf')
    
    def update(self, predictions, targets, masks=None):
        """更新统计信息（支持mask）"""
        # 应用mask过滤
        if masks is not None:
            # 只处理mask=1且targets有效的像素
            valid_mask = (masks > 0.5) & (targets >= 0)
            if valid_mask.sum() == 0:
                return
            predictions = predictions[valid_mask]
            targets = targets[valid_mask]
        
        # 反归一化到真实高度值
        pred_heights = self.height_normalizer.denormalize(predictions.flatten())
        target_heights = self.height_normalizer.denormalize(targets.flatten())
        
        # 移除无效值
        valid_mask = (~np.isnan(pred_heights) & ~np.isnan(target_heights) & 
                     ~np.isinf(pred_heights) & ~np.isinf(target_heights))
        
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
        self.accuracy_1m += np.sum(abs_errors <= 1.0)
        self.accuracy_2m += np.sum(abs_errors <= 2.0)
        self.accuracy_5m += np.sum(abs_errors <= 5.0)
        self.accuracy_10m += np.sum(abs_errors <= 10.0)
        
        # 分层误差（只存储小样本）
        ground_mask = (valid_targets >= -5) & (valid_targets <= 5)
        low_mask = (valid_targets > 5) & (valid_targets <= 20)
        mid_mask = (valid_targets > 20) & (valid_targets <= 50)
        high_mask = valid_targets > 50
        
        if np.sum(ground_mask) > 0:
            self.ground_errors.extend(abs_errors[ground_mask].tolist())
        if np.sum(low_mask) > 0:
            self.low_errors.extend(abs_errors[low_mask].tolist())
        if np.sum(mid_mask) > 0:
            self.mid_errors.extend(abs_errors[mid_mask].tolist())
        if np.sum(high_mask) > 0:
            self.high_errors.extend(abs_errors[high_mask].tolist())
        
        # 限制分层误差数组大小
        max_layer_samples = 2000
        if len(self.ground_errors) > max_layer_samples:
            self.ground_errors = self.ground_errors[-max_layer_samples:]
        if len(self.low_errors) > max_layer_samples:
            self.low_errors = self.low_errors[-max_layer_samples:]
        if len(self.mid_errors) > max_layer_samples:
            self.mid_errors = self.mid_errors[-max_layer_samples:]
        if len(self.high_errors) > max_layer_samples:
            self.high_errors = self.high_errors[-max_layer_samples:]
        
        # 数据范围
        self.min_target = min(self.min_target, valid_targets.min())
        self.max_target = max(self.max_target, valid_targets.max())
        self.min_pred = min(self.min_pred, valid_preds.min())
        self.max_pred = max(self.max_pred, valid_preds.max())
        
        # 相关性计算的采样
        if len(self.sampled_preds) < self.sample_for_correlation:
            sample_size = min(len(valid_preds), self.sample_for_correlation - len(self.sampled_preds))
            indices = np.random.choice(len(valid_preds), sample_size, replace=False)
            self.sampled_preds.extend(valid_preds[indices].tolist())
            self.sampled_targets.extend(valid_targets[indices].tolist())
    
    def compute_metrics(self):
        """计算最终指标"""
        if self.count == 0:
            return {}
        
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
        
        # 相关性指标
        pearson_r = pearson_p = spearman_r = spearman_p = 0.0
        if len(self.sampled_preds) > 10:
            try:
                from scipy.stats import pearsonr, spearmanr
                if len(np.unique(self.sampled_targets)) > 1 and len(np.unique(self.sampled_preds)) > 1:
                    pearson_r, pearson_p = pearsonr(self.sampled_targets, self.sampled_preds)
                    spearman_r, spearman_p = spearmanr(self.sampled_targets, self.sampled_preds)
            except:
                pass
        
        # 精度指标
        accuracy_1m = self.accuracy_1m / self.count
        accuracy_2m = self.accuracy_2m / self.count
        accuracy_5m = self.accuracy_5m / self.count
        accuracy_10m = self.accuracy_10m / self.count
        
        # 相对误差（使用在线计算的均值）
        mean_target_abs = abs(mean_target) if abs(mean_target) > 0.1 else 0.1
        relative_error = (mae / mean_target_abs) * 100
        
        # 分层误差
        mae_ground = np.mean(self.ground_errors) if self.ground_errors else 0.0
        mae_low = np.mean(self.low_errors) if self.low_errors else 0.0
        mae_mid = np.mean(self.mid_errors) if self.mid_errors else 0.0
        mae_high = np.mean(self.high_errors) if self.high_errors else 0.0
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'height_accuracy_1m': float(accuracy_1m),
            'height_accuracy_2m': float(accuracy_2m),
            'height_accuracy_5m': float(accuracy_5m),
            'height_accuracy_10m': float(accuracy_10m),
            'relative_error': float(relative_error),
            'mae_ground_level': float(mae_ground),
            'mae_low_buildings': float(mae_low),
            'mae_mid_buildings': float(mae_mid),
            'mae_high_buildings': float(mae_high),
            'data_range_min': float(self.min_target),
            'data_range_max': float(self.max_target),
            'prediction_range_min': float(self.min_pred),
            'prediction_range_max': float(self.max_pred),
            'total_samples': self.count
        }
        
        return metrics

def test_model_optimized(model, test_loader, device, logger, criterion=None, memory_cleanup_interval=50):
    """优化的模型测试函数（支持mask）"""
    model.eval()
    logger.info("开始模型测试（大数据集优化版本，支持mask）...")
    
    # 获取归一化器
    if hasattr(test_loader.dataset, 'dataset'):
        # Subset情况
        if hasattr(test_loader.dataset.dataset, 'get_normalizer'):
            height_normalizer = test_loader.dataset.dataset.get_normalizer()
        else:
            # 从原始数据集获取
            original_dataset = test_loader.dataset.dataset
            height_normalizer = original_dataset.get_normalizer()
    else:
        height_normalizer = test_loader.dataset.get_normalizer()
    
    metrics_calculator = OnlineMetricsCalculator(height_normalizer)
    
    batch_losses = []
    inference_times = []
    failed_batches = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing Model')
        
        for batch_idx, batch_data in enumerate(test_pbar):
            try:
                # 处理可能包含mask的batch数据
                if len(batch_data) == 3:
                    images, targets, masks = batch_data
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                else:
                    images, targets = batch_data
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    masks = torch.ones_like(targets).to(device)  # 全1mask
                
                # 数据质量检查
                if torch.isnan(images).any() or torch.isinf(images).any():
                    logger.warning(f"测试批次{batch_idx}: 输入图像包含NaN或Inf")
                    failed_batches += 1
                    continue
                    
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    logger.warning(f"测试批次{batch_idx}: nDSM目标包含NaN或Inf")
                    failed_batches += 1
                    continue
                
                # 测量推理时间
                inference_start = time.time()
                predictions = model(images)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # 预测结果检查
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    logger.warning(f"测试批次{batch_idx}: nDSM预测包含NaN或Inf")
                    failed_batches += 1
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
                
                # 计算损失（考虑mask）
                if criterion is not None:
                    try:
                        # 如果损失函数支持mask
                        if hasattr(criterion, 'forward') and 'masks' in criterion.forward.__code__.co_varnames:
                            loss = criterion(predictions, targets, masks)
                        else:
                            # 手动应用mask
                            valid_mask = (masks > 0.5) & (targets >= 0)
                            if valid_mask.sum() > 0:
                                loss = criterion(predictions[valid_mask], targets[valid_mask])
                            else:
                                continue
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            batch_losses.append(loss.item())
                    except Exception as e:
                        logger.warning(f"测试批次{batch_idx}损失计算错误: {e}")
                
                # 更新在线指标（传入mask）
                metrics_calculator.update(
                    predictions.cpu().numpy(), 
                    targets.cpu().numpy(),
                    masks.cpu().numpy() if len(batch_data) == 3 else None
                )
                
                # 更新进度条
                current_metrics = metrics_calculator.compute_metrics()
                test_pbar.set_postfix({
                    'samples': current_metrics.get('total_samples', 0),
                    'mae': f"{current_metrics.get('mae', 0):.3f}",
                    'rmse': f"{current_metrics.get('rmse', 0):.3f}",
                    'r2': f"{current_metrics.get('r2', 0):.3f}"
                })
                
                # 定期清理内存
                if batch_idx % memory_cleanup_interval == 0 and batch_idx > 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # 清理当前批次的张量
                del images, targets, predictions
                if len(batch_data) == 3:
                    del masks
                
            except Exception as e:
                logger.error(f"测试批次{batch_idx}处理错误: {e}")
                failed_batches += 1
                continue
    
    total_time = time.time() - start_time
    
    # 计算最终指标
    final_metrics = metrics_calculator.compute_metrics()
    
    if final_metrics.get('total_samples', 0) == 0:
        logger.error("测试过程中没有收集到有效数据!")
        return None
    
    # 计算平均损失和推理时间
    avg_loss = np.mean(batch_losses) if batch_losses else float('inf')
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    logger.info(f"测试完成:")
    logger.info(f"  总耗时: {total_time:.2f} 秒")
    logger.info(f"  平均推理时间: {avg_inference_time:.4f} 秒/批次")
    logger.info(f"  平均损失: {avg_loss:.6f}")
    logger.info(f"  有效样本数: {final_metrics['total_samples']}")
    logger.info(f"  失败批次数: {failed_batches}")
    
    return {
        'metrics': final_metrics,
        'avg_loss': avg_loss,
        'avg_inference_time': avg_inference_time,
        'total_time': total_time,
        'batch_losses': batch_losses,
        'failed_batches': failed_batches
    }

def save_test_results(test_results, save_dir, logger, model_type):
    """保存测试结果"""
    results_file = os.path.join(save_dir, f'test_results_{model_type}.json')
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': model_type,
        'metrics': test_results['metrics'],
        'performance': {
            'avg_loss': test_results['avg_loss'],
            'avg_inference_time': test_results['avg_inference_time'],
            'total_time': test_results['total_time'],
            'failed_batches': test_results['failed_batches']
        }
    }
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"测试结果已保存: {results_file}")
    except Exception as e:
        logger.error(f"保存测试结果失败: {e}")

def create_quick_visualizations(test_results, save_dir, logger, model_type):
    """创建快速可视化（仅使用采样数据）"""
    logger.info("生成快速可视化图表...")
    
    try:
        # 从结果中获取基本信息
        metrics = test_results['metrics']
        
        # 创建指标总结图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_type.upper()} 模型测试结果', fontsize=16)
        
        # 1. 基础指标条形图
        ax = axes[0, 0]
        basic_metrics = ['mae', 'rmse', 'r2']
        basic_values = [metrics.get(m, 0) for m in basic_metrics]
        basic_labels = ['MAE (m)', 'RMSE (m)', 'R²']
        
        bars = ax.bar(basic_labels, basic_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_title('基础评估指标')
        ax.set_ylabel('值')
        
        # 添加数值标签
        for bar, value in zip(bars, basic_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # 2. 精度指标
        ax = axes[0, 1]
        accuracy_metrics = ['height_accuracy_1m', 'height_accuracy_2m', 'height_accuracy_5m', 'height_accuracy_10m']
        accuracy_values = [metrics.get(m, 0) * 100 for m in accuracy_metrics]  # 转换为百分比
        accuracy_labels = ['±1m', '±2m', '±5m', '±10m']
        
        bars = ax.bar(accuracy_labels, accuracy_values, color='orange', alpha=0.7)
        ax.set_title('高度精度指标')
        ax.set_ylabel('准确率 (%)')
        ax.set_ylim(0, 100)
        
        for bar, value in zip(bars, accuracy_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom')
        
        # 3. 分层误差
        ax = axes[1, 0]
        layer_metrics = ['mae_ground_level', 'mae_low_buildings', 'mae_mid_buildings', 'mae_high_buildings']
        layer_values = [metrics.get(m, 0) for m in layer_metrics]
        layer_labels = ['地面层\n(-5~5m)', '低建筑\n(5~20m)', '中建筑\n(20~50m)', '高建筑\n(>50m)']
        
        bars = ax.bar(layer_labels, layer_values, color='purple', alpha=0.7)
        ax.set_title('分层误差分析')
        ax.set_ylabel('MAE (m)')
        
        for bar, value in zip(bars, layer_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        # 4. 数据范围对比
        ax = axes[1, 1]
        range_data = [
            [metrics.get('data_range_min', 0), metrics.get('data_range_max', 0)],
            [metrics.get('prediction_range_min', 0), metrics.get('prediction_range_max', 0)]
        ]
        range_labels = ['真实值', '预测值']
        
        x = np.arange(len(range_labels))
        width = 0.35
        
        ax.bar(x - width/2, [r[0] for r in range_data], width, label='最小值', alpha=0.7)
        ax.bar(x + width/2, [r[1] for r in range_data], width, label='最大值', alpha=0.7)
        
        ax.set_title('数据范围对比')
        ax.set_ylabel('高度 (m)')
        ax.set_xticks(x)
        ax.set_xticklabels(range_labels)
        ax.legend()
        
        plt.tight_layout()
        
        summary_path = os.path.join(save_dir, f'test_summary_{model_type}.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化图表已保存: {summary_path}")
        
    except Exception as e:
        logger.error(f"生成可视化失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='多模型对比测试脚本（大数据集优化版本）')
    
    # 必需参数
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='训练好的模型检查点路径')
    parser.add_argument('--data_dir', type=str, default='/mnt/data1/UserData/hudong26/HeightData/',
                        help='数据根目录 (包含train/val/test子目录)')
    parser.add_argument('--stats_json_path', type=str, default='./gamus_full_stats.json',
                        help='预计算统计信息JSON文件路径')
    
    # 模型参数（会从检查点自动推断，这里作为备用）
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['auto', 'gamus', 'depth2elevation'],
                        help='模型类型（auto表示从检查点自动推断）')
    parser.add_argument('--encoder', type=str, default='vitb',
                        choices=['vits', 'vitb', 'vitl'],
                        help='编码器类型（需要与训练时一致）')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='预训练模型路径（用于模型结构创建）')
    
    # 数据参数（需要与训练时一致）
    parser.add_argument('--normalization_method', type=str, default='minmax',
                        choices=['minmax', 'percentile', 'zscore'],
                        help='归一化方法（需要与训练时一致）')
    parser.add_argument('--min_height', type=float, default=-5.0,
                        help='最小高度过滤值（米）')
    parser.add_argument('--max_height', type=float, default=200.0,
                        help='最大高度过滤值（米）')
    
    # mask相关参数
    parser.add_argument('--mask_dir', type=str, default='/mnt/data1/UserData/hudong26/HeightData/',
                        help='classes mask根目录')
    parser.add_argument('--building_class_id', type=int, default=3,
                        help='建筑类别ID')
    parser.add_argument('--tree_class_id', type=int, default=6,
                        help='树木类别ID')
    # parser.add_argument('--use_all_classes', action='store_true',
    #                     help='使用所有类别而不只是building+tree')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='测试批次大小')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='数据加载线程数')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                        help='测试结果保存目录')
    
    # 大数据集优化参数
    parser.add_argument('--max_test_samples', type=int, default=0,
                        help='最大测试样本数（0表示使用全部样本）')
    parser.add_argument('--memory_cleanup_interval', type=int, default=50,
                        help='内存清理间隔（批次数）')
    parser.add_argument('--disable_pin_memory', action='store_true',
                        help='禁用pin_memory以节省内存')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'mae', 'huber', 'focal', 'combined'],
                        help='损失函数类型')
    parser.add_argument('--height_aware_loss', action='store_true',
                        help='启用高度感知损失权重')
    
    # 可视化参数
    parser.add_argument('--enable_visualization', action='store_true',
                        help='启用结果可视化')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.save_dir, f'test_log_{timestamp}.log')
    logger = setup_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("多模型对比测试（大数据集优化版本）")
    logger.info("=" * 80)
    logger.info(f"测试参数: {vars(args)}")
    
    try:
        # 设备设置
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                logger.info(f"自动检测到CUDA，使用GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device(args.device)
        
        logger.info(f"使用设备: {device}")
        
        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("启用CUDA优化")
        
        # 加载训练好的权重和模型类型
        model_state_dict, detected_model_type = load_trained_model(args.checkpoint_path, device, logger)
        
        # 确定模型类型
        if args.model_type == 'auto':
            model_type = detected_model_type
        else:
            model_type = args.model_type
        
        logger.info(f"使用模型类型: {model_type}")
        
        # 创建模型结构
        logger.info("创建模型结构...")
        try:
            model_kwargs = {
                'encoder': args.encoder,
                'pretrained_path': args.pretrained_path,
                'freeze_encoder': True,  # 测试时冻结编码器
                'model_type': model_type
            }
            
            # 为Depth2Elevation添加特定参数
            if model_type == 'depth2elevation':
                model_kwargs.update({
                    'img_size': 448,
                    'patch_size': 14,
                    'use_multi_scale_output': False,  # 测试时使用单尺度
                    'loss_config': {},
                    'freezing_config': {}
                })
            
            model = create_gamus_model(**model_kwargs)
            
        except Exception as e:
            logger.error(f"创建模型失败: {e}")
            raise
        
        # 加载权重
        model.load_state_dict(model_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数总数: {total_params:,}")
        
        # 创建测试数据集
        test_loader, test_dataset = create_test_dataset(args.data_dir, args, logger)
        
        # 获取归一化器（用于损失函数）
        if hasattr(test_dataset, 'get_normalizer'):
            height_normalizer = test_dataset.get_normalizer()
        else:
            # Subset情况，从原始数据集获取
            height_normalizer = test_dataset.dataset.get_normalizer()
        
        # 创建损失函数
        criterion = create_height_loss(
            loss_type=args.loss_type,
            height_aware=args.height_aware_loss,
            height_normalizer=height_normalizer,
            min_height=args.min_height,
            max_height=args.max_height
        )
        
        # 执行优化的测试
        test_results = test_model_optimized(
            model, test_loader, device, logger, criterion, 
            args.memory_cleanup_interval
        )
        
        if test_results is None:
            logger.error("测试失败，无法获得有效结果")
            return 1
        
        metrics = test_results['metrics']
        
        # 输出详细结果
        logger.info("\n" + "=" * 60)
        logger.info(f"{model_type.upper()} 模型测试结果")
        logger.info("=" * 60)
        logger.info(f"基础指标:")
        logger.info(f"  MAE: {metrics['mae']:.4f} m")
        logger.info(f"  RMSE: {metrics['rmse']:.4f} m")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  Pearson r: {metrics['pearson_r']:.4f}")
        
        logger.info(f"\nnDSM高度精度:")
        logger.info(f"  ±1m: {metrics['height_accuracy_1m']:.1%}")
        logger.info(f"  ±2m: {metrics['height_accuracy_2m']:.1%}")
        logger.info(f"  ±5m: {metrics['height_accuracy_5m']:.1%}")
        logger.info(f"  ±10m: {metrics['height_accuracy_10m']:.1%}")
        
        logger.info(f"\n分层误差分析:")
        logger.info(f"  地面层(-5~5m): {metrics['mae_ground_level']:.2f} m")
        logger.info(f"  低建筑(5~20m): {metrics['mae_low_buildings']:.2f} m")
        logger.info(f"  中建筑(20~50m): {metrics['mae_mid_buildings']:.2f} m")
        logger.info(f"  高建筑(>50m): {metrics['mae_high_buildings']:.2f} m")
        
        logger.info(f"\n数据范围:")
        logger.info(f"  真实值范围: [{metrics['data_range_min']:.2f}, {metrics['data_range_max']:.2f}] m")
        logger.info(f"  预测值范围: [{metrics['prediction_range_min']:.2f}, {metrics['prediction_range_max']:.2f}] m")
        
        logger.info(f"\n性能指标:")
        logger.info(f"  平均推理时间: {test_results['avg_inference_time']:.4f} 秒/批次")
        logger.info(f"  总测试时间: {test_results['total_time']:.2f} 秒")
        logger.info(f"  有效样本数: {metrics['total_samples']:,}")
        logger.info(f"  失败批次数: {test_results['failed_batches']}")
        
        # 计算每秒处理样本数
        samples_per_second = metrics['total_samples'] / test_results['total_time']
        logger.info(f"  处理速度: {samples_per_second:.1f} 样本/秒")
        
        # 保存测试结果
        save_test_results(test_results, args.save_dir, logger, model_type)
        
        # 生成快速可视化
        if args.enable_visualization:
            create_quick_visualizations(test_results, args.save_dir, logger, model_type)
        
        logger.info("=" * 60)
        logger.info("测试完成!")
        logger.info(f"结果保存在: {args.save_dir}")
        
        # 性能总结
        logger.info(f"\n{model_type.upper()} 模型性能总结:")
        logger.info(f"  最佳指标: MAE={metrics['mae']:.3f}m, RMSE={metrics['rmse']:.3f}m, R²={metrics['r2']:.3f}")
        logger.info(f"  推理效率: {samples_per_second:.1f} 样本/秒")
        
        if metrics['mae'] < 2.0:
            logger.info("  🎉 模型性能优秀 (MAE < 2.0m)")
        elif metrics['mae'] < 5.0:
            logger.info("  ✅ 模型性能良好 (MAE < 5.0m)")
        else:
            logger.info("  ⚠️  模型性能有待改进 (MAE >= 5.0m)")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)