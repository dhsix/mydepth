#!/usr/bin/env python3
"""
GAMUS nDSM模型测试脚本
用于评估已训练模型的精度
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

# 导入您的模块
from improved_dataset import GAMUSDataset
from improved_normalization_loss import ImprovedHeightLoss
from model import GAMUSNDSMPredictor, create_gamus_model
from optimized_validation import GAMUSValidationMetricsCalculator

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
        
        # 提取模型参数
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            logger.info(f"检查点信息:")
            logger.info(f"  训练轮次: {checkpoint.get('epoch', 'N/A')}")
            logger.info(f"  验证损失: {checkpoint.get('val_loss', 'N/A'):.6f}")
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                logger.info(f"  训练时最佳指标:")
                logger.info(f"    MAE: {metrics.get('mae', 'N/A'):.4f}")
                logger.info(f"    RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                logger.info(f"    R²: {metrics.get('r2', 'N/A'):.4f}")
        else:
            model_state_dict = checkpoint
            logger.info("检查点为直接的模型状态字典")
        
        return model_state_dict
        
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        raise

def create_test_dataset(data_dir, args, logger):
    """创建测试数据集"""
    logger.info("创建测试数据集...")
    
    # 测试集路径
    test_image_dir = os.path.join(data_dir, 'test','images')
    test_label_dir = os.path.join(data_dir, 'test','depths')
    
    # 如果没有专门的测试集，使用验证集
    if not os.path.exists(test_image_dir):
        test_image_dir = os.path.join(data_dir, 'val','images')
        test_label_dir = os.path.join(data_dir, 'val','depths')
        logger.info("未找到测试集，使用验证集进行测试")
    
    # 如果还是没有，使用训练集的一部分
    if not os.path.exists(test_image_dir):
        test_image_dir = os.path.join(data_dir, 'train','images')
        test_label_dir = os.path.join(data_dir, 'train','depths')
        logger.info("未找到验证集，使用训练集进行测试（注意：这可能导致过拟合的结果）")
    
    if not os.path.exists(test_image_dir):
        raise FileNotFoundError(f"测试图像目录不存在: {test_image_dir}")
    if not os.path.exists(test_label_dir):
        raise FileNotFoundError(f"测试标签目录不存在: {test_label_dir}")
    
    logger.info(f"测试图像目录: {test_image_dir}")
    logger.info(f"测试标签目录: {test_label_dir}")
    
    # 创建训练数据集以获取归一化器
    train_image_dir = os.path.join(data_dir, 'train','images')
    train_label_dir = os.path.join(data_dir, 'train','depths')
    
    train_dataset = GAMUSDataset(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        normalization_method=args.normalization_method,
        enable_augmentation=False,
        stats_json_path=args.stats_json_path,
        height_filter={'min_height': args.min_height, 'max_height': args.max_height}  # ✅ 添加高度过滤

        # file_extension=args.file_extension
    )
    
    # 创建测试数据集（使用训练集的归一化器）
    test_dataset = GAMUSDataset(
        image_dir=test_image_dir,
        label_dir=test_label_dir,
        normalization_method=args.normalization_method,
        enable_augmentation=False,
        stats_json_path=args.stats_json_path,
        height_filter={'min_height': args.min_height, 'max_height': args.max_height}  # ✅ 添加高度过滤
    )
    
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return test_loader, test_dataset

def test_model(model, test_loader, device, logger, criterion=None):
    """测试模型性能"""
    model.eval()
    logger.info("开始模型测试...")
    
    # 统计变量
    all_predictions = []
    all_targets = []
    batch_losses = []
    inference_times = []
    
    start_time = time.time()
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing Model', leave=False)
        
        for batch_idx, (images, targets) in enumerate(test_pbar):
            try:
                # 移动数据到设备
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # 数据质量检查
                if torch.isnan(images).any() or torch.isinf(images).any():
                    logger.warning(f"测试批次{batch_idx}: 输入图像包含NaN或Inf")
                    continue
                    
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    logger.warning(f"测试批次{batch_idx}: nDSM目标包含NaN或Inf")
                    continue
                
                # 测量推理时间
                inference_start = time.time()
                predictions = model(images)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # 预测结果检查
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    logger.warning(f"测试批次{batch_idx}: nDSM预测包含NaN或Inf")
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
                        logger.warning(f"测试批次{batch_idx}损失计算错误: {e}")
                
                # 收集预测和目标
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # 更新进度条
                if batch_losses:
                    test_pbar.set_postfix({
                        'loss': f'{batch_losses[-1]:.6f}',
                        'avg_inference': f'{np.mean(inference_times):.4f}s'
                    })
                
            except Exception as e:
                logger.error(f"测试批次{batch_idx}处理错误: {e}")
                continue
    
    total_time = time.time() - start_time
    
    if not all_predictions:
        logger.error("测试过程中没有收集到有效数据!")
        return None
    
    # 合并所有预测和目标
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # 计算平均损失
    avg_loss = np.mean(batch_losses) if batch_losses else float('inf')
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    logger.info(f"测试完成:")
    logger.info(f"  总耗时: {total_time:.2f} 秒")
    logger.info(f"  平均推理时间: {avg_inference_time:.4f} 秒/批次")
    logger.info(f"  平均损失: {avg_loss:.6f}")
    logger.info(f"  测试样本数: {len(all_predictions)}")
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'avg_loss': avg_loss,
        'avg_inference_time': avg_inference_time,
        'total_time': total_time,
        'batch_losses': batch_losses
    }

def calculate_detailed_metrics(predictions, targets, height_normalizer, logger):
    """计算详细的评估指标"""
    logger.info("计算详细评估指标...")
    
    # 反归一化到真实高度值
    pred_heights = height_normalizer.denormalize(predictions.flatten())
    target_heights = height_normalizer.denormalize(targets.flatten())
    
    # 移除无效值
    valid_mask = (~np.isnan(pred_heights) & ~np.isnan(target_heights) & 
                 ~np.isinf(pred_heights) & ~np.isinf(target_heights))
    
    if np.sum(valid_mask) < 10:
        logger.error(f"有效样本太少: {np.sum(valid_mask)}")
        return {}
    
    valid_preds = pred_heights[valid_mask]
    valid_targets = target_heights[valid_mask]
    
    # 基础指标
    mse = np.mean((valid_preds - valid_targets) ** 2)
    mae = np.mean(np.abs(valid_preds - valid_targets))
    rmse = np.sqrt(mse)
    
    # R²计算
    ss_res = np.sum((valid_targets - valid_preds) ** 2)
    ss_tot = np.sum((valid_targets - np.mean(valid_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    r2 = max(-10, min(1, r2))
    
    # 相关性指标
    from scipy.stats import pearsonr, spearmanr
    try:
        if len(np.unique(valid_targets)) > 1 and len(np.unique(valid_preds)) > 1:
            pearson_r, pearson_p = pearsonr(valid_targets, valid_preds)
            spearman_r, spearman_p = spearmanr(valid_targets, valid_preds)
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = 0.0
    except:
        pearson_r = pearson_p = spearman_r = spearman_p = 0.0
    
    # nDSM精度指标
    errors = np.abs(valid_preds - valid_targets)
    accuracy_1m = np.mean(errors <= 1.0)
    accuracy_2m = np.mean(errors <= 2.0)
    accuracy_5m = np.mean(errors <= 5.0)
    accuracy_10m = np.mean(errors <= 10.0)
    
    # 相对误差
    denominator = np.maximum(np.abs(valid_targets), 0.1)
    relative_errors = np.abs(valid_preds - valid_targets) / denominator
    relative_errors = np.minimum(relative_errors, 10.0)
    relative_error = np.mean(relative_errors) * 100
    
    # 分层误差分析
    ground_mask = (valid_targets >= -5) & (valid_targets <= 5)
    low_mask = (valid_targets > 5) & (valid_targets <= 20)
    mid_mask = (valid_targets > 20) & (valid_targets <= 50)
    high_mask = valid_targets > 50
    
    mae_ground = np.mean(errors[ground_mask]) if np.sum(ground_mask) > 0 else 0.0
    mae_low = np.mean(errors[low_mask]) if np.sum(low_mask) > 0 else 0.0
    mae_mid = np.mean(errors[mid_mask]) if np.sum(mid_mask) > 0 else 0.0
    mae_high = np.mean(errors[high_mask]) if np.sum(high_mask) > 0 else 0.0
    
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
        'data_range_min': float(valid_targets.min()),
        'data_range_max': float(valid_targets.max()),
        'prediction_range_min': float(valid_preds.min()),
        'prediction_range_max': float(valid_preds.max())
    }
    
    return metrics

def save_test_results(metrics, test_results, save_dir, logger):
    """保存测试结果"""
    results_file = os.path.join(save_dir, 'test_results.json')
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'performance': {
            'avg_loss': test_results['avg_loss'],
            'avg_inference_time': test_results['avg_inference_time'],
            'total_time': test_results['total_time']
        }
    }
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"测试结果已保存: {results_file}")
    except Exception as e:
        logger.error(f"保存测试结果失败: {e}")

def create_test_visualizations(predictions, targets, height_normalizer, save_dir, logger, num_samples=6):
    """创建测试可视化"""
    logger.info("生成测试可视化图表...")
    
    try:
        # 反归一化
        pred_heights = height_normalizer.denormalize(predictions)
        target_heights = height_normalizer.denormalize(targets)
        
        # 移除无效值
        valid_mask = (~np.isnan(pred_heights) & ~np.isnan(target_heights) & 
                     ~np.isinf(pred_heights) & ~np.isinf(target_heights))
        
        valid_preds = pred_heights[valid_mask]
        valid_targets = target_heights[valid_mask]
        
        # 1. 散点图：预测 vs 真实
        plt.figure(figsize=(10, 8))
        
        # 随机采样以减少绘图时间
        sample_size = min(10000, len(valid_preds))
        indices = np.random.choice(len(valid_preds), sample_size, replace=False)
        sample_preds = valid_preds[indices]
        sample_targets = valid_targets[indices]
        
        plt.scatter(sample_targets, sample_preds, alpha=0.5, s=1)
        
        # 完美预测线
        min_val = min(sample_targets.min(), sample_preds.min())
        max_val = max(sample_targets.max(), sample_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('真实nDSM高度 (m)')
        plt.ylabel('预测nDSM高度 (m)')
        plt.title('预测值 vs 真实值散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        scatter_path = os.path.join(save_dir, 'prediction_scatter.png')
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 误差分布直方图
        plt.figure(figsize=(12, 4))
        
        errors = valid_preds - valid_targets
        
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=50, alpha=0.7, density=True)
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        plt.xlabel('预测误差 (m)')
        plt.ylabel('密度')
        plt.title(f'误差分布\nMAE: {np.mean(np.abs(errors)):.2f}m')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        abs_errors = np.abs(errors)
        plt.hist(abs_errors, bins=50, alpha=0.7, density=True, color='orange')
        plt.axvline(np.mean(abs_errors), color='red', linestyle='--', label=f'平均绝对误差: {np.mean(abs_errors):.2f}m')
        plt.xlabel('绝对误差 (m)')
        plt.ylabel('密度')
        plt.title('绝对误差分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        error_path = os.path.join(save_dir, 'error_distribution.png')
        plt.tight_layout()
        plt.savefig(error_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化图表已保存:")
        logger.info(f"  散点图: {scatter_path}")
        logger.info(f"  误差分布: {error_path}")
        
    except Exception as e:
        logger.error(f"生成可视化失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='GAMUS nDSM模型测试脚本')
    
    # 必需参数
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/best_gamus_model.pth',
                        help='训练好的模型检查点路径')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据根目录 (包含images和height子目录)')
    
    # 模型参数（需要与训练时一致）
    parser.add_argument('--encoder', type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl', 'basic_cnn'],
                        help='编码器类型（需要与训练时一致）')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='预训练模型路径（用于模型结构创建）')
    
    # 数据参数（需要与训练时一致）
    parser.add_argument('--normalization_method', type=str, default='minmax',
                        choices=['minmax', 'log_minmax', 'sqrt_minmax', 'percentile', 'zscore_clip'],
                        help='归一化方法（需要与训练时一致）')
    parser.add_argument('--file_extension', type=str, default='auto',
                        choices=['auto', 'tif', 'tiff', 'png', 'jpg', 'jpeg'],
                        help='数据文件扩展名')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='测试批次大小')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                        help='测试结果保存目录')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='huber',
                        choices=['mse', 'mae', 'huber', 'focal', 'combined'],
                        help='损失函数类型')
    parser.add_argument('--height_aware_loss', action='store_true',
                        help='启用高度感知损失权重')
    
    # 可视化参数
    parser.add_argument('--enable_visualization', action='store_true',
                        help='启用结果可视化')
    parser.add_argument('--num_vis_samples', type=int, default=6,
                        help='可视化样本数量')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.save_dir, f'test_log_{timestamp}.log')
    logger = setup_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("GAMUS nDSM模型测试")
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
        
        # 创建模型结构
        logger.info("创建模型结构...")
        try:
            model = create_gamus_model(
                encoder=args.encoder,
                pretrained_path=args.pretrained_path,
                freeze_encoder=True  # 测试时冻结编码器
            )
        except Exception as e:
            logger.warning(f"使用指定编码器失败: {e}, 尝试基础CNN")
            model = GAMUSNDSMPredictor(
                encoder='basic_cnn',
                use_pretrained_dpt=False
            )
        
        # 加载训练好的权重
        model_state_dict = load_trained_model(args.checkpoint_path, device, logger)
        model.load_state_dict(model_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数总数: {total_params:,}")
        
        # 创建测试数据集
        test_loader, test_dataset = create_test_dataset(args.data_dir, args, logger)
        
        # 创建损失函数
        criterion = ImprovedHeightLoss(
            loss_type=args.loss_type,
            height_aware=args.height_aware_loss
        )
        
        # 执行测试
        test_results = test_model(model, test_loader, device, logger, criterion)
        
        if test_results is None:
            logger.error("测试失败，无法获得有效结果")
            return 1
        
        # 计算详细指标
        metrics = calculate_detailed_metrics(
            test_results['predictions'], 
            test_results['targets'],
            test_dataset.get_height_normalizer(),
            logger
        )
        
        # 输出详细结果
        logger.info("\n" + "=" * 60)
        logger.info("GAMUS nDSM模型测试结果")
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
        
        # 保存测试结果
        save_test_results(metrics, test_results, args.save_dir, logger)
        
        # 生成可视化
        if args.enable_visualization:
            create_test_visualizations(
                test_results['predictions'], 
                test_results['targets'],
                test_dataset.get_height_normalizer(),
                args.save_dir, 
                logger, 
                args.num_vis_samples
            )
        
        logger.info("=" * 60)
        logger.info("测试完成!")
        logger.info(f"结果保存在: {args.save_dir}")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)