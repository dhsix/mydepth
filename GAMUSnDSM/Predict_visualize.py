#!/usr/bin/env python3
"""
GAMUS nDSM Model Prediction Visualization Script
Select the first N images from test data to show prediction effects
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
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm

# Import your modules
from improved_dataset import GAMUSDataset
from improved_normalization_loss import ImprovedHeightLoss
from model import GAMUSNDSMPredictor, create_gamus_model

def setup_logger(log_path):
    """Setup logging"""
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
    """Load trained model"""
    logger.info(f"Loading model checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model parameters
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            logger.info(f"Checkpoint info:")
            logger.info(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
            if isinstance(checkpoint.get('val_loss'), (int, float)):
                logger.info(f"  Validation loss: {checkpoint.get('val_loss'):.6f}")
        else:
            model_state_dict = checkpoint
            logger.info("Checkpoint is direct model state dict")
        
        return model_state_dict
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def create_test_dataset(data_dir, args, logger):
    """Create test dataset"""
    logger.info("Creating test dataset...")
    
    # Test set paths
    test_image_dir = os.path.join(data_dir, 'test','images')
    test_label_dir = os.path.join(data_dir, 'test','depths')
    
    # If no dedicated test set, use validation set
    if not os.path.exists(test_image_dir):
        test_image_dir = os.path.join(data_dir, 'val','images')
        test_label_dir = os.path.join(data_dir, 'val','depths')
        logger.info("Test set not found, using validation set for visualization")
    
    # If still not found, use part of training set
    if not os.path.exists(test_image_dir):
        test_image_dir = os.path.join(data_dir, 'train','images')
        test_label_dir = os.path.join(data_dir, 'train','depths')
        logger.info("Validation set not found, using training set for visualization")
    
    if not os.path.exists(test_image_dir):
        raise FileNotFoundError(f"Test image directory does not exist: {test_image_dir}")
    if not os.path.exists(test_label_dir):
        raise FileNotFoundError(f"Test label directory does not exist: {test_label_dir}")
    
    logger.info(f"Test image directory: {test_image_dir}")
    logger.info(f"Test label directory: {test_label_dir}")
    
    # Create training dataset to get normalizer
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
    
    # Create test dataset (using training set's normalizer)
    test_dataset = GAMUSDataset(
        image_dir=test_image_dir,
        label_dir=test_label_dir,
        normalization_method=args.normalization_method,
        enable_augmentation=False,
        stats_json_path=args.stats_json_path,
        height_filter={'min_height': args.min_height, 'max_height': args.max_height}  # ✅ 添加高度过滤
    )
    logger.info(f"Total test set size: {len(test_dataset)}")
    
    return test_dataset

def predict_samples(model, dataset, device, num_samples, logger):
    """Predict specified number of samples"""
    logger.info(f"Starting prediction for first {num_samples} samples...")
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Predicting samples"):
            try:
                # Get single sample
                image, target = dataset[i]
                
                # Add batch dimension
                image_batch = image.unsqueeze(0).to(device)
                target_batch = target.unsqueeze(0).to(device)
                
                # Data quality check
                if torch.isnan(image_batch).any() or torch.isinf(image_batch).any():
                    logger.warning(f"Sample {i}: Input image contains NaN or Inf, skipping")
                    continue
                    
                if torch.isnan(target_batch).any() or torch.isinf(target_batch).any():
                    logger.warning(f"Sample {i}: nDSM target contains NaN or Inf, skipping")
                    continue
                
                # Model prediction
                prediction = model(image_batch)
                
                # Prediction result check
                if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                    logger.warning(f"Sample {i}: nDSM prediction contains NaN or Inf, skipping")
                    continue
                
                # Ensure dimension consistency
                if prediction.shape != target_batch.shape:
                    if prediction.dim() == 3 and target_batch.dim() == 3:
                        if prediction.shape[-2:] != target_batch.shape[-2:]:
                            prediction = F.interpolate(
                                prediction.unsqueeze(1),
                                size=target_batch.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(1)
                    elif prediction.dim() == 4 and target_batch.dim() == 3:
                        prediction = prediction.squeeze(1)
                
                # Numerical range check
                prediction = torch.clamp(prediction, 0, 1)
                target_batch = torch.clamp(target_batch, 0, 1)
                
                # Convert to CPU and remove batch dimension
                image_np = image.cpu().numpy()
                target_np = target_batch.squeeze(0).cpu().numpy()
                prediction_np = prediction.squeeze(0).cpu().numpy()
                
                # Denormalize to real height values
                height_normalizer = dataset.get_normalizer()
                target_real = height_normalizer.denormalize(target_np)
                prediction_real = height_normalizer.denormalize(prediction_np)
                
                # Calculate error
                error = np.abs(prediction_real - target_real)
                mae = np.mean(error)
                rmse = np.sqrt(np.mean((prediction_real - target_real) ** 2))
                
                results.append({
                    'index': i,
                    'image': image_np,
                    'target_normalized': target_np,
                    'prediction_normalized': prediction_np,
                    'target_real': target_real,
                    'prediction_real': prediction_real,
                    'error': error,
                    'mae': mae,
                    'rmse': rmse
                })
                
                logger.info(f"Sample {i}: MAE={mae:.2f}m, RMSE={rmse:.2f}m")
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
    
    logger.info(f"Successfully predicted {len(results)} samples")
    return results

def create_detailed_visualization(results, save_dir, logger, fig_size=(20, 16)):
    """Create detailed visualization charts"""
    logger.info("Creating detailed visualization charts...")
    
    try:
        n_samples = len(results)
        if n_samples == 0:
            logger.error("No valid prediction results")
            return
        
        # Create large figure
        fig, axes = plt.subplots(n_samples, 5, figsize=fig_size)
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        # Set nDSM-specific colormaps
        ndsm_cmap = plt.cm.terrain  # Terrain colormap, suitable for elevation data
        error_cmap = plt.cm.hot     # Heat map, suitable for error visualization
        
        # Calculate global display range
        all_targets = np.concatenate([r['target_real'] for r in results])
        all_predictions = np.concatenate([r['prediction_real'] for r in results])
        all_errors = np.concatenate([r['error'] for r in results])
        
        global_min = min(all_targets.min(), all_predictions.min())
        global_max = max(all_targets.max(), all_predictions.max())
        max_error = min(20, np.percentile(all_errors, 95))  # Limit maximum error display range
        
        logger.info(f"Global height range: [{global_min:.2f}, {global_max:.2f}] m")
        logger.info(f"Error display range: [0, {max_error:.2f}] m")
        
        for i, result in enumerate(results):
            # 1. Input image
            ax = axes[i, 0]
            image = result['image'].transpose(1, 2, 0)  # CHW -> HWC
            
            # Denormalize for display (if ImageNet normalization)
            if image.min() < 0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image * std + mean
                image = np.clip(image, 0, 1)
            
            ax.imshow(image)
            ax.set_title(f'Sample {result["index"]+1}\nInput Image', fontsize=10)
            ax.axis('off')
            
            # 2. Ground Truth nDSM
            ax = axes[i, 1]
            target_real = result['target_real']
            im1 = ax.imshow(target_real, cmap=ndsm_cmap, vmin=global_min, vmax=global_max)
            ax.set_title(f'Ground Truth nDSM\nRange: [{target_real.min():.1f}, {target_real.max():.1f}]m', fontsize=10)
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='Height (m)')
            
            # 3. Predicted nDSM
            ax = axes[i, 2]
            prediction_real = result['prediction_real']
            im2 = ax.imshow(prediction_real, cmap=ndsm_cmap, vmin=global_min, vmax=global_max)
            ax.set_title(f'Predicted nDSM\nRange: [{prediction_real.min():.1f}, {prediction_real.max():.1f}]m', fontsize=10)
            ax.axis('off')
            
            plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Height (m)')
            
            # 4. Error map
            ax = axes[i, 3]
            error = result['error']
            im3 = ax.imshow(error, cmap=error_cmap, vmin=0, vmax=max_error)
            ax.set_title(f'Absolute Error\nMAE: {result["mae"]:.2f}m', fontsize=10)
            ax.axis('off')
            
            plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label='Error (m)')
            
            # 5. Statistics and comparison
            ax = axes[i, 4]
            ax.axis('off')
            
            # Create statistics text
            stats_text = f"""Sample {result['index']+1} Statistics:

MAE: {result['mae']:.2f} m
RMSE: {result['rmse']:.2f} m

Ground Truth Stats:
  Min: {target_real.min():.2f} m
  Max: {target_real.max():.2f} m
  Mean: {target_real.mean():.2f} m
  Std: {target_real.std():.2f} m

Prediction Stats:
  Min: {prediction_real.min():.2f} m
  Max: {prediction_real.max():.2f} m
  Mean: {prediction_real.mean():.2f} m
  Std: {prediction_real.std():.2f} m

Error Stats:
  Max Error: {error.max():.2f} m
  90th Percentile: {np.percentile(error, 90):.2f} m
  95th Percentile: {np.percentile(error, 95):.2f} m"""
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        
        # Save detailed visualization
        detailed_path = os.path.join(save_dir, f'detailed_predictions_{n_samples}_samples.png')
        plt.savefig(detailed_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Detailed visualization saved: {detailed_path}")
        
        # Create error analysis
        create_error_analysis(results, save_dir, logger)
        
        # Create summary statistics
        create_summary_statistics(results, save_dir, logger)
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()

def create_error_analysis(results, save_dir, logger):
    """Create error analysis charts"""
    logger.info("Creating error analysis charts...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract all data
        maes = [r['mae'] for r in results]
        rmses = [r['rmse'] for r in results]
        sample_indices = [r['index'] + 1 for r in results]
        
        # 1. MAE and RMSE comparison
        ax = axes[0, 0]
        x = np.arange(len(results))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, maes, width, label='MAE', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, rmses, width, label='RMSE', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Error (m)')
        ax.set_title('MAE and RMSE Comparison by Sample')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Sample{i}' for i in sample_indices], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Error distribution histogram
        ax = axes[0, 1]
        all_errors = np.concatenate([r['error'].flatten() for r in results])
        
        ax.hist(all_errors, bins=50, alpha=0.7, density=True, color='orange', edgecolor='black')
        ax.axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean Error: {np.mean(all_errors):.2f}m')
        ax.axvline(np.median(all_errors), color='blue', linestyle='--', linewidth=2,
                  label=f'Median Error: {np.median(all_errors):.2f}m')
        
        ax.set_xlabel('Absolute Error (m)')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution for All Samples')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution function
        ax = axes[1, 0]
        sorted_errors = np.sort(all_errors)
        y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        ax.plot(sorted_errors, y, linewidth=2, color='green')
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='1m Error Line')
        ax.axvline(2.0, color='orange', linestyle='--', alpha=0.7, label='2m Error Line')
        ax.axvline(5.0, color='purple', linestyle='--', alpha=0.7, label='5m Error Line')
        
        # Calculate accuracy metrics
        acc_1m = np.mean(all_errors <= 1.0) * 100
        acc_2m = np.mean(all_errors <= 2.0) * 100
        acc_5m = np.mean(all_errors <= 5.0) * 100
        
        ax.set_xlabel('Absolute Error (m)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Error Cumulative Distribution\n±1m: {acc_1m:.1f}%, ±2m: {acc_2m:.1f}%, ±5m: {acc_5m:.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Scatter plot: Ground Truth vs Prediction
        ax = axes[1, 1]
        
        all_targets = np.concatenate([r['target_real'].flatten() for r in results])
        all_predictions = np.concatenate([r['prediction_real'].flatten() for r in results])
        
        # Random sampling to avoid overly dense plots
        sample_size = min(5000, len(all_targets))
        indices = np.random.choice(len(all_targets), sample_size, replace=False)
        
        ax.scatter(all_targets[indices], all_predictions[indices], alpha=0.6, s=1, color='blue')
        
        # Perfect prediction line
        min_val = min(all_targets.min(), all_predictions.min())
        max_val = max(all_targets.max(), all_predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Ground Truth nDSM Height (m)')
        ax.set_ylabel('Predicted nDSM Height (m)')
        ax.set_title('Predicted vs Ground Truth Scatter Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        error_analysis_path = os.path.join(save_dir, 'error_analysis.png')
        plt.savefig(error_analysis_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Error analysis chart saved: {error_analysis_path}")
        
    except Exception as e:
        logger.error(f"Failed to create error analysis chart: {e}")

def create_summary_statistics(results, save_dir, logger):
    """Create overall statistical summary"""
    logger.info("Creating statistical summary...")
    
    try:
        # Calculate overall statistics
        all_errors = np.concatenate([r['error'].flatten() for r in results])
        all_targets = np.concatenate([r['target_real'].flatten() for r in results])
        all_predictions = np.concatenate([r['prediction_real'].flatten() for r in results])
        
        overall_mae = np.mean(all_errors)
        overall_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        
        # Calculate R²
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Accuracy metrics
        acc_1m = np.mean(all_errors <= 1.0) * 100
        acc_2m = np.mean(all_errors <= 2.0) * 100
        acc_5m = np.mean(all_errors <= 5.0) * 100
        acc_10m = np.mean(all_errors <= 10.0) * 100
        
        # Create summary text file
        summary_file = os.path.join(save_dir, 'prediction_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("GAMUS nDSM Model Prediction Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Samples: {len(results)}\n")
            f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Overall Performance Metrics:\n")
            f.write(f"  MAE (Mean Absolute Error): {overall_mae:.4f} m\n")
            f.write(f"  RMSE (Root Mean Square Error): {overall_rmse:.4f} m\n")
            f.write(f"  R² (Coefficient of Determination): {r2:.4f}\n\n")
            
            f.write("Height Accuracy Metrics:\n")
            f.write(f"  ±1m Accuracy: {acc_1m:.2f}%\n")
            f.write(f"  ±2m Accuracy: {acc_2m:.2f}%\n")
            f.write(f"  ±5m Accuracy: {acc_5m:.2f}%\n")
            f.write(f"  ±10m Accuracy: {acc_10m:.2f}%\n\n")
            
            f.write("Data Range:\n")
            f.write(f"  Ground Truth Range: [{all_targets.min():.2f}, {all_targets.max():.2f}] m\n")
            f.write(f"  Prediction Range: [{all_predictions.min():.2f}, {all_predictions.max():.2f}] m\n")
            f.write(f"  Error Range: [0, {all_errors.max():.2f}] m\n\n")
            
            f.write("Individual Sample Results:\n")
            for i, result in enumerate(results):
                f.write(f"  Sample{result['index']+1}: MAE={result['mae']:.3f}m, RMSE={result['rmse']:.3f}m\n")
        
        logger.info(f"Statistical summary saved: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to create statistical summary: {e}")
# 在可视化脚本中添加0值影响分析函数：

def analyze_zero_impact(results, save_dir, logger):
    """分析0值对指标的影响"""
    logger.info("Analyzing impact of zero values on evaluation metrics...")
    
    try:
        all_predictions = np.concatenate([r['prediction_real'].flatten() for r in results])
        all_targets = np.concatenate([r['target_real'].flatten() for r in results])
        
        # 计算包含0值的指标
        errors_with_zeros = np.abs(all_predictions - all_targets)
        mae_with_zeros = np.mean(errors_with_zeros)
        rmse_with_zeros = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        
        # 计算排除0值的指标
        zero_mask = np.abs(all_targets) < 0.1  # 0.1米容差
        non_zero_mask = ~zero_mask
        
        if np.sum(non_zero_mask) > 0:
            errors_without_zeros = errors_with_zeros[non_zero_mask]
            mae_without_zeros = np.mean(errors_without_zeros)
            rmse_without_zeros = np.sqrt(np.mean((all_predictions[non_zero_mask] - all_targets[non_zero_mask]) ** 2))
        else:
            mae_without_zeros = rmse_without_zeros = float('inf')
        
        # 统计0值像素
        zero_count = np.sum(zero_mask)
        total_count = len(all_targets)
        zero_percentage = zero_count / total_count * 100
        
        # 创建对比报告
        comparison_file = os.path.join(save_dir, 'zero_value_impact_analysis.txt')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("Zero Value Impact Analysis on Evaluation Metrics\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Data Statistics:\n")
            f.write(f"  Total Pixels: {total_count:,}\n")
            f.write(f"  Zero Value Pixels: {zero_count:,}\n") 
            f.write(f"  Zero Value Percentage: {zero_percentage:.2f}%\n\n")
            
            f.write(f"Metrics Comparison:\n")
            f.write(f"  Including Zero Values:\n")
            f.write(f"    MAE: {mae_with_zeros:.4f} m\n")
            f.write(f"    RMSE: {rmse_with_zeros:.4f} m\n\n")
            
            f.write(f"  Excluding Zero Values:\n")
            f.write(f"    MAE: {mae_without_zeros:.4f} m\n")
            f.write(f"    RMSE: {rmse_without_zeros:.4f} m\n\n")
            
            if mae_without_zeros != float('inf'):
                mae_diff = mae_without_zeros - mae_with_zeros
                rmse_diff = rmse_without_zeros - rmse_with_zeros
                f.write(f"  Differences:\n")
                f.write(f"    MAE Difference: {mae_diff:.4f} m ({mae_diff/mae_with_zeros*100:+.1f}%)\n")
                f.write(f"    RMSE Difference: {rmse_diff:.4f} m ({rmse_diff/rmse_with_zeros*100:+.1f}%)\n\n")
            
            f.write(f"Recommendations:\n")
            if zero_percentage > 10:
                f.write(f"  - High proportion of zero values ({zero_percentage:.1f}%), consider excluding in training and evaluation\n")
            else:
                f.write(f"  - Low proportion of zero values ({zero_percentage:.1f}%), limited impact on overall metrics\n")
                
            if mae_without_zeros > mae_with_zeros * 1.1:
                f.write(f"  - MAE significantly increases after excluding zeros, indicating worse performance in non-zero regions\n")
            elif mae_without_zeros < mae_with_zeros * 0.9:
                f.write(f"  - MAE significantly decreases after excluding zeros, indicating potential systematic errors in zero regions\n")
        
        logger.info(f"Zero value impact analysis saved: {comparison_file}")
        logger.info(f"Zero value pixel percentage: {zero_percentage:.2f}%")
        logger.info(f"MAE comparison: with zeros={mae_with_zeros:.4f}m, without zeros={mae_without_zeros:.4f}m")
        
        return zero_percentage, mae_with_zeros, mae_without_zeros
        
    except Exception as e:
        logger.error(f"Zero value impact analysis failed: {e}")
        return 0, 0, 0

def main():
    parser = argparse.ArgumentParser(description='GAMUS nDSM Model Prediction Visualization Script')
    
    # Required parameters
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/best_gamus_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data root directory (containing images and height subdirectories)')
    
    # Model parameters (must match training settings)
    parser.add_argument('--encoder', type=str, default='vitb',
                        choices=['vits', 'vitb', 'vitl', 'basic_cnn'],
                        help='Encoder type (must match training settings)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Pretrained model path (for model structure creation)')
    parser.add_argument('--stats_json_path', type=str, default='./gamus_full_stats.json',
                        help='预计算统计信息JSON文件路径')
    parser.add_argument('--min_height', type=float, default=-5.0,
                    help='最小高度过滤值（米）')
    parser.add_argument('--max_height', type=float, default=200.0,
                    help='最大高度过滤值（米）')
    # Data parameters (must match training settings)
    parser.add_argument('--normalization_method', type=str, default='minmax',
                        choices=['minmax', 'log_minmax', 'sqrt_minmax', 'percentile', 'zscore_clip'],
                        help='Normalization method (must match training settings)')
    parser.add_argument('--file_extension', type=str, default='auto',
                        choices=['auto', 'tif', 'tiff', 'png', 'jpg', 'jpeg'],
                        help='Data file extension')
    
    # Visualization parameters
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='auto',
                        help='Computing device')
    parser.add_argument('--save_dir', type=str, default='./visualization_results',
                        help='Visualization results save directory')
    parser.add_argument('--fig_width', type=int, default=20,
                        help='Figure width')
    parser.add_argument('--fig_height', type=int, default=16,
                        help='Figure height')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.save_dir, f'visualization_log_{timestamp}.log')
    logger = setup_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("GAMUS nDSM Model Prediction Visualization")
    logger.info("=" * 80)
    logger.info(f"Visualization parameters: {vars(args)}")
    
    try:
        # Device setup
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                logger.info(f"Auto-detected CUDA, using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device(args.device)
        
        logger.info(f"Using device: {device}")
        
        # Create model structure
        logger.info("Creating model structure...")
        try:
            model = create_gamus_model(
                encoder=args.encoder,
                pretrained_path=args.pretrained_path,
                freeze_encoder=True  # Freeze encoder during testing
            )
        except Exception as e:
            logger.warning(f"Failed to use specified encoder: {e}, trying basic CNN")
            model = GAMUSNDSMPredictor(
                encoder='basic_cnn',
                use_pretrained_dpt=False
            )
        
        # Load trained weights
        model_state_dict = load_trained_model(args.checkpoint_path, device, logger)
        model.load_state_dict(model_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters: {total_params:,}")
        
        # Create test dataset
        test_dataset = create_test_dataset(args.data_dir, args, logger)
        
        # Predict samples
        results = predict_samples(model, test_dataset, device, args.num_samples, logger)
        
        if not results:
            logger.error("No successful predictions")
            return 1
        
        # Create visualization
        create_detailed_visualization(
            results, 
            args.save_dir, 
            logger, 
            fig_size=(args.fig_width, args.fig_height)
        )
        # 新增：0值影响分析
        zero_percentage, mae_with, mae_without = analyze_zero_impact(results, args.save_dir, logger)
            
        logger.info("=" * 60)
        logger.info("Zero Value Analysis Summary:")
        logger.info(f"  Zero pixels: {zero_percentage:.2f}% of total")
        logger.info(f"  MAE impact: {((mae_without - mae_with) / mae_with * 100):+.1f}% when excluding zeros")




        logger.info("=" * 60)
        logger.info("Visualization completed!")
        logger.info(f"Results saved in: {args.save_dir}")
        logger.info(f"Successfully visualized {len(results)} samples")
        
        # Output brief statistics
        avg_mae = np.mean([r['mae'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])
        logger.info(f"Average MAE: {avg_mae:.3f}m")
        logger.info(f"Average RMSE: {avg_rmse:.3f}m")
        
    except Exception as e:
        logger.error(f"Error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)