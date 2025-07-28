#!/usr/bin/env python3
"""
Multi-model Comparison Prediction Visualization Script
Defaults to selecting specified number of images from test directory for prediction visualization
Supports GAMUS and Depth2Elevation models, supports mask functionality
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

# Import updated modules
from improved_dataset_with_mask import create_gamus_dataloader
from improved_normalization_loss import create_height_loss
from model_with_comparison import create_gamus_model

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
        
        # Extract model parameters and type information
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            model_type = checkpoint.get('model_type', 'gamus')
            logger.info(f"Checkpoint info:")
            logger.info(f"  Model type: {model_type}")
            logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            if isinstance(checkpoint.get('loss'), (int, float)):
                logger.info(f"  Validation loss: {checkpoint.get('loss'):.6f}")
        else:
            model_state_dict = checkpoint
            model_type = 'gamus'
            logger.info("Checkpoint is a direct model state dict, assuming GAMUS model")
        
        return model_state_dict, model_type
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def create_test_dataset(data_dir, args, logger):
    """Create test dataset (prioritize test directory)"""
    logger.info("Creating test dataset...")
    
    # Prioritize test directory
    test_image_dir = os.path.join(data_dir, 'test', 'images')
    test_label_dir = os.path.join(data_dir, 'test', 'depths')
    test_mask_dir = None
    
    if args.mask_dir:
        test_mask_dir = os.path.join(args.mask_dir, 'test', 'classes')
    
    # Check if test directory exists
    if os.path.exists(test_image_dir) and os.path.exists(test_label_dir):
        logger.info("✅ Found test directory, will use test data for visualization")
        data_split = 'test'
    else:
        # Fall back to val directory
        test_image_dir = os.path.join(data_dir, 'val', 'images')
        test_label_dir = os.path.join(data_dir, 'val', 'depths')
        if args.mask_dir:
            test_mask_dir = os.path.join(args.mask_dir, 'val', 'classes')
        
        if os.path.exists(test_image_dir) and os.path.exists(test_label_dir):
            logger.info("⚠️ Test directory not found, using val directory for visualization")
            data_split = 'val'
        else:
            # Finally fall back to train directory
            test_image_dir = os.path.join(data_dir, 'train', 'images')
            test_label_dir = os.path.join(data_dir, 'train', 'depths')
            if args.mask_dir:
                test_mask_dir = os.path.join(args.mask_dir, 'train', 'classes')
            
            if os.path.exists(test_image_dir) and os.path.exists(test_label_dir):
                logger.info("⚠️ Val directory also not found, using train directory (note: potential overfitting)")
                data_split = 'train'
            else:
                raise FileNotFoundError(f"All data directories not found: {test_image_dir}")
    
    # Check mask directory
    if args.mask_dir and test_mask_dir and not os.path.exists(test_mask_dir):
        logger.warning(f"Mask directory not found: {test_mask_dir}, will not use mask")
        test_mask_dir = None
    
    logger.info(f"📁 Using dataset: {data_split}")
    logger.info(f"  Image directory: {test_image_dir}")
    logger.info(f"  Label directory: {test_label_dir}")
    if test_mask_dir:
        logger.info(f"  Mask directory: {test_mask_dir}")
    
    # Setup height filter
    height_filter = {
        'min_height': args.min_height,
        'max_height': args.max_height
    }
    
    # Create data loader (supports mask)
    try:
        test_loader, test_dataset = create_gamus_dataloader(
            image_dir=test_image_dir,
            label_dir=test_label_dir,
            mask_dir=test_mask_dir,
            building_class_id=args.building_class_id,
            tree_class_id=args.tree_class_id,
            batch_size=1,  # Use batch_size=1 for visualization
            shuffle=False,  # Don't shuffle, select first N in order
            normalization_method=args.normalization_method,
            enable_augmentation=False,
            stats_json_path=args.stats_json_path,
            height_filter=height_filter,
            force_recompute=False,
            num_workers=0  # Single thread for visualization
        )
        
        logger.info(f"📊 Test set size: {len(test_dataset)}")
        logger.info(f"🎯 Will select from first {min(args.num_samples, len(test_dataset))} samples for visualization")
        
        return test_loader, test_dataset, data_split
        
    except Exception as e:
        logger.error(f"Failed to create test dataset: {e}")
        raise

def predict_samples(model, test_loader, device, num_samples, logger, use_mask=False):
    """Predict specified number of samples (supports mask)"""
    logger.info(f"Starting prediction of first {num_samples} samples...")
    
    model.eval()
    results = []
    
    with torch.no_grad():
        sample_count = 0
        
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Predicting samples")):
            if sample_count >= num_samples:
                break
                
            try:
                # Handle batch data that may contain mask
                if len(batch_data) == 3:
                    image_batch, target_batch, mask_batch = batch_data
                    image_batch = image_batch.to(device)
                    target_batch = target_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    has_mask = True
                else:
                    image_batch, target_batch = batch_data
                    image_batch = image_batch.to(device)
                    target_batch = target_batch.to(device)
                    mask_batch = torch.ones_like(target_batch).to(device)
                    has_mask = False
                
                # Data quality check
                if torch.isnan(image_batch).any() or torch.isinf(image_batch).any():
                    logger.warning(f"Sample {batch_idx}: Input image contains NaN or Inf, skipping")
                    continue
                    
                if torch.isnan(target_batch).any() or torch.isinf(target_batch).any():
                    logger.warning(f"Sample {batch_idx}: nDSM target contains NaN or Inf, skipping")
                    continue
                
                # Model prediction
                prediction = model(image_batch)
                
                # Prediction result check
                if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                    logger.warning(f"Sample {batch_idx}: nDSM prediction contains NaN or Inf, skipping")
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
                image_np = image_batch.squeeze(0).cpu().numpy()
                target_np = target_batch.squeeze(0).cpu().numpy()
                prediction_np = prediction.squeeze(0).cpu().numpy()
                mask_np = mask_batch.squeeze(0).cpu().numpy() if has_mask else None
                
                # Denormalize to real height values
                # Get normalizer from data loader
                height_normalizer = test_loader.dataset.get_normalizer()
                target_real = height_normalizer.denormalize(target_np)
                prediction_real = height_normalizer.denormalize(prediction_np)
                
                # If there's mask, only calculate error in mask=1 regions
                if use_mask and mask_np is not None:
                    valid_mask = (mask_np > 0.5) & (target_np >= 0)
                    if valid_mask.sum() > 0:
                        target_real_masked = target_real[valid_mask]
                        prediction_real_masked = prediction_real[valid_mask]
                        error_masked = np.abs(prediction_real_masked - target_real_masked)
                        mae = np.mean(error_masked)
                        rmse = np.sqrt(np.mean((prediction_real_masked - target_real_masked) ** 2))
                        
                        # Create full-size error map, non-mask regions are 0
                        error_full = np.zeros_like(target_real)
                        error_full[valid_mask] = np.abs(prediction_real - target_real)[valid_mask]
                    else:
                        mae = rmse = 0.0
                        error_full = np.zeros_like(target_real)
                else:
                    # Calculate full-image error
                    error_full = np.abs(prediction_real - target_real)
                    mae = np.mean(error_full)
                    rmse = np.sqrt(np.mean((prediction_real - target_real) ** 2))
                
                results.append({
                    'index': sample_count,
                    'batch_index': batch_idx,
                    'image': image_np,
                    'target_normalized': target_np,
                    'prediction_normalized': prediction_np,
                    'target_real': target_real,
                    'prediction_real': prediction_real,
                    'error': error_full,
                    'mask': mask_np,
                    'has_mask': has_mask,
                    'mae': mae,
                    'rmse': rmse
                })
                
                logger.info(f"Sample {sample_count+1}: MAE={mae:.2f}m, RMSE={rmse:.2f}m")
                sample_count += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {batch_idx}: {e}")
                continue
    
    logger.info(f"Successfully predicted {len(results)} samples")
    return results

def create_detailed_visualization(results, save_dir, logger, model_type, has_mask=False, fig_size=(24, 16)):
    """Create detailed visualization charts"""
    logger.info("Creating detailed visualization charts...")
    
    try:
        n_samples = len(results)
        if n_samples == 0:
            logger.error("No valid prediction results")
            return
        
        # Fixed number of columns (removed mask column)
        n_cols = 5
        
        # Create large figure
        fig, axes = plt.subplots(n_samples, n_cols, figsize=fig_size)
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{model_type.upper()} Model Prediction Results', fontsize=16, fontweight='bold')
        
        # Setup nDSM-specific color mappings
        ndsm_cmap = plt.cm.terrain  # Terrain colormap, suitable for elevation data
        error_cmap = plt.cm.hot     # Heat map, suitable for error visualization
        
        # Calculate global display range
        all_targets = np.concatenate([r['target_real'] for r in results])
        all_predictions = np.concatenate([r['prediction_real'] for r in results])
        all_errors = np.concatenate([r['error'] for r in results])
        
        global_min = min(all_targets.min(), all_predictions.min())
        global_max = max(all_targets.max(), all_predictions.max())
        max_error = min(20, np.percentile(all_errors, 95))  # Limit max error display range
        
        logger.info(f"Global height range: [{global_min:.2f}, {global_max:.2f}] m")
        logger.info(f"Error display range: [0, {max_error:.2f}] m")
        
        for i, result in enumerate(results):
            col = 0
            
            # 1. Input image
            ax = axes[i, col]
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
            col += 1
            
            # 2. Ground Truth nDSM
            ax = axes[i, col]
            target_real = result['target_real']
            im1 = ax.imshow(target_real, cmap=ndsm_cmap, vmin=global_min, vmax=global_max)
            ax.set_title(f'Ground Truth nDSM\n[{target_real.min():.1f}, {target_real.max():.1f}]m', fontsize=10)
            ax.axis('off')
            plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='Height (m)')
            col += 1
            
            # 3. Predicted nDSM
            ax = axes[i, col]
            prediction_real = result['prediction_real']
            im2 = ax.imshow(prediction_real, cmap=ndsm_cmap, vmin=global_min, vmax=global_max)
            ax.set_title(f'Predicted nDSM\n[{prediction_real.min():.1f}, {prediction_real.max():.1f}]m', fontsize=10)
            ax.axis('off')
            plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Height (m)')
            col += 1
            
            # 4. Error map
            ax = axes[i, col]
            error = result['error']
            im3 = ax.imshow(error, cmap=error_cmap, vmin=0, vmax=max_error)
            ax.set_title(f'Absolute Error\nMAE: {result["mae"]:.2f}m', fontsize=10)
            ax.axis('off')
            plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label='Error (m)')
            col += 1
            
            # 5. Statistics and comparison
            ax = axes[i, col]
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
  Max error: {error.max():.2f} m
  90th percentile: {np.percentile(error, 90):.2f} m
  95th percentile: {np.percentile(error, 95):.2f} m"""
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        
        # Save detailed visualization
        detailed_path = os.path.join(save_dir, f'{model_type}_detailed_predictions_{n_samples}_samples.png')
        plt.savefig(detailed_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Detailed visualization saved: {detailed_path}")
        
        # Create error analysis
        create_error_analysis(results, save_dir, logger, model_type)
        
        # Create summary statistics
        create_summary_statistics(results, save_dir, logger, model_type, has_mask)
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()

def create_individual_visualizations(results, save_dir, logger, model_type, has_mask=False):
    """Create individual visualization for each sample"""
    logger.info("Creating individual visualizations for each sample...")
    
    try:
        # Create individual directory
        individual_dir = os.path.join(save_dir, 'individual_samples')
        os.makedirs(individual_dir, exist_ok=True)
        
        # Fixed number of columns (removed mask column)
        n_cols = 5
        
        # Setup color mappings
        ndsm_cmap = plt.cm.terrain
        error_cmap = plt.cm.hot
        
        # Calculate global display range for consistency
        all_targets = np.concatenate([r['target_real'] for r in results])
        all_predictions = np.concatenate([r['prediction_real'] for r in results])
        all_errors = np.concatenate([r['error'] for r in results])
        
        global_min = min(all_targets.min(), all_predictions.min())
        global_max = max(all_targets.max(), all_predictions.max())
        max_error = min(20, np.percentile(all_errors, 95))
        
        for i, result in enumerate(results):
            # Create figure for individual sample
            fig, axes = plt.subplots(1, n_cols, figsize=(20, 4))
            
            fig.suptitle(f'{model_type.upper()} - Sample {result["index"]+1} (MAE: {result["mae"]:.2f}m, RMSE: {result["rmse"]:.2f}m)', 
                        fontsize=14, fontweight='bold')
            
            col = 0
            
            # 1. Input image
            ax = axes[col]
            image = result['image'].transpose(1, 2, 0)  # CHW -> HWC
            
            # Denormalize for display
            if image.min() < 0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image * std + mean
                image = np.clip(image, 0, 1)
            
            ax.imshow(image)
            ax.set_title('Input Image', fontsize=12)
            ax.axis('off')
            col += 1
            
            # 2. Ground Truth nDSM
            ax = axes[col]
            target_real = result['target_real']
            im1 = ax.imshow(target_real, cmap=ndsm_cmap, vmin=global_min, vmax=global_max)
            ax.set_title(f'Ground Truth nDSM\n[{target_real.min():.1f}, {target_real.max():.1f}]m', fontsize=12)
            ax.axis('off')
            plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='Height (m)')
            col += 1
            
            # 3. Predicted nDSM
            ax = axes[col]
            prediction_real = result['prediction_real']
            im2 = ax.imshow(prediction_real, cmap=ndsm_cmap, vmin=global_min, vmax=global_max)
            ax.set_title(f'Predicted nDSM\n[{prediction_real.min():.1f}, {prediction_real.max():.1f}]m', fontsize=12)
            ax.axis('off')
            plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Height (m)')
            col += 1
            
            # 4. Error map
            ax = axes[col]
            error = result['error']
            im3 = ax.imshow(error, cmap=error_cmap, vmin=0, vmax=max_error)
            ax.set_title(f'Absolute Error\nMAE: {result["mae"]:.2f}m', fontsize=12)
            ax.axis('off')
            plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label='Error (m)')
            col += 1
            
            # 5. Statistics
            ax = axes[col]
            ax.axis('off')
            
            # Create statistics text
            stats_text = f"""Performance Metrics:
MAE: {result['mae']:.3f} m
RMSE: {result['rmse']:.3f} m

Ground Truth:
Min: {target_real.min():.2f} m
Max: {target_real.max():.2f} m
Mean: {target_real.mean():.2f} m
Std: {target_real.std():.2f} m

Prediction:
Min: {prediction_real.min():.2f} m
Max: {prediction_real.max():.2f} m
Mean: {prediction_real.mean():.2f} m
Std: {prediction_real.std():.2f} m

Error Analysis:
Max: {error.max():.2f} m
90%: {np.percentile(error, 90):.2f} m
95%: {np.percentile(error, 95):.2f} m"""
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            plt.tight_layout()
            
            # Save individual visualization
            individual_path = os.path.join(individual_dir, f'{model_type}_sample_{result["index"]+1:02d}.png')
            plt.savefig(individual_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Individual visualization saved: sample_{result['index']+1:02d}")
        
        logger.info(f"All individual visualizations saved to: {individual_dir}")
        
    except Exception as e:
        logger.error(f"Failed to create individual visualizations: {e}")
        import traceback
        traceback.print_exc()

def create_error_analysis(results, save_dir, logger, model_type):
    """Create error analysis charts"""
    logger.info("Creating error analysis charts...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_type.upper()} Model Error Analysis', fontsize=16)
        
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
        ax.set_xticklabels([f'S{i}' for i in sample_indices], rotation=45)
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
                  label=f'Mean: {np.mean(all_errors):.2f}m')
        ax.axvline(np.median(all_errors), color='blue', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(all_errors):.2f}m')
        
        ax.set_xlabel('Absolute Error (m)')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution Across All Samples')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution function
        ax = axes[1, 0]
        sorted_errors = np.sort(all_errors)
        y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        ax.plot(sorted_errors, y, linewidth=2, color='green')
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='1m threshold')
        ax.axvline(2.0, color='orange', linestyle='--', alpha=0.7, label='2m threshold')
        ax.axvline(5.0, color='purple', linestyle='--', alpha=0.7, label='5m threshold')
        
        # Calculate accuracy metrics
        acc_1m = np.mean(all_errors <= 1.0) * 100
        acc_2m = np.mean(all_errors <= 2.0) * 100
        acc_5m = np.mean(all_errors <= 5.0) * 100
        
        ax.set_xlabel('Absolute Error (m)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Error Cumulative Distribution\n±1m: {acc_1m:.1f}%, ±2m: {acc_2m:.1f}%, ±5m: {acc_5m:.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Scatter plot: Ground truth vs Prediction
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
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        ax.set_xlabel('Ground Truth nDSM Height (m)')
        ax.set_ylabel('Predicted nDSM Height (m)')
        ax.set_title('Prediction vs Ground Truth Scatter Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        error_analysis_path = os.path.join(save_dir, f'{model_type}_error_analysis.png')
        plt.savefig(error_analysis_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Error analysis chart saved: {error_analysis_path}")
        
    except Exception as e:
        logger.error(f"Failed to create error analysis chart: {e}")

def create_summary_statistics(results, save_dir, logger, model_type, has_mask=False):
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
        summary_file = os.path.join(save_dir, f'{model_type}_prediction_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"{model_type.upper()} nDSM Model Prediction Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test samples: {len(results)}\n")
            f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if has_mask:
                f.write(f"Using mask: Yes\n")
            f.write("\n")
            
            f.write("Overall Performance Metrics:\n")
            f.write(f"  MAE (Mean Absolute Error): {overall_mae:.4f} m\n")
            f.write(f"  RMSE (Root Mean Square Error): {overall_rmse:.4f} m\n")
            f.write(f"  R² (Coefficient of Determination): {r2:.4f}\n\n")
            
            f.write("Height Accuracy Metrics:\n")
            f.write(f"  ±1m accuracy: {acc_1m:.2f}%\n")
            f.write(f"  ±2m accuracy: {acc_2m:.2f}%\n")
            f.write(f"  ±5m accuracy: {acc_5m:.2f}%\n")
            f.write(f"  ±10m accuracy: {acc_10m:.2f}%\n\n")
            
            f.write("Data Range:\n")
            f.write(f"  Ground truth range: [{all_targets.min():.2f}, {all_targets.max():.2f}] m\n")
            f.write(f"  Prediction range: [{all_predictions.min():.2f}, {all_predictions.max():.2f}] m\n")
            f.write(f"  Error range: [0, {all_errors.max():.2f}] m\n\n")
            
            f.write("Individual Sample Results:\n")
            for i, result in enumerate(results):
                f.write(f"  Sample{result['index']+1}: MAE={result['mae']:.3f}m, RMSE={result['rmse']:.3f}m\n")
        
        logger.info(f"Summary statistics saved: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to create summary statistics: {e}")

def analyze_zero_impact(results, save_dir, logger, model_type):
    """Analyze the impact of zero values on metrics"""
    logger.info("Analyzing the impact of zero values on evaluation metrics...")
    
    try:
        all_predictions = np.concatenate([r['prediction_real'].flatten() for r in results])
        all_targets = np.concatenate([r['target_real'].flatten() for r in results])
        
        # Calculate metrics including zero values
        errors_with_zeros = np.abs(all_predictions - all_targets)
        mae_with_zeros = np.mean(errors_with_zeros)
        rmse_with_zeros = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        
        # Calculate metrics excluding zero values
        zero_mask = np.abs(all_targets) < 0.1  # 0.1m tolerance
        non_zero_mask = ~zero_mask
        
        if np.sum(non_zero_mask) > 0:
            errors_without_zeros = errors_with_zeros[non_zero_mask]
            mae_without_zeros = np.mean(errors_without_zeros)
            rmse_without_zeros = np.sqrt(np.mean((all_predictions[non_zero_mask] - all_targets[non_zero_mask]) ** 2))
        else:
            mae_without_zeros = rmse_without_zeros = float('inf')
        
        # Count zero value pixels
        zero_count = np.sum(zero_mask)
        total_count = len(all_targets)
        zero_percentage = zero_count / total_count * 100
        
        # Create comparison report
        comparison_file = os.path.join(save_dir, f'{model_type}_zero_value_impact_analysis.txt')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("Zero Value Impact Analysis on Evaluation Metrics\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Data Statistics:\n")
            f.write(f"  Total pixels: {total_count:,}\n")
            f.write(f"  Zero value pixels: {zero_count:,}\n") 
            f.write(f"  Zero value percentage: {zero_percentage:.2f}%\n\n")
            
            f.write(f"Metric Comparison:\n")
            f.write(f"  Including zero values:\n")
            f.write(f"    MAE: {mae_with_zeros:.4f} m\n")
            f.write(f"    RMSE: {rmse_with_zeros:.4f} m\n\n")
            
            f.write(f"  Excluding zero values:\n")
            f.write(f"    MAE: {mae_without_zeros:.4f} m\n")
            f.write(f"    RMSE: {rmse_without_zeros:.4f} m\n\n")
            
            if mae_without_zeros != float('inf'):
                mae_diff = mae_without_zeros - mae_with_zeros
                rmse_diff = rmse_without_zeros - rmse_with_zeros
                f.write(f"  Difference:\n")
                f.write(f"    MAE difference: {mae_diff:.4f} m ({mae_diff/mae_with_zeros*100:+.1f}%)\n")
                f.write(f"    RMSE difference: {rmse_diff:.4f} m ({rmse_diff/rmse_with_zeros*100:+.1f}%)\n\n")
            
            f.write(f"Recommendations:\n")
            if zero_percentage > 10:
                f.write(f"  - High zero value ratio ({zero_percentage:.1f}%), recommend excluding from training and evaluation\n")
            else:
                f.write(f"  - Low zero value ratio ({zero_percentage:.1f}%), limited impact on overall metrics\n")
                
            if mae_without_zeros > mae_with_zeros * 1.1:
                f.write(f"  - MAE significantly increases after excluding zeros, indicating poor performance in non-zero regions\n")
            elif mae_without_zeros < mae_with_zeros * 0.9:
                f.write(f"  - MAE significantly decreases after excluding zeros, indicating systematic errors in zero regions\n")
        
        logger.info(f"Zero value impact analysis saved: {comparison_file}")
        logger.info(f"Zero value pixel percentage: {zero_percentage:.2f}%")
        logger.info(f"MAE comparison: including zeros={mae_with_zeros:.4f}m, excluding zeros={mae_without_zeros:.4f}m")
        
        return zero_percentage, mae_with_zeros, mae_without_zeros
        
    except Exception as e:
        logger.error(f"Zero value impact analysis failed: {e}")
        return 0, 0, 0

def main():
    parser = argparse.ArgumentParser(description='Multi-model comparison prediction visualization script')
    
    # Required parameters
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='/mnt/data1/UserData/hudong26/HeightData/',
                        help='Data root directory (containing test/val/train subdirectories)')
    parser.add_argument('--stats_json_path', type=str, default='./gamus_full_stats.json',
                        help='Pre-computed statistics JSON file path')
    
    # Model parameters (will be auto-inferred from checkpoint, these are backup)
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['auto', 'gamus', 'depth2elevation'],
                        help='Model type (auto means auto-infer from checkpoint)')
    parser.add_argument('--encoder', type=str, default='vitb',
                        choices=['vits', 'vitb', 'vitl'],
                        help='Encoder type (must be consistent with training)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Pretrained model path (for model structure creation)')
    
    # Data parameters (must be consistent with training)
    parser.add_argument('--normalization_method', type=str, default='minmax',
                        choices=['minmax', 'percentile', 'zscore'],
                        help='Normalization method (must be consistent with training)')
    parser.add_argument('--min_height', type=float, default=-5.0,
                        help='Minimum height filter value (meters)')
    parser.add_argument('--max_height', type=float, default=200.0,
                        help='Maximum height filter value (meters)')
    
    # Mask related parameters
    parser.add_argument('--mask_dir', type=str, default='/mnt/data1/UserData/hudong26/HeightData/',
                        help='Classes mask root directory')
    parser.add_argument('--building_class_id', type=int, default=3,
                        help='Building class ID')
    parser.add_argument('--tree_class_id', type=int, default=6,
                        help='Tree class ID')
    
    # Visualization parameters
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of samples to visualize (default first 10 from test directory)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Compute device')
    parser.add_argument('--save_dir', type=str, default='./visualization_results',
                        help='Visualization results save directory')
    parser.add_argument('--fig_width', type=int, default=24,
                        help='Chart width')
    parser.add_argument('--fig_height', type=int, default=16,
                        help='Chart height')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.save_dir, f'visualization_log_{timestamp}.log')
    logger = setup_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("Multi-model Comparison Prediction Visualization")
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
        
        # Load trained weights and model type
        model_state_dict, detected_model_type = load_trained_model(args.checkpoint_path, device, logger)
        
        # Determine model type
        if args.model_type == 'auto':
            model_type = detected_model_type
        else:
            model_type = args.model_type
        
        logger.info(f"Using model type: {model_type}")
        
        # Create model structure
        logger.info("Creating model structure...")
        try:
            model_kwargs = {
                'encoder': args.encoder,
                'pretrained_path': args.pretrained_path,
                'freeze_encoder': True,  # Freeze encoder during testing
                'model_type': model_type
            }
            
            # Add specific parameters for Depth2Elevation
            if model_type == 'depth2elevation':
                model_kwargs.update({
                    'img_size': 448,
                    'patch_size': 14,
                    'use_multi_scale_output': False,  # Use single scale for visualization
                    'loss_config': {},
                    'freezing_config': {}
                })
            
            model = create_gamus_model(**model_kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
        
        # Load weights
        model.load_state_dict(model_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters: {total_params:,}")
        
        # Create test dataset
        test_loader, test_dataset, data_split = create_test_dataset(args.data_dir, args, logger)
        
        # Check if using mask
        use_mask = args.mask_dir is not None
        
        # Predict samples
        results = predict_samples(model, test_loader, device, args.num_samples, logger, use_mask)
        
        if not results:
            logger.error("No successful prediction results")
            return 1
        
        # Check if results contain mask
        has_mask = any(r['has_mask'] for r in results)
        
        # Create combined visualization
        create_detailed_visualization(
            results, 
            args.save_dir, 
            logger, 
            model_type,
            has_mask,
            fig_size=(args.fig_width, args.fig_height)
        )
        
        # Create individual visualizations
        create_individual_visualizations(
            results,
            args.save_dir,
            logger,
            model_type,
            has_mask
        )
        
        # Zero value impact analysis
        zero_percentage, mae_with, mae_without = analyze_zero_impact(results, args.save_dir, logger, model_type)
        
        logger.info("=" * 60)
        logger.info("Zero Value Analysis Summary:")
        logger.info(f"  Zero value pixels: {zero_percentage:.2f}% of total")
        logger.info(f"  MAE impact: {((mae_without - mae_with) / mae_with * 100):+.1f}% when excluding zeros")
        
        logger.info("=" * 60)
        logger.info("Visualization completed!")
        logger.info(f"Results saved to: {args.save_dir}")
        logger.info(f"Successfully visualized {len(results)} samples")
        logger.info(f"Data source: {data_split} directory")
        
        # Output brief statistics
        avg_mae = np.mean([r['mae'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])
        logger.info(f"Average MAE: {avg_mae:.3f}m")
        logger.info(f"Average RMSE: {avg_rmse:.3f}m")
        
        # Performance evaluation
        if avg_mae < 2.0:
            logger.info("🎉 Excellent model performance (MAE < 2.0m)")
        elif avg_mae < 5.0:
            logger.info("✅ Good model performance (MAE < 5.0m)")
        else:
            logger.info("⚠️ Model performance needs improvement (MAE >= 5.0m)")
        
    except Exception as e:
        logger.error(f"Error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)