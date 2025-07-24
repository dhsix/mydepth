#!/usr/bin/env python3
"""
支持多模型对比的GAMUS nDSM训练脚本
可以训练GAMUS模型和Depth2Elevation模型
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
import time
import argparse
import warnings
from datetime import datetime
from tqdm import tqdm
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# 导入修改后的数据集模块
from improved_dataset_with_mask import create_gamus_dataloader
from improved_normalization_loss import create_height_loss
from model_with_comparison import create_gamus_model  # 使用支持多模型的版本

def setup_logger(log_path):
    """设置日志记录"""
    logger = logging.getLogger('gamus_training')
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_datasets(args, logger):
    """创建训练和验证数据集"""
    logger.info("创建数据集...")
    
    # 构建数据路径
    train_image_dir = os.path.join(args.data_dir, 'train','images' )
    train_label_dir = os.path.join(args.data_dir, 'train','depths' )
    val_image_dir = os.path.join(args.data_dir, 'val','images' )
    val_label_dir = os.path.join(args.data_dir, 'val','depths' )
    
    # mask路径
    train_mask_dir = None
    val_mask_dir = None
    if args.mask_dir:
        train_mask_dir = os.path.join(args.mask_dir, 'train', 'classes')
        val_mask_dir = os.path.join(args.mask_dir, 'val', 'classes')
        
        logger.info(f"检查mask数据路径:")
        logger.info(f"  训练mask目录: {train_mask_dir}")
        logger.info(f"  验证mask目录: {val_mask_dir}")
    
    logger.info(f"检查训练数据路径:")
    logger.info(f"  图像目录: {train_image_dir}")
    logger.info(f"  标签目录: {train_label_dir}")
    
    # 验证路径
    required_paths = [train_image_dir, train_label_dir]
    if train_mask_dir:
        required_paths.append(train_mask_dir)
        
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"目录不存在: {path}")
        else:
            files = os.listdir(path)
            logger.info(f"  {path}: {len(files)} 个文件")
    
    # 检查统计信息文件路径
    if not args.stats_json_path:
        args.stats_json_path = os.path.join(args.save_dir, 'gamus_stats.json')
        logger.warning(f"⚠️ 未指定统计信息文件，使用默认路径: {args.stats_json_path}")
    
    if not os.path.exists(args.stats_json_path):
        logger.error(f"❌ 统计信息文件不存在: {args.stats_json_path}")
        logger.info("💡 请先运行: python precompute_stats.py <data_dir> --output <stats_file>")
        raise FileNotFoundError(f"请先运行预计算脚本生成统计信息文件")    
    
    # 设置高度过滤器
    height_filter = {
        'min_height': args.min_height,
        'max_height': args.max_height
    }
    logger.info(f"高度过滤器: {height_filter}")
    
    # mask配置
    if args.mask_dir:
        logger.info(f"🎯 Mask配置:")
        logger.info(f"   建筑类别ID: {args.building_class_id}")
        logger.info(f"   树木类别ID: {args.tree_class_id}")
    
    # 创建训练数据集
    logger.info("正在创建训练数据集...")
    try:
        train_loader, train_dataset = create_gamus_dataloader(
            image_dir=train_image_dir,
            label_dir=train_label_dir,
            mask_dir=train_mask_dir,
            building_class_id=args.building_class_id,
            tree_class_id=args.tree_class_id,
            batch_size=args.batch_size,
            shuffle=True,
            normalization_method=args.normalization_method,
            enable_augmentation=args.enable_augmentation,
            stats_json_path=args.stats_json_path,
            height_filter=height_filter,
            force_recompute=False,
            num_workers=min(args.num_workers, 2)
        )
        
        if hasattr(train_dataset, 'global_min'):
            logger.info(f"✅ 使用预计算统计信息，数据范围: [{train_dataset.global_min:.2f}, {train_dataset.global_max:.2f}] 米")
        
        # 获取高度归一化器
        height_normalizer = train_dataset.get_normalizer()
        if hasattr(height_normalizer, 'global_min_h') and hasattr(height_normalizer, 'global_max_h'):
            min_height = height_normalizer.global_min_h
            max_height = height_normalizer.global_max_h
            logger.info(f"✅ 从归一化器获取真实高度范围: [{min_height:.2f}, {max_height:.2f}] 米")
        elif hasattr(height_normalizer, 'min_val') and hasattr(height_normalizer, 'max_val'):
            min_height = height_normalizer.min_val
            max_height = height_normalizer.max_val
            logger.info(f"✅ 从归一化器获取真实高度范围: [{min_height:.2f}, {max_height:.2f}] 米")
        else:
            min_height = args.min_height
            max_height = args.max_height
            logger.warning(f"⚠️ 无法从归一化器获取高度范围，使用配置参数: [{min_height:.2f}, {max_height:.2f}] 米")
        
        height_range = max_height - min_height
        logger.info(f"📊 高度范围跨度: {height_range:.2f} 米")
        
        logger.info("✓ 训练数据集创建成功")
    except Exception as e:
        logger.error(f"创建训练数据集失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 创建验证数据集
    val_loader = None
    if os.path.exists(val_image_dir) and os.path.exists(val_label_dir):
        if args.mask_dir and not os.path.exists(val_mask_dir):
            logger.warning(f"验证集mask目录不存在: {val_mask_dir}，将不使用mask")
            val_mask_dir = None
            
        logger.info("正在创建验证数据集...")
        try:
            val_loader, _ = create_gamus_dataloader(
                image_dir=val_image_dir,
                label_dir=val_label_dir,
                mask_dir=val_mask_dir,
                building_class_id=args.building_class_id,
                tree_class_id=args.tree_class_id,
                batch_size=args.batch_size,
                shuffle=False,
                normalization_method=args.normalization_method,
                enable_augmentation=False,
                stats_json_path=args.stats_json_path,
                height_filter=height_filter,
                force_recompute=False,
                num_workers=min(args.num_workers, 2)
            )
            logger.info(f"✓ 验证数据集创建成功: {len(val_loader.dataset)} 个样本")
        except Exception as e:
            logger.error(f"创建验证数据集失败: {e}")
            val_loader = None
    else:
        logger.warning("未找到验证集，将使用训练集进行验证")
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"数据范围: [{height_filter['min_height']}, {height_filter['max_height']}] 米")
    if args.mask_dir:
        logger.info(f"使用mask: 仅训练building和tree区域")
    
    # 测试数据加载
    logger.info("测试数据加载...")
    try:
        test_batch = next(iter(train_loader))
        if len(test_batch) == 3:
            images, labels, masks = test_batch
            logger.info(f"✓ 数据加载测试成功: 图像 {images.shape}, 标签 {labels.shape}, mask {masks.shape}")
            logger.info(f"  mask统计: min={masks.min():.3f}, max={masks.max():.3f}, mean={masks.mean():.3f}")
        else:
            images, labels = test_batch
            logger.info(f"✓ 数据加载测试成功: 图像 {images.shape}, 标签 {labels.shape}")
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}")
        raise
    
    return train_loader, val_loader, train_dataset, height_normalizer, min_height, max_height

class SimpleGAMUSValidator:
    """简化的GAMUS验证器（支持mask）"""
    
    def __init__(self, height_normalizer, logger=None):
        self.height_normalizer = height_normalizer
        self.logger = logger or logging.getLogger(__name__)
        
    def denormalize_height(self, normalized_data):
        """使用归一化器将归一化的nDSM数据还原到真实高度值"""
        return self.height_normalizer.denormalize(normalized_data)
    
    def validate_with_metrics(self, model, val_loader, criterion, device, epoch=None):
        """执行验证并返回详细指标（支持mask）"""
        model.eval()
        
        total_loss = 0.0
        total_count = 0
        
        all_preds_real = []
        all_targets_real = []
        max_samples = 100000
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation', leave=False)
            
            for batch_idx, batch_data in enumerate(pbar):
                if len(batch_data) == 3:
                    images, labels, masks = batch_data
                    images = images.to(device)
                    labels = labels.to(device)
                    masks = masks.to(device)
                else:
                    images, labels = batch_data
                    images = images.to(device)
                    labels = labels.to(device)
                    masks = torch.ones_like(labels).to(device)
                
                try:
                    predictions = model(images)
                    
                    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                        self.logger.warning(f"预测值包含无效值，跳过批次 {batch_idx}")
                        continue
                    
                    # 计算带mask的损失
                    if isinstance(criterion, MaskedLoss):
                        loss = criterion(predictions, labels, masks)
                    else:
                        valid_mask = (masks > 0.5) & (labels >= 0)
                        if valid_mask.sum() > 0:
                            loss = criterion(predictions[valid_mask], labels[valid_mask])
                        else:
                            continue
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"损失值无效，跳过批次 {batch_idx}")
                        continue
                    
                    total_loss += loss.item()
                    total_count += 1
                    
                    # 收集样本用于指标计算
                    if len(all_preds_real) < max_samples:
                        valid_mask = (masks > 0.5) & (labels >= 0)
                        if valid_mask.sum() > 0:
                            preds_cpu = predictions[valid_mask].detach().cpu().numpy().flatten()
                            targets_cpu = labels[valid_mask].detach().cpu().numpy().flatten()
                            
                            n_samples = min(500, len(preds_cpu))
                            if len(preds_cpu) > n_samples:
                                indices = np.random.choice(len(preds_cpu), n_samples, replace=False)
                                preds_cpu = preds_cpu[indices]
                                targets_cpu = targets_cpu[indices]
                            
                            preds_real = self.denormalize_height(preds_cpu)
                            targets_real = self.denormalize_height(targets_cpu)
                            
                            all_preds_real.extend(preds_real)
                            all_targets_real.extend(targets_real)
                    
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                    
                except Exception as e:
                    self.logger.warning(f"验证批次 {batch_idx} 错误: {e}")
                    continue
        
        avg_loss = total_loss / total_count if total_count > 0 else float('inf')
        metrics = {'loss': avg_loss}
        
        if all_preds_real and all_targets_real:
            all_preds_real = np.array(all_preds_real)
            all_targets_real = np.array(all_targets_real)
            
            valid_mask = (~np.isnan(all_preds_real) & ~np.isnan(all_targets_real) & 
                         ~np.isinf(all_preds_real) & ~np.isinf(all_targets_real))
            
            if np.sum(valid_mask) > 10:
                valid_preds = all_preds_real[valid_mask]
                valid_targets = all_targets_real[valid_mask]
                
                mae = mean_absolute_error(valid_targets, valid_preds)
                mse = mean_squared_error(valid_targets, valid_preds)
                rmse = np.sqrt(mse)
                
                ss_res = np.sum((valid_targets - valid_preds) ** 2)
                ss_tot = np.sum((valid_targets - np.mean(valid_targets)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                
                errors = np.abs(valid_preds - valid_targets)
                accuracy_1m = np.mean(errors <= 1.0)
                accuracy_2m = np.mean(errors <= 2.0)
                accuracy_5m = np.mean(errors <= 5.0)
                
                ground_mask = (valid_targets >= -5) & (valid_targets <= 5)
                low_mask = (valid_targets > 5) & (valid_targets <= 20)
                mid_mask = (valid_targets > 20) & (valid_targets <= 50)
                high_mask = valid_targets > 50
                
                mae_ground = np.mean(errors[ground_mask]) if np.sum(ground_mask) > 0 else 0.0
                mae_low = np.mean(errors[low_mask]) if np.sum(low_mask) > 0 else 0.0
                mae_mid = np.mean(errors[mid_mask]) if np.sum(mid_mask) > 0 else 0.0
                mae_high = np.mean(errors[high_mask]) if np.sum(high_mask) > 0 else 0.0
                
                metrics.update({
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'accuracy_1m': accuracy_1m,
                    'accuracy_2m': accuracy_2m,
                    'accuracy_5m': accuracy_5m,
                    'mae_ground': mae_ground,
                    'mae_low': mae_low,
                    'mae_mid': mae_mid,
                    'mae_high': mae_high,
                    'valid_samples': len(valid_preds),
                    'data_range': f'[{valid_targets.min():.1f}, {valid_targets.max():.1f}]m'
                })
        
        return metrics
    
    def log_metrics(self, epoch, metrics, is_best=False):
        """记录指标"""
        self.logger.info(f'验证指标 - Epoch {epoch}:')
        self.logger.info(f'  损失: {metrics.get("loss", "N/A"):.6f}')
        if 'mae' in metrics:
            self.logger.info(f'  MAE: {metrics["mae"]:.4f} m')
            self.logger.info(f'  RMSE: {metrics["rmse"]:.4f} m')
            self.logger.info(f'  R²: {metrics["r2"]:.4f}')
            self.logger.info(f'  精度: ±1m={metrics["accuracy_1m"]:.1%}, ±2m={metrics["accuracy_2m"]:.1%}, ±5m={metrics["accuracy_5m"]:.1%}')
            self.logger.info(f'  分层MAE: 地面={metrics["mae_ground"]:.2f}m, 低建筑={metrics["mae_low"]:.2f}m, 中建筑={metrics["mae_mid"]:.2f}m, 高建筑={metrics["mae_high"]:.2f}m')
            self.logger.info(f'  数据范围: {metrics["data_range"]}, 有效样本: {metrics["valid_samples"]}')
        
        if is_best:
            self.logger.info('  ★ 最佳验证性能 ★')

class MaskedLoss(nn.Module):
    """带mask的损失函数包装器"""
    
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion
    
    def forward(self, predictions, targets, masks):
        valid_mask = (masks > 0.5) & (targets >= 0)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        valid_preds = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        return self.base_criterion(valid_preds, valid_targets)

def validate_model_enhanced(model, val_loader, criterion, device, logger, height_normalizer, epoch=None):
    """增强的验证函数"""
    if val_loader is None:
        return {'loss': 0.0, 'count': 0}
    
    validator = SimpleGAMUSValidator(height_normalizer, logger)
    metrics = validator.validate_with_metrics(model, val_loader, criterion, device, epoch)
    validator.log_metrics(epoch, metrics)
    
    return metrics

def train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch):
    """训练一个epoch（支持mask）"""
    model.train()
    total_loss = 0.0
    total_count = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        if len(batch_data) == 3:
            images, labels, masks = batch_data
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
        else:
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)
            masks = torch.ones_like(labels).to(device)
        
        try:
            optimizer.zero_grad()
            
            predictions = model(images)
            
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                logger.warning(f"预测值包含无效值，跳过批次 {batch_idx}")
                continue
            
            # 计算带mask的损失
            if isinstance(criterion, MaskedLoss):
                loss = criterion(predictions, labels, masks)
            else:
                valid_mask = (masks > 0.5) & (labels >= 0)
                if valid_mask.sum() == 0:
                    continue
                loss = criterion(predictions[valid_mask], labels[valid_mask])
            
            if torch.isnan(loss) or torch.isinf(loss) or loss > 10:
                logger.warning(f"异常损失值 {loss.item():.6f}，跳过批次 {batch_idx}")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_count += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/total_count:.6f}'
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU内存不足: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"训练错误: {e}")
                continue
        except Exception as e:
            logger.error(f"未知错误: {e}")
            continue
    
    avg_loss = total_loss / total_count if total_count > 0 else float('inf')
    logger.info(f"Epoch {epoch} 训练完成 - 平均损失: {avg_loss:.6f}")
    
    return avg_loss

def save_checkpoint(epoch, model, optimizer, loss, save_dir, is_best=False, model_type='gamus'):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_type': model_type,  # 新增：记录模型类型
        'timestamp': datetime.now().isoformat()
    }
    
    # 根据模型类型调整文件名
    prefix = f'{model_type}_'
    
    latest_path = os.path.join(save_dir, f'{prefix}latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = os.path.join(save_dir, f'{prefix}best_model.pth')
        torch.save(checkpoint, best_path)
        return best_path
    
    if epoch % 10 == 0:
        epoch_path = os.path.join(save_dir, f'{prefix}checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
    
    return latest_path

def main():
    parser = argparse.ArgumentParser(description='支持多模型对比的GAMUS nDSM训练脚本')
    
    # 基本参数
    parser.add_argument('--data_dir', type=str, default='/home/hudong26/HeightData/GAMUS/',
                        help='数据根目录 (包含train/val子目录)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--stats_json_path', type=str, default='./gamus_full_stats.json',
                        help='预计算统计信息JSON文件路径')
    
    # 新增：模型选择参数
    parser.add_argument('--model_type', type=str, default='gamus',
                        choices=['gamus', 'depth2elevation'],
                        help='模型类型选择')
    
    # mask相关参数
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='classes mask根目录 (包含train/val/classes子目录)')
    parser.add_argument('--building_class_id', type=int, default=6,
                        help='建筑类别ID')
    parser.add_argument('--tree_class_id', type=int, default=5,
                        help='树木类别ID')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='数据加载线程数')
    
    # 数据参数
    parser.add_argument('--normalization_method', type=str, default='minmax',
                        choices=['minmax', 'percentile', 'zscore'],
                        help='归一化方法')
    parser.add_argument('--min_height', type=float, default=-5.0,
                        help='最小高度过滤值（米）')
    parser.add_argument('--max_height', type=float, default=200.0,
                        help='最大高度过滤值（米）')
    parser.add_argument('--enable_augmentation', action='store_true',
                        help='启用数据增强')
    parser.add_argument('--force_recompute_stats', action='store_true',
                        help='强制重新计算统计信息')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    
    # 模型参数
    parser.add_argument('--encoder', type=str, default='vitb',
                        choices=['vits', 'vitb', 'vitl'],
                        help='编码器类型')
    parser.add_argument('--pretrained_path', type=str, 
                        default='/home/hudong26/hudong26/Height/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth',
                        help='预训练模型路径')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='冻结编码器')
    
    # 新增：Depth2Elevation特定参数
    parser.add_argument('--use_multi_scale_output', action='store_true',
                        help='使用多尺度输出（仅对Depth2Elevation有效）')
    parser.add_argument('--img_size', type=int, default=448,
                        help='输入图像尺寸')
    parser.add_argument('--patch_size', type=int, default=14,
                        help='Patch尺寸（仅对Depth2Elevation有效）')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'mae', 'huber', 'focal', 'combined'],
                        help='损失函数类型')
    parser.add_argument('--height_aware', action='store_true',
                        help='启用高度感知损失')
    
    # 训练控制
    parser.add_argument('--patience', type=int, default=10,
                        help='学习率调度器耐心值')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='早停耐心值')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='验证间隔')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                        help='训练设备')
    
    args = parser.parse_args()
    
    # 验证关键参数
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        return 1
    
    if args.mask_dir and not os.path.exists(args.mask_dir):
        print(f"❌ Mask目录不存在: {args.mask_dir}")
        return 1
    
    if args.stats_json_path and not os.path.exists(args.stats_json_path):
        print(f"❌ 统计信息文件不存在: {args.stats_json_path}")
        print(f"💡 请先运行: python precompute_stats.py {args.data_dir} --output {args.stats_json_path}")
        return 1    
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.save_dir, f'training_{args.model_type}_{timestamp}.log')
    logger = setup_logger(log_file)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 打印配置
    logger.info("=" * 60)
    logger.info(f"多模型对比训练 - 当前模型: {args.model_type.upper()}")
    logger.info("=" * 60)
    logger.info(f"配置参数: {json.dumps(vars(args), indent=2, ensure_ascii=False)}")
    
    # 设备设置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU信息: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # 创建数据集
        train_loader, val_loader, train_dataset, height_normalizer, min_height, max_height = create_datasets(args, logger)
        
        # 创建模型 - 支持多种模型类型
        logger.info(f"创建{args.model_type}模型...")
        
        # 准备模型配置
        model_kwargs = {
            'encoder': args.encoder,
            'pretrained_path': args.pretrained_path,
            'freeze_encoder': args.freeze_encoder,
            'model_type': args.model_type
        }
        
        # 为Depth2Elevation添加特定参数
        if args.model_type == 'depth2elevation':
            model_kwargs.update({
                'img_size': args.img_size,
                'patch_size': args.patch_size,
                'use_multi_scale_output': args.use_multi_scale_output,
                'loss_config': {},  # 可以根据需要配置
                'freezing_config': {}  # 可以根据需要配置
            })
        
        model = create_gamus_model(**model_kwargs).to(device)
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数: 总数={total_params:,}, 可训练={trainable_params:,}")
        
        # 创建优化器
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=1e-4
        )
        
        # 创建学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=args.patience,
            min_lr=1e-6,
            verbose=True
        )
        
        # 创建损失函数
        base_criterion = create_height_loss(
            loss_type=args.loss_type,
            height_aware=args.height_aware,
            height_normalizer=height_normalizer,
            min_height=min_height,
            max_height=max_height
        )
        
        # 包装为带mask的损失函数
        criterion = MaskedLoss(base_criterion)
        
        logger.info(f"📊 损失函数配置:")
        logger.info(f"   类型: {args.loss_type} (带mask)")
        logger.info(f"   高度感知: {args.height_aware}")
        logger.info(f"   高度范围: [{min_height:.2f}, {max_height:.2f}] 米")
        if args.mask_dir:
            logger.info(f"   仅训练: building (ID={args.building_class_id}) 和 tree (ID={args.tree_class_id}) 区域")
        
        # 训练循环
        logger.info("开始训练...")
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, args.num_epochs + 1):
            # 训练
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, logger, epoch
            )
            
            # 验证
            if epoch % args.val_interval == 0:
                val_metrics = validate_model_enhanced(
                    model, val_loader, criterion, device, logger, height_normalizer, epoch
                )
                val_loss = val_metrics['loss']
                
                # 学习率调度
                scheduler.step(val_loss)
                
                # 检查是否为最佳模型
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 保存检查点
                saved_path = save_checkpoint(
                    epoch, model, optimizer, val_loss, args.save_dir, is_best, args.model_type
                )
                
                # 打印结果
                logger.info(f"Epoch {epoch}/{args.num_epochs} 结果:")
                logger.info(f"  训练损失: {train_loss:.6f}")
                logger.info(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
                logger.info(f"  {'🎉 新的最佳模型!' if is_best else ''}")
                logger.info(f"  保存路径: {saved_path}")
                logger.info("-" * 60)
                
                # 早停检查
                if patience_counter >= args.early_stopping_patience:
                    logger.info(f"早停触发 (patience: {patience_counter})")
                    break
            
            # 清理GPU内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"训练完成!")
        logger.info(f"模型类型: {args.model_type}")
        logger.info(f"总耗时: {total_time / 3600:.2f} 小时")
        logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"模型保存在: {args.save_dir}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        logger.info("训练脚本已退出")

if __name__ == '__main__':
    main()