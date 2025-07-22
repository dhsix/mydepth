#!/usr/bin/env python3
"""
内存优化版GAMUS nDSM训练脚本
专注于减少内存占用，提高训练稳定性
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
import gc
import psutil
warnings.filterwarnings('ignore')

# 导入优化的模块
from improved_dataset_optimized import create_gamus_dataloader  # 使用优化版本
from improved_normalization_loss import create_height_loss
from Scale_Modulator import create_gamus_model

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

def get_memory_usage():
    """获取当前内存使用情况（GB）"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 / 1024  # 转换为GB

def create_datasets(args, logger):
    """创建训练和验证数据集 - 内存优化版"""
    logger.info("创建数据集（内存优化模式）...")
    
    # 构建数据路径
    train_image_dir = os.path.join(args.data_dir, 'images', 'train')
    train_label_dir = os.path.join(args.data_dir, 'height', 'train')
    val_image_dir = os.path.join(args.data_dir, 'images', 'val')
    val_label_dir = os.path.join(args.data_dir, 'height', 'val')
    
    logger.info(f"检查训练数据路径:")
    logger.info(f"  图像目录: {train_image_dir}")
    logger.info(f"  标签目录: {train_label_dir}")
    
    # 验证路径
    for path in [train_image_dir, train_label_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"目录不存在: {path}")
        else:
            files = os.listdir(path)
            logger.info(f"  {path}: {len(files)} 个文件")
    
    # 设置高度过滤器
    height_filter = {
        'min_height': args.min_height,
        'max_height': args.max_height
    }
    logger.info(f"高度过滤器: {height_filter}")
    
    # 记录创建前的内存使用
    memory_before = get_memory_usage()
    logger.info(f"创建数据集前内存使用: {memory_before:.1f} GB")
    
    # 创建训练数据集（内存优化）
    logger.info("正在创建训练数据集...")
    try:
        train_loader, train_dataset = create_gamus_dataloader(
            image_dir=train_image_dir,
            label_dir=train_label_dir,
            batch_size=args.batch_size,
            shuffle=True,
            normalization_method=args.normalization_method,
            enable_augmentation=args.enable_augmentation,
            stats_json_path=args.stats_json_path,
            height_filter=height_filter,
            force_recompute=args.force_recompute_stats,
            num_workers=args.num_workers,
            max_memory_samples=args.max_memory_samples  # 限制统计计算的样本数
        )
        height_normalizer = train_dataset.height_normalizer
        logger.info("✓ 训练数据集创建成功")
        
        # 记录创建后的内存使用
        memory_after = get_memory_usage()
        logger.info(f"创建数据集后内存使用: {memory_after:.1f} GB (+{memory_after - memory_before:.1f} GB)")
        
    except Exception as e:
        logger.error(f"创建训练数据集失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 创建验证数据集
    val_loader = None
    if os.path.exists(val_image_dir) and os.path.exists(val_label_dir):
        logger.info("正在创建验证数据集...")
        try:
            val_loader, _ = create_gamus_dataloader(
                image_dir=val_image_dir,
                label_dir=val_label_dir,
                batch_size=args.batch_size,
                shuffle=False,
                normalization_method=args.normalization_method,
                enable_augmentation=False,
                height_filter=height_filter,
                num_workers=max(1, args.num_workers // 2),  # 验证集使用更少worker
                max_memory_samples=args.max_memory_samples // 2  # 验证集使用更少样本
            )
            logger.info(f"✓ 验证数据集创建成功: {len(val_loader.dataset)} 个样本")
        except Exception as e:
            logger.error(f"创建验证数据集失败: {e}")
            val_loader = None
    else:
        logger.warning("未找到验证集，将使用训练集进行验证")
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"数据范围: [{height_filter['min_height']}, {height_filter['max_height']}] 米")
    
    # 测试数据加载
    logger.info("测试数据加载...")
    try:
        test_batch = next(iter(train_loader))
        logger.info(f"✓ 数据加载测试成功: 图像 {test_batch[0].shape}, 标签 {test_batch[1].shape}")
        
        # 清理测试数据
        del test_batch
        gc.collect()
        
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}")
        raise
    
    return train_loader, val_loader, train_dataset, height_normalizer

class SimpleGAMUSValidator:
    """内存优化的GAMUS验证器"""
    
    def __init__(self, height_normalizer, logger=None):
        self.height_normalizer = height_normalizer
        self.logger = logger or logging.getLogger(__name__)
        
    def denormalize_height(self, normalized_data):
        """使用归一化器将归一化的nDSM数据还原到真实高度值"""
        return self.height_normalizer.denormalize(normalized_data)
    
    def validate_with_metrics(self, model, val_loader, criterion, device, epoch=None, max_batches=50):
        """执行验证并返回详细指标 - 内存优化版"""
        model.eval()
        
        total_loss = 0.0
        total_count = 0
        
        # 大大减少收集的样本数量
        all_preds_real = []
        all_targets_real = []
        max_samples = 10000  # 进一步减少样本数量
        
        memory_before = get_memory_usage()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation', leave=False)
            
            for batch_idx, (images, labels) in enumerate(pbar):
                # 限制验证批次数量以节省时间和内存
                if batch_idx >= max_batches:
                    break
                    
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                try:
                    # 前向传播
                    predictions = model(images)
                    
                    # 检查预测值
                    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                        self.logger.warning(f"预测值包含无效值，跳过批次 {batch_idx}")
                        continue
                    
                    # 计算损失
                    loss = criterion(predictions, labels)
                    
                    # 检查损失值
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"损失值无效，跳过批次 {batch_idx}")
                        continue
                    
                    total_loss += loss.item()
                    total_count += 1
                    
                    # 收集少量样本用于指标计算
                    if len(all_preds_real) < max_samples and batch_idx % 5 == 0:  # 每5个批次采样一次
                        # 转换为numpy并反归一化
                        preds_cpu = predictions.detach().cpu().numpy().flatten()
                        targets_cpu = labels.detach().cpu().numpy().flatten()
                        
                        # 进一步采样以节省内存
                        n_samples = min(100, len(preds_cpu))  # 每个批次最多100个样本
                        if len(preds_cpu) > n_samples:
                            indices = np.random.choice(len(preds_cpu), n_samples, replace=False)
                            preds_cpu = preds_cpu[indices]
                            targets_cpu = targets_cpu[indices]
                        
                        # 反归一化到真实高度
                        try:
                            preds_real = self.denormalize_height(preds_cpu)
                            targets_real = self.denormalize_height(targets_cpu)
                            
                            all_preds_real.extend(preds_real)
                            all_targets_real.extend(targets_real)
                        except Exception as e:
                            self.logger.warning(f"反归一化失败: {e}")
                    
                    # 更新进度条
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                    
                    # 立即清理GPU内存
                    del predictions, loss
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    self.logger.warning(f"验证批次 {batch_idx} 错误: {e}")
                    continue
                finally:
                    # 清理CPU内存
                    del images, labels
                    if batch_idx % 10 == 0:
                        gc.collect()
        
        memory_after = get_memory_usage()
        self.logger.debug(f"验证后内存使用: {memory_after:.1f} GB")
        
        # 计算平均损失
        avg_loss = total_loss / total_count if total_count > 0 else float('inf')
        
        # 计算详细指标
        metrics = {'loss': avg_loss}
        
        if all_preds_real and all_targets_real:
            all_preds_real = np.array(all_preds_real)
            all_targets_real = np.array(all_targets_real)
            
            # 移除无效值
            valid_mask = (~np.isnan(all_preds_real) & ~np.isnan(all_targets_real) & 
                         ~np.isinf(all_preds_real) & ~np.isinf(all_targets_real))
            
            if np.sum(valid_mask) > 10:
                valid_preds = all_preds_real[valid_mask]
                valid_targets = all_targets_real[valid_mask]
                
                # 基础指标
                mae = mean_absolute_error(valid_targets, valid_preds)
                mse = mean_squared_error(valid_targets, valid_preds)
                rmse = np.sqrt(mse)
                
                # R²
                ss_res = np.sum((valid_targets - valid_preds) ** 2)
                ss_tot = np.sum((valid_targets - np.mean(valid_targets)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                
                # 精度指标
                errors = np.abs(valid_preds - valid_targets)
                accuracy_1m = np.mean(errors <= 1.0)
                accuracy_2m = np.mean(errors <= 2.0)
                accuracy_5m = np.mean(errors <= 5.0)
                
                metrics.update({
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'accuracy_1m': accuracy_1m,
                    'accuracy_2m': accuracy_2m,
                    'accuracy_5m': accuracy_5m,
                    'valid_samples': len(valid_preds),
                    'data_range': f'[{valid_targets.min():.1f}, {valid_targets.max():.1f}]m'
                })
        
        # 清理内存
        del all_preds_real, all_targets_real
        gc.collect()
        
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
            self.logger.info(f'  数据范围: {metrics["data_range"]}, 有效样本: {metrics["valid_samples"]}')
        
        if is_best:
            self.logger.info('  ★ 最佳验证性能 ★')

def validate_model_enhanced(model, val_loader, criterion, device, logger, height_normalizer, epoch=None):
    """增强的验证函数 - 内存优化版"""
    if val_loader is None:
        return {'loss': 0.0, 'count': 0}
    
    # 创建简化的验证器
    validator = SimpleGAMUSValidator(height_normalizer, logger)
    
    # 执行验证（限制批次数量）
    max_batches = min(50, len(val_loader))  # 最多验证50个批次
    metrics = validator.validate_with_metrics(model, val_loader, criterion, device, epoch, max_batches)
    
    # 记录指标
    validator.log_metrics(epoch, metrics)
    
    return metrics

def train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch, memory_check_interval=50):
    """训练一个epoch - 内存优化版"""
    model.train()
    total_loss = 0.0
    total_count = 0
    
    memory_start = get_memory_usage()
    logger.info(f"Epoch {epoch} 开始，内存使用: {memory_start:.1f} GB")
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        try:
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(images)
            
            # 检查预测值
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                logger.warning(f"预测值包含无效值，跳过批次 {batch_idx}")
                continue
            
            # 计算损失
            loss = criterion(predictions, labels)
            
            # 检查损失值
            if torch.isnan(loss) or torch.isinf(loss) or loss > 10:
                logger.warning(f"异常损失值 {loss.item():.6f}，跳过批次 {batch_idx}")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_count += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/total_count:.6f}'
            })
            
            # 立即清理内存
            del predictions, loss
            
            # 定期清理内存
            if batch_idx % memory_check_interval == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 检查内存使用
                current_memory = get_memory_usage()
                if current_memory > memory_start + 2.0:  # 如果内存增长超过2GB
                    logger.warning(f"内存使用增长较大: {current_memory:.1f} GB (+{current_memory - memory_start:.1f} GB)")
                    gc.collect()  # 强制垃圾回收
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU内存不足: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                logger.error(f"训练错误: {e}")
                continue
        except Exception as e:
            logger.error(f"未知错误: {e}")
            continue
        finally:
            # 确保每个批次后都清理
            del images, labels
    
    avg_loss = total_loss / total_count if total_count > 0 else float('inf')
    
    memory_end = get_memory_usage()
    logger.info(f"Epoch {epoch} 训练完成 - 平均损失: {avg_loss:.6f}, 内存使用: {memory_end:.1f} GB")
    
    return avg_loss

def save_checkpoint(epoch, model, optimizer, loss, save_dir, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存最新检查点
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        return best_path
    
    # 定期保存
    if epoch % 10 == 0:
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
    
    return latest_path

def main():
    parser = argparse.ArgumentParser(description='内存优化版GAMUS nDSM训练脚本')
    
    # 基本参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据根目录 (包含images和height子目录)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--stats_json_path', type=str, default=None,
                        help='预计算统计信息JSON文件路径')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=1,  # 默认减少worker数量
                        help='数据加载线程数')
    
    # 内存优化参数
    parser.add_argument('--max_memory_samples', type=int, default=500,  # 新增：限制统计计算样本数
                        help='用于统计计算的最大样本数（减少内存占用）')
    parser.add_argument('--memory_check_interval', type=int, default=50,
                        help='内存检查间隔（批次数）')
    
    # 数据参数
    parser.add_argument('--normalization_method', type=str, default='minmax',
                        choices=['minmax', 'percentile', 'zscore'],
                        help='归一化方法')
    parser.add_argument('--min_height', type=float, default=-5.0,
                        help='最小高度过滤值（米）')
    parser.add_argument('--max_height', type=float, default=100.0,
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
    parser.add_argument('--pretrained_path', type=str, default='/home/hudong26/hudong26/Height/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth',
                        help='预训练模型路径')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='冻结编码器')
    
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
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.save_dir, f'training_{timestamp}.log')
    logger = setup_logger(log_file)
    
    # 设置调试模式
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 打印系统信息
    logger.info("=" * 60)
    logger.info("内存优化版GAMUS nDSM训练")
    logger.info("=" * 60)
    logger.info(f"系统内存: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    logger.info(f"可用内存: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    logger.info(f"当前内存使用: {get_memory_usage():.1f} GB")
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
        train_loader, val_loader, train_dataset, height_normalizer = create_datasets(args, logger)
        
        # 创建模型
        logger.info("创建模型...")
        model = create_gamus_model(
            encoder=args.encoder,
            pretrained_path=args.pretrained_path,
            freeze_encoder=args.freeze_encoder,
            enable_scale_modulator=True
        ).to(device)
        
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
        criterion = create_height_loss(
            loss_type=args.loss_type,
            height_aware=args.height_aware
        )
        
        # 记录训练开始后的内存使用
        memory_after_init = get_memory_usage()
        logger.info(f"模型初始化后内存使用: {memory_after_init:.1f} GB")
        
        # 训练循环
        logger.info("开始训练...")
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, args.num_epochs + 1):
            # 训练
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, logger, epoch, args.memory_check_interval
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
                    epoch, model, optimizer, val_loss, args.save_dir, is_best
                )
                
                # 打印结果
                current_memory = get_memory_usage()
                logger.info(f"Epoch {epoch}/{args.num_epochs} 结果:")
                logger.info(f"  训练损失: {train_loss:.6f}")
                logger.info(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
                logger.info(f"  内存使用: {current_memory:.1f} GB")
                logger.info(f"  {'🎉 新的最佳模型!' if is_best else ''}")
                logger.info(f"  保存路径: {saved_path}")
                logger.info("-" * 60)
                
                # 早停检查
                if patience_counter >= args.early_stopping_patience:
                    logger.info(f"早停触发 (patience: {patience_counter})")
                    break
            
            # 每个epoch后清理内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        # 训练完成
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        logger.info(f"训练完成!")
        logger.info(f"总耗时: {total_time / 3600:.2f} 小时")
        logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"最终内存使用: {final_memory:.1f} GB")
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
        gc.collect()
        logger.info("训练脚本已退出")

if __name__ == '__main__':
    main()