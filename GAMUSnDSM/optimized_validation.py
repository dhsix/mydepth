import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import time
import logging
from collections import defaultdict
import seaborn as sns
import gc
import psutil

def save_model_checkpoint(epoch, model, optimizer, val_loss, save_dir, is_best=False, metrics=None,
                          checkpoint_interval=10):
    """优化的模型检查点保存函数 - GAMUS版本"""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'timestamp': time.time(),
        'dataset_type': 'GAMUS_nDSM'
    }

    if metrics:
        checkpoint['metrics'] = metrics
        checkpoint['best_metrics'] = {
            'mae': metrics.get('mae', float('inf')),
            'rmse': metrics.get('rmse', float('inf')),
            'r2': metrics.get('r2', -float('inf'))
        }

    # 保存最新模型
    latest_path = os.path.join(save_dir, 'latest_gamus_model.pth')
    torch.save(checkpoint, latest_path)
    logging.info(f"保存最新GAMUS模型: {latest_path}")

    # 保存最佳模型
    if is_best:
        best_path = os.path.join(save_dir, 'best_gamus_model.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"★ 保存最佳GAMUS模型 ★: {best_path} (Val Loss: {val_loss:.4f})")

    # 定期保存检查点
    if epoch % checkpoint_interval == 0:
        epoch_path = os.path.join(save_dir, f'gamus_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        logging.info(f"保存GAMUS周期性检查点: {epoch_path}")
        
    # 清理旧的周期性检查点
    _cleanup_old_checkpoints(save_dir, keep_latest=5)

    return latest_path

def _cleanup_old_checkpoints(save_dir, keep_latest=5):
    """清理旧的检查点文件"""
    try:
        checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('gamus_epoch_') and f.endswith('.pth')]
        if len(checkpoint_files) > keep_latest:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
            for old_file in checkpoint_files[:-keep_latest]:
                os.remove(os.path.join(save_dir, old_file))
                logging.info(f"清理旧检查点: {old_file}")
    except Exception as e:
        logging.warning(f"清理检查点时出错: {e}")

class GAMUSValidationMetricsCalculator:
    """GAMUS nDSM数据集的内存优化验证指标计算器"""

    def __init__(self, save_dir, logger, height_normalizer, visualization_enabled=True, 
                 max_samples_for_metrics=10000, batch_accumulation_limit=100):
        """
        初始化GAMUS验证计算器
        
        Args:
            save_dir: 保存目录
            logger: 日志记录器
            height_normalizer: nDSM高度归一化器
            visualization_enabled: 是否启用可视化
            max_samples_for_metrics: 用于计算指标的最大样本数
            batch_accumulation_limit: 批次累积限制
        """
        self.save_dir = save_dir
        self.logger = logger
        self.height_normalizer = height_normalizer
        self.visualization_enabled = visualization_enabled
        self.vis_dir = os.path.join(save_dir, 'gamus_visualizations')
        # 确保height_normalizer有denormalize方法
        if not hasattr(height_normalizer, 'denormalize'):
            self.logger.error("❌ height_normalizer缺少denormalize方法")
            raise AttributeError("height_normalizer必须有denormalize方法")
         
        
        # 内存优化参数
        self.max_samples_for_metrics = max_samples_for_metrics
        self.batch_accumulation_limit = batch_accumulation_limit

        # 创建可视化目录
        if visualization_enabled:
            os.makedirs(self.vis_dir, exist_ok=True)
            
        # 历史指标记录（限制长度）
        self.metrics_history = defaultdict(lambda: [])
        self.max_history_length = 100
        
        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.logger.info(f"GAMUS内存优化验证计算器初始化完成")
        self.logger.info(f"最大指标样本数: {max_samples_for_metrics}")
        self.logger.info(f"批次累积限制: {batch_accumulation_limit}")

    def denormalize_height(self, normalized_data):
        """使用归一化器将归一化的nDSM数据还原到真实高度值"""
        return self.height_normalizer.denormalize(normalized_data)

    def _log_memory_usage(self, stage=""):
        """记录内存使用情况"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"内存使用 {stage}: {memory_mb:.1f} MB")
        except:
            pass

    def _clear_cache_and_gc(self):
        """清理缓存和强制垃圾回收"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _reservoir_sampling(self, all_preds, all_targets, max_samples):
        """使用水库采样算法减少样本数量"""
        total_samples = len(all_preds)
        if total_samples <= max_samples:
            return all_preds, all_targets
        
        indices = np.random.choice(total_samples, max_samples, replace=False)
        return all_preds[indices], all_targets[indices]

    def safe_ndsm_metric_calculation(self, all_preds_real, all_targets_real):
        """安全的nDSM指标计算，优化内存使用"""
        metrics = {}
        
        try:
            # 基础检查
            if len(all_preds_real) == 0 or len(all_targets_real) == 0:
                raise ValueError("预测值或目标值为空")
            
            # 使用采样减少计算量和内存使用
            if len(all_preds_real) > self.max_samples_for_metrics:
                self.logger.info(f"采样 {self.max_samples_for_metrics} 个样本用于指标计算（总样本数: {len(all_preds_real)}）")
                all_preds_real, all_targets_real = self._reservoir_sampling(
                    all_preds_real, all_targets_real, self.max_samples_for_metrics
                )
            
            # 移除无效值
            valid_mask = (~np.isnan(all_preds_real) & ~np.isnan(all_targets_real) & 
                         ~np.isinf(all_preds_real) & ~np.isinf(all_targets_real))
            
            if np.sum(valid_mask) < 10:
                raise ValueError(f"有效样本太少: {np.sum(valid_mask)}")
            
            valid_preds = all_preds_real[valid_mask]
            valid_targets = all_targets_real[valid_mask]
            
            # 基础指标 - 使用分块计算减少内存占用
            chunk_size = 5000
            if len(valid_preds) > chunk_size:
                # 分块计算MSE和MAE
                mse_sum = 0.0
                mae_sum = 0.0
                for i in range(0, len(valid_preds), chunk_size):
                    end_idx = min(i + chunk_size, len(valid_preds))
                    chunk_pred = valid_preds[i:end_idx]
                    chunk_target = valid_targets[i:end_idx]
                    
                    mse_sum += np.sum((chunk_pred - chunk_target) ** 2)
                    mae_sum += np.sum(np.abs(chunk_pred - chunk_target))
                
                mse = mse_sum / len(valid_preds)
                mae = mae_sum / len(valid_preds)
            else:
                mse = mean_squared_error(valid_targets, valid_preds)
                mae = mean_absolute_error(valid_targets, valid_preds)
            
            rmse = np.sqrt(mse)
            
            # R²计算
            ss_res = np.sum((valid_targets - valid_preds) ** 2)
            ss_tot = np.sum((valid_targets - np.mean(valid_targets)) ** 2)
            
            if ss_tot < 1e-10:
                r2 = 0.0
            else:
                r2 = 1 - (ss_res / ss_tot)
                r2 = max(-10, min(1, r2))
            
            metrics.update({
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            })
            
            # 相关性指标 - 如果样本太多，进一步采样
            correlation_sample_size = min(5000, len(valid_preds))
            if len(valid_preds) > correlation_sample_size:
                indices = np.random.choice(len(valid_preds), correlation_sample_size, replace=False)
                corr_preds = valid_preds[indices]
                corr_targets = valid_targets[indices]
            else:
                corr_preds = valid_preds
                corr_targets = valid_targets
            
            try:
                if len(np.unique(corr_targets)) > 1 and len(np.unique(corr_preds)) > 1:
                    pearson_r, pearson_p = pearsonr(corr_targets, corr_preds)
                    spearman_r, spearman_p = spearmanr(corr_targets, corr_preds)
                else:
                    pearson_r = pearson_p = spearman_r = spearman_p = 0.0
                
                # 处理NaN值
                if np.isnan(pearson_r):
                    pearson_r = 0.0
                if np.isnan(pearson_p):
                    pearson_p = 1.0
                if np.isnan(spearman_r):
                    spearman_r = 0.0
                if np.isnan(spearman_p):
                    spearman_p = 1.0
                    
                metrics.update({
                    'pearson_r': float(pearson_r),
                    'pearson_p': float(pearson_p),
                    'spearman_r': float(spearman_r),
                    'spearman_p': float(spearman_p)
                })
            except Exception as e:
                self.logger.warning(f"相关性计算失败: {e}")
                metrics.update({
                    'pearson_r': 0.0,
                    'pearson_p': 1.0,
                    'spearman_r': 0.0,
                    'spearman_p': 1.0
                })
            
            # nDSM特定的精度指标 - 分块计算
            try:
                accuracy_1m = 0.0
                accuracy_2m = 0.0
                accuracy_5m = 0.0
                accuracy_10m = 0.0  # 对于nDSM，10米的精度也很重要
                
                for i in range(0, len(valid_preds), chunk_size):
                    end_idx = min(i + chunk_size, len(valid_preds))
                    chunk_pred = valid_preds[i:end_idx]
                    chunk_target = valid_targets[i:end_idx]
                    
                    chunk_errors = np.abs(chunk_pred - chunk_target)
                    accuracy_1m += np.sum(chunk_errors <= 1.0)
                    accuracy_2m += np.sum(chunk_errors <= 2.0)
                    accuracy_5m += np.sum(chunk_errors <= 5.0)
                    accuracy_10m += np.sum(chunk_errors <= 10.0)
                
                accuracy_1m /= len(valid_preds)
                accuracy_2m /= len(valid_preds)
                accuracy_5m /= len(valid_preds)
                accuracy_10m /= len(valid_preds)
                
                metrics.update({
                    'height_accuracy_1m': float(accuracy_1m),
                    'height_accuracy_2m': float(accuracy_2m),
                    'height_accuracy_5m': float(accuracy_5m),
                    'height_accuracy_10m': float(accuracy_10m)
                })
            except:
                metrics.update({
                    'height_accuracy_1m': 0.0,
                    'height_accuracy_2m': 0.0,
                    'height_accuracy_5m': 0.0,
                    'height_accuracy_10m': 0.0
                })
            
            # 相对误差 - 分块计算
            try:
                relative_error_sum = 0.0
                for i in range(0, len(valid_preds), chunk_size):
                    end_idx = min(i + chunk_size, len(valid_preds))
                    chunk_pred = valid_preds[i:end_idx]
                    chunk_target = valid_targets[i:end_idx]
                    
                    # 对于nDSM，分母使用绝对值加上小常数
                    denominator = np.maximum(np.abs(chunk_target), 0.1)
                    relative_errors = np.abs(chunk_pred - chunk_target) / denominator
                    relative_errors = np.minimum(relative_errors, 10.0)  # 限制最大相对误差
                    relative_error_sum += np.sum(relative_errors)
                
                metrics['relative_error'] = float(relative_error_sum / len(valid_preds) * 100)
            except:
                metrics['relative_error'] = 100.0
            
            # nDSM高度分层误差分析
            try:
                # 根据nDSM数据特点调整分层
                # 地面/低值区域 (-5 to +5m)
                ground_mask = (valid_targets >= -5) & (valid_targets <= 5)
                if np.sum(ground_mask) > 0:
                    ground_errors = np.abs(valid_preds[ground_mask] - valid_targets[ground_mask])
                    metrics['mae_ground_level'] = float(np.mean(ground_errors))
                else:
                    metrics['mae_ground_level'] = 0.0
                
                # 低建筑物 (5-20m)
                low_mask = (valid_targets > 5) & (valid_targets <= 20)
                if np.sum(low_mask) > 0:
                    low_errors = np.abs(valid_preds[low_mask] - valid_targets[low_mask])
                    metrics['mae_low_buildings'] = float(np.mean(low_errors))
                else:
                    metrics['mae_low_buildings'] = 0.0
                
                # 中等建筑物 (20-50m)
                mid_mask = (valid_targets > 20) & (valid_targets <= 50)
                if np.sum(mid_mask) > 0:
                    mid_errors = np.abs(valid_preds[mid_mask] - valid_targets[mid_mask])
                    metrics['mae_mid_buildings'] = float(np.mean(mid_errors))
                else:
                    metrics['mae_mid_buildings'] = 0.0
                
                # 高建筑物 (>50m)
                high_mask = valid_targets > 50
                if np.sum(high_mask) > 0:
                    high_errors = np.abs(valid_preds[high_mask] - valid_targets[high_mask])
                    metrics['mae_high_buildings'] = float(np.mean(high_errors))
                else:
                    metrics['mae_high_buildings'] = 0.0
                    
            except:
                metrics.update({
                    'mae_ground_level': 0.0,
                    'mae_low_buildings': 0.0,
                    'mae_mid_buildings': 0.0,
                    'mae_high_buildings': 0.0
                })
                
            # 清理临时变量
            del valid_preds, valid_targets
            if 'corr_preds' in locals():
                del corr_preds, corr_targets
            self._clear_cache_and_gc()
                
        except Exception as e:
            self.logger.error(f"nDSM指标计算错误: {e}")
            metrics = self._get_default_ndsm_metrics()
        
        return metrics

    def calculate_metrics(self, model, val_loader, device, epoch, criterion=None):
        """执行验证并计算所有指标 - GAMUS nDSM版本"""
        model.eval()
        start_time = time.time()

        # 初始化指标累积器
        metrics = {
            'loss': 0.0,
            'samples': 0,
            'valid_batches': 0
        }

        # 使用在线统计避免存储所有数据
        running_metrics = {
            'sum_preds': 0.0,
            'sum_targets': 0.0,
            'sum_squared_error': 0.0,
            'sum_absolute_error': 0.0,
            'count': 0
        }

        # 仅存储少量样本用于可视化和详细指标计算
        sampled_preds_norm = []
        sampled_targets_norm = []
        sample_visualization_batch = None
        batch_losses = []
        
        # 采样比例 - 降低内存使用
        sample_ratio = min(1.0, self.max_samples_for_metrics / (len(val_loader) * val_loader.batch_size))

        self._log_memory_usage("GAMUS验证开始前")

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} GAMUS Validation', leave=False)
            
            for i, (images, targets) in enumerate(val_pbar):
                try:
                    # 移动数据到设备
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    # 数据质量检查
                    if torch.isnan(images).any() or torch.isinf(images).any():
                        self.logger.warning(f"验证批次{i}: 输入图像包含NaN或Inf")
                        continue
                        
                    if torch.isnan(targets).any() or torch.isinf(targets).any():
                        self.logger.warning(f"验证批次{i}: nDSM目标包含NaN或Inf")
                        continue

                    # 获取预测结果
                    preds = model(images)
                    
                    # 预测结果检查
                    if torch.isnan(preds).any() or torch.isinf(preds).any():
                        self.logger.warning(f"验证批次{i}: nDSM预测包含NaN或Inf")
                        continue
                    
                    # 确保维度一致性
                    if preds.shape != targets.shape:
                        if preds.dim() == 3 and targets.dim() == 2:
                            targets = targets.unsqueeze(0) if targets.shape[0] != preds.shape[0] else targets
                        elif preds.shape[-2:] != targets.shape[-2:]:
                            preds = F.interpolate(
                                preds.unsqueeze(1) if preds.dim() == 3 else preds,
                                size=targets.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            )
                            if preds.dim() == 4:
                                preds = preds.squeeze(1)

                    # 数值范围检查
                    preds = torch.clamp(preds, 0, 1)
                    targets = torch.clamp(targets, 0, 1)

                    # 计算损失
                    if criterion is not None:
                        try:
                            loss = criterion(preds, targets)
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                batch_losses.append(loss.item())
                        except Exception as e:
                            self.logger.warning(f"验证批次{i}损失计算错误: {e}")

                    # 累积样本数
                    batch_size = images.size(0)
                    metrics['samples'] += batch_size
                    metrics['valid_batches'] += 1

                    # 转换为numpy进行统计计算
                    preds_cpu = preds.detach().cpu().numpy().flatten()
                    targets_cpu = targets.detach().cpu().numpy().flatten()
                    
                    # 反归一化到真实nDSM高度
                    preds_real = self.denormalize_height(preds_cpu)
                    targets_real = self.denormalize_height(targets_cpu)
                    
                    # 更新在线统计
                    valid_mask = (~np.isnan(preds_real) & ~np.isnan(targets_real) & 
                                 ~np.isinf(preds_real) & ~np.isinf(targets_real))
                    
                    if np.any(valid_mask):
                        valid_preds = preds_real[valid_mask]
                        valid_targets = targets_real[valid_mask]
                        
                        running_metrics['count'] += len(valid_preds)
                        running_metrics['sum_preds'] += np.sum(valid_preds)
                        running_metrics['sum_targets'] += np.sum(valid_targets)
                        running_metrics['sum_squared_error'] += np.sum((valid_preds - valid_targets) ** 2)
                        running_metrics['sum_absolute_error'] += np.sum(np.abs(valid_preds - valid_targets))

                    # 采样保存数据用于详细指标计算
                    if np.random.random() < sample_ratio:
                        sampled_preds_norm.append(preds_cpu)
                        sampled_targets_norm.append(targets_cpu)

                    # 存储第一个批次用于可视化
                    if i == 0 and self.visualization_enabled:
                        sample_visualization_batch = (
                            images.cpu(), 
                            targets.cpu(), 
                            preds.cpu()
                        )

                    # 更新进度条
                    if batch_losses:
                        val_pbar.set_postfix({'loss': f'{batch_losses[-1]:.6f}'})
                    
                    # 定期清理内存
                    if i % self.batch_accumulation_limit == 0 and i > 0:
                        self._clear_cache_and_gc()
                        self._log_memory_usage(f"批次 {i}")

                except Exception as e:
                    self.logger.error(f"验证批次{i}处理错误: {e}")
                    continue

        # 计算基础指标
        if running_metrics['count'] > 0:
            metrics['mae'] = running_metrics['sum_absolute_error'] / running_metrics['count']
            metrics['mse'] = running_metrics['sum_squared_error'] / running_metrics['count']
            metrics['rmse'] = np.sqrt(metrics['mse'])
        else:
            self.logger.error("GAMUS验证阶段没有收集到有效数据!")
            return self._get_default_ndsm_metrics(), []

        # 使用采样数据计算详细指标
        if sampled_preds_norm and sampled_targets_norm:
            try:
                # 合并采样数据
                sampled_preds_norm = np.concatenate(sampled_preds_norm)
                sampled_targets_norm = np.concatenate(sampled_targets_norm)
                
                # 反归一化
                all_preds_real = self.denormalize_height(sampled_preds_norm)
                all_targets_real = self.denormalize_height(sampled_targets_norm)

                # 计算详细指标
                detailed_metrics = self.safe_ndsm_metric_calculation(all_preds_real, all_targets_real)
                
                # 用详细指标更新基础指标
                for key in ['mae', 'mse', 'rmse']:
                    if key in detailed_metrics:
                        metrics[key] = detailed_metrics[key]
                
                # 添加其他指标
                for key, value in detailed_metrics.items():
                    if key not in metrics:
                        metrics[key] = value
                
                # 清理大数组
                del sampled_preds_norm, sampled_targets_norm, all_preds_real, all_targets_real
                self._clear_cache_and_gc()
                
            except Exception as e:
                self.logger.error(f"GAMUS详细指标计算阶段错误: {e}")
                # 使用基础指标填充
                metrics.update({
                    'r2': -1.0,
                    'pearson_r': 0.0,
                    'pearson_p': 1.0,
                    'spearman_r': 0.0,
                    'spearman_p': 1.0,
                    'height_accuracy_1m': 0.0,
                    'height_accuracy_2m': 0.0,
                    'height_accuracy_5m': 0.0,
                    'height_accuracy_10m': 0.0,
                    'relative_error': 100.0,
                    'mae_ground_level': metrics['mae'],
                    'mae_low_buildings': metrics['mae'],
                    'mae_mid_buildings': metrics['mae'],
                    'mae_high_buildings': metrics['mae']
                })
        else:
            metrics.update(self._get_default_ndsm_metrics())

        # 计算平均损失
        if batch_losses:
            metrics['loss'] = np.mean(batch_losses)
        else:
            metrics['loss'] = float('inf')

        # 执行可视化（使用有限的数据）
        visualization_paths = []
        if self.visualization_enabled and sample_visualization_batch is not None:
            try:
                # 为可视化创建小样本数据
                viz_preds = self.denormalize_height(sample_visualization_batch[2].numpy())
                viz_targets = self.denormalize_height(sample_visualization_batch[1].numpy())
                
                viz_path = self._save_ndsm_sample_visualization(
                    epoch, sample_visualization_batch, viz_preds, viz_targets
                )
                if viz_path:
                    visualization_paths.append(viz_path)
                
                # 清理可视化数据
                del viz_preds, viz_targets
                
            except Exception as e:
                self.logger.error(f"GAMUS可视化生成错误: {e}")

        # 记录验证耗时
        metrics['duration'] = time.time() - start_time
        
        # 更新历史记录（限制长度）
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                self.metrics_history[key].append(value)
                # 限制历史记录长度
                if len(self.metrics_history[key]) > self.max_history_length:
                    self.metrics_history[key] = self.metrics_history[key][-self.max_history_length:]

        self._log_memory_usage("GAMUS验证结束")
        self._clear_cache_and_gc()

        return metrics, visualization_paths


    # 在 GAMUSValidationMetricsCalculator 类中，添加新的指标计算方法：

    def calculate_metrics_with_zero_handling(self, predictions, targets, height_normalizer, 
                                        exclude_zeros=True, exclude_invalid=True):
        """
        改进的指标计算，可以选择是否排除0值和无效值
        
        Args:
            predictions: 预测值（归一化后）
            targets: 目标值（归一化后）
            height_normalizer: 高度归一化器
            exclude_zeros: 是否排除真实值为0的像素
            exclude_invalid: 是否排除无效值（标记为-1的像素）
        """
        
        # 反归一化到真实高度值
        pred_heights = height_normalizer.denormalize(predictions.flatten())
        target_heights = height_normalizer.denormalize(targets.flatten())
        
        # 基础有效性掩码
        valid_mask = (~np.isnan(pred_heights) & ~np.isnan(target_heights) & 
                    ~np.isinf(pred_heights) & ~np.isinf(target_heights))
        
        # 排除归一化时标记的无效值
        if exclude_invalid:
            # 检查原始归一化值是否为-1（无效标记）
            invalid_in_normalized = (targets.flatten() < 0)
            valid_mask = valid_mask & (~invalid_in_normalized)
        
        # 可选：排除0值区域
        zero_count = 0
        if exclude_zeros:
            # 排除真实值接近0的像素（考虑数值精度）
            zero_mask = np.abs(target_heights) < 0.1  # 0.1米的容差
            valid_mask = valid_mask & (~zero_mask)
            zero_count = np.sum(zero_mask)
            self.logger.debug(f"排除了 {zero_count} 个接近0值的像素")
        
        if np.sum(valid_mask) < 10:
            self.logger.warning(f"有效样本太少: {np.sum(valid_mask)}")
            return {}
        
        valid_preds = pred_heights[valid_mask]
        valid_targets = target_heights[valid_mask]
        
        self.logger.info(f"用于指标计算的有效像素数: {len(valid_preds):,}")
        self.logger.info(f"数据范围: 真实值 [{valid_targets.min():.2f}, {valid_targets.max():.2f}] m")
        self.logger.info(f"数据范围: 预测值 [{valid_preds.min():.2f}, {valid_preds.max():.2f}] m")
        
        # 计算指标
        mse = np.mean((valid_preds - valid_targets) ** 2)
        mae = np.mean(np.abs(valid_preds - valid_targets))
        rmse = np.sqrt(mse)
        
        # R²计算
        ss_res = np.sum((valid_targets - valid_preds) ** 2)
        ss_tot = np.sum((valid_targets - np.mean(valid_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        
        # 精度指标
        errors = np.abs(valid_preds - valid_targets)
        accuracy_1m = np.mean(errors <= 1.0)
        accuracy_2m = np.mean(errors <= 2.0)
        accuracy_5m = np.mean(errors <= 5.0)
        accuracy_10m = np.mean(errors <= 10.0)
        
        # 分层误差分析（基于有效数据）
        ground_mask = (valid_targets >= -5) & (valid_targets <= 5)
        low_mask = (valid_targets > 5) & (valid_targets <= 20)
        mid_mask = (valid_targets > 20) & (valid_targets <= 50)
        high_mask = valid_targets > 50
        
        mae_ground = np.mean(errors[ground_mask]) if np.sum(ground_mask) > 0 else 0.0
        mae_low = np.mean(errors[low_mask]) if np.sum(low_mask) > 0 else 0.0
        mae_mid = np.mean(errors[mid_mask]) if np.sum(mid_mask) > 0 else 0.0
        mae_high = np.mean(errors[high_mask]) if np.sum(high_mask) > 0 else 0.0
        
        # 统计各层级的像素数量
        layer_counts = {
            'ground_pixels': np.sum(ground_mask),
            'low_building_pixels': np.sum(low_mask), 
            'mid_building_pixels': np.sum(mid_mask),
            'high_building_pixels': np.sum(high_mask),
            'total_valid_pixels': len(valid_preds)
        }
        
        if self.logger:
            self.logger.info(f"分层像素统计:")
            self.logger.info(f"  地面层(-5~5m): {layer_counts['ground_pixels']:,} 像素")
            self.logger.info(f"  低建筑(5~20m): {layer_counts['low_building_pixels']:,} 像素") 
            self.logger.info(f"  中建筑(20~50m): {layer_counts['mid_building_pixels']:,} 像素")
            self.logger.info(f"  高建筑(>50m): {layer_counts['high_building_pixels']:,} 像素")
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson_r': 0.0,  # 可以后续添加相关性计算
            'pearson_p': 1.0,
            'spearman_r': 0.0,
            'spearman_p': 1.0,
            'height_accuracy_1m': float(accuracy_1m),
            'height_accuracy_2m': float(accuracy_2m),
            'height_accuracy_5m': float(accuracy_5m),
            'height_accuracy_10m': float(accuracy_10m),
            'relative_error': float(mae / np.mean(np.abs(valid_targets)) * 100) if np.mean(np.abs(valid_targets)) > 0 else 0.0,
            'mae_ground_level': float(mae_ground),
            'mae_low_buildings': float(mae_low),
            'mae_mid_buildings': float(mae_mid),
            'mae_high_buildings': float(mae_high),
            'data_range_min': float(valid_targets.min()),
            'data_range_max': float(valid_targets.max()),
            'prediction_range_min': float(valid_preds.min()),
            'prediction_range_max': float(valid_preds.max()),
            'valid_pixels': len(valid_preds),
            'excluded_zeros': zero_count,
            'excluded_invalid': np.sum(targets.flatten() < 0) if exclude_invalid else 0,
            'layer_counts': layer_counts
        }
        
        return metrics
    def _get_default_ndsm_metrics(self):
        """返回默认的安全nDSM指标值"""
        return {
            'mse': 999.0,
            'mae': 999.0,
            'rmse': 999.0,
            'r2': -1.0,
            'pearson_r': 0.0,
            'pearson_p': 1.0,
            'spearman_r': 0.0,
            'spearman_p': 1.0,
            'height_accuracy_1m': 0.0,
            'height_accuracy_2m': 0.0,
            'height_accuracy_5m': 0.0,
            'height_accuracy_10m': 0.0,
            'relative_error': 999.0,
            'mae_ground_level': 999.0,
            'mae_low_buildings': 999.0,
            'mae_mid_buildings': 999.0,
            'mae_high_buildings': 999.0
        }

    def _save_ndsm_sample_visualization(self, epoch, sample_batch, sample_preds_real, sample_targets_real, max_samples=2):
        """内存优化的nDSM样本可视化"""
        images, targets, preds = sample_batch
        n_samples = min(max_samples, images.shape[0])

        try:
            fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            # 计算显示范围（使用采样数据）
            global_min = min(sample_targets_real.min(), sample_preds_real.min())
            global_max = max(sample_targets_real.max(), sample_preds_real.max())

            for i in range(n_samples):
                # 输入图像
                ax = axes[i, 0]
                image = images[i].permute(1, 2, 0).numpy()
                # 反标准化显示
                if image.min() < 0:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image = image * std + mean
                    image = np.clip(image, 0, 1)
                ax.imshow(image)
                ax.set_title(f'GAMUS样本 {i + 1}: 输入图像')
                ax.axis('off')

                # 真实nDSM
                ax = axes[i, 1]
                target_real = self.denormalize_height(targets[i].numpy())
                # 使用适合nDSM的colormap
                im1 = ax.imshow(target_real, cmap='terrain', vmin=global_min, vmax=global_max)
                ax.set_title(f'真实nDSM\n[{target_real.min():.1f}-{target_real.max():.1f}m]')
                plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='Height (m)')
                ax.axis('off')

                # 预测nDSM
                ax = axes[i, 2]
                pred_real = self.denormalize_height(preds[i].numpy())
                im2 = ax.imshow(pred_real, cmap='terrain', vmin=global_min, vmax=global_max)
                ax.set_title(f'预测nDSM\n[{pred_real.min():.1f}-{pred_real.max():.1f}m]')
                plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Height (m)')
                ax.axis('off')

                # 误差图
                ax = axes[i, 3]
                error = np.abs(pred_real - target_real)
                max_error = min(15, np.percentile(error, 95))  # 对nDSM使用15米作为最大误差显示
                im3 = ax.imshow(error, cmap='hot', vmin=0, vmax=max_error)
                mae_sample = error.mean()
                ax.set_title(f'绝对误差\nMAE: {mae_sample:.2f}m')
                plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label='Error (m)')
                ax.axis('off')

            vis_path = os.path.join(self.vis_dir, f'gamus_epoch_{epoch}_samples.png')
            plt.tight_layout()
            plt.savefig(vis_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()

            return vis_path
            
        except Exception as e:
            self.logger.error(f"GAMUS样本可视化失败: {e}")
            plt.close('all')
            return None

    def log_metrics(self, epoch, metrics, is_best=False):
        """增强的GAMUS指标记录"""
        self.logger.info(f'GAMUS验证指标 - Epoch {epoch}:')
        self.logger.info(f'  损失: {metrics.get("loss", "N/A"):.6f}')
        self.logger.info(f'  MAE: {metrics.get("mae", "N/A"):.4f} m')
        self.logger.info(f'  RMSE: {metrics.get("rmse", "N/A"):.4f} m')
        self.logger.info(f'  R²: {metrics.get("r2", "N/A"):.4f}')
        self.logger.info(f'  Pearson r: {metrics.get("pearson_r", "N/A"):.4f}')
        
        # nDSM特定的精度指标
        self.logger.info(f'  nDSM高度准确率:')
        self.logger.info(f'    ±1m: {metrics.get("height_accuracy_1m", 0):.1%}')
        self.logger.info(f'    ±2m: {metrics.get("height_accuracy_2m", 0):.1%}')
        self.logger.info(f'    ±5m: {metrics.get("height_accuracy_5m", 0):.1%}')
        self.logger.info(f'    ±10m: {metrics.get("height_accuracy_10m", 0):.1%}')
        
        # nDSM分层误差
        self.logger.info(f'  分层MAE:')
        self.logger.info(f'    地面层(-5~5m): {metrics.get("mae_ground_level", "N/A"):.2f} m')
        self.logger.info(f'    低建筑(5~20m): {metrics.get("mae_low_buildings", "N/A"):.2f} m')
        self.logger.info(f'    中建筑(20~50m): {metrics.get("mae_mid_buildings", "N/A"):.2f} m')
        self.logger.info(f'    高建筑(>50m): {metrics.get("mae_high_buildings", "N/A"):.2f} m')
        
        self.logger.info(f'  相对误差: {metrics.get("relative_error", "N/A"):.2f}%')
        self.logger.info(f'  有效批次: {metrics.get("valid_batches", 0)}')
        self.logger.info(f'  耗时: {metrics.get("duration", 0):.2f}秒')

        if is_best:
            self.logger.info('  ★ 最佳GAMUS验证性能 ★')

        return metrics

# 兼容性别名
# OptimizedValidationMetricsCalculator = GAMUSValidationMetricsCalculator