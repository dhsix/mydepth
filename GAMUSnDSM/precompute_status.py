#!/usr/bin/env python3
"""
预计算GAMUS数据集统计信息脚本
计算全量数据的统计信息并保存，用于后续训练时快速加载
"""

import os
import json
import numpy as np
import cv2
import logging
import argparse
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def setup_logger(log_path: Optional[str] = None, log_level: str = 'INFO'):
    """设置日志记录"""
    logger = logging.getLogger('precompute_stats')
    logger.setLevel(getattr(logging, log_level))
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了路径）
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def validate_directories(data_dir: str, logger: logging.Logger) -> Dict[str, Dict]:
    """验证数据目录结构"""
    logger.info("🔍 验证数据目录结构...")
    
    subdirs = [
        ('train/images', 'training images'),
        ('train/depths', 'training depths'),
        ('val/images', 'validation images'),
        ('val/depths', 'validation depths')
    ]
    
    results = {}
    
    for subdir, desc in subdirs:
        full_path = os.path.join(data_dir, subdir)
        
        if os.path.exists(full_path):
            # 统计图像文件
            files = [f for f in os.listdir(full_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            
            results[subdir] = {
                'exists': True,
                'path': full_path,
                'files': files,
                'count': len(files)
            }
            
            logger.info(f"  ✓ {desc}: {len(files)} 个文件")
            if len(files) > 0:
                logger.info(f"    示例: {files[0]}")
                
        else:
            results[subdir] = {
                'exists': False,
                'path': full_path,
                'files': [],
                'count': 0
            }
            logger.warning(f"  ⚠️  {desc}: 目录不存在")
    
    return results

def extract_base_name(filename: str, is_image: bool = True) -> Optional[str]:
    """提取文件的基础名称用于匹配"""
    try:
        # 处理不同的命名模式
        if '_RGB_' in filename and is_image:
            base_name = filename.replace('_RGB_', '_').rsplit('.', 1)[0]
        elif '_AGL_' in filename and not is_image:
            base_name = filename.replace('_AGL_', '_').rsplit('.', 1)[0]
        elif '_DSM_' in filename and not is_image:
            base_name = filename.replace('_DSM_', '_').rsplit('.', 1)[0]
        elif '_depth_' in filename and not is_image:
            base_name = filename.replace('_depth_', '_').rsplit('.', 1)[0]
        else:
            # 通用处理：去掉扩展名
            base_name = Path(filename).stem
        
        return base_name
    except Exception:
        return None

def match_file_pairs(image_files: List[str], depth_files: List[str], logger: logging.Logger) -> List[Tuple[str, str]]:
    """匹配图像和深度文件对"""
    logger.info("🔗 匹配图像和深度文件...")
    
    # 创建映射字典
    image_dict = {}
    depth_dict = {}
    
    # 处理图像文件
    for img_file in image_files:
        base_name = extract_base_name(img_file, is_image=True)
        if base_name:
            image_dict[base_name] = img_file
    
    # 处理深度文件
    for depth_file in depth_files:
        base_name = extract_base_name(depth_file, is_image=False)
        if base_name:
            depth_dict[base_name] = depth_file
    
    # 匹配文件对
    matched_pairs = []
    for base_name in image_dict:
        if base_name in depth_dict:
            matched_pairs.append((image_dict[base_name], depth_dict[base_name]))
    
    logger.info(f"  ✓ 成功匹配: {len(matched_pairs)} 对文件")
    logger.info(f"  ⚠️  未匹配图像: {len(image_files) - len(matched_pairs)} 个")
    logger.info(f"  ⚠️  未匹配深度: {len(depth_files) - len(matched_pairs)} 个")
    
    return matched_pairs

def get_valid_mask(data: np.ndarray) -> np.ndarray:
    """获取有效数据掩码"""
    invalid_values = [-9999, -32768, 9999, 32767]
    valid_mask = ~(np.isnan(data) | np.isinf(data))
    
    for invalid_val in invalid_values:
        valid_mask = valid_mask & (data != invalid_val)
    
    return valid_mask

def compute_full_statistics(depth_dir: str, 
                          file_pairs: List[Tuple[str, str]], 
                          height_filter: Dict[str, float],
                          logger: logging.Logger) -> Dict:
    """计算全量数据统计信息"""
    logger.info("📊 开始计算全量数据统计信息...")
    logger.info(f"  总文件数: {len(file_pairs)}")
    logger.info(f"  高度过滤范围: [{height_filter['min_height']:.1f}, {height_filter['max_height']:.1f}] 米")
    
    all_heights = []
    processed_count = 0
    error_count = 0
    total_pixels = 0
    valid_pixels = 0
    
    start_time = time.time()
    
    # 使用进度条处理所有文件
    with tqdm(total=len(file_pairs), desc="处理深度文件", unit="文件") as pbar:
        for idx, (image_file, depth_file) in enumerate(file_pairs):
            try:
                depth_path = os.path.join(depth_dir, depth_file)
                
                # 更新进度条描述
                pbar.set_postfix({'当前': depth_file[:30] + '...' if len(depth_file) > 30 else depth_file})
                
                # 加载深度数据
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                
                if depth_data is None:
                    error_count += 1
                    logger.debug(f"无法加载: {depth_file}")
                    pbar.update(1)
                    continue
                
                # 转换数据类型
                depth_data = depth_data.astype(np.float32)
                total_pixels += depth_data.size
                
                # 过滤无效值
                valid_mask = get_valid_mask(depth_data)
                valid_heights = depth_data[valid_mask]
                valid_pixels += len(valid_heights)
                
                if len(valid_heights) > 0:
                    # 单位转换：厘米转米（假设原始数据是厘米）
                    valid_heights = valid_heights / 100.0
                    
                    # 应用高度过滤
                    filtered_heights = valid_heights[
                        (valid_heights >= height_filter['min_height']) & 
                        (valid_heights <= height_filter['max_height'])
                    ]
                    
                    if len(filtered_heights) > 0:
                        all_heights.append(filtered_heights)
                        processed_count += 1
                
                pbar.update(1)
                
                # 每处理1000个文件输出一次中间状态
                if (idx + 1) % 1000 == 0:
                    current_samples = sum(len(h) for h in all_heights)
                    elapsed = time.time() - start_time
                    speed = (idx + 1) / elapsed
                    pbar.set_description(f"已采集 {current_samples:,} 数据点 ({speed:.1f} 文件/秒)")
                
            except Exception as e:
                error_count += 1
                logger.debug(f"处理文件 {depth_file} 失败: {e}")
                pbar.update(1)
                continue
    
    processing_time = time.time() - start_time
    
    # 处理结果
    logger.info(f"📈 处理完成 (耗时 {processing_time:.1f}s):")
    logger.info(f"  成功处理: {processed_count}/{len(file_pairs)} 个文件 ({processed_count/len(file_pairs)*100:.1f}%)")
    logger.info(f"  错误文件: {error_count} 个")
    logger.info(f"  总像素数: {total_pixels:,}")
    logger.info(f"  有效像素: {valid_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
    
    if not all_heights:
        raise ValueError("❌ 无法提取有效的高度数据")
    
    # 合并所有数据
    logger.info("🔄 合并并计算最终统计信息...")
    combined_heights = np.concatenate(all_heights)
    total_samples = len(combined_heights)
    
    logger.info(f"  合并数据点: {total_samples:,} 个")
    
    # 计算基础统计信息
    global_min = float(np.min(combined_heights))
    global_max = float(np.max(combined_heights))
    global_mean = float(np.mean(combined_heights))
    global_std = float(np.std(combined_heights))
    global_median = float(np.median(combined_heights))
    
    # 计算分位数
    percentiles = np.percentile(combined_heights, [1, 5, 25, 50, 75, 95, 99])
    percentile_dict = {
        'p1': float(percentiles[0]),
        'p5': float(percentiles[1]),
        'p25': float(percentiles[2]),
        'p50': float(percentiles[3]),
        'p75': float(percentiles[4]),
        'p95': float(percentiles[5]),
        'p99': float(percentiles[6])
    }
    
    # 计算直方图
    hist, bin_edges = np.histogram(combined_heights, bins=100)
    hist_dict = {
        'counts': hist.tolist(),
        'bin_edges': bin_edges.tolist()
    }
    
    # 统计结果
    stats = {
        'processing_info': {
            'total_files': len(file_pairs),
            'processed_files': processed_count,
            'error_files': error_count,
            'processing_time': processing_time,
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixels,
            'height_filter': height_filter,
            'timestamp': datetime.now().isoformat()
        },
        'global_statistics': {
            'min': global_min,
            'max': global_max,
            'mean': global_mean,
            'std': global_std,
            'median': global_median,
            'range': global_max - global_min,
            'total_samples': total_samples
        },
        'percentiles': percentile_dict,
        'histogram': hist_dict
    }
    
    # 输出详细统计信息
    logger.info("✅ 统计信息计算完成:")
    logger.info(f"  📊 高度范围: [{global_min:.2f}, {global_max:.2f}] 米")
    logger.info(f"  📊 均值±标准差: {global_mean:.2f} ± {global_std:.2f} 米")
    logger.info(f"  📊 中位数: {global_median:.2f} 米")
    logger.info(f"  📊 分位数 [5%, 25%, 75%, 95%]: {[percentile_dict['p5'], percentile_dict['p25'], percentile_dict['p75'], percentile_dict['p95']]}")
    logger.info(f"  📊 有效数据点: {total_samples:,} 个")
    
    return stats

def save_statistics(stats: Dict, output_path: str, logger: logging.Logger):
    """保存统计信息到JSON文件"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 统计信息已保存: {output_path}")
        
        # 输出文件大小
        file_size = os.path.getsize(output_path)
        logger.info(f"  文件大小: {file_size / 1024:.1f} KB")
        
    except Exception as e:
        logger.error(f"❌ 保存统计信息失败: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='预计算GAMUS数据集统计信息')
    
    parser.add_argument('--data_dir', type=str, help='数据根目录路径')
    parser.add_argument('--output', type=str, default='./gamus_full_stats.json',
                       help='输出统计信息文件路径')
    parser.add_argument('--log_file', type=str, default='./precompute_stats.log',
                       help='日志文件路径')
    parser.add_argument('--min_height', type=float, default=-5.0,
                       help='最小高度过滤值（米）')
    parser.add_argument('--max_height', type=float, default=200.0,
                       help='最大高度过滤值（米）')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(args.log_file, args.log_level)
    
    logger.info("🚀 GAMUS数据集统计信息预计算")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"高度范围: [{args.min_height}, {args.max_height}] 米")
    
    try:
        # 1. 验证数据目录
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")
        
        dir_results = validate_directories(args.data_dir, logger)
        
        # 2. 检查训练数据
        train_images = dir_results.get('train/images', {})
        train_depths = dir_results.get('train/depths', {})
        
        if not train_images['exists'] or not train_depths['exists']:
            raise ValueError("缺少训练数据目录")
        
        if train_images['count'] == 0 or train_depths['count'] == 0:
            raise ValueError("训练数据为空")
        
        # 3. 匹配文件对
        file_pairs = match_file_pairs(
            train_images['files'], 
            train_depths['files'], 
            logger
        )
        
        if len(file_pairs) == 0:
            raise ValueError("没有找到匹配的图像-深度文件对")
        
        # 4. 设置高度过滤器
        height_filter = {
            'min_height': args.min_height,
            'max_height': args.max_height
        }
        
        # 5. 计算统计信息
        stats = compute_full_statistics(
            train_depths['path'],
            file_pairs,
            height_filter,
            logger
        )
        
        # 6. 保存结果
        save_statistics(stats, args.output, logger)
        
        # 7. 总结
        logger.info("=" * 60)
        logger.info("🎉 统计信息预计算完成!")
        logger.info(f"  处理文件: {len(file_pairs)} 对")
        logger.info(f"  数据点数: {stats['global_statistics']['total_samples']:,}")
        logger.info(f"  输出文件: {args.output}")
        logger.info("")
        logger.info("💡 现在你可以在训练时使用这个统计信息文件:")
        logger.info(f"   python train.py --stats_json_path {args.output} ...")
        
    except Exception as e:
        logger.error(f"❌ 预计算失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())