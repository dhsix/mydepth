#!/usr/bin/env python3
"""
通用工具函数模块
统一管理项目中常用的工具函数，避免重复代码
"""

import os
import json
import time
import logging
import psutil
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple


def setup_logger(name: str = 'gamus', log_path: Optional[str] = None, 
                level: str = 'INFO', console: bool = True) -> logging.Logger:
    """
    统一的日志设置函数
    
    Args:
        name: logger名称
        log_path: 日志文件路径，None则不保存文件
        level: 日志级别
        console: 是否输出到控制台
    
    Returns:
        配置好的logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device(device_str: str = 'auto') -> torch.device:
    """
    统一的设备获取函数
    
    Args:
        device_str: 设备字符串 ('auto', 'cuda', 'cpu')
    
    Returns:
        torch.device对象
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    return device


def get_device_info(device: torch.device, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    获取设备详细信息
    
    Args:
        device: torch设备对象
        logger: 日志记录器
    
    Returns:
        设备信息字典
    """
    info = {'device': str(device)}
    
    if device.type == 'cuda' and torch.cuda.is_available():
        info.update({
            'device_name': torch.cuda.get_device_name(),
            'device_count': torch.cuda.device_count(),
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'memory_free': torch.cuda.memory_reserved(0) / 1024**3
        })
        
        if logger:
            logger.info(f"GPU设备: {info['device_name']}")
            logger.info(f"总内存: {info['memory_total']:.1f} GB")
    elif device.type == 'cpu':
        info.update({
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024**3,
            'memory_available': psutil.virtual_memory().available / 1024**3
        })
        
        if logger:
            logger.info(f"CPU设备: {info['cpu_count']} 核心")
            logger.info(f"可用内存: {info['memory_available']:.1f} GB")
    
    return info


def load_json_config(config_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    加载JSON配置文件
    
    Args:
        config_path: 配置文件路径
        logger: 日志记录器
    
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if logger:
            logger.info(f"成功加载配置文件: {config_path}")
        
        return config
        
    except Exception as e:
        raise ValueError(f"加载配置文件失败: {e}")


def save_json_config(config: Dict[str, Any], save_path: str, 
                    logger: Optional[logging.Logger] = None) -> None:
    """
    保存JSON配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
        logger: 日志记录器
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        if logger:
            logger.info(f"配置已保存: {save_path}")
            
    except Exception as e:
        raise IOError(f"保存配置文件失败: {e}")


def validate_paths(*paths: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    验证路径是否存在
    
    Args:
        *paths: 要验证的路径列表
        logger: 日志记录器
    
    Returns:
        是否所有路径都存在
    """
    missing_paths = []
    
    for path in paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        if logger:
            logger.error(f"以下路径不存在: {missing_paths}")
        return False
    
    return True


def count_files(directory: str, extensions: Tuple[str, ...] = None) -> int:
    """
    统计目录中文件数量
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名元组，None表示所有文件
    
    Returns:
        文件数量
    """
    if not os.path.exists(directory):
        return 0
    
    files = os.listdir(directory)
    
    if extensions:
        files = [f for f in files if f.lower().endswith(extensions)]
    
    return len(files)


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.2f}小时"


def format_size(bytes_size: int) -> str:
    """
    格式化文件大小显示
    
    Args:
        bytes_size: 字节数
    
    Returns:
        格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"


def create_timestamp() -> str:
    """
    创建时间戳字符串
    
    Returns:
        时间戳字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        a: 被除数
        b: 除数
        default: 除数为0时的默认值
    
    Returns:
        除法结果
    """
    return a / b if abs(b) > 1e-10 else default


def clip_and_check(tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0,
                  name: str = "tensor") -> torch.Tensor:
    """
    裁剪张量并检查异常值
    
    Args:
        tensor: 输入张量
        min_val: 最小值
        max_val: 最大值
        name: 张量名称（用于日志）
    
    Returns:
        裁剪后的张量
    """
    # 检查异常值
    if torch.isnan(tensor).any():
        logging.warning(f"{name}包含NaN值")
    if torch.isinf(tensor).any():
        logging.warning(f"{name}包含Inf值")
    
    # 裁剪
    return torch.clamp(tensor, min_val, max_val)


def clear_memory(device: torch.device = None) -> None:
    """
    清理内存
    
    Args:
        device: 设备对象，None则自动检测
    """
    import gc
    
    # 清理Python垃圾回收
    gc.collect()
    
    # 清理CUDA缓存
    if device is None:
        device = get_device()
    
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage() -> Dict[str, float]:
    """
    获取内存使用情况
    
    Returns:
        内存使用信息字典
    """
    memory_info = {'cpu_memory_mb': psutil.virtual_memory().used / 1024**2}
    
    if torch.cuda.is_available():
        memory_info['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024**2
        memory_info['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024**2
    
    return memory_info


def log_memory_usage(logger: logging.Logger, stage: str = "") -> None:
    """
    记录内存使用情况
    
    Args:
        logger: 日志记录器
        stage: 阶段描述
    """
    memory_info = get_memory_usage()
    
    stage_str = f"[{stage}] " if stage else ""
    logger.debug(f"{stage_str}内存使用: CPU {memory_info['cpu_memory_mb']:.1f}MB")
    
    if 'gpu_memory_mb' in memory_info:
        logger.debug(f"{stage_str}GPU内存: {memory_info['gpu_memory_mb']:.1f}MB "
                    f"(缓存: {memory_info['gpu_memory_cached_mb']:.1f}MB)")


def ensure_dir(path: str) -> str:
    """
    确保目录存在，不存在则创建
    
    Args:
        path: 目录路径
    
    Returns:
        目录路径
    """
    os.makedirs(path, exist_ok=True)
    return path


class Timer:
    """计时器工具类"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """停止计时并返回耗时"""
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """获取已用时间"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(f"第{attempt + 1}次尝试失败: {e}, {delay}秒后重试...")
                        time.sleep(delay)
                    else:
                        logging.error(f"所有{max_retries + 1}次尝试都失败了")
            
            raise last_exception
        
        return wrapper
    return decorator