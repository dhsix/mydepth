#!/usr/bin/env python3
"""
GAMUS nDSM 核心模块
提供数据集、模型、损失函数和归一化器的统一接口
消除重复代码，提供标准化的核心功能
"""

__version__ = "1.0.0"
__author__ = "GAMUS Team"
__description__ = "GAMUS nDSM核心模块 - 统一的数据集、模型、损失函数和归一化器"

# 核心功能导入
try:
    # 数据集相关
    from .dataset import (
        GAMUSDataset,
        GAMUSDatasetConfig, 
        create_gamus_dataloader,
        DatasetValidator
    )
    
    # 归一化器
    from .normalizer import (
        HeightNormalizer,
        create_height_normalizer,
        NormalizationMethod
    )
    
    # 损失函数
    from .loss import (
        # 基础损失类
        BaseLoss,
        MSELoss,
        MAELoss,
        HuberLoss,
        FocalLoss,
        GradientLoss,
        HeightAwareLoss,
        
        # 组合损失
        CombinedLoss,
        ImprovedHeightLoss,
        
        # 工具函数
        create_height_loss,
        create_loss_from_config
    )
    
    # 模型相关 (待实现)
    try:
        from .model import (
            BaseGAMUSModel,
            GAMUSNDSMPredictor,
            create_model,
            load_model_weights
        )
        _MODEL_AVAILABLE = True
    except ImportError:
        _MODEL_AVAILABLE = False
        
except ImportError as e:
    import warnings
    warnings.warn(f"部分核心模块导入失败: {e}", ImportWarning)
    raise

# 导出的公共接口
__all__ = [
    # 版本信息
    '__version__',
    '__author__', 
    '__description__',
    
    # 数据集
    'GAMUSDataset',
    'GAMUSDatasetConfig',
    'create_gamus_dataloader',
    'DatasetValidator',
    
    # 归一化器
    'HeightNormalizer',
    'create_height_normalizer', 
    'NormalizationMethod',
    
    # 损失函数
    'BaseLoss',
    'MSELoss',
    'MAELoss', 
    'HuberLoss',
    'FocalLoss',
    'GradientLoss',
    'HeightAwareLoss',
    'CombinedLoss',
    'ImprovedHeightLoss',
    'create_height_loss',
    'create_loss_from_config',
]

# 条件导出模型相关
if _MODEL_AVAILABLE:
    __all__.extend([
        'BaseGAMUSModel',
        'GAMUSNDSMPredictor', 
        'create_model',
        'load_model_weights'
    ])

# 常用配置预设
DEFAULT_DATASET_CONFIG = {
    'normalization_method': 'minmax',
    'enable_augmentation': False,
    'target_size': (448, 448),
    'height_filter': {'min_height': -5.0, 'max_height': 200.0},
    'batch_size': 8,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True,
    'drop_last': True
}

DEFAULT_LOSS_CONFIG = {
    'loss_type': 'huber',
    'height_aware': True,
    'huber_delta': 0.1,
    'focal_alpha': 2.0,
    'weights': {
        'mse': 1.0,
        'mae': 0.3,
        'huber': 0.5,
        'focal': 0.2,
        'gradient': 0.1
    }
}

DEFAULT_NORMALIZATION_CONFIG = {
    'method': 'minmax',
    'height_filter': {'min_height': -5.0, 'max_height': 200.0},
    'clamp_values': True,
    'epsilon': 1e-8
}

# 便利函数
def create_default_dataset(image_dir: str, label_dir: str, 
                          stats_json_path: str, **kwargs):
    """
    使用默认配置创建数据集
    
    Args:
        image_dir: 影像目录
        label_dir: 标签目录  
        stats_json_path: 统计信息文件路径
        **kwargs: 覆盖默认配置的参数
        
    Returns:
        GAMUSDataset: 配置好的数据集实例
    """
    config = DEFAULT_DATASET_CONFIG.copy()
    config.update(kwargs)
    config.update({
        'image_dir': image_dir,
        'label_dir': label_dir,
        'stats_json_path': stats_json_path
    })
    
    dataset_config = GAMUSDatasetConfig(**config)
    return dataset_config.create_dataset()

def create_default_loss(height_normalizer=None, **kwargs):
    """
    使用默认配置创建损失函数
    
    Args:
        height_normalizer: 高度归一化器
        **kwargs: 覆盖默认配置的参数
        
    Returns:
        ImprovedHeightLoss: 配置好的损失函数实例
    """
    config = DEFAULT_LOSS_CONFIG.copy()
    config.update(kwargs)
    
    return create_height_loss(
        height_normalizer=height_normalizer,
        **config
    )

def create_default_normalizer(stats_json_path: str, **kwargs):
    """
    使用默认配置创建归一化器
    
    Args:
        stats_json_path: 统计信息文件路径
        **kwargs: 覆盖默认配置的参数
        
    Returns:
        HeightNormalizer: 配置好的归一化器实例
    """
    config = DEFAULT_NORMALIZATION_CONFIG.copy()
    config.update(kwargs)
    
    return create_height_normalizer(
        stats_json_path=stats_json_path,
        **config
    )

# 核心模块信息
def get_core_info():
    """获取核心模块信息"""
    info = {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': {
            'dataset': True,
            'normalizer': True, 
            'loss': True,
            'model': _MODEL_AVAILABLE
        },
        'available_classes': len(__all__),
        'default_configs': {
            'dataset': list(DEFAULT_DATASET_CONFIG.keys()),
            'loss': list(DEFAULT_LOSS_CONFIG.keys()),
            'normalizer': list(DEFAULT_NORMALIZATION_CONFIG.keys())
        }
    }
    return info

# 模块初始化检查
def _check_dependencies():
    """检查核心依赖"""
    required_packages = ['torch', 'torchvision', 'numpy', 'opencv-python']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        raise ImportError(f"缺少必要依赖包: {', '.join(missing)}")

# 执行初始化检查
_check_dependencies()

# 模块级日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"GAMUS核心模块 v{__version__} 初始化成功")
logger.info(f"可用模块: dataset={'✓' if True else '✗'}, "
           f"normalizer={'✓' if True else '✗'}, " 
           f"loss={'✓' if True else '✗'}, "
           f"model={'✓' if _MODEL_AVAILABLE else '✗'}")