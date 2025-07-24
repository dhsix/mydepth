#!/usr/bin/env python3
"""
GAMUS nDSM 配置管理模块
集中管理所有配置参数，支持配置文件和命令行参数
提供配置验证、默认值管理和实验管理功能
"""

__version__ = "1.0.0"
__author__ = "GAMUS Team"
__description__ = "GAMUS nDSM配置管理模块 - 统一的配置管理系统"

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import warnings

# 导入配置类
try:
    from .base_config import BaseConfig, ConfigError, ConfigValidator
    from .data_config import DataConfig, create_data_config
    from .model_config import ModelConfig, create_model_config
    from .training_config import TrainingConfig, create_training_config
    
    _CONFIG_IMPORT_SUCCESS = True
    
except ImportError as e:
    warnings.warn(f"部分配置模块导入失败: {e}", ImportWarning)
    _CONFIG_IMPORT_SUCCESS = False

# 导出的公共接口
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__description__',
    
    # 基础配置
    'BaseConfig',
    'ConfigError',
    'ConfigValidator',
    
    # 具体配置类
    'DataConfig',
    'ModelConfig', 
    'TrainingConfig',
    
    # 配置创建函数
    'create_data_config',
    'create_model_config',
    'create_training_config',
    
    # 统一配置管理
    'GAMUSConfig',
    'create_gamus_config',
    'load_config_from_file',
    'save_config_to_file',
    'merge_configs',
    'validate_config',
    
    # 便利函数
    'get_default_config',
    'create_config_from_args',
    'setup_experiment_config'
]

class GAMUSConfig:
    """GAMUS统一配置管理器"""
    
    def __init__(self, 
                 data_config: Optional[DataConfig] = None,
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 experiment_name: str = "gamus_experiment",
                 logger: Optional[logging.Logger] = None):
        """
        初始化GAMUS配置管理器
        
        Args:
            data_config: 数据配置
            model_config: 模型配置
            training_config: 训练配置
            experiment_name: 实验名称
            logger: 日志记录器
        """
        self.experiment_name = experiment_name
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化各子配置
        self.data = data_config or DataConfig()
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        
        # 元数据
        self.metadata = {
            'experiment_name': experiment_name,
            'config_version': __version__,
            'created_at': None,
            'updated_at': None
        }
        
        # 验证配置
        self._validate_all_configs()
    
    def _validate_all_configs(self):
        """验证所有配置"""
        try:
            self.data.validate()
            self.model.validate()
            self.training.validate()
            self.logger.info("所有配置验证通过")
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            raise ConfigError(f"配置验证失败: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'metadata': self.metadata,
            'data': self.data.to_dict(),
            'model': self.model.to_dict(),
            'training': self.training.to_dict()
        }
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """从字典加载配置"""
        self.metadata.update(config_dict.get('metadata', {}))
        
        if 'data' in config_dict:
            self.data.from_dict(config_dict['data'])
        if 'model' in config_dict:
            self.model.from_dict(config_dict['model'])
        if 'training' in config_dict:
            self.training.from_dict(config_dict['training'])
        
        self._validate_all_configs()
    
    def save(self, file_path: str, format: str = 'yaml'):
        """
        保存配置到文件
        
        Args:
            file_path: 文件路径
            format: 文件格式 ('yaml', 'json')
        """
        from datetime import datetime
        
        # 更新时间戳
        current_time = datetime.now().isoformat()
        if self.metadata['created_at'] is None:
            self.metadata['created_at'] = current_time
        self.metadata['updated_at'] = current_time
        
        # 保存配置
        save_config_to_file(self.to_dict(), file_path, format)
        self.logger.info(f"配置已保存到: {file_path}")
    
    def load(self, file_path: str):
        """
        从文件加载配置
        
        Args:
            file_path: 文件路径
        """
        config_dict = load_config_from_file(file_path)
        self.from_dict(config_dict)
        self.logger.info(f"配置已从 {file_path} 加载")
    
    def update_from_args(self, args: argparse.Namespace):
        """
        从命令行参数更新配置
        
        Args:
            args: 命令行参数
        """
        # 更新数据配置
        if hasattr(args, 'data_dir'):
            self.data.data_dir = args.data_dir
        if hasattr(args, 'batch_size'):
            self.data.batch_size = args.batch_size
        if hasattr(args, 'normalization_method'):
            self.data.normalization_method = args.normalization_method
        if hasattr(args, 'stats_json_path'):
            self.data.stats_json_path = args.stats_json_path
        
        # 更新模型配置
        if hasattr(args, 'encoder'):
            self.model.encoder = args.encoder
        if hasattr(args, 'pretrained_path'):
            self.model.pretrained_path = args.pretrained_path
        if hasattr(args, 'freeze_encoder'):
            self.model.freeze_encoder = args.freeze_encoder
        
        # 更新训练配置
        if hasattr(args, 'num_epochs'):
            self.training.num_epochs = args.num_epochs
        if hasattr(args, 'learning_rate'):
            self.training.learning_rate = args.learning_rate
        if hasattr(args, 'loss_type'):
            self.training.loss_type = args.loss_type
        if hasattr(args, 'save_dir'):
            self.training.save_dir = args.save_dir
        if hasattr(args, 'device'):
            self.training.device = args.device
        
        # 重新验证配置
        self._validate_all_configs()
    
    def get_summary(self) -> str:
        """获取配置摘要"""
        summary = f"""
GAMUS配置摘要 - {self.experiment_name}
{'='*50}

数据配置:
  数据目录: {self.data.data_dir}
  批次大小: {self.data.batch_size}
  归一化方法: {self.data.normalization_method}
  统计文件: {self.data.stats_json_path}

模型配置:
  编码器: {self.model.encoder}
  预训练路径: {self.model.pretrained_path}
  冻结编码器: {self.model.freeze_encoder}
  目标尺寸: {self.model.target_size}

训练配置:
  训练轮数: {self.training.num_epochs}
  学习率: {self.training.learning_rate}
  损失函数: {self.training.loss_type}
  保存目录: {self.training.save_dir}
  设备: {self.training.device}

元数据:
  创建时间: {self.metadata.get('created_at', 'N/A')}
  更新时间: {self.metadata.get('updated_at', 'N/A')}
"""
        return summary.strip()


def create_gamus_config(experiment_name: str = "gamus_experiment",
                       **kwargs) -> GAMUSConfig:
    """
    创建GAMUS配置的便利函数
    
    Args:
        experiment_name: 实验名称
        **kwargs: 其他配置参数
    
    Returns:
        GAMUS配置实例
    """
    # 提取各部分配置
    data_kwargs = {k: v for k, v in kwargs.items() 
                   if k in ['data_dir', 'batch_size', 'normalization_method', 
                           'stats_json_path', 'enable_augmentation', 'num_workers']}
    
    model_kwargs = {k: v for k, v in kwargs.items()
                    if k in ['encoder', 'pretrained_path', 'freeze_encoder', 
                            'target_size', 'use_pretrained_dpt']}
    
    training_kwargs = {k: v for k, v in kwargs.items()
                       if k in ['num_epochs', 'learning_rate', 'loss_type',
                               'save_dir', 'device', 'early_stopping_patience']}
    
    # 创建子配置
    data_config = create_data_config(**data_kwargs) if data_kwargs else None
    model_config = create_model_config(**model_kwargs) if model_kwargs else None
    training_config = create_training_config(**training_kwargs) if training_kwargs else None
    
    return GAMUSConfig(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        experiment_name=experiment_name
    )


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    从文件加载配置
    
    Args:
        file_path: 配置文件路径
    
    Returns:
        配置字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif file_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
    
    return config


def save_config_to_file(config: Dict[str, Any], 
                       file_path: str, 
                       format: str = 'yaml'):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        file_path: 文件路径
        format: 文件格式 ('yaml', 'json')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        if format.lower() == 'yaml':
            yaml.dump(config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        elif format.lower() == 'json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典
    
    Args:
        *configs: 配置字典列表
    
    Returns:
        合并后的配置字典
    """
    merged = {}
    
    for config in configs:
        if config is None:
            continue
        
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any], 
                   config_type: str = 'gamus') -> bool:
    """
    验证配置字典
    
    Args:
        config: 配置字典
        config_type: 配置类型
    
    Returns:
        是否验证通过
    """
    try:
        if config_type == 'gamus':
            gamus_config = GAMUSConfig()
            gamus_config.from_dict(config)
        elif config_type == 'data':
            data_config = DataConfig()
            data_config.from_dict(config)
        elif config_type == 'model':
            model_config = ModelConfig()
            model_config.from_dict(config)
        elif config_type == 'training':
            training_config = TrainingConfig()
            training_config.from_dict(config)
        else:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        return True
        
    except Exception:
        return False


def get_default_config() -> GAMUSConfig:
    """
    获取默认配置
    
    Returns:
        默认的GAMUS配置
    """
    return GAMUSConfig(
        experiment_name="default_gamus_experiment"
    )


def create_config_from_args(args: argparse.Namespace,
                           experiment_name: Optional[str] = None) -> GAMUSConfig:
    """
    从命令行参数创建配置
    
    Args:
        args: 命令行参数
        experiment_name: 实验名称
    
    Returns:
        GAMUS配置实例
    """
    if experiment_name is None:
        experiment_name = getattr(args, 'experiment_name', 'gamus_from_args')
    
    config = get_default_config()
    config.experiment_name = experiment_name
    config.update_from_args(args)
    
    return config


def setup_experiment_config(experiment_dir: str,
                           config_name: str = "config.yaml",
                           **kwargs) -> GAMUSConfig:
    """
    设置实验配置
    
    Args:
        experiment_dir: 实验目录
        config_name: 配置文件名
        **kwargs: 配置参数
    
    Returns:
        GAMUS配置实例
    """
    experiment_path = Path(experiment_dir)
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    config_path = experiment_path / config_name
    
    # 如果配置文件存在，加载现有配置
    if config_path.exists():
        config = GAMUSConfig()
        config.load(str(config_path))
        
        # 用新参数更新配置
        if kwargs:
            config_dict = config.to_dict()
            merged_dict = merge_configs(config_dict, kwargs)
            config.from_dict(merged_dict)
    else:
        # 创建新配置
        experiment_name = experiment_path.name
        config = create_gamus_config(experiment_name=experiment_name, **kwargs)
    
    # 保存配置
    config.save(str(config_path))
    
    return config


# 预定义的配置模板
CONFIG_TEMPLATES = {
    'quick_test': {
        'data': {'batch_size': 2, 'num_workers': 1},
        'training': {'num_epochs': 5, 'learning_rate': 1e-4},
        'model': {'encoder': 'basic_cnn', 'freeze_encoder': False}
    },
    
    'production': {
        'data': {'batch_size': 8, 'num_workers': 4},
        'training': {'num_epochs': 100, 'learning_rate': 1e-4},
        'model': {'encoder': 'vitb', 'freeze_encoder': True}
    },
    
    'debug': {
        'data': {'batch_size': 1, 'num_workers': 0},
        'training': {'num_epochs': 2, 'learning_rate': 1e-3},
        'model': {'encoder': 'basic_cnn', 'freeze_encoder': False}
    }
}


def get_config_template(template_name: str) -> Dict[str, Any]:
    """
    获取配置模板
    
    Args:
        template_name: 模板名称
    
    Returns:
        配置模板字典
    """
    if template_name not in CONFIG_TEMPLATES:
        available = list(CONFIG_TEMPLATES.keys())
        raise ValueError(f"未知模板: {template_name}. 可用模板: {available}")
    
    return CONFIG_TEMPLATES[template_name].copy()


def create_config_from_template(template_name: str,
                               experiment_name: Optional[str] = None,
                               **overrides) -> GAMUSConfig:
    """
    从模板创建配置
    
    Args:
        template_name: 模板名称
        experiment_name: 实验名称
        **overrides: 覆盖参数
    
    Returns:
        GAMUS配置实例
    """
    template = get_config_template(template_name)
    merged_config = merge_configs(template, overrides)
    
    if experiment_name is None:
        experiment_name = f"gamus_{template_name}"
    
    return create_gamus_config(experiment_name=experiment_name, **merged_config)


# 配置健康检查
def check_config_health() -> Dict[str, Any]:
    """
    检查配置模块健康状态
    
    Returns:
        健康状态字典
    """
    health = {
        'import_success': _CONFIG_IMPORT_SUCCESS,
        'templates_available': list(CONFIG_TEMPLATES.keys()),
        'config_classes': ['DataConfig', 'ModelConfig', 'TrainingConfig'],
        'version': __version__
    }
    
    # 测试配置创建
    try:
        test_config = get_default_config()
        health['default_config_ok'] = True
    except Exception:
        health['default_config_ok'] = False
    
    # 测试模板
    try:
        test_template = get_config_template('quick_test')
        health['templates_ok'] = True
    except Exception:
        health['templates_ok'] = False
    
    health['overall_healthy'] = all([
        health['import_success'],
        health['default_config_ok'],
        health['templates_ok']
    ])
    
    return health


# 模块初始化日志
if _CONFIG_IMPORT_SUCCESS:
    _logger = logging.getLogger(__name__)
    _logger.info(f"GAMUS Config模块 v{__version__} 初始化成功")
    
    health = check_config_health()
    if health['overall_healthy']:
        _logger.info("配置模块运行正常")
    else:
        _logger.warning(f"配置模块状态: {health}")

# 使用示例
USAGE_EXAMPLES = {
    'basic_usage': '''
# 基础使用
from config import create_gamus_config, GAMUSConfig
config = create_gamus_config('my_experiment', data_dir='/path/to/data')
config.save('config.yaml')
''',
    
    'from_template': '''
# 从模板创建
from config import create_config_from_template
config = create_config_from_template('production', 'my_exp')
''',
    
    'from_args': '''
# 从命令行参数
from config import create_config_from_args
config = create_config_from_args(args, 'experiment_name')
''',
    
    'experiment_setup': '''
# 实验设置
from config import setup_experiment_config
config = setup_experiment_config('./experiments/exp1', 
                                data_dir='/data', 
                                num_epochs=50)
'''
}

def print_config_examples():
    """打印配置使用示例"""
    print("GAMUS Config 使用示例:")
    print("=" * 40)
    
    for name, example in USAGE_EXAMPLES.items():
        print(f"\n{name.upper()}:")
        print(example.strip())

if __name__ == '__main__':
    print(f"GAMUS Config模块 v{__version__}")
    print(f"健康状态: {check_config_health()}")
    print_config_examples()