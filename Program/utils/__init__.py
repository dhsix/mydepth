#!/usr/bin/env python3
"""
GAMUS nDSM 工具模块
提供数据处理、模型管理、指标计算、可视化等统一工具函数
消除重复代码，提供标准化的工具接口
"""

__version__ = "1.0.0"
__author__ = "GAMUS Team"
__description__ = "GAMUS nDSM工具模块 - 统一的工具函数库"

import warnings
import logging
from typing import Dict, Any, List, Optional

# 核心工具导入
try:
    # 通用工具
    from .common import (
        # 日志相关
        setup_logger,
        get_default_logger,
        
        # 设备管理
        get_device,
        clear_memory,
        
        # 文件系统
        ensure_dir,
        create_timestamp,
        safe_copy_file,
        
        # 配置管理
        load_config,
        save_config,
        
        # 验证工具
        validate_paths,
        count_files,
        
        # 格式化工具
        format_time,
        format_size,
        
        # 计时器
        Timer
    )
    
    # 数据处理工具
    from .data_utils import (
        # 文件匹配
        extract_base_name,
        match_file_pairs,
        
        # 数据验证
        validate_data_structure,
        validate_single_split,
        
        # 统计信息
        load_statistics_config,
        
        # 掩码处理
        get_valid_mask,
        
        # 数据结构验证
        check_data_directory_structure
    )
    
    # 指标计算
    from .metrics import (
        # 配置
        MetricsConfig,
        
        # 计算器
        BaseMetricsCalculator,
        OnlineMetricsCalculator,
        ValidationMetricsCalculator,
        
        # 验证函数
        validate_model_enhanced,
        
        # 便利函数
        calculate_regression_metrics,
        calculate_accuracy_metrics
    )
    
    # 模型工具
    from .model_utils import (
        # 模型配置
        create_model_config,
        
        # 参数统计
        count_parameters,
        log_model_info,
        
        # 检查点管理
        save_model_checkpoint,
        load_model_checkpoint,
        find_latest_checkpoint,
        
        # 模型状态
        freeze_model_layers,
        unfreeze_model_layers,
        get_model_summary,
        
        # 训练工具
        setup_training_environment,
        cleanup_training_environment
    )
    
    # 可视化工具
    from .visualization import (
        # 主要可视化器
        GAMUSVisualizer,
        
        # 便利函数
        create_quick_visualization,
        create_batch_quick_visualization,
        
        # 配置
        VISUALIZATION_CONFIG,
        NDSM_COLORMAPS,
        ERROR_COLORMAPS
    )
    
    _IMPORT_SUCCESS = True
    
except ImportError as e:
    import warnings
    warnings.warn(f"部分工具模块导入失败: {e}", ImportWarning)
    _IMPORT_SUCCESS = False

# 导出的公共接口
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__description__',
    
    # 通用工具 - 日志
    'setup_logger',
    'get_default_logger',
    
    # 通用工具 - 设备
    'get_device',
    'clear_memory',
    
    # 通用工具 - 文件系统
    'ensure_dir',
    'create_timestamp',
    'safe_copy_file',
    
    # 通用工具 - 配置
    'load_config',
    'save_config',
    
    # 通用工具 - 验证
    'validate_paths',
    'count_files',
    
    # 通用工具 - 格式化
    'format_time',
    'format_size',
    
    # 通用工具 - 计时
    'Timer',
    
    # 数据处理工具
    'extract_base_name',
    'match_file_pairs',
    'validate_data_structure',
    'validate_single_split',
    'load_statistics_config',
    'get_valid_mask',
    'check_data_directory_structure',
    
    # 指标计算
    'MetricsConfig',
    'BaseMetricsCalculator',
    'OnlineMetricsCalculator',
    'ValidationMetricsCalculator',
    'validate_model_enhanced',
    'calculate_regression_metrics',
    'calculate_accuracy_metrics',
    
    # 模型工具
    'create_model_config',
    'count_parameters',
    'log_model_info',
    'save_model_checkpoint',
    'load_model_checkpoint',
    'find_latest_checkpoint',
    'freeze_model_layers',
    'unfreeze_model_layers',
    'get_model_summary',
    'setup_training_environment',
    'cleanup_training_environment',
    
    # 可视化工具
    'GAMUSVisualizer',
    'create_quick_visualization',
    'create_batch_quick_visualization',
    'VISUALIZATION_CONFIG',
    'NDSM_COLORMAPS',
    'ERROR_COLORMAPS',
]

# 常用工具组合预设
COMMON_TOOL_PRESETS = {
    'basic_logging': {
        'setup_logger': setup_logger,
        'get_device': get_device,
        'ensure_dir': ensure_dir,
        'format_time': format_time
    },
    
    'data_processing': {
        'validate_data_structure': validate_data_structure,
        'load_statistics_config': load_statistics_config,
        'match_file_pairs': match_file_pairs,
        'get_valid_mask': get_valid_mask
    },
    
    'model_management': {
        'count_parameters': count_parameters,
        'log_model_info': log_model_info,
        'save_model_checkpoint': save_model_checkpoint,
        'load_model_checkpoint': load_model_checkpoint
    },
    
    'metrics_evaluation': {
        'MetricsConfig': MetricsConfig,
        'OnlineMetricsCalculator': OnlineMetricsCalculator,
        'validate_model_enhanced': validate_model_enhanced
    },
    
    'visualization': {
        'GAMUSVisualizer': GAMUSVisualizer,
        'create_quick_visualization': create_quick_visualization,
        'create_batch_quick_visualization': create_batch_quick_visualization
    }
}

# 便利函数
def get_tool_preset(preset_name: str) -> Dict[str, Any]:
    """
    获取预设的工具组合
    
    Args:
        preset_name: 预设名称 ('basic_logging', 'data_processing', 
                    'model_management', 'metrics_evaluation', 'visualization')
    
    Returns:
        工具函数字典
    
    Raises:
        ValueError: 如果预设名称不存在
    """
    if preset_name not in COMMON_TOOL_PRESETS:
        available_presets = list(COMMON_TOOL_PRESETS.keys())
        raise ValueError(f"未知的预设名称: {preset_name}. 可用预设: {available_presets}")
    
    return COMMON_TOOL_PRESETS[preset_name].copy()

def list_available_tools() -> Dict[str, List[str]]:
    """
    列出所有可用的工具函数
    
    Returns:
        按模块分组的工具函数字典
    """
    tools = {
        'common': [
            'setup_logger', 'get_default_logger', 'get_device', 'clear_memory',
            'ensure_dir', 'create_timestamp', 'safe_copy_file', 'load_config',
            'save_config', 'validate_paths', 'count_files', 'format_time',
            'format_size', 'Timer'
        ],
        'data_utils': [
            'extract_base_name', 'match_file_pairs', 'validate_data_structure',
            'validate_single_split', 'load_statistics_config', 'get_valid_mask',
            'check_data_directory_structure'
        ],
        'metrics': [
            'MetricsConfig', 'BaseMetricsCalculator', 'OnlineMetricsCalculator',
            'ValidationMetricsCalculator', 'validate_model_enhanced',
            'calculate_regression_metrics', 'calculate_accuracy_metrics'
        ],
        'model_utils': [
            'create_model_config', 'count_parameters', 'log_model_info',
            'save_model_checkpoint', 'load_model_checkpoint', 'find_latest_checkpoint',
            'freeze_model_layers', 'unfreeze_model_layers', 'get_model_summary',
            'setup_training_environment', 'cleanup_training_environment'
        ],
        'visualization': [
            'GAMUSVisualizer', 'create_quick_visualization',
            'create_batch_quick_visualization', 'VISUALIZATION_CONFIG',
            'NDSM_COLORMAPS', 'ERROR_COLORMAPS'
        ]
    }
    return tools

def create_quick_setup(data_dir: str, 
                      save_dir: str,
                      log_level: str = 'INFO') -> Dict[str, Any]:
    """
    快速设置常用工具的便利函数
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        log_level: 日志级别
    
    Returns:
        设置好的工具字典
    """
    # 确保目录存在
    ensure_dir(save_dir)
    
    # 设置日志
    log_file = f"{save_dir}/utils_{create_timestamp()}.log"
    logger = setup_logger(log_file, level=getattr(logging, log_level.upper()))
    
    # 获取设备
    device = get_device()
    
    # 验证数据目录
    data_valid = validate_paths(data_dir)
    
    # 创建计时器
    timer = Timer()
    
    setup_info = {
        'logger': logger,
        'device': device,
        'data_dir': data_dir,
        'save_dir': save_dir,
        'data_valid': data_valid,
        'timer': timer,
        'timestamp': create_timestamp()
    }
    
    logger.info(f"快速设置完成:")
    logger.info(f"  数据目录: {data_dir} ({'✓' if data_valid else '✗'})")
    logger.info(f"  保存目录: {save_dir}")
    logger.info(f"  设备: {device}")
    logger.info(f"  时间戳: {setup_info['timestamp']}")
    
    return setup_info

def check_utils_health() -> Dict[str, Any]:
    """
    检查 utils 模块的健康状态
    
    Returns:
        健康状态字典
    """
    health_status = {
        'import_success': _IMPORT_SUCCESS,
        'modules_available': {},
        'presets_available': list(COMMON_TOOL_PRESETS.keys()),
        'total_tools': len(__all__),
        'version': __version__
    }
    
    # 检查各个模块的可用性
    modules_to_check = ['common', 'data_utils', 'metrics', 'model_utils', 'visualization']
    
    for module_name in modules_to_check:
        try:
            module = __import__(f'utils.{module_name}', fromlist=[module_name])
            health_status['modules_available'][module_name] = True
        except ImportError:
            health_status['modules_available'][module_name] = False
    
    # 整体健康状态
    all_modules_available = all(health_status['modules_available'].values())
    health_status['overall_healthy'] = _IMPORT_SUCCESS and all_modules_available
    
    return health_status

def get_utils_info() -> str:
    """
    获取 utils 模块的详细信息
    
    Returns:
        格式化的信息字符串
    """
    health = check_utils_health()
    tools = list_available_tools()
    
    info = f"""
GAMUS Utils 模块信息 v{__version__}
{'='*50}

健康状态: {'✓ 健康' if health['overall_healthy'] else '✗ 有问题'}
导入状态: {'✓ 成功' if health['import_success'] else '✗ 失败'}
可用工具数: {health['total_tools']}

模块状态:"""
    
    for module, available in health['modules_available'].items():
        status = '✓' if available else '✗'
        tool_count = len(tools.get(module, []))
        info += f"\n  {status} {module}: {tool_count} 个工具"
    
    info += f"\n\n可用预设: {', '.join(health['presets_available'])}"
    
    info += f"\n\n使用示例:"
    info += f"\n  from utils import setup_logger, get_device"
    info += f"\n  from utils import GAMUSVisualizer, validate_data_structure"
    info += f"\n  setup = create_quick_setup('/data', './output')"
    
    return info

# 模块初始化检查
def _check_dependencies():
    """检查必要依赖"""
    required_packages = ['torch', 'numpy', 'matplotlib', 'sklearn', 'opencv-python']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('opencv', 'cv2'))
        except ImportError:
            missing.append(package)
    
    if missing:
        warnings.warn(f"缺少推荐依赖包: {', '.join(missing)}", UserWarning)

# 执行依赖检查
_check_dependencies()

# 模块级日志
if _IMPORT_SUCCESS:
    _logger = logging.getLogger(__name__)
    _logger.info(f"GAMUS Utils模块 v{__version__} 初始化成功")
    
    # 健康检查
    health = check_utils_health()
    if health['overall_healthy']:
        _logger.info("所有工具模块运行正常")
    else:
        _logger.warning(f"部分模块不可用: {health['modules_available']}")

# 使用示例和文档
USAGE_EXAMPLES = {
    'basic_setup': '''
# 基础设置
from utils import create_quick_setup
setup = create_quick_setup('/path/to/data', './output')
logger = setup['logger']
device = setup['device']
''',
    
    'data_processing': '''
# 数据处理
from utils import validate_data_structure, load_statistics_config
data_structure = validate_data_structure('/path/to/data')
stats = load_statistics_config('stats.json')
''',
    
    'model_management': '''
# 模型管理
from utils import count_parameters, log_model_info, save_model_checkpoint
params = count_parameters(model)
log_model_info(model, 'GAMUS', logger)
save_model_checkpoint(epoch, model, optimizer, loss, './checkpoints')
''',
    
    'visualization': '''
# 可视化
from utils import GAMUSVisualizer, create_quick_visualization
visualizer = GAMUSVisualizer(logger=logger)
path = visualizer.create_single_sample_visualization(result, 'output.png')
''',
    
    'metrics': '''
# 指标计算
from utils import OnlineMetricsCalculator, validate_model_enhanced
calculator = OnlineMetricsCalculator(height_normalizer)
metrics = validate_model_enhanced(model, val_loader, criterion, device)
'''
}

def print_usage_examples():
    """打印使用示例"""
    print("GAMUS Utils 使用示例:")
    print("=" * 40)
    
    for name, example in USAGE_EXAMPLES.items():
        print(f"\n{name.upper()}:")
        print(example.strip())

# 条件导出使用示例
if __name__ == '__main__':
    print(get_utils_info())
    print("\n" + "="*50)
    print_usage_examples()