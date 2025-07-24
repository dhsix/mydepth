#!/usr/bin/env python3
"""
基础配置类模块
提供配置管理的基础功能和接口
"""

import os
import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import warnings


class ConfigError(Exception):
    """配置相关异常"""
    pass


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_path(path: str, 
                     must_exist: bool = True,
                     path_type: str = 'any') -> bool:
        """
        验证路径
        
        Args:
            path: 路径字符串
            must_exist: 是否必须存在
            path_type: 路径类型 ('file', 'dir', 'any')
        
        Returns:
            是否有效
        
        Raises:
            ConfigError: 路径无效时
        """
        if not path:
            raise ConfigError("路径不能为空")
        
        path_obj = Path(path)
        
        if must_exist:
            if not path_obj.exists():
                raise ConfigError(f"路径不存在: {path}")
            
            if path_type == 'file' and not path_obj.is_file():
                raise ConfigError(f"路径不是文件: {path}")
            elif path_type == 'dir' and not path_obj.is_dir():
                raise ConfigError(f"路径不是目录: {path}")
        
        return True
    
    @staticmethod
    def validate_positive_number(value: Union[int, float],
                                name: str = "值") -> bool:
        """
        验证正数
        
        Args:
            value: 要验证的值
            name: 参数名称
        
        Returns:
            是否有效
        
        Raises:
            ConfigError: 值无效时
        """
        if not isinstance(value, (int, float)):
            raise ConfigError(f"{name}必须是数字，当前类型: {type(value)}")
        
        if value <= 0:
            raise ConfigError(f"{name}必须是正数，当前值: {value}")
        
        return True
    
    @staticmethod
    def validate_in_choices(value: Any, 
                           choices: List[Any],
                           name: str = "值") -> bool:
        """
        验证值是否在选择列表中
        
        Args:
            value: 要验证的值
            choices: 有效选择列表
            name: 参数名称
        
        Returns:
            是否有效
        
        Raises:
            ConfigError: 值无效时
        """
        if value not in choices:
            raise ConfigError(f"{name}必须是以下之一: {choices}，当前值: {value}")
        
        return True
    
    @staticmethod
    def validate_range(value: Union[int, float],
                      min_val: Optional[Union[int, float]] = None,
                      max_val: Optional[Union[int, float]] = None,
                      name: str = "值") -> bool:
        """
        验证值是否在指定范围内
        
        Args:
            value: 要验证的值
            min_val: 最小值
            max_val: 最大值
            name: 参数名称
        
        Returns:
            是否有效
        
        Raises:
            ConfigError: 值无效时
        """
        if not isinstance(value, (int, float)):
            raise ConfigError(f"{name}必须是数字，当前类型: {type(value)}")
        
        if min_val is not None and value < min_val:
            raise ConfigError(f"{name}不能小于{min_val}，当前值: {value}")
        
        if max_val is not None and value > max_val:
            raise ConfigError(f"{name}不能大于{max_val}，当前值: {value}")
        
        return True
    
    @staticmethod
    def validate_tuple_length(value: tuple,
                             expected_length: int,
                             name: str = "元组") -> bool:
        """
        验证元组长度
        
        Args:
            value: 要验证的元组
            expected_length: 期望长度
            name: 参数名称
        
        Returns:
            是否有效
        
        Raises:
            ConfigError: 值无效时
        """
        if not isinstance(value, tuple):
            raise ConfigError(f"{name}必须是元组，当前类型: {type(value)}")
        
        if len(value) != expected_length:
            raise ConfigError(f"{name}长度必须是{expected_length}，当前长度: {len(value)}")
        
        return True


class BaseConfig(ABC):
    """配置基类"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化基础配置
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._config_name = self.__class__.__name__
        self._validator = ConfigValidator()
        
        # 初始化默认值
        self._set_defaults()
    
    @abstractmethod
    def _set_defaults(self):
        """设置默认值（子类实现）"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """验证配置（子类实现）"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            配置字典
        """
        config_dict = {}
        
        for key, value in self.__dict__.items():
            # 跳过私有属性和特殊属性
            if key.startswith('_') or key in ['logger']:
                continue
            
            # 处理嵌套配置对象
            if hasattr(value, 'to_dict'):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = copy.deepcopy(value)
        
        return config_dict
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """
        从字典加载配置
        
        Args:
            config_dict: 配置字典
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                # 处理嵌套配置对象
                attr = getattr(self, key)
                if hasattr(attr, 'from_dict') and isinstance(value, dict):
                    attr.from_dict(value)
                else:
                    setattr(self, key, value)
            else:
                self.logger.warning(f"未知配置项: {key} = {value}")
    
    def update(self, **kwargs):
        """
        更新配置
        
        Args:
            **kwargs: 要更新的配置项
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.logger.warning(f"尝试设置未知配置项: {key} = {value}")
    
    def merge(self, other_config: 'BaseConfig'):
        """
        合并另一个配置对象
        
        Args:
            other_config: 另一个配置对象
        """
        if not isinstance(other_config, BaseConfig):
            raise ConfigError("只能合并BaseConfig的子类实例")
        
        other_dict = other_config.to_dict()
        self.from_dict(other_dict)
    
    def copy(self) -> 'BaseConfig':
        """
        创建配置的深拷贝
        
        Returns:
            配置副本
        """
        new_config = self.__class__(logger=self.logger)
        new_config.from_dict(self.to_dict())
        return new_config
    
    def get_diff(self, other_config: 'BaseConfig') -> Dict[str, Any]:
        """
        获取与另一个配置的差异
        
        Args:
            other_config: 另一个配置对象
        
        Returns:
            差异字典
        """
        if not isinstance(other_config, BaseConfig):
            raise ConfigError("只能比较BaseConfig的子类实例")
        
        self_dict = self.to_dict()
        other_dict = other_config.to_dict()
        
        diff = {}
        
        # 检查自己有但对方没有的
        for key, value in self_dict.items():
            if key not in other_dict:
                diff[f'+{key}'] = value
            elif self_dict[key] != other_dict[key]:
                diff[f'~{key}'] = {'self': value, 'other': other_dict[key]}
        
        # 检查对方有但自己没有的
        for key, value in other_dict.items():
            if key not in self_dict:
                diff[f'-{key}'] = value
        
        return diff
    
    def validate_and_fix(self) -> List[str]:
        """
        验证配置并尝试修复
        
        Returns:
            修复信息列表
        """
        fixes = []
        
        try:
            self.validate()
        except ConfigError as e:
            self.logger.warning(f"配置验证失败: {e}")
            # 子类可以重写此方法来实现自动修复
            fixes.append(f"验证失败: {e}")
        
        return fixes
    
    def get_summary(self) -> str:
        """
        获取配置摘要
        
        Returns:
            格式化的配置摘要
        """
        config_dict = self.to_dict()
        
        summary = f"{self._config_name} 配置摘要:\n"
        summary += "=" * (len(self._config_name) + 8) + "\n"
        
        for key, value in config_dict.items():
            if isinstance(value, dict):
                summary += f"{key}:\n"
                for sub_key, sub_value in value.items():
                    summary += f"  {sub_key}: {sub_value}\n"
            else:
                summary += f"{key}: {value}\n"
        
        return summary.strip()
    
    def check_required_fields(self, required_fields: List[str]) -> bool:
        """
        检查必需字段
        
        Args:
            required_fields: 必需字段列表
        
        Returns:
            是否所有必需字段都存在
        
        Raises:
            ConfigError: 缺少必需字段时
        """
        missing_fields = []
        
        for field in required_fields:
            if not hasattr(self, field) or getattr(self, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise ConfigError(f"缺少必需字段: {missing_fields}")
        
        return True
    
    def set_from_env(self, env_mapping: Dict[str, str]):
        """
        从环境变量设置配置
        
        Args:
            env_mapping: 环境变量映射 {config_key: env_var_name}
        """
        for config_key, env_var in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # 尝试类型转换
                try:
                    # 检查现有属性的类型
                    if hasattr(self, config_key):
                        current_value = getattr(self, config_key)
                        if isinstance(current_value, bool):
                            env_value = env_value.lower() in ['true', '1', 'yes', 'on']
                        elif isinstance(current_value, int):
                            env_value = int(env_value)
                        elif isinstance(current_value, float):
                            env_value = float(env_value)
                    
                    setattr(self, config_key, env_value)
                    self.logger.info(f"从环境变量 {env_var} 设置 {config_key} = {env_value}")
                    
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"无法从环境变量 {env_var} 设置 {config_key}: {e}")
    
    def get_config_hash(self) -> str:
        """
        获取配置的哈希值（用于缓存等）
        
        Returns:
            配置的MD5哈希值
        """
        import hashlib
        import json
        
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def is_compatible_with(self, other_config: 'BaseConfig',
                          ignore_fields: Optional[List[str]] = None) -> bool:
        """
        检查与另一个配置的兼容性
        
        Args:
            other_config: 另一个配置对象
            ignore_fields: 要忽略的字段列表
        
        Returns:
            是否兼容
        """
        ignore_fields = ignore_fields or []
        
        self_dict = self.to_dict()
        other_dict = other_config.to_dict()
        
        # 移除要忽略的字段
        for field in ignore_fields:
            self_dict.pop(field, None)
            other_dict.pop(field, None)
        
        # 比较关键字段
        critical_fields = self._get_critical_fields()
        
        for field in critical_fields:
            if field in self_dict and field in other_dict:
                if self_dict[field] != other_dict[field]:
                    self.logger.warning(f"关键字段不匹配: {field}")
                    return False
        
        return True
    
    def _get_critical_fields(self) -> List[str]:
        """
        获取关键字段列表（子类可重写）
        
        Returns:
            关键字段列表
        """
        return []
    
    def export_to_file(self, file_path: str, format: str = 'yaml'):
        """
        导出配置到文件
        
        Args:
            file_path: 文件路径
            format: 文件格式 ('yaml', 'json')
        """
        import json
        try:
            import yaml
            YAML_AVAILABLE = True
        except ImportError:
            YAML_AVAILABLE = False
        
        config_dict = self.to_dict()
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml':
                if not YAML_AVAILABLE:
                    raise ImportError("需要安装PyYAML: pip install pyyaml")
                yaml.dump(config_dict, f, default_flow_style=False,
                         allow_unicode=True, indent=2)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的格式: {format}")
        
        self.logger.info(f"配置已导出到: {file_path}")
    
    def import_from_file(self, file_path: str):
        """
        从文件导入配置
        
        Args:
            file_path: 文件路径
        """
        import json
        try:
            import yaml
            YAML_AVAILABLE = True
        except ImportError:
            YAML_AVAILABLE = False
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("需要安装PyYAML: pip install pyyaml")
                config_dict = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        self.from_dict(config_dict)
        self.logger.info(f"配置已从 {file_path} 导入")
    
    def __str__(self) -> str:
        """字符串表示"""
        return self.get_summary()
    
    def __repr__(self) -> str:
        """详细表示"""
        return f"{self.__class__.__name__}({self.to_dict()})"
    
    def __eq__(self, other) -> bool:
        """相等性比较"""
        if not isinstance(other, BaseConfig):
            return False
        return self.to_dict() == other.to_dict()


# 配置管理器类
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._configs: Dict[str, BaseConfig] = {}
    
    def register_config(self, name: str, config: BaseConfig):
        """注册配置"""
        self._configs[name] = config
        self.logger.info(f"注册配置: {name}")
    
    def get_config(self, name: str) -> BaseConfig:
        """获取配置"""
        if name not in self._configs:
            raise ConfigError(f"未找到配置: {name}")
        return self._configs[name]
    
    def list_configs(self) -> List[str]:
        """列出所有配置名称"""
        return list(self._configs.keys())
    
    def validate_all(self) -> Dict[str, bool]:
        """验证所有配置"""
        results = {}
        for name, config in self._configs.items():
            try:
                config.validate()
                results[name] = True
            except Exception as e:
                self.logger.error(f"配置 {name} 验证失败: {e}")
                results[name] = False
        return results


# 使用示例
if __name__ == '__main__':
    # 创建一个简单的配置类用于测试
    class TestConfig(BaseConfig):
        def _set_defaults(self):
            self.test_value = 42
            self.test_path = "/tmp"
        
        def validate(self) -> bool:
            self._validator.validate_positive_number(self.test_value, "test_value")
            self._validator.validate_path(self.test_path, must_exist=False)
            return True
    
    # 测试配置类
    config = TestConfig()
    print("默认配置:")
    print(config.get_summary())
    
    # 测试验证
    try:
        config.validate()
        print("✓ 验证通过")
    except ConfigError as e:
        print(f"✗ 验证失败: {e}")
    
    # 测试序列化
    config_dict = config.to_dict()
    print(f"\n配置字典: {config_dict}")
    
    # 测试导入导出
    config.export_to_file("test_config.json", "json")
    print("✓ 配置已导出")