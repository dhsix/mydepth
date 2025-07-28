# models/imele/__init__.py
from .imele_model import IMELEModel

def create_imele_model(config):
    """创建IMELE模型的工厂函数"""
    return IMELEModel(config)

__all__ = ['IMELEModel', 'create_imele_model']