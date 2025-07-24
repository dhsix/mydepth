import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

class BaseDepthModel(nn.Module, ABC):
    """深度估计模型的基础接口类"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = config.get('model_name', 'base_model')
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            **kwargs: 其他参数
            
        Returns:
            深度/高程预测结果
        """
        pass
    
    @abstractmethod
    def compute_loss(self, 
                    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                    targets: torch.Tensor,
                    masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            masks: 可选的掩码
            
        Returns:
            包含各项损失的字典
        """
        pass
    
    def predict(self, 
               x: torch.Tensor, 
               return_multi_scale: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """预测接口，默认实现"""
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_multi_scale=return_multi_scale)
    
    @torch.no_grad()
    def infer_image(self, 
                   raw_image: np.ndarray, 
                   input_size: int = 448) -> np.ndarray:
        """单张图像推理接口
        
        Args:
            raw_image: 原始图像 [H, W, 3], BGR格式
            input_size: 输入尺寸
            
        Returns:
            高程图 [H, W]
        """
        # 图像预处理
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        # 模型推理
        prediction = self.predict(image)
        
        # 后处理
        if isinstance(prediction, dict):
            # 多尺度输出，选择最高分辨率
            prediction = prediction.get('scale_4', list(prediction.values())[-1])
        
        # 调整到原始尺寸
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), 
            (h, w), 
            mode="bilinear", 
            align_corners=True
        )[0, 0]
        
        return prediction.cpu().numpy()
    
    def image2tensor(self, raw_image: np.ndarray, input_size: int = 448) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """图像预处理，转换为tensor"""
        # 这里可以复用data/transforms.py中的逻辑
        # 或者子类重写这个方法
        import cv2
        from data.transforms import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose
        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        sample = {'image': image}
        sample = transform(sample)
        image = torch.from_numpy(sample['image']).unsqueeze(0)
        
        # 移动到设备
        device = next(self.parameters()).device
        image = image.to(device)
        
        return image, (h, w)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config
        }
    
    def freeze_encoder(self):
        """冻结编码器参数，子类可重写"""
        pass
    
    def unfreeze_encoder(self):
        """解冻编码器参数，子类可重写"""
        pass