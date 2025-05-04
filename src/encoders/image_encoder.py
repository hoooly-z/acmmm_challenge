# src/encoders/image_encoder.py
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import logging

logger = logging.getLogger(__name__)

class ImageEncoder:
    def __init__(self, config):
        """
        图像编码器类
        
        Args:
            config: 配置信息，包含模型路径等参数
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        try:
            logger.info(f"Loading image encoder model from {config['model']['model_path']}")
            # 使用Qwen2.5-VL的视觉编码器
            self.processor = AutoProcessor.from_pretrained(config["model"]["model_path"])
            self.model = AutoModel.from_pretrained(
                config["model"]["model_path"], 
                device_map=config["model"]["device_map"],
                torch_dtype=self._get_precision(config["model"]["precision"])
            )
            logger.info("Image encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load image encoder: {e}")
            raise
    
    def _get_precision(self, precision):
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        return torch.float32
    
    def encode(self, image, return_tensors=True):
        """
        编码单张图像
        
        Args:
            image: PIL图像或图像路径
            return_tensors: 是否返回张量，否则返回numpy数组
            
        Returns:
            图像的特征向量
        """
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to open image {image}: {e}")
                raise
        
        # 使用处理器处理图像
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # 获取图像嵌入
        with torch.no_grad():
            outputs = self.model.get_vision_embeddings(**inputs)
            # 获取[CLS]令牌或池化的表示
            image_features = outputs.last_hidden_state[:, 0, :]
        
        if not return_tensors:
            image_features = image_features.cpu().numpy()
            
        return image_features
    
    def batch_encode(self, images, batch_size=8):
        """
        批量编码图像
        
        Args:
            images: 图像列表（PIL对象或路径）
            batch_size: 批处理大小
            
        Returns:
            图像特征向量列表
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_images = []
            
            for img in batch:
                if isinstance(img, str):
                    try:
                        img = Image.open(img).convert("RGB")
                    except Exception as e:
                        logger.error(f"Failed to open image {img}: {e}")
                        # 使用空白图像替代
                        img = Image.new("RGB", (224, 224), (255, 255, 255))
                batch_images.append(img)
                
            # 批量处理
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_vision_embeddings(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :]
                all_features.append(batch_features)
        
        # 连接所有批次
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            return all_features.cpu().numpy()
        return []