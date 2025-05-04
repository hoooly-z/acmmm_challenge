
# src/encoders/text_encoder.py
import torch
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class TextEncoder:
    def __init__(self, config):
        """
        文本编码器类
        
        Args:
            config: 配置信息，包含模型路径等参数
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        try:
            logger.info(f"Loading text encoder model from {config['model']['model_path']}")
            # 使用Qwen2.5-VL的文本编码器
            self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_path"])
            self.model = AutoModel.from_pretrained(
                config["model"]["model_path"],
                device_map=config["model"]["device_map"],
                torch_dtype=self._get_precision(config["model"]["precision"])
            )
            logger.info("Text encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text encoder: {e}")
            raise
    
    def _get_precision(self, precision):
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        return torch.float32
    
    def encode(self, text, return_tensors=True):
        """
        编码单个文本
        
        Args:
            text: 输入文本字符串
            return_tensors: 是否返回张量，否则返回numpy数组
            
        Returns:
            文本的特征向量
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_text_embeddings(**inputs)
            # 获取[CLS]令牌或池化的表示
            text_features = outputs.last_hidden_state[:, 0, :]
        
        if not return_tensors:
            text_features = text_features.cpu().numpy()
            
        return text_features
    
    def batch_encode(self, texts, batch_size=16):
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            文本特征向量列表
        """
        all_features = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
                
            # 批量处理
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_text_embeddings(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :]
                all_features.append(batch_features)
        
        # 连接所有批次
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            return all_features.cpu().numpy()
        return []