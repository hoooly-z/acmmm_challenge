# src/data/dataset.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class OpenEventsDataset(Dataset):
    """
    OpenEvents数据集类
    """
    
    def __init__(self, config, split="train", task="image_description"):
        """
        初始化数据集
        
        Args:
            config: 配置信息
            split: 数据集分割（train/val/test）
            task: 任务类型（image_description/image_retrieval）
        """
        self.config = config
        self.split = split
        self.task = task
        self.data_path = config["data"]["openevents_path"]
        self.images_path = config["data"]["images_path"]
        self.annotations_path = config["data"]["annotations_path"]
        
        # 加载数据
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples for {task} task ({split} split)")
    
    def _load_samples(self):
        """
        加载数据样本
        
        Returns:
            数据样本列表
        """
        # 加载标注文件
        annotation_file = os.path.join(self.annotations_path, f"{self.split}_{self.task}.json")
        
        if not os.path.exists(annotation_file):
            logger.warning(f"Annotation file not found: {annotation_file}")
            return []
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # 处理样本
            samples = []
            for item in annotations:
                if self.task == "image_description":
                    # 任务1: 图像描述
                    sample = {
                        "image_id": item.get("image_id", ""),
                        "image_path": os.path.join(self.images_path, f"{item.get('image_id', '')}.jpg"),
                        "caption": item.get("caption", "")
                    }
                else:
                    # 任务2: 图像检索
                    sample = {
                        "event_description": item.get("event_description", ""),
                        "image_ids": item.get("image_ids", []),
                        "image_paths": [os.path.join(self.images_path, f"{img_id}.jpg") 
                                        for img_id in item.get("image_ids", [])]
                    }
                
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load samples: {e}")
            return []
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            样本数据
        """
        sample = self.samples[idx]
        
        if self.task == "image_description":
            # 任务1: 图像描述
            try:
                image_path = sample["image_path"]
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                else:
                    # 使用空白图像替代
                    logger.warning(f"Image not found: {image_path}")
                    image = Image.new("RGB", (224, 224), (255, 255, 255))
                    
                return {
                    "image": image,
                    "image_id": sample["image_id"],
                    "caption": sample["caption"]
                }
            except Exception as e:
                logger.error(f"Error loading image {sample['image_path']}: {e}")
                # 返回空白图像
                return {
                    "image": Image.new("RGB", (224, 224), (255, 255, 255)),
                    "image_id": sample["image_id"],
                    "caption": sample["caption"]
                }
        else:
            # 任务2: 图像检索
            return {
                "event_description": sample["event_description"],
                "image_ids": sample["image_ids"]
            }

def get_dataloader(config, split="train", task="image_description", batch_size=None):
    """
    获取数据加载器
    
    Args:
        config: 配置信息
        split: 数据集分割
        task: 任务类型
        batch_size: 批量大小（可选）
        
    Returns:
        数据加载器
    """
    if batch_size is None:
        batch_size = config["data"]["batch_size"]
        
    dataset = OpenEventsDataset(config, split, task)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config["compute"]["num_workers"]
    )
    
    return dataloader