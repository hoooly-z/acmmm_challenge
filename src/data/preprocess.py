# src/data/preprocess.py
import os
import json
import logging
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
from ..encoders.image_encoder import ImageEncoder
from ..encoders.text_encoder import TextEncoder
from ..retrieval.vector_db import VectorDatabase

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    OpenEvents数据预处理类
    """
    
    def __init__(self, config):
        """
        初始化预处理器
        
        Args:
            config: 配置信息
        """
        self.config = config
        self.data_path = config["data"]["openevents_path"]
        self.news_path = config["data"]["news_articles_path"]
        self.images_path = config["data"]["images_path"]
        self.annotations_path = config["data"]["annotations_path"]
        
        # 创建所需目录
        for path in [self.data_path, self.news_path, self.images_path, self.annotations_path]:
            os.makedirs(path, exist_ok=True)
        
        # 初始化编码器
        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        
    def process_news_articles(self, articles_file, max_articles=None):
        """
        处理新闻文章数据
        
        Args:
            articles_file: 新闻文章JSON文件路径
            max_articles: 处理的最大文章数（用于测试）
            
        Returns:
            处理的文章数量
        """
        logger.info(f"Processing news articles from {articles_file}")
        
        try:
            # 读取文章数据
            with open(articles_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            if max_articles:
                articles = articles[:max_articles]
                
            # 创建向量数据库
            vector_db = VectorDatabase(self.config)
            vector_db.create_index("IP")  # 使用内积相似度
            
            # 处理每篇文章
            processed_count = 0
            for article in tqdm(articles, desc="Processing articles"):
                try:
                    # 提取文章信息
                    article_id = article.get("id", str(processed_count))
                    title = article.get("title", "")
                    content = article.get("content", "")
                    
                    if not content:
                        continue
                    
                    # 分段处理长文本
                    paragraphs = self._split_text(content)
                    
                    # 为每个段落编码
                    for i, para in enumerate(paragraphs):
                        # 添加标题信息
                        if i == 0 and title:
                            para_text = f"{title}\n\n{para}"
                        else:
                            para_text = para
                            
                        # 编码段落文本
                        vector = self.text_encoder.encode(para_text, return_tensors=False)
                        
                        # 添加到向量数据库
                        metadata = {
                            "id": f"{article_id}_{i}",
                            "article_id": article_id,
                            "title": title,
                            "content": para,
                            "paragraph_idx": i
                        }
                        
                        vector_db.add_vectors([vector], [metadata])
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing article: {e}")
                    continue
            
            # 保存索引
            vector_db.save("news_articles")
            logger.info(f"Processed {processed_count} articles, saved to news_articles index")
            return processed_count
            
        except Exception as e:
            logger.error(f"Failed to process news articles: {e}")
            raise
    
    def process_news_images(self, images_dir, annotations_file=None, max_images=None):
        """
        处理新闻图像数据
        
        Args:
            images_dir: 图像目录路径
            annotations_file: 可选，图像标注文件路径
            max_images: 处理的最大图像数（用于测试）
            
        Returns:
            处理的图像数量
        """
        logger.info(f"Processing news images from {images_dir}")
        
        try:
            # 获取图像文件列表
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            
            if max_images:
                image_files = image_files[:max_images]
            
            # 加载标注（如果有）
            annotations = {}
            if annotations_file and os.path.exists(annotations_file):
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations_data = json.load(f)
                    # 转换为以图像ID为键的字典
                    for item in annotations_data:
                        img_id = item.get("image_id")
                        if img_id:
                            annotations[img_id] = item
            
            # 创建向量数据库
            vector_db = VectorDatabase(self.config)
            vector_db.create_index("IP")  # 使用内积相似度
            
            # 处理每张图像
            processed_count = 0
            for i, img_file in enumerate(tqdm(image_files, desc="Processing images")):
                try:
                    # 加载图像
                    img_path = os.path.join(images_dir, img_file)
                    image = Image.open(img_path).convert("RGB")
                    
                    # 获取图像ID
                    img_id = os.path.splitext(img_file)[0]
                    
                    # 编码图像
                    vector = self.image_encoder.encode(image, return_tensors=False)
                    
                    # 获取标注（如果有）
                    annotation = annotations.get(img_id, {})
                    caption = annotation.get("caption", "")
                    
                    # 添加到向量数据库
                    metadata = {
                        "id": img_id,
                        "file_name": img_file,
                        "path": img_path,
                        "caption": caption
                    }
                    
                    vector_db.add_vectors([vector], [metadata])
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_file}: {e}")
                    continue
            
            # 保存索引
            vector_db.save("news_images")
            logger.info(f"Processed {processed_count} images, saved to news_images index")
            return processed_count
            
        except Exception as e:
            logger.error(f"Failed to process news images: {e}")
            raise
    
    def _split_text(self, text, max_length=512):
        """
        将长文本分割为段落
        
        Args:
            text: 输入文本
            max_length: 每段最大长度
            
        Returns:
            段落列表
        """
        # 按段落分割
        paragraphs = text.split("\n\n")
        
        # 合并短段落，分割长段落
        result = []
        current = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current) + len(para) <= max_length:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    result.append(current)
                
                # 处理超长段落
                if len(para) > max_length:
                    words = para.split()
                    current = ""
                    for word in words:
                        if len(current) + len(word) + 1 <= max_length:
                            current = current + " " + word if current else word
                        else:
                            result.append(current)
                            current = word
                    if current:
                        result.append(current)
                        current = ""
                else:
                    current = para
        
        if current:
            result.append(current)
            
        return result