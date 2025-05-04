# src/pipelines/task1_pipeline.py
import logging
from ..retrieval.text_retriever import TextRetriever
from ..generation.generator import QwenGenerator
from ..utils.prompt_templates import PromptTemplates
from ..encoders.image_encoder import ImageEncoder

logger = logging.getLogger(__name__)

class EventEnhancedImageDescriptionPipeline:
    """
    任务1: 事件增强图像描述任务的处理流水线
    """
    
    def __init__(self, config):
        """
        初始化流水线
        
        Args:
            config: 配置信息
        """
        self.config = config
        
        # 初始化组件
        logger.info("Initializing Task 1 pipeline components")
        self.image_encoder = ImageEncoder(config)
        self.text_retriever = TextRetriever(config)
        self.generator = QwenGenerator(config)
        
        # 加载文本检索数据库
        try:
            self.text_retriever.load_database("news_articles")
        except Exception as e:
            logger.warning(f"Could not load news article database: {e}")
            logger.warning("Text retrieval will not be available until database is built")
    
    def process(self, image, use_cot=True, top_k=None):
        """
        处理单个图像，生成事件增强描述
        
        Args:
            image: 输入图像 (PIL图像或路径)
            use_cot: 是否使用Chain-of-Thought提示策略
            top_k: 检索的新闻文章数量
            
        Returns:
            生成的图像描述文本
        """
        logger.info("Starting Task 1 pipeline processing")
        
        # 步骤1: 编码图像
        logger.debug("Encoding input image")
        image_vector = self.image_encoder.encode(image, return_tensors=False)
        
        # 步骤2: 通过图像向量检索相关新闻文章
        logger.debug("Retrieving related news articles")
        try:
            retrieved_articles = self.text_retriever.retrieve(image_vector, top_k=top_k)
            logger.debug(f"Retrieved {len(retrieved_articles)} related articles")
        except Exception as e:
            logger.warning(f"Article retrieval failed: {e}")
            retrieved_articles = []
        
        # 步骤3: 生成描述文本
        logger.debug("Generating enhanced description")
        if use_cot:
            # 使用Chain-of-Thought提示策略
            prompt = PromptTemplates.task1_cot_image_description(retrieved_articles)
            response = self.generator.generate(image=image, prompt=prompt)
        else:
            # 使用标准提示策略
            prompt = PromptTemplates.task1_image_description(retrieved_articles)
            response = self.generator.generate(image=image, prompt=prompt, contexts=retrieved_articles)
        
        logger.info("Task 1 pipeline processing completed")
        return response
