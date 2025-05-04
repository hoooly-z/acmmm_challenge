# src/pipelines/task2_pipeline.py
import logging
from ..retrieval.image_retriever import ImageRetriever
from ..generation.generator import QwenGenerator
from ..utils.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class EventDrivenImageRetrievalPipeline:
    """
    任务2: 基于事件的图像检索任务的处理流水线
    """
    
    def __init__(self, config):
        """
        初始化流水线
        
        Args:
            config: 配置信息
        """
        self.config = config
        
        # 初始化组件
        logger.info("Initializing Task 2 pipeline components")
        self.image_retriever = ImageRetriever(config)
        self.generator = QwenGenerator(config)
        
        # 加载图像检索数据库
        try:
            self.image_retriever.load_database("news_images")
        except Exception as e:
            logger.warning(f"Could not load news image database: {e}")
            logger.warning("Image retrieval will not be available until database is built")
    
    def process(self, event_description, use_cot=True, top_k=None):
        """
        处理单个事件描述，检索相关图像
        
        Args:
            event_description: 输入的事件描述文本
            use_cot: 是否使用Chain-of-Thought提示策略
            top_k: 检索的图像数量
            
        Returns:
            检索到的图像元数据列表
        """
        logger.info("Starting Task 2 pipeline processing")
        
        # 步骤1: 分析事件描述，提取关键元素（可选）
        if use_cot:
            logger.debug("Analyzing event description with CoT")
            prompt = PromptTemplates.task2_cot_image_retrieval()
            analysis = self.generator.generate(prompt=prompt + "\n\n事件描述: " + event_description)
            logger.debug(f"Event analysis: {analysis}")
        
        # 步骤2: 检索相关图像
        logger.debug("Retrieving related images")
        try:
            retrieved_images = self.image_retriever.retrieve_by_text(event_description, top_k=top_k)
            logger.debug(f"Retrieved {len(retrieved_images)} related images")
        except Exception as e:
            logger.error(f"Image retrieval failed: {e}")
            retrieved_images = []
        
        logger.info("Task 2 pipeline processing completed")
        return retrieved_images