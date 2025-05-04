# src/retrieval/text_retriever.py
import logging
from ..encoders.text_encoder import TextEncoder

logger = logging.getLogger(__name__)

class TextRetriever:
    def __init__(self, config, vector_db=None):
        """
        文本检索器，用于查询相关新闻文章
        
        Args:
            config: 配置信息
            vector_db: 可选，预加载的向量数据库实例
        """
        self.config = config
        self.vector_db = vector_db
        self.encoder = TextEncoder(config)
        self.top_k = config["retrieval"]["text_retrieval"]["top_k"]
        
    def load_database(self, index_name="news_articles"):
        """
        加载新闻文章向量数据库
        
        Args:
            index_name: 索引名称
            
        Returns:
            加载的向量数据库实例
        """
        if self.vector_db is None:
            from .vector_db import VectorDatabase
            self.vector_db = VectorDatabase(self.config)
            
        try:
            self.vector_db.load(index_name)
            logger.info(f"Loaded news article index: {index_name}")
            return self.vector_db
        except Exception as e:
            logger.error(f"Failed to load news article index: {e}")
            raise
    
    def retrieve(self, query_text, top_k=None):
        """
        根据文本查询相关新闻文章
        
        Args:
            query_text: 查询文本
            top_k: 可选，返回的结果数量
            
        Returns:
            相关文章元数据列表和相似度分数
        """
        if self.vector_db is None or self.vector_db.index is None:
            logger.warning("Vector database not loaded, attempting to load default news_articles index")
            self.load_database()
            
        if top_k is None:
            top_k = self.top_k
            
        # 编码查询文本
        query_vector = self.encoder.encode(query_text, return_tensors=False)
        
        # 执行向量搜索
        distances, indices = self.vector_db.search(query_vector, top_k)
        
        # 获取元数据
        metadata = self.vector_db.get_metadata(indices)
        
        # 计算相似度分数 (将距离转换为相似度)
        if distances[0] > 0:  # 如果是L2距离
            similarity_scores = [1.0 / (1.0 + dist) for dist in distances]
        else:  # 如果是内积距离 (已经是相似度)
            similarity_scores = [-dist for dist in distances]
            
        results = []
        if metadata:
            for i, (meta, score) in enumerate(zip(metadata, similarity_scores)):
                results.append({
                    "rank": i + 1,
                    "metadata": meta,
                    "score": float(score)
                })
                
        return results
