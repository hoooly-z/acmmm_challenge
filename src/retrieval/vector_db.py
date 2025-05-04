# src/retrieval/vector_db.py
import os
import numpy as np
import faiss
import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, config):
        """
        向量数据库类，支持FAISS索引
        
        Args:
            config: 配置信息，包含索引路径等参数
        """
        self.config = config
        self.index_path = config["retrieval"]["vector_db"]["index_path"]
        self.dimension = config["retrieval"]["vector_db"]["dimension"]
        self.index = None
        self.metadata = []  # 存储向量对应的元数据
        
        os.makedirs(self.index_path, exist_ok=True)
    
    def create_index(self, index_type="L2"):
        """
        创建FAISS索引
        
        Args:
            index_type: 索引类型，如"L2", "IP"(内积)等
            
        Returns:
            创建的索引对象
        """
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        logger.info(f"Created FAISS index with dimension {self.dimension}, type {index_type}")
        return self.index
    
    def add_vectors(self, vectors, metadata=None):
        """
        向索引添加向量
        
        Args:
            vectors: numpy数组，形状为(n, dimension)
            metadata: 可选，与向量关联的元数据列表
            
        Returns:
            添加的向量数量
        """
        if self.index is None:
            logger.warning("Index not initialized. Creating default L2 index.")
            self.create_index("L2")
            
        # 确保向量是float32类型
        vectors = np.array(vectors, dtype=np.float32)
        
        # 添加到索引
        self.index.add(vectors)
        
        # 保存元数据
        if metadata:
            self.metadata.extend(metadata)
            
        logger.info(f"Added {len(vectors)} vectors to index")
        return len(vectors)
    
    def search(self, query_vector, k=5):
        """
        在索引中搜索最相似的向量
        
        Args:
            query_vector: 查询向量，形状为(1, dimension)
            k: 返回的最相似向量数量
            
        Returns:
            (距离列表, 索引列表) 元组
        """
        if self.index is None:
            raise ValueError("Index not initialized or empty")
            
        # 确保查询向量是float32类型且形状正确
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        # 执行搜索
        distances, indices = self.index.search(query_vector, k)
        
        logger.debug(f"Search completed, found {len(indices[0])} results")
        return distances[0], indices[0]
    
    def get_metadata(self, indices):
        """
        获取指定索引的元数据
        
        Args:
            indices: 索引列表
            
        Returns:
            元数据列表
        """
        if not self.metadata:
            return None
            
        return [self.metadata[i] for i in indices if 0 <= i < len(self.metadata)]
    
    def save(self, index_name="vector_index"):
        """
        保存索引和元数据
        
        Args:
            index_name: 索引文件名前缀
            
        Returns:
            保存的文件路径
        """
        if self.index is None:
            raise ValueError("No index to save")
            
        # 保存FAISS索引
        index_file = os.path.join(self.index_path, f"{index_name}.faiss")
        faiss.write_index(self.index, index_file)
        
        # 保存元数据
        meta_file = os.path.join(self.index_path, f"{index_name}_meta.pkl")
        with open(meta_file, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        logger.info(f"Saved index to {index_file} and metadata to {meta_file}")
        return index_file, meta_file
    
    def load(self, index_name="vector_index"):
        """
        加载索引和元数据
        
        Args:
            index_name: 索引文件名前缀
            
        Returns:
            加载的索引对象
        """
        # 加载FAISS索引
        index_file = os.path.join(self.index_path, f"{index_name}.faiss")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        self.index = faiss.read_index(index_file)
        
        # 加载元数据
        meta_file = os.path.join(self.index_path, f"{index_name}_meta.pkl")
        if os.path.exists(meta_file):
            with open(meta_file, 'rb') as f:
                self.metadata = pickle.load(f)
                
        logger.info(f"Loaded index from {index_file} with {self.index.ntotal} vectors")
        return self.index

