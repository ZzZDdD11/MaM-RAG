# app/core/vector.py
import logging
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    _instance = None
    _embeddings = None

    @classmethod
    def get_embeddings(cls) -> HuggingFaceEmbeddings:
        """获取 HuggingFace Embedding 模型单例"""
        if cls._embeddings is None:
            # 这里的 model_name 可以是 HuggingFace Hub ID (如 "BAAI/bge-m3")
            # 也可以是本地下载好的模型路径
            model_name = settings.embedding_model # 确保 config.py 里配的是 "BAAI/bge-m3" 或本地路径
            
            logger.info(f"⚡️ 正在加载 HuggingFace Embedding 模型: {model_name} ...")
            
            # encode_kwargs={'normalize_embeddings': True} 对于某些模型（如 BGE）很重要
            cls._embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}, # 如果有显卡改成 'cuda'
                encode_kwargs={'normalize_embeddings': True} 
            )
            logger.info("✅ Embedding 模型加载完成")
            
        return cls._embeddings

    @classmethod
    def get_instance(cls) -> Milvus:
        """获取 Milvus 向量库实例"""
        if cls._instance is None:
            logger.info("🔌 正在连接 Milvus 向量数据库...")
            
            # 获取 Embedding 实例
            embeddings = cls.get_embeddings()
            
            # 初始化 Milvus
            # 注意：collection_name 建议用英文，避免潜在的编码问题
            cls._instance = Milvus(
                embedding_function=embeddings,
                collection_name="mineral_rag_collection",
                connection_args={
                    "uri": "http://localhost:19530", # Milvus 默认端口
                    # 如果设置了用户名密码:
                    # "user": "root",
                    # "password": "..." 
                },
                # 启用自动 ID 生成 (这对 LangChain 来说通常比较方便)
                auto_id=True,
                # 确保持久化数据
                drop_old=False 
            )
            
            logger.info("✅ Milvus 连接成功")
            
        return cls._instance

# 工厂函数
def get_vector_store() -> Milvus:
    return VectorStoreService.get_instance()

def get_embeddings() -> HuggingFaceEmbeddings:
    return VectorStoreService.get_embeddings()