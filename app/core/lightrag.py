# app/core/rag.py
import logging
import os
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from app.core.config import settings
from lightrag.utils import EmbeddingFunc
logger = logging.getLogger(__name__)

class LightRAGService:
    _instance = None
    _initialized = False # 标记是否已完成异步初始化
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._init_rag()
        return cls._instance

    @staticmethod
    def _init_rag():
        logger.info(f"正在初始化全局 LightRAG 实例 (Dir: {settings.working_dir})...")
        
        # 1. 确保目录存在
        if not os.path.exists(settings.working_dir):
            os.makedirs(settings.working_dir)

        # 2. 定义 Embedding 函数 (复用之前的逻辑)
        async def _embedding_func(texts: list[str]):
            from langchain_community.llms.ollama import Ollama # 借用 langchain 的或者直接用 ollama 库
            # 这里保持和你之前 ingest 脚本一致的逻辑，使用 ollama 原生库
            return await ollama_embed(
                texts, 
                embed_model=settings.embedding_model, 
                host="http://localhost:11434"
            )
        embded_obj = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=_embedding_func

        )
        # 3. 初始化 LightRAG
        rag = LightRAG(
            working_dir=settings.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=settings.llm_model_name,
            embedding_func=embded_obj,
            addon_params={"embedding_batch_size": 4} 
        )
        logger.info("✅ 全局 LightRAG 初始化完成")
        return rag
    

    @classmethod
    async def initialize(cls):
        """
        🚀 关键修复：显式异步初始化存储
        必须在 FastAPI 启动时的 lifespan 中调用
        """
        if cls._initialized:
            return

        rag = cls.get_instance()
        logger.info("⚡️ 正在异步初始化 LightRAG 存储 (Storage & Pipeline)...")
        
        # 1. 初始化内部存储 (KV, VectorDB, GraphDB)
        if hasattr(rag, "initialize_storages"):
            await rag.initialize_storages()
        
        # 2. 初始化 Pipeline 状态 (新版 LightRAG 必需)
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status
            await initialize_pipeline_status()
        except ImportError:
            logger.warning("未找到 initialize_pipeline_status，跳过 (可能是旧版本)")
        except Exception as e:
            logger.error(f"Pipeline status 初始化失败: {e}")

        cls._initialized = True
        logger.info("✅ LightRAG 存储初始化完成！")


# 方便外部导入的单例获取函数
def get_rag():
    return LightRAGService.get_instance()