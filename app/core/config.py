from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):

    # LLM
    
    # =========================================================
    # 核心路径与模型配置 (对应 Agent 初始化需求)
    # =========================================================
    project_name: str = "M-RAG Service"
    
    # 兼容 MRetrievalAgent 和 VectorRetrieval 的 config.working_dir
    working_dir: str = Field("lightrag_store", description="LightRAG 存储目录")
    
    # 兼容 config.llm_model_name
    llm_model_name: str = Field("qwen2.5:7b", description="LLM 模型名称")
    
    # 兼容 config.debug_dump_dir (SummaryAgent 需要)
    debug_dump_dir: str = "outputs/debug_runs"

    embedding_model: str = "BAAI/bge-m3"
    
    # =========================================================
    # 检索开关与参数 (对应 MRetrievalAgent 初始化逻辑)
    # =========================================================
    # 必须小写，否则 getattr(config, 'disable_vector') 会失败
    disable_vector: bool = False
    disable_graph: bool = False
    enable_web: bool = False  # 对应 config.enable_web
    
    top_k: int = 4            # 对应 config.top_k
    mode: str = "mix"         # 对应 config.mode (VectorRetrieval 使用)
    
    # 选项列表 (SummaryAgent 需要 config.options)
    options: List[str] = ["A", "B", "C", "D", "E"]
    
    # =========================================================
    # 外部服务凭证
    # =========================================================
    # 对应 WebRetrieval 的 config.serper_api_key
    serper_api_key: Optional[str] = None
    
    # 对应 GraphRetrieval 的 config.neo4j_*
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "12345678"
    neo4j_db: str = "neo4j"
    
    # 词表映射路径 (GraphRetrieval 可能用到)
    term_map_path: Optional[str] = None

    # =========================================================
    # Pydantic 配置
    # =========================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # 允许从大写环境变量读取 (例如环境变量定义的 DISABLE_VECTOR 会自动赋值给 disable_vector)
        case_sensitive=False, 
        extra="ignore"
    )

# 实例化单例
settings = Settings() # type: ignore