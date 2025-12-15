# app/core/graph_extract.py
import logging
from typing import List
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
# 导入图数据库连接
from app.core.graph_store import get_graph_store
from app.core.config import settings
import os
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class GraphExtractor:
    _llm = None
    _transformer = None

    @classmethod
    def _init_llm(cls):
        # 专门用于抽取的 LLM
        # 建议设置 temperature=0，让提取结果更稳定
        cls._llm = ChatOllama(
            base_url="http://localhost:11434",
            # 建议用 qwen2.5:7b 或 qwen2.5:1.5b
            # 7b 抽取效果更好，1.5b 速度更快
            model="qwen2.5:7b", 
            temperature=0,
        )
        
        # 初始化转换器
        # 你可以在这里限制允许的节点类型和关系类型，或者让它自由发挥
        cls._transformer = LLMGraphTransformer(
            llm=cls._llm,
            # allowed_nodes=["Mineral", "Rock", "Location", "Property"], # 可选：限制节点类型
            # allowed_relationships=["ASSOCIATED_WITH", "LOCATED_IN", "HAS_PROPERTY"], # 可选：限制关系
        )

    @classmethod
    def process_and_store(cls, chunks: List[Document]):
        """
        核心方法：提取 -> 存储
        """
        if cls._transformer is None:
            cls._init_llm()
            
        logger.info(f"⛏️ [Graph] 开始从 {len(chunks)} 个文本块中抽取知识 (这可能需要一点时间)...")
        
        try:
            # 1. LLM 抽取 (这一步最慢)
            # convert_to_graph_documents 会把 Document 列表转换成 GraphDocument 列表
            graph_documents = cls._transformer.convert_to_graph_documents(chunks) # type: ignore
            
            logger.info(f"🧩 [Graph] 抽取完成，准备写入 Neo4j...")
            
            # 2. 写入 Neo4j
            graph_store = get_graph_store()
            # include_source=True 会把原始文本作为属性存到节点里，方便溯源
            graph_store.add_graph_documents(
                graph_documents, 
                include_source=True
            )
            
            logger.info(f"✅ [Graph] 知识图谱入库成功！生成的节点和关系已保存。")
            
        except Exception as e:
            logger.error(f"❌ [Graph] 抽取或存储失败: {e}", exc_info=True)
            # 注意：图谱失败不应影响向量库的成功，所以这里只记录日志，不抛出异常中断流程

# 方便调用的函数
def extract_and_store_graph(chunks: List[Document]):
    return GraphExtractor.process_and_store(chunks)