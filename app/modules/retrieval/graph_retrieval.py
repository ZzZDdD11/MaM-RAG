# app/modules/retrieval/graph_retrieval.py
from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field, PrivateAttr

# 导入图数据库连接
from app.core.graph_store import get_graph_store
# 导入 LLM 用于提取实体
from langchain_ollama import ChatOllama
from app.core.config import settings

class MineralGraphRetriever(BaseRetriever):
    """
    基于 Neo4j 的子图检索器。
    逻辑：Query -> 提取实体 -> 查找子图 -> 返回关系文本
    """
    level: int = Field(1, description="图谱扩展深度 (1-hop 或 2-hop)")
    
    _graph: Any = PrivateAttr()
    _llm: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._graph = get_graph_store()
        # 初始化一个轻量级 LLM 用于提取实体 (可以用 1.5b 或 3b)
        self._llm = ChatOllama(
            base_url="http://localhost:11434",
            model=settings.llm_model_name, # 或者写死 "qwen2.5:1.5b" 以求速度
            temperature=0
        )

    def _extract_entities(self, query: str) -> List[str]:
        """
        利用 LLM 从问题中提取关键实体名称
        """
        prompt = f"""
        请从以下用户问题中提取关键实体（矿物名称、岩石名称、属性等）。
        只输出实体名称，用逗号分隔，不要有任何其他解释。
        
        问题: {query}
        实体:
        """
        try:
            response = self._llm.invoke(prompt)
            content = response.content.strip()
            # 处理分隔符 (中英文逗号)
            entities = [e.strip() for e in content.replace("，", ",").split(",") if e.strip()]
            return entities
        except Exception:
            return []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        同步检索逻辑
        """
        # 1. 提取实体
        entities = self._extract_entities(query)
        if not entities:
            return []
        
        # 2. 构造 Cypher 查询
        # 逻辑：找到名字包含这些实体的节点，并返回它们周围 1 跳的关系
        # (?i) 表示忽略大小写
        cypher_query = f"""
        MATCH (n)
        WHERE any(e IN $entities WHERE n.id CONTAINS e)
        MATCH (n)-[r]-(m)
        RETURN n.id AS source, type(r) AS rel, m.id AS target
        LIMIT 100
        """
        
        # 3. 执行查询
        try:
            results = self._graph.query(cypher_query, params={"entities": entities})
        except Exception as e:
            print(f"Graph query error: {e}")
            return []

        if not results:
            return []

        # 4. 格式化结果为 Document
        # 我们把每一条关系变成一个文档，或者把所有关系合并成一个文档
        # 这里选择合并成一个大文档，方便阅读
        triples = []
        for row in results:
            # 格式: 石膏 -[共生]-> 硬石膏
            triple = f"{row['source']} -[{row['rel']}]-> {row['target']}"
            triples.append(triple)
            
        # 去重
        triples = list(set(triples))
        
        full_text = "Graph Knowledge:\n" + "\n".join(triples)
        
        return [Document(
            page_content=full_text, 
            metadata={"source": "neo4j", "entities": entities}
        )]