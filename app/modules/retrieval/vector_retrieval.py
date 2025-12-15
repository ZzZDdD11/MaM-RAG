from typing import List, Optional, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field, PrivateAttr
from langchain_milvus import Milvus

# 导入你的基础设施单例
from app.core.vector import get_vector_store
from app.core.rerank import rerank_documents # 假设你之前写好了 rerank



class MineralVectorRetriever(BaseRetriever):
    """
    基于 Milvus + BGE-Reranker 的企业级向量检索器。
    实现了 LangChain 标准接口。
    """
    # Pydantic 字段（对外暴露的配置参数）
    top_k: int = Field(3, description="最终返回给 LLM 的文档数量")
    search_k: int = Field(50, description="向量库初筛召回的数量")
    use_rerank: bool = Field(True, description="是否开启重排序")

    # --- 2. 声明内部私有属性 ---
    # 这告诉 Pydantic："_vector_store" 是我自己用的，你别管，也别尝试校验它
    _vector_store: Milvus = PrivateAttr()

    def __init__(self, **kwargs):
        """
        初始化方法
        """
        super().__init__(**kwargs)
        self._vector_store = get_vector_store()

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
        """
        同步检索逻辑：Milvus 粗排 -> BGE Rerank 精排
        """
        if self.use_rerank:
            initial_k = self.search_k
        else:
            initial_k = self.top_k

        docs = self._vector_store.similarity_search(query=query, k=initial_k)

        if not docs:
            return []
        
        doc_contents = [doc.page_content for doc in docs]
        ranked_results = rerank_documents(query, doc_contents, top_k=self.top_k)

        final_docs=[]

        for index, score in ranked_results:

            target_doc = docs[index]
            target_doc.metadata["rerank_score"] = score
            final_docs.append(target_doc)
        return final_docs