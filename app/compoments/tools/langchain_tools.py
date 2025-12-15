# tools/langchain_tools.py
from langchain_core.tools import tool
from legacy.vector_retrieval import VectorRetrieval
from legacy.graph_retrieval import GraphRetrieval
from legacy.web_retrieval import WebRetrieval
import os

# 假设 config 是一个简单的对象，这里为了演示简化处理
class MockConfig:
    def __init__(self):
        self.working_dir = "lightrag_store" # 你的数据路径
        self.top_k = 3
        self.mode = "mix"
        self.llm_model_name = "qwen2.5:7b"
        self.serper_api_key = os.environ.get("SERPER_API_KEY", "")

# 初始化原始检索器实例 (单例模式，避免重复加载)
config = MockConfig()
_vector_retriever = VectorRetrieval(config)
_graph_retriever = GraphRetrieval(config)
# _web_retriever = WebRetrieval(config) # 如果你有 key 可以开启

@tool
def vector_search_tool(query: str) -> str:
    """
    利用向量数据库检索非结构化的文本信息。
    适用于查询具体的描述、定义、或者文档片段。
    """
    try:
        return _vector_retriever.find_top_k(query)
    except Exception as e:
        return f"Vector search error: {e}"

@tool
def graph_search_tool(query: str) -> str:
    """
    利用知识图谱检索结构化的关系信息。
    适用于查询矿物的共生关系、晶系、分类等属性，或者是多跳推理问题。
    """
    try:
        return _graph_retriever.find_top_k(query)
    except Exception as e:
        return f"Graph search error: {e}"

# 将工具列表打包
def get_retrieval_tools():
    return [vector_search_tool, graph_search_tool]