# app/core/graph.py
import operator
import logging
from typing import Annotated, List, TypedDict, Dict, Any
from app.modules.retrieval.graph_retrieval import MineralGraphRetriever
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from app.modules.retrieval.vector_retrieval import MineralVectorRetriever
# 导入配置单例
from app.core.config import settings
from app.modules.retrieval.web_retrieval import MineralWebRetriever
# 导入现有的业务组件
from app.modules.generation.answer_generator import generator
from app.core.router import router
# from legacy.vector_retrieval import VectorRetrieval
# from legacy.graph_retrieval import GraphRetrieval

# 配置日志
logger = logging.getLogger(__name__)

# --- 1. 定义状态 (State) ---
# 这里的字段必须覆盖 app/api/routers/chat.py 中 initial_state 的所有 key
class AgentState(TypedDict):
    original_query: str
    sub_queries: List[str]
    # 使用 operator.add 实现列表自动合并 (并行检索时不会互相覆盖)
    retrieved_contents: Annotated[List[str], operator.add]
    final_answer: str
    routes: List[str]

# --- 2. 初始化工具实例 ---
# 我们利用全局 settings 初始化单例，避免每次请求都重新加载模型
try:
    logger.info("正在初始化 Graph 组件...")
    _decompose_agent = DecomposeAgent(settings)
    _summary_agent = SummaryAgent(settings)
    
    # 根据配置决定是否初始化检索器 (虽然这里初始化了，但在 Node 中我们会再次检查请求级开关)
    # _vector_retriever = VectorRetrieval(settings)
    # _graph_retriever = GraphRetrieval(settings)
    logger.info("Graph 组件初始化完成")
except Exception as e:
    logger.error(f"Graph 组件初始化失败: {e}")
    # 这里不抛出异常，允许服务启动，但在调用时可能会报错
    _decompose_agent = None
    _summary_agent = None
    _vector_retriever = None
    _graph_retriever = None

# --- 3. 定义节点 (Nodes) ---

def node_decompose(state: AgentState, config: RunnableConfig):
    """
    节点：问题分解
    """
    query = state["original_query"]
    # 从 metadata 获取可能的 trace_id 用于日志
    tid = config.get("configurable", {}).get("thread_id", "N/A")
    logger.info(f"[{tid}] [Node: Decompose] 处理: {query}")
    
    if not _decompose_agent:
        return {"sub_queries": [query]}

    try:
        # 调用原有逻辑
        sub_queries = _decompose_agent.decompose(query)
        # 归一化为列表
        if isinstance(sub_queries, str):
            sub_queries = [sub_queries]
        return {"sub_queries": sub_queries}
    except Exception as e:
        logger.error(f"分解失败: {e}")
        return {"sub_queries": [query]}

def node_vector_search(state: AgentState, config: RunnableConfig):
    """
    节点：向量检索
    """
    # 1. 获取运行时配置 (来自 API 请求)
    meta = config.get("metadata", {})
    # 默认为 True，除非显式设为 False
    if meta.get("enable_vector", True) is False: 
        return {"retrieved_contents": []}
    # 获取动态参数
    top_k = meta.get("top_k", 3)
    # [修改点 2]: 实例化你的新检索器
    # 思考：这里需要传哪些参数？提示：看你在 MineralVectorRetriever 里定义的 Field
    retriever = MineralVectorRetriever(
        top_k=top_k,
        use_rerank=True,      # 默认开启重排序
        search_k=top_k * 10   # 自动设定粗排数量
    )

    queries = state["sub_queries"]
    top_k = meta.get("top_k", 3)
    
    results = []

    for q in queries:
        docs = retriever.invoke(q)
        for i, doc in enumerate(docs):
            score = doc.metadata.get("rerank_score", 0)
            # 构造字符串
            formatted = f"[Vector Source] (Score:{score:.2f})\n Content:{doc.page_content}"
            results.append(formatted)
    return {"retrieved_contents": results}


def node_graph_search(state: AgentState, config: RunnableConfig):
    """
    节点：图谱检索 (升级版)
    """
    meta = config.get("metadata", {})
    if meta.get("enable_graph", True) is False:
        return {"retrieved_contents": []}

    # 实例化检索器
    retriever = MineralGraphRetriever(level=1)
    
    queries = state["sub_queries"]
    results = []
    
    for q in queries:
        try:
            # 调用 invoke
            docs = retriever.invoke(q)
            
            for doc in docs:
                # 加上 [Graph Source] 标记
                formatted = f"[Graph Source] (Entities: {doc.metadata.get('entities')})\nContent: {doc.page_content}"
                results.append(formatted)
                
        except Exception as e:
            logger.error(f"图谱检索出错: {e}")
            
    return {"retrieved_contents": results}

def node_web_search(state: AgentState, config: RunnableConfig):
    """
    节点：联网检索
    """
    meta = config.get("metadata", {})
    # 检查开关，默认关闭 (False)，因为联网比较慢
    if meta.get("enable_web", False) is False:
        return {"retrieved_contents": []}

    # 实例化检索器
    retriever = MineralWebRetriever(top_k=3)
    
    queries = state["sub_queries"]
    results = []
    
    # 通常联网搜索只需要搜原始问题，或者第一个子问题
    # 搜太多会被封 IP，所以这里我们只搜第一个 query
    target_query = queries[0] if queries else state["original_query"]
    
    try:
        # 调用 invoke
        docs = retriever.invoke(target_query)
        
        for doc in docs:
            # 格式化输出
            formatted = f"[Web Source] ({doc.metadata.get('source')})\nContent: {doc.page_content}"
            results.append(formatted)
            
    except Exception as e:
        print(f"联网检索失败: {e}")
            
    return {"retrieved_contents": results}

# app/core/graph.py

def node_generate(state: AgentState):
    """
    节点：生成回答 (Final Synthesis)
    """
    query = state["original_query"]
    contexts = state["retrieved_contents"]
    routes = state.get("routes", []) # 获取路由结果
    
    # 🔍 核心修改：判断逻辑
    
    # 情况 1: 如果路由器明确说是 'generate' (闲聊)，直接走闲聊模式
    if "generate" in routes:
        answer = generator.chitchat(query)
        return {"final_answer": answer}

    # 情况 2: 如果路由器想查，但没查到东西 (Context 为空)
    if not contexts:
        return {"final_answer": "抱歉，经过多源检索（向量、图谱、网络），我没有找到任何与您问题相关的信息。"}

    # 情况 3: 有上下文，走 RAG 模式
    try:
        answer = generator.generate(query, contexts)
        return {"final_answer": answer}
        
    except Exception as e:
        return {"final_answer": f"生成过程中发生错误: {e}"}
    
def node_router(state: AgentState):
    """
    第一站：分析用户意图
    """
    question = state["original_query"]
    # 调用路由器
    decision = router.route(question)
    
    # 这里的 sub_queries 暂时直接用原问题
    # (如果保留之前的 Decompose 逻辑，可以把 Decompose 放在 Router 之后)
    return {
        "routes": decision, 
        "sub_queries": [question] 
    }

def route_decision(state: AgentState):
    """
    交通指挥官：根据 State 中的 routes 决定下一步去哪里
    返回的是一个 list，LangGraph 会并发执行这些节点
    """
    routes = state["routes"]
    next_nodes = []
    
    if "vector" in routes:
        next_nodes.append("vector_search")
    if "graph" in routes:
        next_nodes.append("graph_search")
    if "web" in routes:
        next_nodes.append("web_search")
        
    # 如果列表为空 (比如 routes=['generate'])，或者没选中任何检索源
    if not next_nodes:
        return ["generate"] # 直接去生成
        
    return next_nodes

# --- 4. 构建工作流 (Graph Construction) ---

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("router_node", node_router)
workflow.add_node("decompose", node_decompose)
workflow.add_node("vector_search", node_vector_search)
workflow.add_node("graph_search", node_graph_search)
workflow.add_node("web_search", node_web_search) # 需要时取消注释
workflow.add_node("generate", node_generate)

# 定义流程
# 1. 
workflow.set_entry_point("router_node")
# 2. 
workflow.add_conditional_edges(
    "router_node",
    route_decision,
    # 映射字典 (可选，但写上更规范)
    {
        "vector_search": "vector_search",
        "graph_search": "graph_search",
        "web_search": "web_search",
        "generate": "generate"
    }
)

# 4. 汇聚 (检索节点 -> 生成节点)
workflow.add_edge("vector_search", "generate")
workflow.add_edge("graph_search", "generate")
workflow.add_edge("web_search", "generate")

# 5. 终点
workflow.add_edge("generate", END)


# 编译应用
app_graph = workflow.compile()