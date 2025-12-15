# app/api/routers/chat.py
import logging
import time
import uuid
import re
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from langchain_core.runnables import RunnableConfig

# 导入契约 (Schema)
from app.schemas.chat import ChatRequest, ChatResponse, SourceDocument
# 导入图谱实例
from app.core.gprah import app_graph
# 导入配置
from app.core.config import settings

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, summary="多源检索问答接口")
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    核心问答端点：
    1. 接收用户 Query 和 开关配置
    2. 触发 LangGraph 执行 (Decompose -> Retrieve -> Generate)
    3. 解析返回的非结构化证据为 SourceDocument 对象
    """
    req_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"[{req_id}] 收到请求: {body.query} | Config: Graph={body.enable_graph}, Web={body.enable_web}")

    try:
        # 1. 构造图初始状态 (Input State)
        # 必须与 app/core/graph.py 中的 AgentState 对应
        initial_state = {
            "original_query": body.query,
            "sub_queries": [],          
            "retrieved_contents": [],   
            "final_answer": ""          
        }

        # 2. 构造运行时配置 (Runtime Config)
        # 将请求参数通过 metadata 传递给 Graph 中的 Node 使用
        run_config: RunnableConfig = {
            "configurable": {"thread_id": req_id},
            "metadata": {
                "top_k": body.top_k,
                "enable_graph": body.enable_graph,
                "enable_web": body.enable_web
            }
        }

        # 3. 异步执行图
        # 使用 ainvoke 非阻塞调用
        final_state = await app_graph.ainvoke(initial_state, config=run_config) # type: ignore

        # 4. 提取结果
        answer = final_state.get("final_answer", "抱歉，未能生成回答。")
        raw_contents = final_state.get("retrieved_contents", [])
        sub_queries = final_state.get("sub_queries", [])

        # 5. 数据清洗与转换
        # 5.1 解析证据来源
        structured_sources = _parse_sources(raw_contents)
        
        # 5.2 构建推理轨迹 (Reasoning Trace)
        # 这里我们将图谱执行过程中的关键中间状态可视化给前端
        trace = _build_trace(body.query, sub_queries, structured_sources)

        # 6. 计算耗时并返回
        latency = round(time.time() - start_time, 3)
        
        return ChatResponse(
            answer=answer,
            sources=structured_sources,
            latency=latency,
            reasoning_trace=trace
        )

    except Exception as e:
        logger.error(f"[{req_id}] 处理异常: {str(e)}", exc_info=True)
        # 生产环境建议隐藏具体堆栈，仅返回通用错误信息
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# --- 辅助函数 ---

def _parse_sources(raw_contents: List[str]) -> List[SourceDocument]:
    """
    将 LangGraph 返回的字符串列表解析为结构化的 SourceDocument 对象。
    假设 raw_content 格式形如: "[Vector Source] Q:xxx\nContent:..."
    """
    parsed_docs = []
    
    for idx, raw_text in enumerate(raw_contents):
        source_type = "unknown"
        clean_content = raw_text
        score = None
        
        # 简单的规则提取来源类型
        if "[Vector Source]" in raw_text:
            source_type = "vector"
            # 去除标签，清理内容
            clean_content = raw_text.replace("[Vector Source]", "").strip()
        elif "[Graph Source]" in raw_text:
            source_type = "graph"
            clean_content = raw_text.replace("[Graph Source]", "").strip()
        elif "[Web Source]" in raw_text:
            source_type = "web"
            clean_content = raw_text.replace("[Web Source]", "").strip()
            
        # 尝试移除可能存在的 "Q:..." 前缀，保留纯 Content
        # (这取决于你在 graph node 里是怎么拼字符串的)
        if "Content:" in clean_content:
            parts = clean_content.split("Content:", 1)
            if len(parts) > 1:
                clean_content = parts[1].strip()

        # 构造对象
        doc = SourceDocument(
            source_type=source_type,
            content=clean_content,
            score=None,  # 如果后续你的检索器返回了分数，可以在这里解析
            metadata={"original_index": idx}
        )
        parsed_docs.append(doc)
        
    return parsed_docs

def _build_trace(original_query: str, sub_queries: List[str], sources: List[SourceDocument]) -> List[str]:
    """
    构建推理轨迹，告诉用户 Agent 做了什么
    """
    trace = []
    trace.append(f"1. 接收问题: '{original_query}'")
    
    if sub_queries:
        trace.append(f"2. 意图分解: 模型将其拆解为 {len(sub_queries)} 个子问题 -> {sub_queries}")
    else:
        trace.append("2. 意图分解: 保持原问题直接检索")
        
    # 统计来源分布
    source_counts = {"vector": 0, "graph": 0, "web": 0, "unknown": 0}
    for s in sources:
        source_counts[s.source_type] = source_counts.get(s.source_type, 0) + 1
    
    retrieval_summary = ", ".join([f"{k}={v}" for k, v in source_counts.items() if v > 0])
    if retrieval_summary:
        trace.append(f"3. 多源检索: 执行并行检索，共获取 {len(sources)} 条证据 ({retrieval_summary})")
    else:
        trace.append("3. 多源检索: 未找到相关信息")
        
    trace.append("4. 答案生成: 综合证据，生成最终回答")
    
    return trace