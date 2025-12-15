# app/schemas/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SourceDocument(BaseModel):
    source_type: str = Field(..., description="来源类型: vector/graph/web")
    content: str = Field(..., description="文档片段内容")
    score: Optional[float] = Field(None, description="相关性分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外信息")

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, example="石膏的用途是什么？") # type: ignore
    top_k: int = Field(3, ge=1, le=10)
    enable_graph: bool = True
    enable_web: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = []
    latency: float
    # 【修改点】添加 reasoning_trace，并给默认值 []
    reasoning_trace: List[str] = Field(default_factory=list, description="Agent的中间思考过程")