from typing import Literal, List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from app.core.config import settings


RouteTarget = Literal["vector", "graph", "web", "generate"]

class RouteQuery(BaseModel):
    """
    路由决策模型：决定将问题分发到哪些数据源。
    """
    datasources: List[RouteTarget] = Field(
        ...,
        description="Given a user question, choose one or more datasources to retrieve information from."
    )

class SemanticRouter:
    def __init__(self):
        # 路由任务相对简单，建议用小模型 (qwen2.5:1.5b) 以求极速
        # 如果没有 1.5b，用 7b 也行，就是稍慢一点点
        self.llm = ChatOllama(
            base_url="http://localhost:11434",
            model=settings.llm_model_name, 
            temperature=0,
        )
        # 绑定结构化输出
        self.structured_llm = self.llm.with_structured_output(RouteQuery)

    def route(self, question:str)-> List[str]: # type: ignore
        system_prompt = """你是一个智能的语义路由器。你的任务是将用户的问题分发到最合适的数据源。
        
        可选数据源：
        1. 'vector': 适用于具体的矿物定义、化学成分、理化性质、实验数据、具体开采技术等（查本地文档）。
        2. 'graph': 适用于查询实体间的关系、共生矿物、所属分类、层级结构等（查知识图谱）。
        3. 'web': 适用于最新的市场价格、行业新闻、产量排名、2024/2025年的实时信息（查互联网）。
        4. 'generate': 适用于简单的问候、感谢、通用常识或完全无关的问题（不检索，直接回答）。
        
        指导原则：
        - 如果问题涉及具体事实，优先选 'vector' 或 'graph'。
        - 如果问题涉及时间敏感信息（如“最近”、“今年”），必须选 'web'。
        - 如果不确定，可以选择多个源（例如既查 vector 又查 web）。
        """

        router_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human","{question}"),
        ])

        router_chain = router_prompt | self.structured_llm 
        try:
            print(f"🚦 [Router] 正在分析意图: {question}")
            result = router_chain.invoke({"question": question})
            
            # 打印决策结果，方便调试
            print(f"👉 [Router] 决策结果: {result.datasources}") # type: ignore
            return result.datasources # type: ignore
            
        except Exception as e:
            print(f"❌ [Router] 路由失败，默认回退到全量检索: {e}")
            # 兜底：如果出错，默认查本地文档和图谱
            return ["vector", "graph"]

# 单例
router = SemanticRouter()