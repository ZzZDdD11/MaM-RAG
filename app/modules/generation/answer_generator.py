import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from app.core.config import settings

logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        self.llm = ChatOllama(
            base_url="http://localhost:11434",
            model=settings.llm_model_name,
            temperature=0.1
        )

    def _format_context(self, retrieval_content: List[str]) -> str:
        
        if retrieval_content is None:
            return "无相关检索结果"
        
        formatted_content = []
        for i,content in enumerate(retrieval_content):
            formatted_content.append(f"--- 证据{i+1} ---\n{content}")

        return "\n\n".join(formatted_content)
    
    def generate(self, query:str, retrieval_content: List[str]) -> str:

        # 格式化上下文
        context = self._format_context(retrieval_content)
        
        system_prompt = """你是一个专业、严谨的矿物地质学专家助手。你的任务是基于提供的【检索上下文】回答用户的【问题】。

        ### 来源优先级说明：
        1. **[Graph Source] (知识图谱)**：最权威。涉及矿物分类、共生关系、晶系等结构化知识时，优先采信。
        2. **[Vector Source] (本地文档)**：非常可靠。涉及具体描述、性质定义、实验数据时，优先采信。
        3. **[Web Source] (互联网)**：补充参考。主要用于回答最新的数据、新闻或本地知识库中没有的概念。

        ### 回答要求：
        1. **基于证据**：严格根据上下文回答，不要编造。如果上下文没有相关信息，请直接说“根据现有知识库无法回答”。
        2. **结构清晰**：使用 Markdown 格式，分点作答。
        3. **标注来源**：在关键结论后面，尽量尝试标注来源，例如 "(参考图谱)" 或 "(参考文档)"。
        4. **融合信息**：不要把三个来源割裂开，要将它们融合成一段通顺的文字。
        5. **语言风格**：学术、客观、简洁。

        ### 检索上下文：
        {context}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            ("human","{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        # 5. 执行
        try:
            return chain.invoke({"context": context, "question": query})
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return "抱歉，生成回答时发生系统错误。"
        pass
        
    def chitchat(self, query: str) -> str:
        """
        闲聊模式：不依赖检索结果，直接用 LLM 自身知识回答
        """
        logger.info(f"🗣️ [Generate] 进入闲聊模式: {query}")
        
        system_prompt = """你是一个友好、专业的矿物地质学专家助手。
        当前用户的问题不需要查阅资料，请直接用你自己的知识库，以自然、流畅的语气进行对话。
        如果用户是在问候（如“你好”），请礼貌回应并简要介绍自己（我是MineralRAG助手）。
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        
        try:
            return chain.invoke({"question": query})
        except Exception as e:
            logger.error(f"闲聊生成失败: {e}")
            return "你好！我是 MineralRAG 助手，很高兴为您服务。"
generator = AnswerGenerator()