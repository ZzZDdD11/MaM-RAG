from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field
# 导入 DuckDuckGo 搜索工具
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

class MineralWebRetriever(BaseRetriever):
    """
    基于 DuckDuckGo 的联网检索器。
    不需要 API Key，即插即用。
    """
    top_k: int = Field(5, description="搜索结果数量")
    
    # 私有属性
    _search_tool: DuckDuckGoSearchResults = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化搜索包装器
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=self.top_k)
        # backend="news" 可以搜新闻，默认搜网页
        self._search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        同步执行搜索
        """
        try:
            #1. 执行搜索
            # DuckDuckGoSearchResults 返回的是一个字典列表的字符串或者是对象列表
            # 我们需要处理一下返回结果
            raw_results = self._search_tool.invoke(query)

            docs = []
            if isinstance(raw_results, list):
                for res in raw_results:
                    # res 通常包含: {'snippet': '...', 'title': '...', 'link': '...'}
                    content = f"Title: {res.get('title')}\nSnippet: {res.get('snippet')}"
                    metadata = {"source": res.get('link'), "type": "web"}
                    
                    docs.append(Document(page_content=content, metadata=metadata))
            
            return docs
        except Exception as e:
            # 联网搜索很容易超时或报错，一定要捕获异常，不要让整个系统崩溃
            print(f"Web search error: {e}")
            return []