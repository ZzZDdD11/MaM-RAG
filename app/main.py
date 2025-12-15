# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from app.core.config import settings
from app.api.routers import chat,ingest  # 导入刚才写的路由模块
#from agents.multi_retrieval_agents import MRetrievalAgent
from app.core.gprah import app_graph
#from app.core.lightrag import LightRAGService
# 配置日志
logging.basicConfig(level=logging.INFO if not settings.debug_dump_dir else logging.DEBUG)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动阶段 ---
    logger.info(f"🚀 {settings.project_name} 正在启动...")
    logger.info(f"配置信息: Working Dir={settings.working_dir}, LLM={settings.llm_model_name}")
    
    try:

        logger.info("✅ 新架构 (Milvus + Neo4j + LangGraph) 就绪")
    except Exception as e:
        logger.error(f"❌ 引擎初始化失败: {e}")
        # 这里可以选择 raise e 让服务启动失败，或者保留 app.state.agent = None
        app.state.agent = None
        
    yield
    
    # --- 关闭阶段 ---
    logger.info("🛑 服务正在关闭...")
    # 如果 agent 有 close() 方法，可以在这里调用
    # if app.state.agent:
    #     app.state.agent.close()

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.project_name, 
        lifespan=lifespan
    )
    
    # 注册路由
    # prefix="/v1" 意味着接口地址是 http://localhost:8000/v1/chat
    app.include_router(chat.router, prefix="/v1", tags=["Chat"])
    app.include_router(ingest.router,prefix="/v1", tags=["Ingest"])
    return app

app = create_app()

# 开发模式下运行：
# uvicorn app.main:app --reload