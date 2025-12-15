from langchain_community.graphs import Neo4jGraph
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class GraphStoreService:
    _instance = None

    @classmethod
    def get_instance(cls) -> Neo4jGraph:
        if cls._instance is None:
            logger.info("正在连接 Neo4j图数据库")

            try:
                cls._instance = Neo4jGraph(
                    url= settings.neo4j_uri,
                    username=settings.neo4j_user,
                    password=settings.neo4j_password
                )

                cls._instance.refresh_schema()

                logger.info("Neo4j 连接成功")
            except Exception as e:
                logger.error(f"Neo4j连接失败:{e}")
                raise e
            
        return cls._instance
    
# 工厂函数
def get_graph_store() -> Neo4jGraph:
    return GraphStoreService.get_instance()

    