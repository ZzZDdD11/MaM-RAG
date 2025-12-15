import argparse
from collections import defaultdict
from neo4j import GraphDatabase
from neo4j.exceptions import ConfigurationError
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
import asyncio
try:
    from lightrag.utils import EmbeddingFunc  # 旧版才有
except Exception:
    EmbeddingFunc = None
CY = """
MATCH (m:Mineral)-[:`共伴生`]->(n:Mineral) RETURN m.name AS h, '共伴生' AS r, n.name AS t
UNION ALL
MATCH (m:Mineral)-[:`晶系`]->(c:`晶系晶类`) RETURN m.name AS h, '晶系' AS r, c.name AS t
UNION ALL
MATCH (m:Mineral)-[:`晶类`]->(c:`晶系晶类`) RETURN m.name AS h, '晶类' AS r, c.name AS t
UNION ALL
MATCH (m:Mineral)-[:`分类`]->(cat:`矿物类别`) RETURN m.name AS h, '分类' AS r, cat.name AS t
UNION ALL
MATCH (c:`晶系晶类`)-[:`属于`]->(p:`晶系晶类`) RETURN c.name AS h, '属于' AS r, p.name AS t
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--neo4j_uri', default='bolt://localhost:7687')
    ap.add_argument('--neo4j_user', default='neo4j')
    ap.add_argument('--neo4j_password', default='12345678')
    ap.add_argument('--neo4j_db', default='neo4j')
    ap.add_argument('--working_dir', required=True)
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        with driver.session(database=args.neo4j_db) as sess:
            rows = list(sess.run(CY))
    except ConfigurationError:
        # Bolt 3.0 不支持选择 DB，改用默认数据库会话
        with driver.session() as sess:
            rows = list(sess.run(CY))
    docs_by_head = defaultdict(list)
    for r in rows:
        h, rel, t = r['h'], r['r'], r['t']
        # 用自然语言句式，便于 LightRAG 从文本抽取图关系
        docs_by_head[h].append(f"{h} 的{rel} 是 {t}。")

    # 用本地 Ollama 嵌入（bge-m3）以保证可离线且维度匹配 LightRAG 服务配置
    import inspect
    import logging
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    lr_kwargs = {"working_dir": args.working_dir}
    sig = inspect.signature(LightRAG)
    if "llm_model_func" in sig.parameters:
        lr_kwargs["llm_model_func"] = ollama_model_complete
    if "llm_model_name" in sig.parameters:
        lr_kwargs["llm_model_name"] = "qwen2.5:7b"
    # 使用 bge-m3 嵌入模型（维度 1024），匹配 LightRAG 服务配置
    if "embedding_function" in sig.parameters:
        lr_kwargs["embedding_function"] = lambda texts: ollama_embed(
            texts, embed_model="bge-m3", host="http://localhost:11434"
        )
        if "embedding_dim" in sig.parameters:
            lr_kwargs["embedding_dim"] = 1024
    elif "embedding_func" in sig.parameters and EmbeddingFunc is not None:
        lr_kwargs["embedding_func"] = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="bge-m3", host="http://localhost:11434"
            ),
        )
    
    # 增加超时时间和并发设置
    if "worker_timeout" in sig.parameters:
        lr_kwargs["worker_timeout"] = 300  # 从60s改为300s
    if "max_async_workers" in sig.parameters:
        lr_kwargs["max_async_workers"] = 1  # 减少并发数，避免超时
    if "max_llm_async_workers" in sig.parameters:
        lr_kwargs["max_llm_async_workers"] = 1
    try:
        rag = LightRAG(**lr_kwargs)
        logger.info(f"✅ LightRAG 初始化成功，工作目录: {args.working_dir}")
    except TypeError as e:
        # 兜底最小构造
        logger.warning(f"⚠️ LightRAG 初始化参数不匹配，使用最小构造: {e}")
        rag = LightRAG(working_dir=args.working_dir)

    # 初始化 LightRAG 存储（新版本必需）
    async def _init_lightrag():
        await rag.initialize_storages()
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()

    try:
        asyncio.run(_init_lightrag())
    except RuntimeError:
        # 若已有事件循环在运行，使用新循环执行
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_init_lightrag())
        finally:
            loop.close()

    cnt = 0
    total_minerals = len(docs_by_head)
    logger.info(f"开始写入 {total_minerals} 个矿物到 LightRAG...")
    
    for h, lines in docs_by_head.items():
        doc = "\n".join(lines)
        try:
            rag.insert(doc)  # LightRAG 会基于文本构建/更新其向量与图信息
            cnt += 1
            if cnt % 10 == 0:
                logger.info(f"✅ 已插入 {cnt}/{total_minerals} 个矿物文档")
        except Exception as e:
            logger.error(f"❌ 插入矿物 {h} 失败: {e}")

    logger.info(f"✅ 完成！共插入 {cnt} 个矿物文档到 {args.working_dir}")

if __name__ == "__main__":
    main()