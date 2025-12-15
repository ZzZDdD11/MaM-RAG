# tools/test_multihop.py
import asyncio, os
import requests

async def main():
    # 尝试用本地 LightRAG 实例调用（无需 LLM）
    try:
        from lightrag import LightRAG, QueryParam
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status
        except Exception:
            initialize_pipeline_status = None

        rag = LightRAG(working_dir="lightrag_store")
        if hasattr(rag, "initialize_storages"):
            await rag.initialize_storages()
        if initialize_pipeline_status is not None:
            await initialize_pipeline_status()

        label = "羟碳钠铍石"
        depth = 2
        max_nodes = 200

        print(f"[Local] get_knowledge_graph(label={label}, depth={depth}, max_nodes={max_nodes})")
        if hasattr(rag, "get_knowledge_graph"):
            subgraph = await rag.get_knowledge_graph(node_label=label, max_depth=depth, max_nodes=max_nodes)
            print(f"Nodes: {len(subgraph.get('nodes', []))}, Edges: {len(subgraph.get('edges', []))}")
        else:
            raise AttributeError("LightRAG.get_knowledge_graph not available, fallback to HTTP")

        # 简单再做一次 mix 查询（演示）
        try:
            param = QueryParam(mode="mix", top_k=3, enable_rerank=False)
            q = f"{label}的晶系是什么？"
            print(f"\n[Local] rag.query: {q}")
            res = await rag.query(q, param=param)
            print(str(res)[:400])
        except Exception as e:
            print(f"[Local] rag.query failed: {e}")

    except Exception as e:
        # 回退到 HTTP /graphs（服务需已在 9621 端口运行）
        print(f"[HTTP fallback] Reason: {e}")
        url = "http://127.0.0.1:9621/graphs"
        params = {"label": "羟碳钠铍石", "max_depth": 2, "max_nodes": 200}
        r = requests.get(url, params=params, timeout=30)
        print(f"HTTP {r.status_code}")
        print(r.text[:800])

if __name__ == "__main__":
    asyncio.run(main())