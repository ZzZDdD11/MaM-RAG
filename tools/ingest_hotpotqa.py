import sys
import os
from langchain_core.documents import Document

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from tools.load_hotpotqa import load_hotpot_samples
from app.core.vector import get_vector_store
from app.core.graph_extract import extract_and_store_graph

def ingest_hotpot_data(limit=10):
    # 1. 加载数据
    samples = load_hotpot_samples(limit)
    
    vector_store = get_vector_store()
    
    print(f"🚀 开始将 {limit} 条 HotpotQA 数据的上下文入库...")
    print("⚠️ 警告：这将调用 LLM 进行图谱抽取，速度较慢，请耐心等待...")

    total_docs = []
    
    for i, sample in enumerate(samples):
        print(f"\n--- 处理第 {i+1}/{limit} 个问题上下文 ---")
        
        # 将字符串转为 Document 对象
        chunks = [
            Document(page_content=txt, metadata={"source": "hotpotqa", "question_id": i}) 
            for txt in sample["context_docs"]
        ]
        
        # 2. 向量入库
        print(f"💾 [Vector] 存入 Milvus ({len(chunks)} chunks)...")
        vector_store.add_documents(chunks)
        
        # 3. 图谱抽取与入库
        # HotpotQA 的核心就在这里！看看 LLM 能不能把 Wiki 里的实体关系抽出来
        print(f"⛏️ [Graph] 抽取图谱知识...")
        extract_and_store_graph(chunks)
        
    print("\n🎉 入库完成！现在你的数据库里已经有了 Wikipedia 的知识。")

if __name__ == "__main__":
    # 先跑 5 个试试水，别贪多，否则跑一天
    ingest_hotpot_data(limit=5)