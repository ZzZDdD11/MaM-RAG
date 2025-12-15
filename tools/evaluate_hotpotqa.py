import sys
import os
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_ollama import ChatOllama, OllamaEmbeddings
from load_hotpotqa import load_hotpot_samples

# 配置裁判
judge_llm = ChatOllama(model="qwen2.5:7b", temperature=0)
judge_embeddings = OllamaEmbeddings(model="bge-m3")

# 1. 加载刚才入库的那几条数据
# 必须和入库时的 limit 保持一致，或者是其子集
LIMIT = 5
test_data = load_hotpot_samples(LIMIT)

API_URL = "http://localhost:8000/v1/chat"

data_samples = {
    'question': [],
    'answer': [],
    'contexts': [],
    'ground_truth': []
}

print("\n🚀 开始 HotpotQA 挑战赛...")

for item in test_data:
    q = item["question"]
    truth = item["answer"]
    
    print(f"\n❓ 问题: {q}")
    print(f"✅ 答案: {truth}")
    
    # 2. 调用 Agent
    # 关闭 Web 搜索，因为我们要测的是内部检索能力 (Vector + Graph)
    # 如果开了 Web，它直接去谷歌搜答案了，就测不出我们架构的水平了
    try:
        response = requests.post(API_URL, json={
            "query": q,
            "enable_vector": True,
            "enable_graph": True, 
            "enable_web": False # 🔴 关掉联网！只测内功！
        }).json()
        
        ans = response.get("final_answer", "")
        print(f"🤖 回答: {ans}")
        
        # 提取上下文
        source_list = response.get("sources", [])
        ctxs = [src["content"] for src in source_list]
        
        data_samples['question'].append(q)
        data_samples['answer'].append(ans)
        data_samples['contexts'].append(ctxs)
        data_samples['ground_truth'].append(truth)
        
    except Exception as e:
        print(f"❌ 错误: {e}")

# 3. Ragas 评分
print("\n⚖️ 裁判打分中...")
dataset = Dataset.from_dict(data_samples)
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=judge_llm,
    embeddings=judge_embeddings
)

print("\n🏆 HotpotQA 成绩单:")
print(results)