import json
import requests
import os
from typing import List, Dict

# 下载地址 (HotpotQA 官方验证集 distractor 版本)
DATA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
SAVE_PATH = "data/hotpot_dev.json"

def download_data():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(SAVE_PATH):
        print("📥 正在下载 HotpotQA 数据集 (约 40MB)...")
        response = requests.get(DATA_URL)
        with open(SAVE_PATH, "wb") as f:
            f.write(response.content)
        print("✅ 下载完成")
    else:
        print("✅ 数据集已存在")

def load_hotpot_samples(limit: int = 20) -> List[Dict]:
    """
    加载并解析 HotpotQA 数据
    返回结构:
    [
        {
            "question": "...",
            "answer": "...",
            "context_docs": ["doc_content_1", "doc_content_2"...] 
        },
        ...
    ]
    """
    download_data()
    
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    # 我们只取前 limit 个样本进行测试
    for item in data[:limit]:
        question = item["question"]
        answer = item["answer"]
        
        # HotpotQA 的 context 格式是: [ [title, [sent1, sent2...]], ... ]
        # 我们需要把它拼接成纯文本
        context_texts = []
        for ctx in item["context"]:
            title = ctx[0]
            sentences = "".join(ctx[1])
            # 拼成一段完整的文本，模拟 PDF 的一段
            full_text = f"Title: {title}\nContent: {sentences}"
            context_texts.append(full_text)
            
        samples.append({
            "question": question,
            "answer": answer,
            "context_docs": context_texts
        })
        
    print(f"✅ 已加载 {len(samples)} 条测试样本")
    return samples

if __name__ == "__main__":
    load_hotpot_samples(5)