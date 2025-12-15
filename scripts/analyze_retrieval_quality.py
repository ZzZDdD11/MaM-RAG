#!/usr/bin/env python3
"""
检索质量分析脚本 - 分析向量检索vs图谱检索的贡献
"""

import json
import re
from collections import defaultdict

def analyze_retrieval_logs(log_file: str):
    """分析检索日志，统计各检索源的贡献"""
    
    stats = {
        'vector_hits': 0,
        'graph_hits': 0, 
        'kg_first_wins': 0,
        'llm_fusion_wins': 0,
        'query_types': defaultdict(int),
        'retrieval_quality': defaultdict(list)
    }
    
    # 这里需要解析实际的日志文件
    # 由于没有具体日志，提供分析框架
    
    print("检索质量分析结果:")
    print(f"向量检索命中: {stats['vector_hits']}")
    print(f"图谱检索命中: {stats['graph_hits']}")
    print(f"KG-first策略获胜: {stats['kg_first_wins']}")
    print(f"LLM融合策略获胜: {stats['llm_fusion_wins']}")

def compare_retrieval_results(multi_source_file: str, graph_only_file: str, vector_only_file: str):
    """对比不同检索方式的结果"""
    
    with open(multi_source_file, 'r', encoding='utf-8') as f:
        multi_results = json.load(f)
    
    with open(graph_only_file, 'r', encoding='utf-8') as f:
        graph_results = json.load(f)
        
    with open(vector_only_file, 'r', encoding='utf-8') as f:
        vector_results = json.load(f)
    
    # 分析哪些问题向量检索有帮助
    vector_helps = []
    graph_sufficient = []
    
    for qid in multi_results:
        multi_ans = multi_results[qid]
        graph_ans = graph_results.get(qid, -1)
        vector_ans = vector_results.get(qid, -1)
        
        if multi_ans == graph_ans and graph_ans != vector_ans:
            graph_sufficient.append(qid)
        elif multi_ans != graph_ans and multi_ans == vector_ans:
            vector_helps.append(qid)
    
    print(f"\n检索贡献分析:")
    print(f"图谱检索足够的问题: {len(graph_sufficient)} ({graph_sufficient})")
    print(f"向量检索有帮助的问题: {len(vector_helps)} ({vector_helps})")
    
    return {
        'graph_sufficient': graph_sufficient,
        'vector_helps': vector_helps
    }

if __name__ == "__main__":
    # 对比现有结果
    compare_retrieval_results(
        "outputs/multi_source_full_test.json",
        "outputs/graph_only_test.json", 
        "outputs/vector_only_test.json"
    )
