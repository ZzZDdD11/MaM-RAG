#!/usr/bin/env python3
"""
修复Neo4j数据集格式，添加缺失的split字段
"""

import json
import os

def fix_neo4j_dataset():
    """修复Neo4j数据集格式"""
    
    dataset_dir = "dataset/mineral_neo4j_based"
    
    # 读取现有数据
    with open(f"{dataset_dir}/problems.json", 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    with open(f"{dataset_dir}/pid_splits.json", 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    # 为每个问题添加split字段
    for qid, problem in problems.items():
        if qid in splits["train"]:
            problem["split"] = "train"
        elif qid in splits["val"]:
            problem["split"] = "val"
        elif qid in splits["test"]:
            problem["split"] = "test"
        else:
            problem["split"] = "train"  # 默认
        
        # 添加其他必需字段
        if "image" not in problem:
            problem["image"] = ""
        if "id" not in problem:
            problem["id"] = qid
    
    # 保存修复后的数据
    with open(f"{dataset_dir}/problems.json", 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    
    print(f"修复完成:")
    print(f"  总问题数: {len(problems)}")
    print(f"  训练集: {len(splits['train'])}")
    print(f"  验证集: {len(splits['val'])}")
    print(f"  测试集: {len(splits['test'])}")

if __name__ == "__main__":
    fix_neo4j_dataset()
