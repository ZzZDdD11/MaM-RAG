#!/usr/bin/env python3
"""
为Neo4j数据集创建包含所有数据的split
"""

import json

def create_all_split():
    """创建包含所有数据的split"""
    
    # 读取现有数据
    with open('dataset/mineral_neo4j_based/problems.json', 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    # 获取所有问题ID
    all_qids = list(problems.keys())
    
    # 创建新的split，将所有数据放入test
    new_splits = {
        "train": [],
        "val": [],
        "test": all_qids  # 所有150题都作为测试集
    }
    
    # 保存新的split
    with open('dataset/mineral_neo4j_based/pid_splits.json', 'w', encoding='utf-8') as f:
        json.dump(new_splits, f, indent=2)
    
    print(f"已创建新的数据集划分:")
    print(f"  训练集: {len(new_splits['train'])}")
    print(f"  验证集: {len(new_splits['val'])}")
    print(f"  测试集: {len(new_splits['test'])}")

if __name__ == "__main__":
    create_all_split()
