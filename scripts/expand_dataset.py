#!/usr/bin/env python3
"""
数据集扩充脚本 - 基于现有问题生成更多地质问答
"""

import json
import random
from typing import Dict, List

def load_problems(file_path: str) -> Dict:
    """加载问题数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_variants(problem: Dict) -> List[Dict]:
    """为单个问题生成变体"""
    variants = []
    question = problem['question']
    
    # 问法变体模板
    question_templates = {
        '分类查询': [
            "{mineral}属于什么矿物分类？",
            "{mineral}的矿物分类是什么？", 
            "{mineral}被归类为哪种矿物？",
            "请问{mineral}是什么类型的矿物？"
        ],
        '晶系查询': [
            "{mineral}的晶系是？",
            "{mineral}属于哪个晶系？",
            "请问{mineral}是什么晶系？",
            "{mineral}的晶体系统是什么？"
        ],
        '共生查询': [
            "以下哪种矿物与{mineral}共伴生？",
            "{mineral}通常与哪种矿物共生？",
            "与{mineral}有共伴生关系的矿物是？",
            "{mineral}的共生矿物包括？"
        ]
    }
    
    # 识别问题类型
    query_type = None
    mineral_name = None
    
    if '属于什么矿物分类' in question or '分类' in question:
        query_type = '分类查询'
        # 提取矿物名称 (简化版)
        if '属于什么矿物分类' in question:
            mineral_name = question.replace('属于什么矿物分类？', '').strip()
    elif '晶系' in question:
        query_type = '晶系查询'
        mineral_name = question.replace('的晶系是？', '').strip()
    elif '共伴生' in question:
        query_type = '共生查询'
        # 从选择题中提取矿物名称
        import re
        match = re.search(r'与(.+?)共伴生', question)
        if match:
            mineral_name = match.group(1)
    
    if query_type and mineral_name and query_type in question_templates:
        templates = question_templates[query_type]
        for i, template in enumerate(templates[1:], 1):  # 跳过原始模板
            new_question = template.format(mineral=mineral_name)
            new_problem = problem.copy()
            new_problem['question'] = new_question
            # 生成新的ID
            original_id = problem.get('id', '0')
            new_problem['id'] = f"{original_id}_var{i}"
            variants.append(new_problem)
    
    return variants

def generate_caption(question: str) -> str:
    """根据问题生成caption"""
    if '共伴生' in question or '共生' in question:
        return "矿物共生关系问题"
    elif '晶系' in question:
        return "矿物晶系属性问题"
    elif '分类' in question or '属于' in question:
        return "矿物分类属性问题"
    elif '成分' in question or '化学式' in question:
        return "矿物成分问题"
    elif '硬度' in question or '比重' in question or '颜色' in question:
        return "矿物物理性质问题"
    elif '产地' in question or '分布' in question:
        return "矿物产地分布问题"
    else:
        return "地质矿物问题"

def expand_dataset(input_file: str, output_file: str, target_size: int = 200):
    """扩充数据集到目标大小"""
    problems = load_problems(input_file)
    expanded_problems = {}
    captions = {}
    
    # 保留原始问题
    for qid, problem in problems.items():
        problem['id'] = qid
        expanded_problems[qid] = problem
        captions[qid] = generate_caption(problem['question'])
    
    # 生成变体直到达到目标大小
    original_problems = list(problems.values())
    current_size = len(expanded_problems)
    variant_counter = 0
    
    while current_size < target_size:
        # 随机选择一个原始问题
        base_problem = random.choice(original_problems)
        variants = generate_variants(base_problem)
        
        for variant in variants:
            if current_size >= target_size:
                break
            
            # 生成唯一ID
            new_id = f"expanded_{variant_counter}"
            variant_counter += 1
            
            expanded_problems[new_id] = variant
            captions[new_id] = generate_caption(variant['question'])
            current_size += 1
    
    # 保存扩充后的数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_problems, f, indent=2, ensure_ascii=False)
    
    # 保存captions
    captions_file = output_file.replace('problems.json', 'captions.json')
    captions_data = {"captions": captions}
    with open(captions_file, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)
    
    print(f"数据集扩充完成:")
    print(f"  原始问题: {len(problems)}")
    print(f"  扩充后: {len(expanded_problems)}")
    print(f"  保存到: {output_file}")
    print(f"  Captions保存到: {captions_file}")

def create_splits(problems_file: str, splits_file: str, train_ratio: float = 0.7):
    """创建训练/测试/验证集划分"""
    with open(problems_file, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    qids = list(problems.keys())
    random.shuffle(qids)
    
    n_total = len(qids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * 0.15)
    
    splits = {
        "train": qids[:n_train],
        "val": qids[n_train:n_train + n_val],
        "test": qids[n_train + n_val:]
    }
    
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    
    print(f"数据集划分:")
    print(f"  训练集: {len(splits['train'])}")
    print(f"  验证集: {len(splits['val'])}")
    print(f"  测试集: {len(splits['test'])}")

if __name__ == "__main__":
    # 扩充mineral_test_kg数据集
    expand_dataset(
        input_file="dataset/mineral_test_kg/problems.json",
        output_file="dataset/mineral_expanded/problems.json",
        target_size=200
    )
    
    # 创建新的数据集划分
    create_splits(
        problems_file="dataset/mineral_expanded/problems.json",
        splits_file="dataset/mineral_expanded/pid_splits.json"
    )
