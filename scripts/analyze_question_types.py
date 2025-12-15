#!/usr/bin/env python3
"""
分析不同问题类型的准确率
"""

import json

def analyze_question_types():
    """按问题类型分析准确率"""
    
    # 读取数据
    with open('dataset/mineral_neo4j_based/problems.json', 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    with open('dataset/mineral_neo4j_based/captions.json', 'r', encoding='utf-8') as f:
        captions = json.load(f)['captions']
    
    # 读取三种检索模式的结果
    results = {}
    result_files = {
        '双源检索': 'outputs/neo4j_all_150_test_test.json',
        '仅图谱': 'outputs/neo4j_graph_only_test.json', 
        '仅向量': 'outputs/neo4j_vector_only_test.json'
    }
    
    for mode, file_path in result_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results[mode] = json.load(f)
        except FileNotFoundError:
            print(f"警告: 文件 {file_path} 不存在")
            results[mode] = {}
    
    # 按问题类型分组
    type_groups = {
        '矿物共生关系问题': [],
        '矿物晶系属性问题': [],
        '矿物晶类属性问题': [],
        '矿物分类属性问题': [],
        '其他': []
    }
    
    for qid, problem in problems.items():
        caption = captions.get(qid, '其他')
        if caption not in type_groups:
            caption = '其他'
        type_groups[caption].append(qid)
    
    # 分析每种类型的准确率
    print("=" * 80)
    print("问题类型准确率分析")
    print("=" * 80)
    
    for question_type, qids in type_groups.items():
        if not qids:
            continue
            
        print(f"\n【{question_type}】({len(qids)}题)")
        print("-" * 60)
        
        for mode, result_data in results.items():
            if not result_data:
                continue
                
            correct = sum(1 for qid in qids if result_data.get(qid, 0) != 0)
            total = len(qids)
            accuracy = correct / total if total > 0 else 0
            
            print(f"{mode:>8}: {correct:>3}/{total:<3} = {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # 总体统计
    print(f"\n{'='*80}")
    print("总体准确率对比")
    print("=" * 80)
    
    for mode, result_data in results.items():
        if not result_data:
            continue
            
        correct = sum(1 for v in result_data.values() if v != 0)
        total = len(result_data)
        accuracy = correct / total if total > 0 else 0
        
        print(f"{mode:>8}: {correct:>3}/{total:<3} = {accuracy:.4f} ({accuracy*100:.1f}%)")

if __name__ == "__main__":
    analyze_question_types()
