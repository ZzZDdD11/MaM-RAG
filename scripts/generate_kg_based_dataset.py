#!/usr/bin/env python3
"""
基于现有知识图谱生成测试数据集
从LightRAG存储的知识图谱中提取关系，生成高质量的测试问题
"""

import json
import random
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict

def extract_kg_relations(kg_file_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """从知识图谱文件中提取关系三元组"""
    relations = defaultdict(list)
    
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 解析每个chunk中的关系
    for chunk_id, chunk_data in data.items():
        content = chunk_data.get('content', '')
        
        # 提取关系三元组：实体1 的关系 是 实体2
        pattern = r'(\S+?)\s+的(\S+?)\s+是\s+(\S+?)。'
        matches = re.findall(pattern, content)
        
        for match in matches:
            entity1, relation, entity2 = match
            relations[relation].append((entity1, entity2))
    
    return relations

def generate_questions_from_relations(relations: Dict[str, List[Tuple[str, str]]], 
                                    target_size: int = 100) -> List[Dict]:
    """基于关系生成问题"""
    questions = []
    question_id = 0
    
    # 问题模板
    templates = {
        '共伴生': [
            "以下哪种矿物与{entity}共伴生？",
            "{entity}通常与哪种矿物共生？",
            "与{entity}有共伴生关系的矿物是？"
        ],
        '晶系': [
            "{entity}的晶系是？",
            "{entity}属于哪个晶系？",
            "请问{entity}是什么晶系？"
        ],
        '晶类': [
            "{entity}的晶类是？",
            "{entity}属于哪个晶类？",
            "请问{entity}是什么晶类？"
        ],
        '分类': [
            "{entity}属于什么矿物分类？",
            "{entity}的矿物分类是什么？",
            "{entity}被归类为哪种矿物？"
        ]
    }
    
    # 为每种关系类型生成问题
    for relation_type, entity_pairs in relations.items():
        if relation_type not in templates:
            continue
            
        # 按实体分组，找出每个实体的所有相关值
        entity_to_values = defaultdict(set)
        for entity1, entity2 in entity_pairs:
            entity_to_values[entity1].add(entity2)
        
        # 为每个实体生成问题
        for entity, values in entity_to_values.items():
            if len(values) == 0:
                continue
                
            # 随机选择一个模板
            template = random.choice(templates[relation_type])
            question = template.format(entity=entity)
            
            # 生成选项
            correct_answer = random.choice(list(values))
            
            # 从其他实体的值中选择干扰项
            all_other_values = set()
            for other_entity, other_values in entity_to_values.items():
                if other_entity != entity:
                    all_other_values.update(other_values)
            
            # 移除正确答案，避免重复
            all_other_values.discard(correct_answer)
            
            if len(all_other_values) < 3:
                # 如果干扰项不够，跳过这个问题
                continue
            
            # 随机选择3个干扰项
            distractors = random.sample(list(all_other_values), min(3, len(all_other_values)))
            
            # 构建选项
            choices = [correct_answer] + distractors
            random.shuffle(choices)
            
            # 找到正确答案的索引
            correct_index = choices.index(correct_answer)
            
            # 创建问题
            question_data = {
                'id': str(question_id),
                'question': question,
                'choices': choices,
                'answer': correct_index,
                'hint': f"这是一个关于{entity}的{relation_type}问题",
                'lecture': f"根据知识图谱，{entity}的{relation_type}是{correct_answer}。",
                'image': None,
                'relation_type': relation_type,
                'entity': entity
            }
            
            questions.append(question_data)
            question_id += 1
            
            if len(questions) >= target_size:
                break
        
        if len(questions) >= target_size:
            break
    
    return questions[:target_size]

def create_balanced_dataset(questions: List[Dict], output_dir: str):
    """创建平衡的数据集"""
    
    # 按关系类型分组
    by_relation = defaultdict(list)
    for q in questions:
        by_relation[q['relation_type']].append(q)
    
    print("数据集统计:")
    for rel_type, qs in by_relation.items():
        print(f"  {rel_type}: {len(qs)}个问题")
    
    # 转换为所需格式
    problems = {}
    for q in questions:
        problems[q['id']] = {
            'question': q['question'],
            'choices': q['choices'],
            'answer': q['answer'],
            'hint': q['hint'],
            'lecture': q['lecture'],
            'image': q['image']
        }
    
    # 保存问题
    problems_file = f"{output_dir}/problems.json"
    with open(problems_file, 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    
    # 生成captions
    captions = {}
    for q in questions:
        if q['relation_type'] == '共伴生':
            captions[q['id']] = "矿物共生关系问题"
        elif q['relation_type'] == '晶系':
            captions[q['id']] = "矿物晶系属性问题"
        elif q['relation_type'] == '晶类':
            captions[q['id']] = "矿物晶类属性问题"
        elif q['relation_type'] == '分类':
            captions[q['id']] = "矿物分类属性问题"
        else:
            captions[q['id']] = "地质矿物问题"
    
    captions_file = f"{output_dir}/captions.json"
    with open(captions_file, 'w', encoding='utf-8') as f:
        json.dump({"captions": captions}, f, indent=2, ensure_ascii=False)
    
    # 创建数据集划分 (70% train, 15% val, 15% test)
    qids = list(problems.keys())
    random.shuffle(qids)
    
    n_total = len(qids)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    splits = {
        "train": qids[:n_train],
        "val": qids[n_train:n_train + n_val],
        "test": qids[n_train + n_val:]
    }
    
    splits_file = f"{output_dir}/pid_splits.json"
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n数据集生成完成:")
    print(f"  总问题数: {len(problems)}")
    print(f"  训练集: {len(splits['train'])}")
    print(f"  验证集: {len(splits['val'])}")
    print(f"  测试集: {len(splits['test'])}")
    print(f"  保存到: {output_dir}")

def analyze_kg_coverage(kg_file_path: str):
    """分析知识图谱的覆盖情况"""
    relations = extract_kg_relations(kg_file_path)
    
    print("知识图谱关系统计:")
    total_triples = 0
    for relation_type, pairs in relations.items():
        print(f"  {relation_type}: {len(pairs)}个三元组")
        total_triples += len(pairs)
    
    print(f"  总计: {total_triples}个三元组")
    
    # 统计实体数量
    all_entities = set()
    for pairs in relations.values():
        for entity1, entity2 in pairs:
            all_entities.add(entity1)
            all_entities.add(entity2)
    
    print(f"  唯一实体数: {len(all_entities)}")
    
    return relations

if __name__ == "__main__":
    import os
    
    try:
        # 分析现有知识图谱
        kg_file = "lightrag_store/kv_store_text_chunks.json"
        print("分析现有知识图谱...")
        
        if not os.path.exists(kg_file):
            print(f"错误: 文件 {kg_file} 不存在")
            exit(1)
            
        relations = analyze_kg_coverage(kg_file)
        
        if not relations:
            print("错误: 没有找到任何关系")
            exit(1)
        
        # 生成基于KG的测试数据集
        print("\n生成测试数据集...")
        questions = generate_questions_from_relations(relations, target_size=150)
        
        if not questions:
            print("错误: 没有生成任何问题")
            exit(1)
        
        # 创建输出目录
        output_dir = "dataset/mineral_kg_based"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据集
        create_balanced_dataset(questions, output_dir)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
