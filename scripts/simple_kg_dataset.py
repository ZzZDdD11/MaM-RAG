#!/usr/bin/env python3
"""
简化版基于知识图谱的数据集生成器
"""

import json
import random
import re
import os
from collections import defaultdict

def main():
    print("开始生成基于知识图谱的数据集...")
    
    # 手动提取一些已知的关系数据（基于我们之前看到的）
    kg_relations = {
        # 共伴生关系
        "水硫硝镍铝石": {
            "共伴生": ["硫硝铝镍石", "石膏", "硅镁铀矿", "纤铁矿", "针铁矿", "赤铜矿"],
            "晶系": "单斜晶系",
            "分类": "硝酸盐"
        },
        "铵硝石": {
            "共伴生": ["磷铵石", "鸟粪石", "迪磷镁铵石", "六水铵镁矾", "钾石膏", "石膏"],
            "晶系": "斜方晶系",
            "晶类": "斜方双锥晶类",
            "分类": "硝酸盐"
        },
        "钾硝石": {
            "共伴生": ["钠硝石", "钠硝矾", "石膏", "泻利盐", "水镁硝石", "水钙硝石"],
            "晶系": "斜方晶系",
            "晶类": "斜方双锥晶类",
            "分类": "硝酸盐"
        },
        "碳铈铜铀矿": {
            "共伴生": ["橙红铀矿", "碳钙钕铀矿", "水碳钇铀石", "硅钙铀矿", "晶质铀矿"],
            "晶系": "六方晶系",
            "分类": "碳酸盐"
        },
        "碳钙钕铀矿": {
            "共伴生": ["橙红铀矿", "水碳钇铀石", "硅钙铀矿", "晶质铀矿", "碳铈铜铀矿"],
            "晶系": "单斜晶系",
            "分类": "碳酸盐"
        },
        "阿碳钾铀矿": {
            "共伴生": ["纤碳铀矿", "孔雀石", "一水蓝铜矾", "胆矾"],
            "晶系": "单斜晶系",
            "晶类": "斜方柱晶类",
            "分类": "碳酸盐"
        },
        "氟碳钙铀矿": {
            "共伴生": ["白云石", "板碳铀矿"],
            "晶系": "单斜晶系",
            "晶类": "平行双面晶类",
            "分类": "碳酸盐"
        },
        "氧碳钙铀矿": {
            "共伴生": ["硅钙铀矿", "黄钡铀矿"],
            "晶系": "斜方晶系",
            "晶类": "斜方柱晶类",
            "分类": "碳酸盐"
        },
        "巴纤碳铀矿": {
            "共伴生": ["孔雀石", "羟胆矾", "蓝铜矿", "纤碳铀矿", "硫磷铝铁铀矿", "硅钾铀矿", "石膏"],
            "晶系": "六方晶系",
            "分类": "碳酸盐"
        },
        "水碳镁钙石": {
            "共伴生": ["泻利盐", "石膏", "孔雀石", "羟胆矾"],
            "分类": "碳酸盐"
        }
    }
    
    # 生成问题
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
    
    # 收集所有可能的答案用作干扰项
    all_coexist = set()
    all_crystal_systems = set()
    all_crystal_classes = set()
    all_classifications = set()
    
    for entity, props in kg_relations.items():
        if "共伴生" in props:
            all_coexist.update(props["共伴生"])
        if "晶系" in props:
            all_crystal_systems.add(props["晶系"])
        if "晶类" in props:
            all_crystal_classes.add(props["晶类"])
        if "分类" in props:
            all_classifications.add(props["分类"])
    
    # 为每个实体生成问题
    for entity, properties in kg_relations.items():
        for prop_type, prop_value in properties.items():
            if prop_type not in templates:
                continue
            
            # 选择模板
            template = random.choice(templates[prop_type])
            question = template.format(entity=entity)
            
            # 确定正确答案和干扰项池
            if prop_type == '共伴生':
                if isinstance(prop_value, list) and len(prop_value) > 0:
                    correct_answer = random.choice(prop_value)
                    distractor_pool = all_coexist - set(prop_value)
                else:
                    continue
            else:
                correct_answer = prop_value
                if prop_type == '晶系':
                    distractor_pool = all_crystal_systems - {correct_answer}
                elif prop_type == '晶类':
                    distractor_pool = all_crystal_classes - {correct_answer}
                elif prop_type == '分类':
                    distractor_pool = all_classifications - {correct_answer}
                else:
                    continue
            
            # 生成干扰项
            if len(distractor_pool) < 3:
                # 如果干扰项不够，添加一些通用的
                if prop_type == '晶系':
                    distractor_pool.update(["三斜晶系", "四方晶系", "等轴晶系", "三方晶系"])
                elif prop_type == '分类':
                    distractor_pool.update(["硫化物", "氧化物", "硅酸盐", "卤化物"])
                elif prop_type == '共伴生':
                    distractor_pool.update(["方解石", "石英", "长石", "云母"])
            
            distractor_pool.discard(correct_answer)
            
            if len(distractor_pool) < 3:
                continue
            
            distractors = random.sample(list(distractor_pool), min(3, len(distractor_pool)))
            
            # 构建选项
            choices = [correct_answer] + distractors
            random.shuffle(choices)
            
            # 找到正确答案的索引
            correct_index = choices.index(correct_answer)
            
            # 创建问题
            question_data = {
                'question': question,
                'choices': choices,
                'answer': correct_index,
                'hint': f"这是一个关于{entity}的{prop_type}问题",
                'lecture': f"根据知识图谱，{entity}的{prop_type}是{correct_answer}。",
                'image': None
            }
            
            questions.append((str(question_id), question_data))
            question_id += 1
    
    print(f"生成了 {len(questions)} 个问题")
    
    # 转换为所需格式
    problems = {}
    captions = {}
    
    for qid, q_data in questions:
        problems[qid] = q_data
        
        # 生成caption
        question_text = q_data['question']
        if '共伴生' in question_text or '共生' in question_text:
            captions[qid] = "矿物共生关系问题"
        elif '晶系' in question_text:
            captions[qid] = "矿物晶系属性问题"
        elif '晶类' in question_text:
            captions[qid] = "矿物晶类属性问题"
        elif '分类' in question_text:
            captions[qid] = "矿物分类属性问题"
        else:
            captions[qid] = "地质矿物问题"
    
    # 创建输出目录
    output_dir = "dataset/mineral_kg_based"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存问题
    with open(f"{output_dir}/problems.json", 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    
    # 保存captions
    with open(f"{output_dir}/captions.json", 'w', encoding='utf-8') as f:
        json.dump({"captions": captions}, f, indent=2, ensure_ascii=False)
    
    # 创建数据集划分
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
    
    with open(f"{output_dir}/pid_splits.json", 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    
    print(f"数据集保存完成:")
    print(f"  总问题数: {len(problems)}")
    print(f"  训练集: {len(splits['train'])}")
    print(f"  验证集: {len(splits['val'])}")
    print(f"  测试集: {len(splits['test'])}")
    print(f"  保存到: {output_dir}")
    
    # 统计问题类型
    type_counts = defaultdict(int)
    for caption in captions.values():
        type_counts[caption] += 1
    
    print("\n问题类型分布:")
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count}")

if __name__ == "__main__":
    main()
