#!/usr/bin/env python3
"""
修复captions.json文件 - 为所有200个问题生成captions
"""

import json

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

def fix_captions():
    """为所有问题生成captions"""
    # 读取问题数据
    with open('dataset/mineral_expanded/problems.json', 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    # 生成所有captions
    captions = {}
    for qid, problem in problems.items():
        captions[qid] = generate_caption(problem['question'])
    
    # 保存captions
    captions_data = {"captions": captions}
    with open('dataset/mineral_expanded/captions.json', 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)
    
    print(f"Captions生成完成:")
    print(f"  问题总数: {len(problems)}")
    print(f"  Captions总数: {len(captions)}")

if __name__ == "__main__":
    fix_captions()
