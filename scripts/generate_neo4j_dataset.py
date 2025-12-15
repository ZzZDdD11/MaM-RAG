#!/usr/bin/env python3
"""
基于Neo4j知识图谱生成测试数据集
直接从Neo4j数据库中提取关系，生成高质量的测试问题
"""

import json
import random
import os
from typing import Dict, List, Tuple, Set
from collections import defaultdict

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("警告: neo4j库未安装，请运行: pip install neo4j")

class Neo4jKGExtractor:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="12345678"):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j库未安装")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def extract_relations(self) -> Dict[str, List[Tuple[str, str]]]:
        """从Neo4j中提取关系三元组"""
        relations = defaultdict(list)
        
        with self.driver.session() as session:
            # 查询所有关系
            query = """
            MATCH (n1)-[r]->(n2)
            RETURN n1.name as entity1, type(r) as relation, n2.name as entity2
            LIMIT 10000
            """
            
            result = session.run(query)
            
            for record in result:
                entity1 = record["entity1"]
                relation = record["relation"]
                entity2 = record["entity2"]
                
                if entity1 and relation and entity2:
                    relations[relation].append((entity1, entity2))
        
        return relations
    
    def get_mineral_properties(self) -> Dict[str, Dict[str, str]]:
        """获取矿物的属性信息"""
        mineral_props = {}
        
        with self.driver.session() as session:
            # 查询矿物及其属性
            queries = [
                # 共伴生关系
                """
                MATCH (m:Mineral)-[:共伴生]->(other)
                RETURN m.name as mineral, collect(other.name) as coexist
                """,
                # 晶系属性
                """
                MATCH (m:Mineral)-[:晶系]->(cs)
                RETURN m.name as mineral, cs.name as crystal_system
                """,
                # 晶类属性
                """
                MATCH (m:Mineral)-[:晶类]->(cc)
                RETURN m.name as mineral, cc.name as crystal_class
                """,
                # 分类属性
                """
                MATCH (m:Mineral)-[:分类]->(cat)
                RETURN m.name as mineral, cat.name as classification
                """
            ]
            
            # 执行共伴生查询
            result = session.run(queries[0])
            for record in result:
                mineral = record["mineral"]
                if mineral not in mineral_props:
                    mineral_props[mineral] = {}
                mineral_props[mineral]["共伴生"] = record["coexist"]
            
            # 执行晶系查询
            result = session.run(queries[1])
            for record in result:
                mineral = record["mineral"]
                if mineral not in mineral_props:
                    mineral_props[mineral] = {}
                mineral_props[mineral]["晶系"] = record["crystal_system"]
            
            # 执行晶类查询
            result = session.run(queries[2])
            for record in result:
                mineral = record["mineral"]
                if mineral not in mineral_props:
                    mineral_props[mineral] = {}
                mineral_props[mineral]["晶类"] = record["crystal_class"]
            
            # 执行分类查询
            result = session.run(queries[3])
            for record in result:
                mineral = record["mineral"]
                if mineral not in mineral_props:
                    mineral_props[mineral] = {}
                mineral_props[mineral]["分类"] = record["classification"]
        
        return mineral_props

def generate_questions_from_neo4j(mineral_props: Dict[str, Dict], target_size: int = 100) -> List[Dict]:
    """基于Neo4j数据生成问题"""
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
    
    for mineral, props in mineral_props.items():
        if "共伴生" in props and isinstance(props["共伴生"], list):
            all_coexist.update(props["共伴生"])
        if "晶系" in props:
            all_crystal_systems.add(props["晶系"])
        if "晶类" in props:
            all_crystal_classes.add(props["晶类"])
        if "分类" in props:
            all_classifications.add(props["分类"])
    
    # 为每个矿物生成问题
    for mineral, properties in mineral_props.items():
        for prop_type, prop_value in properties.items():
            if prop_type not in templates:
                continue
            
            # 选择模板
            template = random.choice(templates[prop_type])
            question = template.format(entity=mineral)
            
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
                elif prop_type == '晶类':
                    distractor_pool.update(["单锥晶类", "双锥晶类", "柱晶类", "面体晶类"])
            
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
                'hint': f"这是一个关于{mineral}的{prop_type}问题",
                'lecture': f"根据知识图谱，{mineral}的{prop_type}是{correct_answer}。",
                'image': None,
                'relation_type': prop_type,
                'entity': mineral
            }
            
            questions.append(question_data)
            question_id += 1
            
            if len(questions) >= target_size:
                break
        
        if len(questions) >= target_size:
            break
    
    return questions[:target_size]

def create_dataset_from_neo4j(output_dir: str, target_size: int = 100):
    """从Neo4j创建数据集"""
    
    if not NEO4J_AVAILABLE:
        print("错误: neo4j库未安装，无法连接数据库")
        return
    
    try:
        # 连接Neo4j
        print("连接Neo4j数据库...")
        extractor = Neo4jKGExtractor()
        
        # 提取矿物属性
        print("提取矿物属性...")
        mineral_props = extractor.get_mineral_properties()
        
        print(f"找到 {len(mineral_props)} 个矿物")
        
        # 统计属性类型
        prop_counts = defaultdict(int)
        for mineral, props in mineral_props.items():
            for prop_type in props.keys():
                prop_counts[prop_type] += 1
        
        print("属性统计:")
        for prop_type, count in prop_counts.items():
            print(f"  {prop_type}: {count}个")
        
        # 生成问题
        print(f"生成 {target_size} 个问题...")
        questions = generate_questions_from_neo4j(mineral_props, target_size)
        
        if not questions:
            print("错误: 没有生成任何问题")
            return
        
        # 转换为所需格式
        problems = {}
        captions = {}
        
        for i, q_data in enumerate(questions):
            qid = str(i)
            problems[qid] = {
                'question': q_data['question'],
                'choices': q_data['choices'],
                'answer': q_data['answer'],
                'hint': q_data['hint'],
                'lecture': q_data['lecture'],
                'image': q_data['image']
            }
            
            # 生成caption
            if q_data['relation_type'] == '共伴生':
                captions[qid] = "矿物共生关系问题"
            elif q_data['relation_type'] == '晶系':
                captions[qid] = "矿物晶系属性问题"
            elif q_data['relation_type'] == '晶类':
                captions[qid] = "矿物晶类属性问题"
            elif q_data['relation_type'] == '分类':
                captions[qid] = "矿物分类属性问题"
            else:
                captions[qid] = "地质矿物问题"
        
        # 创建输出目录
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
        
        print(f"\n数据集生成完成:")
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
        
        # 关闭连接
        extractor.close()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 生成基于Neo4j的测试数据集
    create_dataset_from_neo4j("dataset/mineral_neo4j_based", target_size=300)
