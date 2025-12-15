import argparse
import json
import random
from neo4j import GraphDatabase
from neo4j.exceptions import ConfigurationError
from typing import List, Dict, Any
import os


def generate_test_data(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_db: str,
    output_path: str,
    num_questions: int = 50,
    seed: int = 42,
):
    """
    Generate test dataset from Neo4j mineral knowledge graph.
    Creates multiple-choice questions (4-5 options, 1 correct answer).
    """
    random.seed(seed)

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    # Test if we can use database parameter (Bolt 4.0+) or fall back to default session (Bolt 3.0)
    use_db_param = True
    try:
        with driver.session(database=neo4j_db) as sess:
            sess.run("RETURN 1")
    except ConfigurationError:
        print("[WARN] Bolt Protocol 3.0 detected - falling back to default database")
        use_db_param = False
    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
        return

    # Load minerals
    try:
        if use_db_param:
            with driver.session(database=neo4j_db) as sess:
                minerals = list(sess.run("MATCH (m:Mineral) RETURN m.name AS name LIMIT 200"))
        else:
            with driver.session() as sess:
                minerals = list(sess.run("MATCH (m:Mineral) RETURN m.name AS name LIMIT 200"))
    except Exception as e:
        print(f"[ERROR] Failed to load minerals: {e}")
        return

    if not minerals:
        print("[ERROR] No minerals found in Neo4j database")
        return

    mineral_names = [m['name'] for m in minerals]
    print(f"[INFO] Loaded {len(mineral_names)} minerals from Neo4j")

    problems = {}
    pid_splits = {"train": [], "test": [], "val": []}

    # Question types to generate
    question_types = [
        "classification",    # 某矿物属于什么分类？
        "crystal_system",    # 某矿物的晶系是？
        "association",       # 与某矿物共伴生的矿物是？
    ]

    qid = 0
    generated_count = 0

    while generated_count < num_questions and qid < num_questions * 5:
        q_type = random.choice(question_types)
        problem = None

        if q_type == "classification":
            # 某矿物属于什么矿物分类？
            mineral = random.choice(mineral_names)
            try:
                if use_db_param:
                    with driver.session(database=neo4j_db) as sess:
                        result = list(sess.run(
                            "MATCH (m:Mineral {name: $name})-[:`分类`]->(cat:`矿物类别`) RETURN cat.name AS cat",
                            name=mineral
                        ))
                else:
                    with driver.session() as sess:
                        result = list(sess.run(
                            "MATCH (m:Mineral {name: $name})-[:`分类`]->(cat:`矿物类别`) RETURN cat.name AS cat",
                            name=mineral
                        ))
                
                if result:
                    correct_cat = result[0]['cat']
                    # Get all categories for distractors
                    if use_db_param:
                        with driver.session(database=neo4j_db) as sess:
                            all_cats = [r['c'] for r in sess.run("MATCH (c:`矿物类别`) RETURN c.name AS c LIMIT 50")]
                    else:
                        with driver.session() as sess:
                            all_cats = [r['c'] for r in sess.run("MATCH (c:`矿物类别`) RETURN c.name AS c LIMIT 50")]

                    distractors = [c for c in all_cats if c != correct_cat]
                    if len(distractors) >= 3:
                        choices = [correct_cat] + random.sample(distractors, 3)
                        random.shuffle(choices)
                        answer_idx = choices.index(correct_cat)
                        problem = {
                            "question": f"{mineral}属于什么矿物分类？",
                            "choices": choices,
                            "answer": answer_idx,
                            "split": "test" if qid % 10 < 8 else "val",
                            "image": "",
                            "caption": f"{mineral}的矿物分类属性问题",
                        }
            except Exception as e:
                print(f"[WARN] Classification query failed for {mineral}: {e}")

        elif q_type == "crystal_system":
            # 某矿物的晶系是？
            mineral = random.choice(mineral_names)
            try:
                if use_db_param:
                    with driver.session(database=neo4j_db) as sess:
                        result = list(sess.run(
                            "MATCH (m:Mineral {name: $name})-[:`晶系`]->(cs:`晶系晶类`) RETURN cs.name AS cs",
                            name=mineral
                        ))
                else:
                    with driver.session() as sess:
                        result = list(sess.run(
                            "MATCH (m:Mineral {name: $name})-[:`晶系`]->(cs:`晶系晶类`) RETURN cs.name AS cs",
                            name=mineral
                        ))
                
                if result:
                    correct_cs = result[0]['cs']
                    if use_db_param:
                        with driver.session(database=neo4j_db) as sess:
                            all_cs = [r['c'] for r in sess.run("MATCH (c:`晶系晶类`) WHERE c.name CONTAINS '晶系' RETURN c.name AS c LIMIT 20")]
                    else:
                        with driver.session() as sess:
                            all_cs = [r['c'] for r in sess.run("MATCH (c:`晶系晶类`) WHERE c.name CONTAINS '晶系' RETURN c.name AS c LIMIT 20")]

                    distractors = [c for c in all_cs if c != correct_cs]
                    if len(distractors) >= 2:
                        choices = [correct_cs] + random.sample(distractors, min(2, len(distractors)))
                        all_cs_for_pad = [c for c in all_cs if c not in choices]
                        while len(choices) < 4 and all_cs_for_pad:
                            choices.append(all_cs_for_pad.pop())
                        random.shuffle(choices)
                        answer_idx = choices.index(correct_cs)
                        problem = {
                            "question": f"{mineral}的晶系是？",
                            "choices": choices,
                            "answer": answer_idx,
                            "split": "test" if qid % 10 < 8 else "val",
                            "image": "",
                            "caption": f"{mineral}的晶系属性问题",
                        }
            except Exception as e:
                print(f"[WARN] Crystal system query failed for {mineral}: {e}")

        elif q_type == "association":
            # 与某矿物共伴生的矿物有哪些？
            mineral = random.choice(mineral_names)
            try:
                if use_db_param:
                    with driver.session(database=neo4j_db) as sess:
                        result = list(sess.run(
                            "MATCH (m:Mineral {name: $name})-[:`共伴生`]->(n:Mineral) RETURN n.name AS assoc LIMIT 10",
                            name=mineral
                        ))
                else:
                    with driver.session() as sess:
                        result = list(sess.run(
                            "MATCH (m:Mineral {name: $name})-[:`共伴生`]->(n:Mineral) RETURN n.name AS assoc LIMIT 10",
                            name=mineral
                        ))
                
                if result and len(result) >= 1:
                    associated = [r['assoc'] for r in result]
                    correct_assoc = random.choice(associated)
                    other_minerals = [m for m in mineral_names if m not in [mineral] + associated]
                    distractors = random.sample(other_minerals, min(3, len(other_minerals)))
                    choices = [correct_assoc] + distractors
                    random.shuffle(choices)
                    answer_idx = choices.index(correct_assoc)
                    problem = {
                        "question": f"以下哪种矿物与{mineral}共伴生？",
                        "choices": choices,
                        "answer": answer_idx,
                        "split": "test" if qid % 10 < 8 else "val",
                        "image": "",
                        "caption": f"与{mineral}共伴生的矿物关系问题",
                    }
            except Exception as e:
                print(f"[WARN] Association query failed for {mineral}: {e}")

        if problem:
            problems[str(qid)] = problem
            pid_splits[problem["split"]].append(str(qid))
            generated_count += 1
            if generated_count % 10 == 0:
                print(f"[GEN] Generated {generated_count} questions...")

        qid += 1

    # Save to files
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    problems_file = os.path.join(output_dir, "problems.json")
    splits_file = os.path.join(output_dir, "pid_splits.json")
    captions_file = os.path.join(output_dir, "captions.json")

    with open(problems_file, "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Saved {len(problems)} problems to {problems_file}")

    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(pid_splits, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Saved splits to {splits_file}")

    # Create captions
    captions = {qid: problems[qid]["caption"] for qid in problems}
    with open(captions_file, "w", encoding="utf-8") as f:
        json.dump({"captions": captions}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Saved captions to {captions_file}")

    print(f"\n[DONE] Generated {len(problems)} test questions")
    print(f"  - Test set: {len(pid_splits['test'])}")
    print(f"  - Val set: {len(pid_splits['val'])}")
    
    driver.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j_uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j_user", default="neo4j")
    ap.add_argument("--neo4j_password", default="12345678")
    ap.add_argument("--neo4j_db", default="neo4j")
    ap.add_argument("--output_dir", default="dataset/mineral_test_kg")
    ap.add_argument("--num_questions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    generate_test_data(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.neo4j_db,
        args.output_dir,
        args.num_questions,
        args.seed,
    )
