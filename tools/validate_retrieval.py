import argparse
import json
import os
from pathlib import Path


def extract_evidence_sections(debug_file):
    """Extract evidence text from each source in debug file."""
    if not os.path.exists(debug_file):
        return {}
    
    with open(debug_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = {
        'graph': '',
        'vector': '',
        'web': ''
    }
    
    # Split by [Source X]
    parts = content.split('[Source ')
    
    for part in parts:
        if 'Graph Retrieval Agent:' in part:
            # Extract text after "Graph Retrieval Agent:"
            start = part.find('Graph Retrieval Agent:') + len('Graph Retrieval Agent:')
            end = part.find('[Source') if '[Source' in part else len(part)
            sections['graph'] = part[start:end].strip()
        elif 'Vector Retrieval Agent:' in part:
            start = part.find('Vector Retrieval Agent:') + len('Vector Retrieval Agent:')
            end = part.find('[Source') if '[Source' in part else len(part)
            sections['vector'] = part[start:end].strip()
        elif 'Web Retrieval Agent:' in part:
            start = part.find('Web Retrieval Agent:') + len('Web Retrieval Agent:')
            end = part.find('[Source') if '[Source' in part else len(part)
            sections['web'] = part[start:end].strip()
    
    return sections


def analyze_evidence(evidence_text, mode='graph'):
    """Analyze if evidence contains real knowledge or LLM hallucination."""
    metrics = {
        'length': len(evidence_text),
        'lines': len(evidence_text.split('\n')),
        'is_disabled': 'Disabled' in evidence_text,
        'has_specific_info': False,
        'keyword_count': 0,
        'confidence_score': 0.0,
    }
    
    # Key indicators of real retrieval vs hallucination
    graph_keywords = ['共伴生', '晶系', '晶类', '分类', '属于', '成分', '性质']
    vector_keywords = ['矿物', '晶体', '晶系', '性质', '特征', '应用', '产地']
    
    keywords = graph_keywords if mode == 'graph' else vector_keywords
    
    # Count keywords
    for kw in keywords:
        metrics['keyword_count'] += evidence_text.count(kw)
    
    # Check for specific relationship indicators (not generic text)
    specific_indicators = [
        '与' in evidence_text and '共伴生' in evidence_text,
        '属于' in evidence_text and ('硫酸盐' in evidence_text or '氧化物' in evidence_text or '硅酸盐' in evidence_text),
        '晶系' in evidence_text and ('三方晶系' in evidence_text or '立方晶系' in evidence_text or '四方晶系' in evidence_text),
    ]
    
    metrics['has_specific_info'] = any(specific_indicators)
    
    # Calculate confidence score
    if metrics['is_disabled']:
        metrics['confidence_score'] = 0.0
    elif metrics['has_specific_info'] and metrics['keyword_count'] >= 2:
        metrics['confidence_score'] = 0.9  # High confidence - specific real knowledge
    elif metrics['keyword_count'] >= 1 and metrics['length'] > 200:
        metrics['confidence_score'] = 0.6  # Medium confidence - has some relevant keywords
    elif metrics['length'] < 100 and 'error' not in evidence_text.lower():
        metrics['confidence_score'] = 0.3  # Low confidence - minimal content, possible LLM generation
    else:
        metrics['confidence_score'] = 0.1  # Very low - likely hallucination
    
    return metrics


def validate_debug_files(debug_dir, problems_file=None):
    """Validate all debug files in a directory."""
    
    if not os.path.exists(debug_dir):
        print(f"[ERROR] Debug directory not found: {debug_dir}")
        return
    
    # Load problems for reference
    problems = {}
    if problems_file and os.path.exists(problems_file):
        with open(problems_file, 'r', encoding='utf-8') as f:
            problems = json.load(f)
    
    debug_files = sorted([f for f in os.listdir(debug_dir) if f.endswith('.txt')])
    
    if not debug_files:
        print(f"[ERROR] No debug files found in {debug_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"RETRIEVAL VALIDATION REPORT")
    print(f"Debug directory: {debug_dir}")
    print(f"Total files: {len(debug_files)}")
    print(f"{'='*80}\n")
    
    # Summary statistics
    stats = {
        'total': len(debug_files),
        'graph_high_conf': 0,
        'graph_medium_conf': 0,
        'graph_low_conf': 0,
        'vector_high_conf': 0,
        'vector_medium_conf': 0,
        'vector_low_conf': 0,
    }
    
    issue_count = 0
    
    for debug_file in debug_files[:10]:  # Analyze first 10 for detailed report
        qid = debug_file.replace('.txt', '')
        debug_path = os.path.join(debug_dir, debug_file)
        
        sections = extract_evidence_sections(debug_path)
        
        print(f"[Q{qid}]")
        if qid in problems:
            print(f"  Question: {problems[qid].get('question', 'N/A')[:80]}")
            print(f"  Answer: {problems[qid].get('choices', [])[problems[qid].get('answer', -1)]}")
        
        # Analyze each source
        for source in ['graph', 'vector']:
            evidence = sections.get(source, '')
            metrics = analyze_evidence(evidence, mode=source)
            
            conf = metrics['confidence_score']
            if conf >= 0.8:
                conf_label = "HIGH (likely real retrieval)"
                stats[f'{source}_high_conf'] += 1
            elif conf >= 0.5:
                conf_label = "MEDIUM (partial retrieval)"
                stats[f'{source}_medium_conf'] += 1
            else:
                conf_label = "LOW (possible hallucination)"
                stats[f'{source}_low_conf'] += 1
                issue_count += 1
            
            print(f"  {source.upper():8} | Len:{metrics['length']:5} Keywords:{metrics['keyword_count']:2} Conf:{conf:.2f} ({conf_label})")
            if not metrics['is_disabled'] and metrics['keyword_count'] == 0:
                print(f"           First 100 chars: {evidence[:100]}")
        
        print()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (first 10 files)")
    print(f"{'='*80}")
    print(f"Graph Retrieval:")
    print(f"  High confidence (likely real):   {stats['graph_high_conf']:2}/10")
    print(f"  Medium confidence (partial):     {stats['graph_medium_conf']:2}/10")
    print(f"  Low confidence (hallucination):  {stats['graph_low_conf']:2}/10")
    print(f"\nVector Retrieval:")
    print(f"  High confidence (likely real):   {stats['vector_high_conf']:2}/10")
    print(f"  Medium confidence (partial):     {stats['vector_medium_conf']:2}/10")
    print(f"  Low confidence (hallucination):  {stats['vector_low_conf']:2}/10")
    
    print(f"\n{'='*80}")
    if issue_count > 3:
        print("WARNING: High number of potentially hallucinated results!")
        print("Recommendation: Check if KG/Vector data was properly ingested into LightRAG")
    elif issue_count > 0:
        print("INFO: Some retrieval quality issues detected")
    else:
        print("OK: All retrievals appear to contain real knowledge")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_dir", required=True, help="Path to debug_runs directory")
    ap.add_argument("--problems", default="", help="Path to problems.json for context")
    args = ap.parse_args()
    
    validate_debug_files(args.debug_dir, args.problems)
