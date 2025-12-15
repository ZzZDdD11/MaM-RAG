import json, os, re, argparse
from typing import List, Optional, Set, Tuple

PURE_CN = re.compile(r"^[\u4e00-\u9fa5·]{2,32}$")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_alias2std(term_map_path: str):
    if not term_map_path or not os.path.exists(term_map_path):
        return {}
    m = json.load(open(term_map_path, "r", encoding="utf-8"))
    a2s = {}
    for std, meta in m.items():
        for a in meta.get("aliases") or []:
            if a and PURE_CN.match(a):
                a2s[a] = std
        if PURE_CN.match(std):
            a2s.setdefault(std, std)
    return a2s

def normalize_token(tok: str, a2s: dict):
    tok = tok.strip()
    if not tok: return ""
    if tok in a2s: return a2s[tok]
    return tok

def split_cn_phrases(s: str):
    parts = re.split(r"[、和及与/，,；;\s]+", str(s).strip())
    return [p for p in parts if p and len(p) >= 2]

def option_tokens(option_text: str, a2s: dict):
    toks = split_cn_phrases(option_text)
    toks = [normalize_token(t, a2s) for t in toks]
    toks = [t for t in toks if t]
    return set(toks)

def evidence_tokens(
    evidences: List[str],
    a2s: dict,
    allow: Optional[Set[str]] = None,
    relation_keywords: Optional[List[str]] = None,
    window: int = 32,
):
    """Extract normalized entity tokens from evidence text.
    Simple approach: find all entities from a2s in the text, normalize them, and return.
    """
    text = "\n".join(evidences)[:8000]
    cands = set()
    for a in a2s.keys():
        if not a or len(a) < 2:
            continue
        if a in text:
            norm = normalize_token(a, a2s)
            if norm:
                cands.add(norm)
    return cands


def score_options(
    evidence_text: str,
    opt_sets: List[Set[str]],
    a2s: dict,
    allow: Set[str],
    relation_keywords: Optional[List[str]] = None,
    window: int = 32,
) -> Tuple[List[Tuple[int, int]], Set[str]]:
    # If evidence is just a short "disabled" message, return zero scores to avoid spurious hits.
    if ("Vector Disabled" in evidence_text or "Graph Disabled" in evidence_text) and len(evidence_text) < 100:
        ev_set = set()
        scores = [(i, 0) for i in range(len(opt_sets))]
        return sorted(scores, key=lambda x: x[0]), ev_set

    # Extract all entities from evidence
    ev_set = evidence_tokens([evidence_text], a2s)
    # Filter to only entities that appear in options (allow set)
    ev_set_filtered = ev_set & allow
    
    # Score each option by the count of matching entities
    scores = [(i, len(ev_set_filtered & opt_sets[i])) for i in range(len(opt_sets))]
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores_sorted, ev_set_filtered

def jaccard(a: set, b: set):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)

def read_debug_sources(debug_fp):
    """Read debug file and split evidence by source (graph/vector/other)."""
    src = {"graph": "", "vector": "", "other": ""}
    if not os.path.exists(debug_fp):
        return src
    raw = open(debug_fp, "r", encoding="utf-8").read()
    # Remove Choices line to avoid leaking option text into evidence
    raw = "\n".join([ln for ln in raw.splitlines() if not ln.strip().startswith("Choices:")])
    blocks = raw.split("[Source ")
    for b in blocks:
        if "Graph Retrieval Agent:" in b:
            src["graph"] += b + "\n"
        elif "Vector Retrieval Agent:" in b:
            src["vector"] += b + "\n"
        else:
            src["other"] += b + "\n"
    return src

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--debug_dir", default="outputs/debug_runs")
    ap.add_argument("--term_map_path", default="")
    ap.add_argument("--options", default="A,B,C,D,E")
    ap.add_argument("--label", default="")
    ap.add_argument("--metrics_out", default="", help="Path to save metrics JSON (optional)")
    args = ap.parse_args()

    problems = load_json(args.problems)
    preds = load_json(args.outputs)
    a2s = load_alias2std(args.term_map_path)
    relation_keywords = ["属于", "分类", "类", "类型", "共伴生", "伴生", "共生", "晶系", "晶类", "产地", "组成", "成分", "矿物", "矿物分类"]

    n = 0
    exact = 0
    cov2_graph = 0
    cov2_vector = 0
    cov2_hybrid = 0
    cov1_graph = 0
    cov1_vector = 0
    cov1_hybrid = 0
    r1_graph = 0
    r1_vector = 0
    r1_hybrid = 0
    margin_graph_sum = 0.0
    margin_vector_sum = 0.0
    margin_hybrid_sum = 0.0
    set_overlap = 0
    set_j_sum = 0.0
    ev_any = 0
    ev_all = 0

    for qid, prob in problems.items():
        if qid not in preds: 
            continue
        n += 1
        gold_idx = prob["answer"]
        gold_text = prob["choices"][gold_idx]
        pred_idx = preds[qid]

        # Exact
        if pred_idx == gold_idx:
            exact += 1

        # Option entity sets
        gold_set = option_tokens(gold_text, a2s)
        opt_sets = [option_tokens(prob["choices"][i], a2s) for i in range(len(prob["choices"]))]

        # Set overlap/jaccard (between predicted option and gold option)
        pred_set = opt_sets[pred_idx]
        if gold_set & pred_set:
            set_overlap += 1
        set_j_sum += jaccard(gold_set, pred_set)

        # Read evidences by source
        debug_fp = os.path.join(args.debug_dir, f"{qid}.txt")
        src_txt = read_debug_sources(debug_fp)
        hybrid_ev = "\n".join([src_txt["graph"], src_txt["vector"], src_txt["other"]])

        # Option unions as whitelist for evidence entity extraction
        allow_entities = set()
        for s in opt_sets:
            allow_entities.update(s)

        def coverage_k_and_stats(evidence_text: str, k: int, relation_aware: bool = False):
            rel_keys = relation_keywords if relation_aware else None
            scores_sorted, _ = score_options(evidence_text, opt_sets, a2s, allow_entities, relation_keywords=rel_keys)
            
            # A score of 0 means no entities matched. It shouldn't count as a hit.
            # Filter to only consider choices with actual evidence support.
            positive_scores = [(i, s) for i, s in scores_sorted if s > 0]

            if not positive_scores:
                return False, False, 0.0

            # Find the rank of the gold answer among choices with positive scores
            gold_rank = -1
            for rank, (i, s) in enumerate(positive_scores):
                if i == gold_idx:
                    gold_rank = rank
                    break
            
            hitk = (gold_rank != -1 and gold_rank < k)

            # margin: gold_score - best_other_score
            gold_score = next((s for i, s in positive_scores if i == gold_idx), 0)
            best_other = 0
            for i, s in positive_scores:
                if i != gold_idx:
                    # The list is already sorted, so the first non-gold is the best other
                    best_other = s
                    break

            margin = float(gold_score - best_other)
            # hits@1
            hit1 = (len(positive_scores) > 0 and positive_scores[0][0] == gold_idx and positive_scores[0][1] > 0)
            return hitk, hit1, margin

        # Per-source coverage
        hit2_g, hit1_g, margin_g = coverage_k_and_stats(src_txt["graph"], 2, relation_aware=False)
        if hit2_g:
            cov2_graph += 1
        if hit1_g:
            cov1_graph += 1
        margin_graph_sum += margin_g

        hit2_v, hit1_v, margin_v = coverage_k_and_stats(src_txt["vector"], 2, relation_aware=False)
        if hit2_v:
            cov2_vector += 1
        if hit1_v:
            cov1_vector += 1
        margin_vector_sum += margin_v

        hit2_h, hit1_h, margin_h = coverage_k_and_stats(hybrid_ev, 2, relation_aware=False)
        if hit2_h:
            cov2_hybrid += 1
        if hit1_h:
            cov1_hybrid += 1
        margin_hybrid_sum += margin_h

        # Relation-aware hits@1
        _, r_hit1_g, _ = coverage_k_and_stats(src_txt["graph"], 1, relation_aware=True)
        if r_hit1_g:
            r1_graph += 1
        _, r_hit1_v, _ = coverage_k_and_stats(src_txt["vector"], 1, relation_aware=True)
        if r_hit1_v:
            r1_vector += 1
        _, r_hit1_h, _ = coverage_k_and_stats(hybrid_ev, 1, relation_aware=True)
        if r_hit1_h:
            r1_hybrid += 1

        # Evidence support (hybrid evidence)
        _, ev_set = score_options(hybrid_ev, opt_sets, a2s, allow_entities)
        if ev_set & gold_set:
            ev_any += 1
        if gold_set and gold_set.issubset(ev_set):
            ev_all += 1

    if n == 0:
        print("Count=0")
        return

    exact_acc = exact/n
    cov2_g = cov2_graph/n
    cov2_v = cov2_vector/n
    cov2_h = cov2_hybrid/n
    cov1_g = cov1_graph/n
    cov1_v = cov1_vector/n
    cov1_h = cov1_hybrid/n
    r1_g = r1_graph/n
    r1_v = r1_vector/n
    r1_h = r1_hybrid/n
    m_g = margin_graph_sum/n
    m_v = margin_vector_sum/n
    m_h = margin_hybrid_sum/n
    opt_overlap = set_overlap/n
    opt_j = set_j_sum/n
    ev_any_r = ev_any/n
    ev_all_r = ev_all/n

    print(f"Count={n}")
    print(f"Exact Acc = {exact_acc:.4f}")
    print(f"Coverage@2_graph  = {cov2_g:.4f}")
    print(f"Coverage@2_vector = {cov2_v:.4f}")
    print(f"Coverage@2_hybrid = {cov2_h:.4f}")
    print(f"Coverage@1_graph  = {cov1_g:.4f}")
    print(f"Coverage@1_vector = {cov1_v:.4f}")
    print(f"Coverage@1_hybrid = {cov1_h:.4f}")
    print(f"Relation@1_graph  = {r1_g:.4f}")
    print(f"Relation@1_vector = {r1_v:.4f}")
    print(f"Relation@1_hybrid = {r1_h:.4f}")
    print(f"SupportMargin_graph  = {m_g:.4f}")
    print(f"SupportMargin_vector = {m_v:.4f}")
    print(f"SupportMargin_hybrid = {m_h:.4f}")
    print(f"OptionSet-Overlap = {opt_overlap:.4f}")
    print(f"OptionSet-Jaccard = {opt_j:.4f}")
    print(f"Evidence-any (gold & evidence != empty) = {ev_any_r:.4f}")
    print(f"Evidence-all (gold subset-of evidence)  = {ev_all_r:.4f}")

    # Optional: dump metrics JSON for visualization
    if args.metrics_out:
        out_dir = os.path.dirname(args.metrics_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        label = (args.label or "").lower()
        def pick_primary(g, v, h):
            if "kg_only" in label:
                return g
            if "vec_only" in label or "vector_only" in label or "veconly" in label:
                return v
            return h  # default hybrid
        metrics = {
            "label": args.label or "",
            "count": n,
            "exact_acc": exact_acc,
            # coverage2 输出主值；明细放到 coverage2_detail
            "coverage2": pick_primary(cov2_g, cov2_v, cov2_h),
            "coverage2_detail": {"graph": cov2_g, "vector": cov2_v, "hybrid": cov2_h},
            # 其余保持原样（若需也可改为主值策略）
            "coverage1": {"graph": cov1_g, "vector": cov1_v, "hybrid": cov1_h},
            "relation1": {"graph": r1_g, "vector": r1_v, "hybrid": r1_h},
            "support_margin": {"graph": m_g, "vector": m_v, "hybrid": m_h},
            "option_overlap": opt_overlap,
            "option_jaccard": opt_j,
            "evidence_any": ev_any_r,
            "evidence_all": ev_all_r,
        }
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics JSON to {args.metrics_out}")

if __name__ == "__main__":
    main()