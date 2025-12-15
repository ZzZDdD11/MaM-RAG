import sys, os, networkx as nx

g = nx.read_graphml(r"D:\Code\HMRAG-main\lightrag_store\graph_chunk_entity_relation.graphml")
def search(q):
    hits = []
    q = q.replace("'","").replace("'","").strip()
    for n,attrs in g.nodes(data=True):
        for k,v in attrs.items():
            if isinstance(v,str) and q in v:
                hits.append((n,attrs)); break
    return hits
def show_edges(n, limit=10):
    out=[]
    # 改用 edges() 而非 out_edges()，并处理无向图的双向边
    for u,v,attrs in g.edges(n, data=True):
        out.append(f"{u} --[{attrs.get('label',attrs.get('type','rel'))}]--> {v}")
        if len(out)>=limit: break
    return "\n".join(out) or "(no outgoing edges)"
for q in ["石膏","针铁矿","水硫硝镍铝石"]:
    print("Q:", q)
    hs = search(q)
    if not hs: print("No node hit"); print("-"*40); continue
    nid,_=hs[0]
    print(show_edges(nid)); print("-"*40)