import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from legacy.graph_retrieval import GraphRetrieval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--working_dir', required=True)
    ap.add_argument('--top_k', type=int, default=3)
    ap.add_argument('--mode', type=str, default='mix')
    ap.add_argument('--queries', nargs='*', default=[
        "‘石膏’的分类是什么？",
        "‘针铁矿’属于什么晶系？",
        "‘水硫硝镍铝石’的共伴生矿物有哪些？",
    ])
    args = ap.parse_args()

    class C:
        pass
    C.working_dir = args.working_dir
    C.top_k = args.top_k
    C.mode = args.mode

    gr = GraphRetrieval(C)
    for q in args.queries:
        print("Q:", q)
        try:
            ans = gr.find_top_k(q)
            print("A:\n", ans)
        except Exception as e:
            print("Error:", e)
        print("-" * 60)


if __name__ == "__main__":
    main()


