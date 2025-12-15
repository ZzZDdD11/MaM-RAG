import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from legacy.vector_retrieval import VectorRetrieval


def main():
    ap = argparse.ArgumentParser(description="Test Vector Retrieval via LightRAG")
    ap.add_argument("--working_dir", required=True)
    ap.add_argument("--mode", default="naive", help="naive|mix|global (naive=vector only)")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument(
        "--queries",
        nargs="*",
        default=[
            "石膏的用途是什么？",
            "针铁矿的产地有哪些？",
            "羟碳钠铍石的特征是什么？",
        ],
    )
    args = ap.parse_args()

    class C:
        pass

    C.working_dir = args.working_dir
    C.top_k = args.top_k
    C.mode = args.mode

    vr = VectorRetrieval(C)
    for q in args.queries:
        print("Q:", q)
        try:
            ans = vr.find_top_k(q)
            print("A:\n", ans)
        except Exception as e:
            print("Error:", e)
        print("-" * 60)


if __name__ == "__main__":
    main()


