import argparse
import sys
import requests


def fetch_subgraph(base_url: str, label: str, depth: int, max_nodes: int) -> dict:
    params = {"label": label, "max_depth": depth, "max_nodes": max_nodes}
    r = requests.get(f"{base_url}/graphs", params=params, timeout=60)
    r.raise_for_status()
    return r.json() if r.text else {}


def main():
    ap = argparse.ArgumentParser(description="Verify LightRAG Neo4JStorage /graphs endpoint")
    ap.add_argument("--host", default="127.0.0.1", help="LightRAG server host")
    ap.add_argument("--port", type=int, default=9621, help="LightRAG server port")
    ap.add_argument("--label", dest="labels", action="append", help="Label to query; repeatable", required=True)
    ap.add_argument("--depths", default="1,2", help="Comma-separated depths to query, e.g. 1,2,3")
    ap.add_argument("--max_nodes", type=int, default=200, help="Max nodes per subgraph")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]

    all_ok = True
    print(f"Base URL: {base_url}\nLabels: {args.labels}\nDepths: {depths}\nMax nodes: {args.max_nodes}\n")
    for label in args.labels:
        for d in depths:
            try:
                data = fetch_subgraph(base_url, label, d, args.max_nodes)
                nodes = len(data.get("nodes", []) or [])
                edges = len(data.get("edges", []) or [])
                truncated = bool(data.get("is_truncated", False))
                status = "OK" if (nodes > 0 or edges > 0) else "EMPTY"
                print(f"[label={label!s}][depth={d}] nodes={nodes} edges={edges} truncated={truncated} -> {status}")
                if status == "EMPTY":
                    all_ok = False
            except Exception as e:
                all_ok = False
                print(f"[label={label!s}][depth={d}] ERROR: {e}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()


