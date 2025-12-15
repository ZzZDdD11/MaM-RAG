# tools/run_three_evals.py
import argparse, subprocess, os

def run(cmd):
    print("\n>>>", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    print(p.stdout)
    return p.returncode

def build_base_args(a):
    base = [
        "python", "main.py",
        "--data_root", a.data_root,
        "--image_root", a.image_root,
        "--output_root", a.output_root,
        "--caption_file", a.caption_file,
        "--working_dir", a.working_dir,
        "--llm_model_name", a.llm_model_name,
        "--top_k", str(a.top_k),
        "--seed", str(a.seed),
        "--test_split", a.test_split,
        "--test_number", str(a.test_number),
        "--save_every", str(a.save_every),
        "--shot_number", str(a.shot_number),
        "--use_caption",
    ]
    if a.term_map_path:
        base += ["--term_map_path", a.term_map_path]
    return base

def outputs_path(a, label):
    return os.path.join(a.output_root, f"{label}_{a.test_split}.json")

def metrics_path(a, label):
    return os.path.join(a.output_root, f"{label}_{a.test_split}.metrics.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--image_root", default="")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--caption_file", required=True)
    ap.add_argument("--working_dir", required=True)
    ap.add_argument("--llm_model_name", default="qwen2.5:7b")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--test_number", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--shot_number", type=int, default=0)
    ap.add_argument("--term_map_path", default="")
    ap.add_argument("--debug_dir", default="outputs/debug_runs")
    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    # 1) KG-only
    kg_label = "kg_only"
    kg_debug = os.path.join(args.debug_dir, kg_label)
    os.makedirs(kg_debug, exist_ok=True)
    kg_cmd = build_base_args(args) + [
        "--mode", "mix",
        "--label", kg_label,
        "--disable_vector",
        "--debug_dump_dir", kg_debug,
    ]
    run(kg_cmd)

    # 2) Vec-only (naive)
    vec_label = "vec_only_naive"
    vec_debug = os.path.join(args.debug_dir, vec_label)
    os.makedirs(vec_debug, exist_ok=True)
    vec_cmd = build_base_args(args) + [
        "--mode", "naive",
        "--label", vec_label,
        "--disable_graph",
        "--debug_dump_dir", vec_debug,
    ]
    run(vec_cmd)

    # 3) Hybrid
    hy_label = "hybrid"
    hy_debug = os.path.join(args.debug_dir, hy_label)
    os.makedirs(hy_debug, exist_ok=True)
    hy_cmd = build_base_args(args) + [
        "--mode", "mix",
        "--label", hy_label,
        "--debug_dump_dir", hy_debug,
    ]
    run(hy_cmd)

    # Lenient evaluation for all three
    eval_script = ["python", "tools/evaluate_lenient.py", "--problems", os.path.join(args.data_root, "problems.json")]
    if args.term_map_path:
        eval_script += ["--term_map_path", args.term_map_path]

    print("\n=== Lenient metrics: KG-only ===")
    run(eval_script + [
        "--outputs", outputs_path(args, kg_label),
        "--debug_dir", kg_debug,
        "--label", kg_label,
        "--metrics_out", metrics_path(args, kg_label),
    ])

    print("\n=== Lenient metrics: Vec-only (naive) ===")
    run(eval_script + [
        "--outputs", outputs_path(args, vec_label),
        "--debug_dir", vec_debug,
        "--label", vec_label,
        "--metrics_out", metrics_path(args, vec_label),
    ])

    print("\n=== Lenient metrics: Hybrid ===")
    run(eval_script + [
        "--outputs", outputs_path(args, hy_label),
        "--debug_dir", hy_debug,
        "--label", hy_label,
        "--metrics_out", metrics_path(args, hy_label),
    ])

    # Plot chart comparing modes
    chart_png = os.path.join(args.output_root, f"metrics_{args.test_split}.html")
    run([
        "python", "tools/plot_metrics.py",
        "--kg_metrics", metrics_path(args, kg_label),
        "--vec_metrics", metrics_path(args, vec_label),
        "--hybrid_metrics", metrics_path(args, hy_label),
        "--out", chart_png,
    ])

if __name__ == "__main__":
    main()