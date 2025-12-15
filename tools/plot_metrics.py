import argparse
import json
import os


def load_metrics(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def render_html(metrics_files, out_html):
    data = [load_metrics(p) for p in metrics_files]
    labels = [d.get('label') or os.path.splitext(os.path.basename(path))[0] for d, path in zip(data, metrics_files)]

    # We compare three intuitive metrics per mode
    # exact_acc; coverage1(mode); relation1(mode)
    def mode_key(label):
        if 'hybrid' in label:
            return 'hybrid'
        if 'vec' in label:
            return 'vector'
        return 'graph'

    exact = [d.get('exact_acc', 0.0) for d in data]
    cov1 = [d.get('coverage1', {}).get(mode_key(d.get('label') or ''), 0.0) for d in data]
    rel1 = [d.get('relation1', {}).get(mode_key(d.get('label') or ''), 0.0) for d in data]

    series = [
        ('Exact Acc', exact, '#4C78A8'),
        ('Coverage@1', cov1, '#72B7B2'),
        ('Relation@1', rel1, '#F58518'),
    ]

    # simple inline SVG bar chart
    width = 800
    height = 360
    margin_left = 80
    margin_bottom = 40
    chart_w = width - margin_left - 20
    chart_h = height - 20 - margin_bottom

    max_y = 1.0
    num_groups = len(labels)
    num_series = len(series)
    group_w = chart_w / max(1, num_groups)
    bar_w = group_w / (num_series + 1)

    def bars():
        elems = []
        for gi, label in enumerate(labels):
            gx = margin_left + gi * group_w
            for si, (name, vals, color) in enumerate(series):
                val = min(max(vals[gi], 0.0), 1.0)
                h = val * chart_h
                x = gx + si * bar_w
                y = height - margin_bottom - h
                elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.9:.1f}" height="{h:.1f}" fill="{color}" />')
        return '\n'.join(elems)

    def axes():
        # y-axis line
        axis = [
            f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{margin_left}" y2="20" stroke="#333" stroke-width="1" />',
            f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-20}" y2="{height-margin_bottom}" stroke="#333" stroke-width="1" />',
        ]
        # y ticks
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ty = height - margin_bottom - t * chart_h
            axis.append(f'<line x1="{margin_left-4}" y1="{ty:.1f}" x2="{margin_left}" y2="{ty:.1f}" stroke="#333" />')
            axis.append(f'<text x="{margin_left-8}" y="{ty+4:.1f}" font-size="12" text-anchor="end">{t:.2f}</text>')
        # x labels
        for gi, label in enumerate(labels):
            gx = margin_left + gi * group_w + (num_series-1) * bar_w / 2
            axis.append(f'<text x="{gx:.1f}" y="{height-10}" font-size="12" text-anchor="middle">{label}</text>')
        return '\n'.join(axis)

    def legend():
        items = []
        lx = margin_left
        ly = 4
        for name, _, color in series:
            items.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{color}" />')
            items.append(f'<text x="{lx+18}" y="{ly+11}" font-size="12">{name}</text>')
            lx += 120
        return '\n'.join(items)

    html = f"""
<!doctype html>
<meta charset="utf-8" />
<title>RAG Retrieval Comparison</title>
<div style="font-family: -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; max-width: 860px; margin: 16px auto;">
  <h3 style="margin: 8px 0 12px;">RAG Retrieval Comparison (Higher is Better)</h3>
  <svg width="{width}" height="{height}">
    {axes()}
    {bars()}
    {legend()}
  </svg>
  <div style="margin-top: 8px; color:#666; font-size: 12px;">Metrics: Exact Acc, Coverage@1, Relation@1</div>
  <pre style="background:#f7f7f7; padding:8px; border-radius:6px; overflow:auto; font-size:12px;">{json.dumps(data, ensure_ascii=False, indent=2)}</pre>
  <div style="color:#666; font-size: 12px;">Chart saved to: {out_html}</div>
  </div>
"""

    out_dir = os.path.dirname(out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved chart to {out_html}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kg_metrics', required=True)
    ap.add_argument('--vec_metrics', required=True)
    ap.add_argument('--hybrid_metrics', required=True)
    ap.add_argument('--out', default='outputs/metrics_chart.html')
    args = ap.parse_args()

    render_html([args.kg_metrics, args.vec_metrics, args.hybrid_metrics], args.out)


if __name__ == '__main__':
    main()


