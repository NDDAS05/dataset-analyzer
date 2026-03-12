"""
report_generator.py - HTML report generation for the Automated Dataset Analyzer.
Uses Jinja2 to render a light, cartoon-style HTML report.
"""

import os
import base64
from datetime import datetime
from jinja2 import Environment, FileSystemLoader


def _img_to_b64(img_path: str, outputs_dir: str) -> str:
    """Convert an image file to a base64 data URI for embedding in HTML."""
    full_path = os.path.join(outputs_dir, img_path)
    if not os.path.exists(full_path):
        return ''
    with open(full_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def generate_report(
    df,
    analysis: dict,
    chart_paths: dict,
    ml_results: dict,
    insights: list,
    output_dir: str,
    dataset_name: str = "Dataset"
) -> str:
    """
    Render the full HTML report and save to output_dir/dataset_report.html.

    Args:
        df: Original DataFrame.
        analysis: Output from analyzer.analyze_dataset().
        chart_paths: Output from visualizer.generate_visualizations().
        ml_results: Output from ml_insights.run_ml_analysis() (or None).
        insights: List of insight strings.
        output_dir: Directory where plots are stored.
        dataset_name: Display name for the report header.
    Returns:
        Path to the generated HTML report.
    """
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('report_template.html')

    def embed(path):
        return _img_to_b64(path, output_dir) if path else ''

    # Per-feature distribution images  →  [{col, img}, ...]
    distributions = [
        {'col': item['col'], 'img': embed(item['path'])}
        for item in chart_paths.get('distributions', [])
    ]

    # Per-feature categorical images  →  [{col, img}, ...]
    cat_charts = [
        {'col': item['col'], 'img': embed(item['path'])}
        for item in chart_paths.get('categorical', [])
    ]

    hm_img  = embed(chart_paths.get('correlation_heatmap'))
    mv_img  = embed(chart_paths.get('missing_values'))
    out_img = embed(chart_paths.get('outliers'))
    fi_img  = embed(chart_paths.get('feature_importance'))
    cm_img  = embed(chart_paths.get('confusion_matrix'))

    # Descriptive stats
    desc = analysis.get('descriptive_stats', {})
    stats_keys = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    desc_rows = []
    for col, stats in desc.items():
        row = {'column': col}
        row.update(stats)
        desc_rows.append(row)

    # Missing values table rows
    mv_rows = [
        {'column': col, 'count': v['missing_count'], 'pct': v['missing_pct']}
        for col, v in analysis.get('missing_values', {}).items()
    ]

    # Outlier table rows
    out_rows = [
        {'column': col, 'count': v['count'], 'pct': v['pct'],
         'lower': v['lower_bound'], 'upper': v['upper_bound']}
        for col, v in analysis.get('outliers', {}).items()
        if v['count'] > 0
    ]

    ctx = dict(
        dataset_name        = dataset_name,
        generated_at        = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        overview            = analysis['overview'],
        desc_rows           = desc_rows,
        stats_keys          = stats_keys,
        mv_rows             = mv_rows,
        total_missing       = analysis.get('total_missing', 0),
        total_missing_pct   = analysis.get('total_missing_pct', 0),
        duplicates          = analysis.get('duplicates', {}),
        out_rows            = out_rows,
        strong_corrs        = analysis.get('strong_correlations', []),
        cat_analysis        = analysis.get('categorical_analysis', {}),
        insights            = insights,
        ml                  = ml_results or {},
        # Chart data
        distributions       = distributions,
        cat_charts          = cat_charts,
        hm_img              = hm_img,
        mv_img              = mv_img,
        out_img             = out_img,
        fi_img              = fi_img,
        cm_img              = cm_img,
    )

    html = template.render(**ctx)
    report_path = os.path.join(output_dir, 'dataset_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return report_path
