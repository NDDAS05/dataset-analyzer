"""
main.py — Automated Dataset Analyzer
=====================================
After installing with  pip install -e .  it can be can run from anywhere:

    analyze dataset.csv
    analyze dataset.csv --target price
    analyze dataset.csv --no-ml
    analyze dataset.csv --no-launch

Or without installing:

    python main.py dataset.csv
"""

import sys
import os
import argparse
import time
import webbrowser

# Paths
# ROOT_DIR  = wherever main.py / the installed package lives
# REPORTS_DIR = ./reports/ relative to wherever the user runs the command from
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.join(ROOT_DIR, 'src')
REPORTS_DIR = os.path.join(os.getcwd(), 'reports')
sys.path.insert(0, SRC_DIR)

from analyzer        import load_data, analyze_dataset, detect_target_column, generate_insights
from visualizer      import generate_visualizations, plot_feature_importance, plot_confusion_matrix
from ml_insights     import run_ml_analysis
from report_generator import generate_report


def _banner():
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║    Automated Dataset Analyzer  v1.0          ║")
    print("  ║    By Nirupam Das                            ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()


def _step(msg: str):
    print(f"  ➜  {msg}")


def _info(msg: str):
    print(f"     {msg}")


def main():
    _banner()

    parser = argparse.ArgumentParser(
        description='Automated Dataset Analyzer — full EDA report from a CSV.'
    )
    parser.add_argument('csv_file',      help='Path to input CSV file')
    parser.add_argument('--target',      default=None,
                        help='Target column for ML (auto-detected if omitted)')
    parser.add_argument('--no-ml',       action='store_true',
                        help='Skip machine learning analysis')
    parser.add_argument('--no-launch',   action='store_true',
                        help='Do not auto-open the report in a browser')
    args = parser.parse_args()

    start = time.time()

    # Output dir: reports/<dataset_name>/
    dataset_stem = os.path.splitext(os.path.basename(args.csv_file))[0]
    dataset_name = dataset_stem.replace('_', ' ').replace('-', ' ').title()
    output_dir   = os.path.join(REPORTS_DIR, dataset_stem)
    plots_dir    = os.path.join(output_dir, '_assets')
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load 
    _step(f"Loading dataset: {args.csv_file}")
    try:
        df = load_data(args.csv_file)
    except Exception as e:
        print(f"\n  ✗  Could not load dataset: {e}")
        sys.exit(1)
    _info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # 2. Analyze 
    _step("Running analysis...")
    try:
        analysis = analyze_dataset(df)
    except Exception as e:
        print(f"\n  ✗  Analysis failed: {e}")
        sys.exit(1)
    ov = analysis['overview']
    _info(f"Numeric: {len(ov['numeric_columns'])} cols  |  "
          f"Categorical: {len(ov['categorical_columns'])} cols")
    _info(f"Missing: {analysis['total_missing_pct']}%  |  "
          f"Duplicates: {analysis['duplicates']['count']:,}")

    # 3. Charts 
    _step("Generating charts...")
    try:
        chart_paths = generate_visualizations(df, analysis, output_dir)
    except Exception as e:
        _info(f"Warning: charts partially failed — {e}")
        chart_paths = {}

    # 4. ML
    ml_results = {'skipped': True}
    if args.no_ml:
        _step("Skipping ML (--no-ml)")
    else:
        target_col = args.target or detect_target_column(df)
        _step(f"Training model (target: '{target_col}')...")
        try:
            ml_results = run_ml_analysis(df, target_col)
            if ml_results.get('success'):
                task    = ml_results['task']
                metrics = ml_results['metrics']
                if task == 'classification':
                    _info(f"Classification  |  Accuracy: {metrics['accuracy']*100:.1f}%")
                else:
                    _info(f"Regression  |  R²: {metrics['r2']}  MAE: {metrics['mae']}")

                if ml_results.get('feature_importances'):
                    fi = plot_feature_importance(ml_results['feature_importances'], plots_dir)
                    if fi:
                        chart_paths['feature_importance'] = fi

                if ml_results.get('confusion_matrix'):
                    cm = plot_confusion_matrix(
                        ml_results['confusion_matrix'],
                        ml_results.get('confusion_matrix_labels', []),
                        plots_dir
                    )
                    if cm:
                        chart_paths['confusion_matrix'] = cm
            else:
                _info(f"ML skipped: {ml_results.get('error', 'unknown')}")
        except Exception as e:
            _info(f"Warning: ML failed — {e}")
            ml_results = {'skipped': True}

    # 5. Insights
    _step("Building insights...")
    insights = generate_insights(analysis, ml_results)

    # 6. Report
    _step("Building report...")
    try:
        report_path = generate_report(
            df           = df,
            analysis     = analysis,
            chart_paths  = chart_paths,
            ml_results   = ml_results,
            insights     = insights,
            output_dir   = output_dir,
            dataset_name = dataset_name,
        )
    except Exception as e:
        print(f"\n  ✗  Report generation failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print(f" ║  > Done in {elapsed:.1f}s")
    print(f" ║  > Report:  {report_path}")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    # 7. Auto-launch browser
    if not args.no_launch:
        _step("Opening report in browser...")
        try:
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
        except Exception as e:
            _info(f"Could not open browser automatically: {e}")
            _info(f"Open manually: {report_path}")


if __name__ == '__main__':
    main()
