"""
visualizer.py - Visualization module for the Automated Dataset Analyzer.
Light, calm, cartoon-like aesthetic. One chart per feature.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BG      = "#FAFBFF"
SURFACE = "#FFFFFF"
ACCENT  = "#6C8EF5"
ACCENT2 = "#F4845F"
ACCENT3 = "#52C17E"
ACCENT4 = "#B97FF5"
WARN    = "#F5C842"
TEXT    = "#2D3250"
SUBTLE  = "#8A93B2"
GRID    = "#E8ECF8"

PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4, WARN,
           "#5BC4F5", "#F5A3C0", "#7ED8C8", "#F5D07A", "#A8D8A8"]


def _setup_style():
    plt.rcParams.update({
        'figure.facecolor':  BG,
        'axes.facecolor':    SURFACE,
        'axes.edgecolor':    GRID,
        'axes.labelcolor':   SUBTLE,
        'axes.titlecolor':   TEXT,
        'xtick.color':       SUBTLE,
        'ytick.color':       SUBTLE,
        'text.color':        TEXT,
        'grid.color':        GRID,
        'grid.linewidth':    1.0,
        'axes.grid':         True,
        'axes.titlesize':    12,
        'axes.labelsize':    10,
        'font.family':       'DejaVu Sans',
        'figure.dpi':        130,
    })

_setup_style()


def generate_visualizations(df: pd.DataFrame, analysis: dict, output_dir: str) -> dict:
    """
    Generate all charts and save to output_dir/plots/.
    Returns dict with chart path lists. Each numeric/categorical feature
    gets its own individual full-size chart.
    """
    plots_dir = os.path.join(output_dir, '_assets')
    os.makedirs(plots_dir, exist_ok=True)
    chart_paths = {}

    numeric_cols = analysis['overview']['numeric_columns']
    cat_cols     = analysis['overview']['categorical_columns']

    print("  -> Numeric distributions (individual)...")
    chart_paths['distributions'] = _plot_distributions_individual(df, numeric_cols, plots_dir)

    print("  -> Correlation heatmap...")
    if len(numeric_cols) >= 2:
        chart_paths['correlation_heatmap'] = _plot_correlation_heatmap(df, numeric_cols, plots_dir)

    print("  -> Categorical charts (individual)...")
    chart_paths['categorical'] = _plot_categorical_individual(df, cat_cols, plots_dir)

    print("  -> Missing values chart...")
    mv = _plot_missing_values(analysis, plots_dir)
    if mv:
        chart_paths['missing_values'] = mv

    print("  -> Outlier chart...")
    out = _plot_outliers(analysis, plots_dir)
    if out:
        chart_paths['outliers'] = out

    return chart_paths


def _plot_distributions_individual(df, numeric_cols, plots_dir):
    results = []
    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()
        if data.empty:
            continue
        color = PALETTE[i % len(PALETTE)]

        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
        ax.set_facecolor(SURFACE)

        ax.hist(data, bins=35, color=color, alpha=0.80,
                edgecolor='white', linewidth=1.2, zorder=3)

        mean_v   = data.mean()
        median_v = data.median()
        ax.axvline(mean_v,   color=ACCENT2, linestyle='--', linewidth=2.2,
                   label=f'Mean: {mean_v:.2f}', zorder=4)
        ax.axvline(median_v, color=ACCENT3, linestyle=':',  linewidth=2.2,
                   label=f'Median: {median_v:.2f}', zorder=4)

        ax.set_title(f'Distribution of  {col}', color=TEXT,
                     fontsize=14, fontweight='bold', pad=14)
        ax.set_xlabel(col, color=SUBTLE, fontsize=11)
        ax.set_ylabel('Count', color=SUBTLE, fontsize=11)
        ax.legend(fontsize=10, framealpha=0.95, facecolor=SURFACE,
                  edgecolor=GRID, labelcolor=TEXT)
        _style_ax(ax)
        fig.tight_layout()

        fname = f'dist_{_safe_name(col)}.png'
        fig.savefig(os.path.join(plots_dir, fname),
                    bbox_inches='tight', facecolor=BG, dpi=140)
        plt.close(fig)
        results.append({'col': col, 'path': f'_assets/{fname}'})
    return results


def _plot_categorical_individual(df, cat_cols, plots_dir):
    results = []
    for i, col in enumerate(cat_cols[:12]):
        vc = df[col].value_counts().head(10)
        if vc.empty:
            continue
        color = PALETTE[i % len(PALETTE)]
        light  = _lighten(color, 0.40)

        n_bars = len(vc)
        fig, ax = plt.subplots(
            figsize=(10, max(4, n_bars * 0.60 + 1.8)), facecolor=BG)
        ax.set_facecolor(SURFACE)

        colors = [color if j % 2 == 0 else light for j in range(n_bars)]
        bars = ax.barh(vc.index[::-1].astype(str), vc.values[::-1],
                       color=colors[::-1], edgecolor='white',
                       linewidth=1.0, height=0.62, zorder=3)

        for bar, val in zip(bars, vc.values[::-1]):
            ax.text(bar.get_width() + vc.max() * 0.013,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:,}', va='center', ha='left',
                    color=SUBTLE, fontsize=10, fontweight='500')

        ax.set_title(f'Category Counts — {col}', color=TEXT,
                     fontsize=14, fontweight='bold', pad=14)
        ax.set_xlabel('Count', color=SUBTLE, fontsize=11)
        ax.set_xlim(0, vc.max() * 1.20)
        _style_ax(ax)
        fig.tight_layout()

        fname = f'cat_{_safe_name(col)}.png'
        fig.savefig(os.path.join(plots_dir, fname),
                    bbox_inches='tight', facecolor=BG, dpi=140)
        plt.close(fig)
        results.append({'col': col, 'path': f'_assets/{fname}'})
    return results


def _plot_correlation_heatmap(df, numeric_cols, plots_dir):
    corr = df[numeric_cols].corr()
    n    = len(numeric_cols)
    size = max(7, min(n * 1.0, 16))

    fig, ax = plt.subplots(figsize=(size, size * 0.85), facecolor=BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, s=65, l=62, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                annot=len(numeric_cols) <= 15, fmt='.2f',
                square=True, ax=ax,
                linewidths=2.5, linecolor=BG,
                cbar_kws={'shrink': 0.7},
                annot_kws={'size': 9, 'color': TEXT, 'weight': 'bold'})

    ax.set_title('Correlation Heatmap', color=TEXT,
                 fontsize=14, fontweight='bold', pad=14)
    ax.tick_params(colors=SUBTLE, labelsize=10)
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()

    path = os.path.join(plots_dir, 'correlation_heatmap.png')
    fig.savefig(path, bbox_inches='tight', facecolor=BG, dpi=140)
    plt.close(fig)
    return '_assets/correlation_heatmap.png'


def _plot_missing_values(analysis, plots_dir):
    mv = analysis['missing_values']
    if not mv:
        return None
    cols = list(mv.keys())[:15]
    pcts = [mv[c]['missing_pct'] for c in cols]

    fig, ax = plt.subplots(
        figsize=(10, max(3, len(cols) * 0.55 + 1.5)), facecolor=BG)
    ax.set_facecolor(SURFACE)

    colors = [ACCENT2 if p > 50 else WARN if p > 20 else ACCENT3 for p in pcts]
    ax.barh(cols[::-1], pcts[::-1], color=colors[::-1],
            edgecolor='white', linewidth=1.0, height=0.6, zorder=3)

    for i, (col, pct) in enumerate(zip(cols[::-1], pcts[::-1])):
        ax.text(pct + 0.5, i, f'{pct}%', va='center',
                color=SUBTLE, fontsize=10)

    ax.set_xlim(0, 110)
    ax.set_xlabel('Missing %', color=SUBTLE)
    ax.set_title('Missing Values by Column', color=TEXT,
                 fontsize=14, fontweight='bold', pad=14)
    _style_ax(ax)
    fig.tight_layout()

    path = os.path.join(plots_dir, 'missing_values.png')
    fig.savefig(path, bbox_inches='tight', facecolor=BG, dpi=140)
    plt.close(fig)
    return '_assets/missing_values.png'


def _plot_outliers(analysis, plots_dir):
    out = {c: v for c, v in analysis['outliers'].items() if v['count'] > 0}
    if not out:
        return None
    cols = list(out.keys())[:12]
    pcts = [out[c]['pct'] for c in cols]

    fig, ax = plt.subplots(
        figsize=(10, max(3, len(cols) * 0.55 + 1.5)), facecolor=BG)
    ax.set_facecolor(SURFACE)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(cols))]
    ax.barh(cols[::-1], pcts[::-1], color=colors[::-1],
            edgecolor='white', linewidth=1.0, height=0.6, zorder=3)

    for i, pct in enumerate(pcts[::-1]):
        ax.text(pct + 0.2, i, f'{pct}%', va='center', color=SUBTLE, fontsize=10)

    ax.set_xlabel('Outlier %', color=SUBTLE)
    ax.set_title('Outliers per Column (IQR Method)', color=TEXT,
                 fontsize=14, fontweight='bold', pad=14)
    _style_ax(ax)
    fig.tight_layout()

    path = os.path.join(plots_dir, 'outliers.png')
    fig.savefig(path, bbox_inches='tight', facecolor=BG, dpi=140)
    plt.close(fig)
    return '_assets/outliers.png'


def plot_feature_importance(feature_importances: list, plots_dir: str) -> str:
    if not feature_importances:
        return None
    fi = feature_importances[:15]
    features    = [x['feature'] for x in fi]
    importances = [x['importance'] for x in fi]

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(features) * 0.55 + 1.5)), facecolor=BG)
    ax.set_facecolor(SURFACE)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(features))]
    ax.barh(features[::-1], importances[::-1], color=colors[::-1],
            edgecolor='white', linewidth=1.0, height=0.6, zorder=3)

    for i, val in enumerate(importances[::-1]):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center',
                color=SUBTLE, fontsize=10)

    ax.set_xlabel('Importance', color=SUBTLE)
    ax.set_title('Feature Importance (Random Forest)', color=TEXT,
                 fontsize=14, fontweight='bold', pad=14)
    _style_ax(ax)
    fig.tight_layout()

    path = os.path.join(plots_dir, 'feature_importance.png')
    fig.savefig(path, bbox_inches='tight', facecolor=BG, dpi=140)
    plt.close(fig)
    return '_assets/feature_importance.png'


def plot_confusion_matrix(cm: list, labels: list, plots_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    cmap = sns.light_palette(ACCENT, as_cmap=True)
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=2.5, linecolor=BG,
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 12, 'weight': 'bold', 'color': TEXT})

    ax.set_xlabel('Predicted', color=SUBTLE, fontsize=11)
    ax.set_ylabel('Actual',    color=SUBTLE, fontsize=11)
    ax.set_title('Confusion Matrix', color=TEXT, fontsize=14,
                 fontweight='bold', pad=14)
    ax.tick_params(colors=SUBTLE, labelsize=10)
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()

    path = os.path.join(plots_dir, 'confusion_matrix.png')
    fig.savefig(path, bbox_inches='tight', facecolor=BG, dpi=140)
    plt.close(fig)
    return '_assets/confusion_matrix.png'


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=SUBTLE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)


def _safe_name(s: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in s)


def _lighten(hex_color: str, amount: float) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return '#{:02x}{:02x}{:02x}'.format(
        int(r + (255 - r) * amount),
        int(g + (255 - g) * amount),
        int(b + (255 - b) * amount),
    )
