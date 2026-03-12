"""
analyzer.py - Core dataset analysis module for the Automated Dataset Analyzer.
Performs data loading, overview generation, quality checks, and insight extraction.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the given filepath.
    Handles encoding errors and basic validation.
    
    Args:
        filepath: Path to the CSV file.
    Returns:
        Loaded pandas DataFrame.
    Raises:
        ValueError: If the file is empty or cannot be parsed.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip')
            break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Could not read file: {filepath}. Check encoding or format.")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if len(df.columns) < 2:
        raise ValueError("Dataset has fewer than 2 columns.")

    # Drops completely empty rows
    df = df.dropna(how='all')
    return df


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive overview and quality report for the dataset.
    
    Args:
        df: The input DataFrame.
    Returns:
        Dictionary containing all analysis results.
    """
    results = {}

    # Basic Info
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = _detect_datetime_cols(df, categorical_cols)
    categorical_cols = [c for c in categorical_cols if c not in datetime_cols]

    results['overview'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'datetime_columns': datetime_cols,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        'column_dtypes': df.dtypes.astype(str).to_dict(),
    }

    # Descriptive Stats
    if numeric_cols:
        results['descriptive_stats'] = df[numeric_cols].describe().round(4).to_dict()
    else:
        results['descriptive_stats'] = {}

    # Missing Values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    }).query('missing_count > 0').sort_values('missing_count', ascending=False)
    results['missing_values'] = missing_df.to_dict('index')
    results['total_missing'] = int(missing.sum())
    results['total_missing_pct'] = round(missing.sum() / (len(df) * len(df.columns)) * 100, 2)

    # Duplicates
    dup_count = int(df.duplicated().sum())
    results['duplicates'] = {
        'count': dup_count,
        'pct': round(dup_count / len(df) * 100, 2)
    }

    # Outliers (IQR method)
    results['outliers'] = detect_outliers(df, numeric_cols)

    # Correlations
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(4)
        results['correlation_matrix'] = corr.to_dict()
        # Find strong correlations (>0.7, excluding diagonal)
        strong_corrs = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    val = corr.loc[col1, col2]
                    if abs(val) >= 0.7:
                        strong_corrs.append((col1, col2, round(val, 3)))
        results['strong_correlations'] = strong_corrs
    else:
        results['correlation_matrix'] = {}
        results['strong_correlations'] = []

    # Categorical Value Counts
    cat_value_counts = {}
    for col in categorical_cols[:10]:  # limit to first 10
        vc = df[col].value_counts().head(10)
        cat_value_counts[col] = {
            'counts': vc.to_dict(),
            'unique': int(df[col].nunique())
        }
    results['categorical_analysis'] = cat_value_counts

    return results


def detect_outliers(df: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Detect outliers in numeric columns using the IQR method.
    
    Args:
        df: The input DataFrame.
        numeric_cols: List of numeric column names.
    Returns:
        Dictionary mapping column names to outlier counts.
    """
    outliers = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((series < lower) | (series > upper)).sum())
        outliers[col] = {
            'count': n_out,
            'pct': round(n_out / len(series) * 100, 2),
            'lower_bound': round(lower, 4),
            'upper_bound': round(upper, 4),
        }
    return outliers


def generate_insights(analysis: dict, ml_results: dict = None) -> list:
    """
    Auto-generate human-readable insight bullet points from analysis results.
    
    Args:
        analysis: Output from analyze_dataset().
        ml_results: Optional ML results dict.
    Returns:
        List of insight strings.
    """
    insights = []
    ov = analysis['overview']

    insights.append(f"Dataset contains {ov['rows']:,} rows and {ov['columns']} columns "
                    f"({len(ov['numeric_columns'])} numeric, {len(ov['categorical_columns'])} categorical).")

    # Missing values
    if analysis['missing_values']:
        worst_col = list(analysis['missing_values'].keys())[0]
        worst_pct = analysis['missing_values'][worst_col]['missing_pct']
        insights.append(f"Column \"{worst_col}\" has the most missing data: {worst_pct}% missing values.")
    else:
        insights.append("No missing values detected — the dataset is complete.")

    # Duplicates
    dup = analysis['duplicates']
    if dup['count'] > 0:
        insights.append(f"Dataset contains {dup['count']:,} duplicate rows ({dup['pct']}%).")
    else:
        insights.append("No duplicate rows found in the dataset.")

    # Outliers
    if analysis['outliers']:
        worst_out = max(analysis['outliers'].items(), key=lambda x: x[1]['count'])
        insights.append(
            f"Column \"{worst_out[0]}\" has the most outliers: {worst_out[1]['count']:,} "
            f"({worst_out[1]['pct']}% of values)."
        )

    # Correlations
    if analysis['strong_correlations']:
        c1, c2, val = analysis['strong_correlations'][0]
        direction = "positive" if val > 0 else "negative"
        insights.append(f"Strong {direction} correlation ({val}) between \"{c1}\" and \"{c2}\".")

    # Memory
    insights.append(f"Dataset uses {ov['memory_usage_mb']} MB in memory.")

    # ML insights
    if ml_results and ml_results.get('success'):
        task = ml_results.get('task', '')
        target = ml_results.get('target_column', '')
        if task == 'classification':
            acc = ml_results.get('metrics', {}).get('accuracy', 0)
            insights.append(f"ML classification on \"{target}\" achieved {acc*100:.1f}% accuracy.")
        elif task == 'regression':
            r2 = ml_results.get('metrics', {}).get('r2', 0)
            insights.append(f"ML regression on \"{target}\" achieved R² = {r2:.3f}.")
        fi = ml_results.get('feature_importances', [])
        if fi:
            top_feat = fi[0]['feature']
            insights.append(f"Top predictive feature: \"{top_feat}\".")

    return insights


def _detect_datetime_cols(df: pd.DataFrame, candidate_cols: list) -> list:
    """Try to detect datetime columns among string columns."""
    dt_cols = []
    for col in candidate_cols[:20]:
        try:
            sample = df[col].dropna().head(50)
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
            if parsed.notna().mean() > 0.8:
                dt_cols.append(col)
        except Exception:
            pass
    return dt_cols


def detect_target_column(df: pd.DataFrame) -> str | None:
    """
    Detect a likely target column for ML. Checks common names first,
    then falls back to low-cardinality columns.
    
    Args:
        df: The input DataFrame.
    Returns:
        Column name or None.
    """
    common_names = ['target', 'label', 'class', 'y', 'outcome', 'output',
                    'result', 'prediction', 'response']
    cols_lower = {c.lower(): c for c in df.columns}
    for name in common_names:
        if name in cols_lower:
            return cols_lower[name]

    # Guess: last column with low cardinality or numeric
    last_col = df.columns[-1]
    return last_col
