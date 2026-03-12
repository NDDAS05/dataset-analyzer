"""
ml_insights.py - Machine learning analysis module for the Automated Dataset Analyzer.
Automatically detects the task type, trains a Random Forest model, and returns metrics.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)


def run_ml_analysis(df: pd.DataFrame, target_col: str) -> dict:
    """
    Run automated ML analysis on the dataset.

    Detects classification vs regression, preprocesses data, trains a
    Random Forest model, and returns metrics + feature importances.

    Args:
        df: The input DataFrame.
        target_col: Name of the target column.
    Returns:
        Dictionary with ML results.
    """
    results = {'success': False, 'target_column': target_col}

    if target_col not in df.columns:
        results['error'] = f"Target column '{target_col}' not found."
        return results

    # Drop rows where target is missing
    df_ml = df.dropna(subset=[target_col]).copy()
    if len(df_ml) < 30:
        results['error'] = "Not enough rows for ML (need >= 30)."
        return results

    y_raw = df_ml[target_col]
    X_raw = df_ml.drop(columns=[target_col])

    # Detect task type
    task = _detect_task(y_raw)
    results['task'] = task

    # Prepare target
    le_target = None
    if task == 'classification':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_raw.astype(str))
        results['classes'] = list(le_target.classes_)
    else:
        y = y_raw.values.astype(float)

    # Prepare features
    X, feature_names = _preprocess_features(X_raw)
    results['feature_names'] = feature_names

    if X.shape[1] == 0:
        results['error'] = "No usable feature columns after preprocessing."
        return results

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except Exception as e:
        results['error'] = f"Train/test split failed: {e}"
        return results

    # Train model
    try:
        if task == 'classification':
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=42, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=42, n_jobs=-1
            )
        model.fit(X_train, y_train)
    except Exception as e:
        results['error'] = f"Model training failed: {e}"
        return results

    # Metrics
    y_pred = model.predict(X_test)
    metrics = {}
    if task == 'classification':
        metrics['accuracy'] = round(float(accuracy_score(y_test, y_pred)), 4)
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()
        results['confusion_matrix_labels'] = [str(c) for c in le_target.classes_]
    else:
        metrics['r2']   = round(float(r2_score(y_test, y_pred)), 4)
        metrics['mae']  = round(float(mean_absolute_error(y_test, y_pred)), 4)
        metrics['rmse'] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

    results['metrics'] = metrics

    # Feature importances
    importances = model.feature_importances_
    fi_list = sorted(
        [{'feature': name, 'importance': round(float(imp), 5)}
         for name, imp in zip(feature_names, importances)],
        key=lambda x: x['importance'], reverse=True
    )
    results['feature_importances'] = fi_list[:15]

    results['train_samples'] = int(len(X_train))
    results['test_samples']  = int(len(X_test))
    results['success'] = True
    return results





# Helpers

def _detect_task(y: pd.Series) -> str:
    """Classify target as 'classification' or 'regression'."""
    if y.dtype == object or str(y.dtype) == 'category':
        return 'classification'
    n_unique = y.nunique()
    if n_unique <= 20 or n_unique / len(y) < 0.05:
        return 'classification'
    return 'regression'


def _preprocess_features(X: pd.DataFrame):
    """
    Encode categoricals, fill missing values, return numpy array + feature names.
    Drops datetime and high-cardinality text columns.
    """
    X = X.copy()
    drop_cols = []
    for col in X.columns:
        if X[col].dtype == object:
            n_unique = X[col].nunique()
            if n_unique > 50 or n_unique / max(len(X), 1) > 0.5:
                drop_cols.append(col)
    X = X.drop(columns=drop_cols)

    # Try to parse potential datetime columns
    for col in X.select_dtypes(include='object').columns:
        try:
            X[col] = pd.to_datetime(X[col], infer_datetime_format=True, errors='coerce')
        except Exception:
            pass

    # Convert datetime to numeric timestamp
    for col in X.select_dtypes(include='datetime').columns:
        X[col] = X[col].astype('int64') // 10**9  # unix seconds

    # Fill missing
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    # Encode remaining categoricals
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].fillna('__missing__')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    feature_names = list(X.columns)
    return X.values, feature_names
