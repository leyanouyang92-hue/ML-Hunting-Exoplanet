# src/model.py
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from .visualize import plot_dashboard

__all__ = ["train_and_eval"]

def train_and_eval(df_all: pd.DataFrame, random_state: int = 42) -> Tuple[object, pd.Series]:
    """Train RF + evaluate + draw charts. Returns (model, y_proba_on_test)."""
    # Prevent accidental deletion of column names
    assert "label" in df_all.columns, "df_all 缺少 label 列"
    X = df_all.drop(columns=[c for c in ["label", "label_raw"] if c in df_all.columns])
    y = df_all["label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    model.fit(X_tr, y_tr)

    # Probability and optimal threshold (using PR curve)
    y_proba = model.predict_proba(X_te)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_te, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1))
    # The length of thresholds is 1 less than precision/recall, perform some boundary handling
    thr = thresholds[max(min(best_idx, len(thresholds) - 1), 0)] if len(thresholds) else 0.5

    y_pred_best = (y_proba >= thr).astype(int)

    print(f"Best threshold = {thr:.3f}, P={precision[best_idx]:.2f}, R={recall[best_idx]:.2f}, F1={f1[best_idx]:.2f}")
    print("Report @ best threshold:\n", classification_report(y_te, y_pred_best))
    print(confusion_matrix(y_te, y_pred_best))

    # Visualisation
    rf = model.named_steps["randomforestclassifier"]
    plot_dashboard(rf, X.columns.tolist(), y_te, y_pred_best, y_proba, X_te_yte=(X_te, y_te))

    return model, y_proba
