# --- 放在 src/visualize.py 末尾 ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

def build_dashboard_fig(model, feature_names, y_true, y_pred, y_proba):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Feature Importance
    ax = axes[0,0]
    if hasattr(model, "feature_importances_"):
        imps = np.asarray(model.feature_importances_)
        order = np.argsort(imps)
        ax.barh(np.array(feature_names)[order], imps[order])
        ax.set_xlabel("Feature Importance")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No feature importance", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Which physical parameters matter most?")

    # 2) Confusion Matrix
    ax = axes[0,1]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    # 3) ROC
    ax = axes[1,0]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1],'--'); ax.legend()
    ax.set_title("ROC Curve"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")

    # 4) PR
    ax = axes[1,1]
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.legend()
    ax.set_title("Precision–Recall"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")

    plt.tight_layout()
    return fig
