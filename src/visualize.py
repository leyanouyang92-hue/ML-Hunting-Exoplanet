# --- 放在 src/visualize.py 末尾 ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

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

# 简明解释（按你的项目）
DESC = {
    "prad":    "Planet radius",
    "srad":    "Stellar radius",
    "period":  "Orbital period",
    "duration":"Transit duration",
    "depth":   "Transit depth",
    "steff":   "Stellar effective temperature",
}

def fig_feature_importance(model, feature_names):
    """
    深色主题 + 银色横条；右边按重要性顺序给出文字解释；整体偏左。
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Model has no feature_importances_.")

    # 排序（从大到小）
    order = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in order]
    vals  = importances[order]

    # 画布：左图 + 右侧说明
    fig = plt.figure(figsize=(9.8, 4.0), facecolor="#0B1020")
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.9], wspace=0.25)
    ax  = fig.add_subplot(gs[0, 0])     # 左：条形图
    axR = fig.add_subplot(gs[0, 1])     # 右：文字说明

    # 左侧轴背景 & 样式
    ax.set_facecolor("#101735")
    axR.set_facecolor("#0B1020")  # 与整体背景一致
    for spine in ax.spines.values():
        spine.set_edgecolor("#1D2A55")

    # 只取前若干（可改）；或保留全部
    top_k = len(names)  # 改成 6/8 也行
    names_k = names[:top_k]
    vals_k  = vals[:top_k]

    # 银色横条（带亮色描边）
    bars = ax.barh(names_k, vals_k,
                   color="#C0C0C0", edgecolor="#E6EDF3", linewidth=0.6)
    ax.invert_yaxis()

    ax.set_title("Which physical parameters matter most?",
                 color="#FFFFFF", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Feature importance", color="#E6EDF3")
    ax.tick_params(colors="#E6EDF3", labelsize=10)
    ax.grid(axis="x", color="#24355C", alpha=0.35, linewidth=0.6)
    ax.margins(x=0.02, y=0.08)

    # 右侧说明：按重要性顺序逐行写明缩写含义 + 数值
    axR.set_axis_off()
    y0 = 0.95
    dy = 1.0 / (top_k + 1.5)  # 行距
    for i, (n, v) in enumerate(zip(names_k, vals_k)):
        y = y0 - i * dy

        # 小银色方块作为 bullet
        axR.add_patch(mpatches.Rectangle((0.02, y-0.012), 0.012, 0.012,
                                         transform=axR.transAxes,
                                         facecolor="#C0C0C0", edgecolor="#E6EDF3", lw=0.4))
        # 名称 + 解释
        desc = DESC.get(n, "")  # 未收录则给空字符串
        label = f"{n} — {desc}"
        axR.text(0.04, y, label, transform=axR.transAxes,
                 va="center", ha="left", fontsize=10, color="#E6EDF3")
        # 右侧给个重要性数值
        axR.text(0.98, y, f"{v:.3f}", transform=axR.transAxes,
                 va="center", ha="right", fontsize=10, color="#9fb3c8")

    # 让右侧说明顶端略贴左、整体看起来更“偏左”
    axR.set_xlim(0, 1.0)
    axR.set_ylim(0, 1.0)

    plt.tight_layout()
    return fig

def plot_confusion_matrix_dark(
    y_true, y_pred,
    labels=(0, 1),
    label_names=None,
    normalize=False,
    title="Confusion Matrix"
):
    # 先拿“整数计数”的混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)
        fmt = ".2f"            # 归一化 -> 小数
        cbar_label = "Proportion"
        vmin, vmax = 0.0, 1.0
        ttl = f"{title} (normalized)"
    else:
        # 保持整数，不转 float；或者也可以 fmt=".0f"
        fmt = ".0f"            # 计数 -> 整数显示
        cbar_label = "Count"
        vmin, vmax = None, None
        ttl = title

    if label_names is None:
        label_names = [str(l) for l in labels]

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    fig.patch.set_facecolor("#0B1020")
    ax.set_facecolor("#0B1020")

    sns.heatmap(
        cm,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap=sns.color_palette(["#0B1020", "#30384f", "#bfc4cc"], as_cmap=True),
        linewidths=1,
        linecolor="#1D2A55",
        cbar_kws={"label": cbar_label},
        vmin=vmin, vmax=vmax
    )

    ax.set_title(ttl, color="#E6EDF3", pad=12, fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted", color="#E6EDF3")
    ax.set_ylabel("True", color="#E6EDF3")
    ax.set_xticklabels(label_names, color="#E6EDF3", rotation=0)
    ax.set_yticklabels(label_names, color="#E6EDF3", rotation=0)

    for spine in ax.spines.values():
        spine.set_edgecolor("#1D2A55")

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#E6EDF3")
    cbar.ax.tick_params(colors="#E6EDF3")

    return fig

_DARK_BG   = "#0B1020"
_DARK_EDGE = "#1D2A55"
_TEXT_COL  = "#E6EDF3"
_LINE_SILV = "#D0D5DD"
_LINE_BASE = "#6B7280"  # 参考线

def plot_roc_dark(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)

    # ROC 主曲线（银色）
    ax.plot(fpr, tpr, color=_LINE_SILV, lw=2.4, label=f"AUC = {roc_auc:.3f}")

    # 随机猜测基线：加 label 进图例
    ax.plot([0, 1], [0, 1],
            ls="--", color=_LINE_BASE, lw=1.4,
            label="Random guess (TPR = FPR)")

    # 在图中沿虚线给个斜着的说明（位置/旋转可按需微调）
    ax.text(0.64, 0.60, "Random guess",
            color=_LINE_BASE, fontsize=10.5, rotation=34,
            ha="center", va="center", alpha=0.95)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC Curve", color=_TEXT_COL, fontsize=15, fontweight="bold", pad=8)
    ax.set_xlabel("False Positive Rate", color=_TEXT_COL)
    ax.set_ylabel("True Positive Rate",  color=_TEXT_COL)

    ax.grid(True, color=_DARK_EDGE, alpha=.35)
    ax.tick_params(colors=_TEXT_COL)
    for s in ax.spines.values():
        s.set_color(_DARK_EDGE)

    # 图例：显示 AUC + Random guess
    leg = ax.legend(loc="lower right",
                    facecolor="#101735", edgecolor=_DARK_EDGE, framealpha=.95)
    for t in leg.get_texts():
        t.set_color(_TEXT_COL)

    return fig


def plot_pr_dark(y_true, y_proba):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)

    ax.plot(rec, prec, color=_LINE_SILV, lw=2.4, label=f"AP = {ap:.3f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision–Recall", color=_TEXT_COL, fontsize=15, fontweight="bold", pad=8)
    ax.set_xlabel("Recall",    color=_TEXT_COL)
    ax.set_ylabel("Precision", color=_TEXT_COL)

    ax.grid(True, color=_DARK_EDGE, alpha=.35)
    ax.tick_params(colors=_TEXT_COL)
    for s in ax.spines.values():
        s.set_color(_DARK_EDGE)

    ax.legend(loc="lower left", facecolor="#101735", edgecolor=_DARK_EDGE, labelcolor=_TEXT_COL)
    return fig