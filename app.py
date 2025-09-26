# app.py —— Streamlit 前端，复用 src/ 内模块
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.harmonize import harmonize_table
from src.config import CANON_ORDER
from src.visualize import fig_feature_importance, plot_confusion_matrix_dark, plot_roc_dark, plot_pr_dark
from src.ui import inject_css, title_bar, upload_card, kpi_grid, render_universe_html
from streamlit.components.v1 import html

# ---------------- UI --------------
st.set_page_config(page_title="Exoplanet ML Explorer", layout="wide")
inject_css()
title_bar()

# ---------------- 读上传文件 --------------
def read_uploaded(file):
    if file is None:
        return None
    name = file.name.lower()
    data = file.read()
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(data))
    # CSV/TSV（带 # 注释也兼容）
    try:
        return pd.read_csv(io.BytesIO(data), comment="#", low_memory=False)
    except Exception:
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python", comment="#", low_memory=False)

with st.sidebar:
    st.header("1) Data Upload")
    koi_up  = upload_card("Kepler cumulative", key="koi",  types=["csv","tsv","txt","xlsx","xls"])
    tess_up = upload_card("TESS TOI",         key="tess", types=["csv","tsv","txt","xlsx","xls"])
    k2_up   = upload_card("K2 P&C",           key="k2",   types=["csv","tsv","txt","xlsx","xls"])

    st.header("2) Training parameters")
    test_size    = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("RF trees", 100, 1000, 400, 50)
    rs           = st.number_input("Random state", 0, 9999, 42, step=1)

# ---------------- 统一与合并 --------------
dfs = []
for up, mission in [(koi_up, "Kepler"), (tess_up, "TESS"), (k2_up, "K2")]:
    raw = read_uploaded(up)
    if raw is not None:
        h = harmonize_table(raw, mission)  # 内部已尽量保留 ra/dec/dist_raw
        st.sidebar.write(f"{mission}: {len(h)} row | with label: {h['label'].notna().sum()}")
        dfs.append(h)

if not dfs:
    st.info("👉 At least one file need to be uploaded（Kepler/TESS/K2）。")
    st.stop()

df_all = pd.concat(dfs, ignore_index=True)

# 3D 宇宙视图（使用合并后的全量数据；_to_xyz 会优先用 RA/DEC）
html(render_universe_html(df_all, color_by="label", max_points=6000, point_size=1.8), height=560)

# ---------------- 训练数据准备（仅物理特征，不把距离当特征） ----------------
# 只保留有标签 0/1 的样本
df_lab = df_all[df_all["label"].isin([0, 1])].copy()
if df_lab.empty:
    st.error("Examples with no available labels (label is 0/1).")
    st.stop()

# 物理特征白名单（先去掉真实行星半径，防止泄漏）
LEAKY = {"prad", "pl_rade", "planet_radius"}
feature_cols = [c for c in CANON_ORDER if c in df_lab.columns and c not in LEAKY]

# ——可选：把 depth 统一成分数（0.01 表示 1%），避免某些数据用 % 或 ppm——
def depth_to_fraction(s):
    d = pd.to_numeric(s, errors="coerce")
    out = d.copy()
    # 0~1 视为分数；1~100 视为百分比；>100 视为 ppm
    out[(d > 1) & (d <= 100)] = d[(d > 1) & (d <= 100)] / 100.0
    out[d > 100] = d[d > 100] / 1e6
    return out

# 工程化半径：prad_est = srad * sqrt(depth_fraction)
if {"srad", "depth"}.issubset(df_lab.columns):
    srad  = pd.to_numeric(df_lab["srad"], errors="coerce")
    depth = depth_to_fraction(df_lab["depth"])
    df_lab["prad_est"] = srad * np.sqrt(depth)
    feature_cols.append("prad_est")

# 生成数值特征矩阵（现在会包含 prad_est，而不含 prad）
num_feats = df_lab[feature_cols].apply(pd.to_numeric, errors="coerce")


# 丢掉“整列皆缺失”的特征，避免插补失败
all_nan_cols = [c for c in num_feats.columns if num_feats[c].notna().sum() == 0]
if all_nan_cols:
    st.warning(f"These features are all missing in the current data and have been ignored：{all_nan_cols}")
    num_feats = num_feats.drop(columns=all_nan_cols)

# mission one-hot（不保留原始 mission 列）
if "mission" in df_lab.columns and df_lab["mission"].nunique() > 1:
    df_lab = pd.get_dummies(df_lab, columns=["mission"], drop_first=True)
    mission_cols = [c for c in df_lab.columns if c.startswith("mission_")]
else:
    df_lab = df_lab.drop(columns=[c for c in ["mission"] if c in df_lab.columns])
    mission_cols = []

# 组装 X / y（仅物理特征 + mission one-hot）
X = pd.concat([num_feats, df_lab[mission_cols]], axis=1).select_dtypes(include=["number"])
y = df_lab["label"].astype(int)

# 基本检查
if X.shape[1] == 0:
    st.error("Available physical features are listed as 0. Please check your upload data or reduce the required features.")
    st.stop()
if len(X) < 2:
    st.error(f"Insufficient number of trainable samples after cleaning ({len(X)}).")
    st.stop()

# 分层仅当类别≥2
stratify_y = y if y.nunique() > 1 else None

# 计算“可行”的 test_size：至少 1 条训练 + 1 条测试
n = len(X)
n_test = max(1, int(round(n * test_size)))
n_test = min(n - 1, n_test)
test_size_eff = n_test / n

# 切分：分层失败就降级非分层
try:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size_eff, random_state=rs, stratify=stratify_y
    )
except ValueError:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size_eff, random_state=rs, stratify=None
    )

# ---------------- 训练 ----------------
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced",
        random_state=rs, n_jobs=-1
    )
)
model.fit(X_tr, y_tr)
rf = model.named_steps["randomforestclassifier"]

# hold-out 概率与 “参考阈值”
y_proba = model.predict_proba(X_te)[:, 1]
prec0, rec0, thr0 = precision_recall_curve(y_te, y_proba)
f10 = 2 * prec0 * rec0 / (prec0 + rec0 + 1e-9)
best_idx0 = int(np.nanargmax(f10)) if len(thr0) else 0
best_thr0 = float(thr0[max(min(best_idx0, len(thr0) - 1), 0)]) if len(thr0) else 0.5

# 统一的阈值状态（让“表格在上，滑杆在下”也能同步）
st.session_state.setdefault("thr", float(best_thr0))
thr_now = float(st.session_state["thr"])

# ---------------- 结果浏览 / 下载（放在前面，用当前阈值） ----------------
st.subheader("3) Browse / Download Results")
pred_df = pd.DataFrame(
    {"y_true": y_te.values, "proba": y_proba, "pred": (y_proba >= thr_now).astype(int)},
    index=y_te.index
).join(X_te)

mission_cols_in_pred = [c for c in pred_df.columns if c.startswith("mission_")]
if mission_cols_in_pred:
    ms = st.multiselect("Filter by mission", mission_cols_in_pred, default=mission_cols_in_pred)
    mask = pred_df[ms].sum(axis=1) > 0
    view = pred_df[mask]
else:
    view = pred_df

topk = st.slider("Top-N by probability", 10, 200, 50, 10)
st.dataframe(view.sort_values("proba", ascending=False).head(topk))
st.download_button(
    "Download current view CSV",
    data=view.sort_values("proba", ascending=False).to_csv(index=False).encode("utf-8"),
    file_name="predictions_view.csv",
    mime="text/csv"
)

# ---------------- 评估模式 & 指标（放在后面，滑杆更新全局阈值） ----------------
st.subheader("4) Thresholds and indicators")
mode = st.radio(
    "Evaluate on",
    ["Hold-out test set", "All labeled (in-sample)", "5-fold CV (out-of-fold)"],
    index=0, horizontal=True
)

use_mode = mode
if mode == "5-fold CV (out-of-fold)" and (len(X) < 5 or y.nunique() < 2):
    st.warning("If there are not enough samples to perform 5-fold CV or there is only one category, it will automatically be changed to: All labeled (in-sample)")
    use_mode = "All labeled (in-sample)"

if use_mode == "Hold-out test set":
    y_eval = y_te
    proba_eval = y_proba
elif use_mode == "All labeled (in-sample)":
    y_eval = y
    proba_eval = model.predict_proba(X)[:, 1]
else:  # 5-fold CV（管道与训练一致，含插补）
    base_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=rs, n_jobs=-1
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs) if y.nunique() > 1 else None
    if skf is None:
        y_eval = y
        proba_eval = model.predict_proba(X)[:, 1]
        st.warning("With only one category, CV falls back to in-sample evaluation.")
    else:
        proba_eval = cross_val_predict(base_pipe, X, y, cv=skf, method="predict_proba")[:, 1]
        y_eval = y

# 用 proba_eval + 阈值做所有指标/曲线/混淆矩阵
prec, rec, thr = precision_recall_curve(y_eval, proba_eval)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = int(np.nanargmax(f1)) if len(thr) else 0
best_thr = float(thr[max(min(best_idx, len(thr) - 1), 0)]) if len(thr) else best_thr0

# 阈值滑杆（同步 st.session_state['thr']）
sel_thr = st.slider("Decision threshold", 0.0, 1.0, step=0.01, key="thr")
y_pred_eval = (proba_eval >= sel_thr).astype(int)

cm = confusion_matrix(y_eval, y_pred_eval, labels=[0, 1])
TN, FP, FN, TP = cm.ravel()
acc = (TP + TN) / cm.sum()
P = TP / (TP + FP + 1e-9)
R = TP / (TP + FN + 1e-9)
F1 = 2 * P * R / (P + R + 1e-9)

kpi_grid({"Accuracy": f"{acc:.3f}", "Precision": f"{P:.3f}", "Recall": f"{R:.3f}", "F1": f"{F1:.3f}"})

# ① Which physical parameters matter most?
st.subheader("Which physical parameters matter most?")
fig_imp = fig_feature_importance(rf, X.columns.tolist())
st.pyplot(fig_imp, clear_figure=True, use_container_width=False)

# ② Confusion Matrix
st.subheader("Confusion Matrix")

# 1) 用 session_state 记住当前是否归一化
if "cm_norm" not in st.session_state:
    st.session_state.cm_norm = False

def _toggle_cm_norm():
    st.session_state.cm_norm = not st.session_state.cm_norm

c1, c2 = st.columns([3, 2], vertical_alignment="top")

with c1:
    fig_cm = plot_confusion_matrix_dark(
        y_eval, y_pred_eval,
        labels=(0, 1),
        label_names=("No exoplanet (0)", "Exoplanet (1)"),
        normalize=st.session_state.cm_norm,
        title="Confusion Matrix (normalized)" if st.session_state.cm_norm else "Confusion Matrix"
    )
    st.pyplot(fig_cm, clear_figure=True)

with c2:
    st.markdown(
        """
**How to read this?**  
- **Top-left (TN)**: true 0 predicted 0  
- **Top-right (FP)**: true 0 predicted 1 (**false alarm**)  
- **Bottom-left (FN)**: true 1 predicted 0 (**miss**)  
- **Bottom-right (TP)**: true 1 predicted 1  

> If **Normalization** is enabled, each row is scaled to 0–1 by its row total, making it easier to compare class proportions when classes are imbalanced.
        """
    )

    # 2) 这个按钮就放在这句旁边：做两列，左边放这句话，右边放按钮
    msg_col, btn_col = st.columns([4, 2])
    with msg_col:
        st.write("")  # 占位，让两列对齐
    with btn_col:
        st.button(
            "Change to normalization graph" if not st.session_state.cm_norm else "Show counts",
            on_click=_toggle_cm_norm,
            use_container_width=True
        )

# ③ ROC & Precision–Recall
# ----- ROC & PR section -----
st.subheader("ROC & Precision–Recall")
c1, c2 = st.columns(2, gap="large")

with c1:
    st.pyplot(plot_roc_dark(y_eval, proba_eval), clear_figure=True)

with c2:
    st.pyplot(plot_pr_dark(y_eval, proba_eval), clear_figure=True)

# 简短说明（放在两个折线图下面）
st.markdown(
    """
**What they mean:**
- **ROC curve**（TPR vs FPR）Shows the overall recognition ability from low to high thresholds; the larger the **AUC**, the better the overall ranking.
- **Precision–Recall**（P vs R）It is more sensitive to **positive example scarcity/class imbalance**: the closer the curve is to the upper right corner, the better; **AP** is its area metric.
"""
)
