# app.py —— Streamlit 前端，复用 src/ 内模块
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.harmonize import harmonize_table
from src.config import CANON_ORDER
from src.visualize import build_dashboard_fig

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
    koi_up  = upload_card("Kepler cumulative", key="koi",
                          types=["csv", "tsv", "txt", "xlsx", "xls"])
    tess_up = upload_card("TESS TOI",         key="tess",
                          types=["csv", "tsv", "txt", "xlsx", "xls"])
    k2_up   = upload_card("K2 P&C",           key="k2",
                          types=["csv", "tsv", "txt", "xlsx", "xls"])

    st.header("2) Training parameters")
    test_size    = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("RF trees", 100, 1000, 400, 50)
    rs           = st.number_input("Random state", 0, 9999, 42, step=1)


# ---------------- 统一与合并 --------------
dfs = []
for up, mission in [(koi_up, "Kepler"), (tess_up, "TESS"), (k2_up, "K2")]:
    raw = read_uploaded(up)
    if raw is not None:
        h = harmonize_table(raw, mission)  # 这里已包含 ra/dec/dist_raw 的保留
        st.sidebar.write(f"{mission}: {len(h)} row | with label: {h['label'].notna().sum()}")
        dfs.append(h)

if not dfs:
    st.info("👉 At least one data file must be uploaded on the left side.（Kepler/TESS/K2）。")
    st.stop()

df_all = pd.concat(dfs, ignore_index=True)

# 3D 宇宙视图（使用全体合并数据；_to_xyz 会优先用 RA/DEC）
html(render_universe_html(df_all, color_by="label",
                          max_points=6000, point_size=1.8),
     height=560)


# ---------------- 训练用数据（稳健版；不使用距离作为特征） ----------------
from sklearn.impute import SimpleImputer

# 只保留有标签的行
df_lab = df_all[df_all["label"].isin([0, 1])].copy()
if df_lab.empty:
    st.error("没有可用标签（label 为 0/1）的样本。")
    st.stop()

# 物理特征白名单（不包含距离）
feature_cols = [c for c in CANON_ORDER if c in df_lab.columns]  # period,duration,depth,prad,srad,steff

# 强制数值化（非数值→NaN）
num_feats = df_lab[feature_cols].apply(pd.to_numeric, errors="coerce")

# 丢掉整列全缺失的特征（避免 Imputer 无法计算中位数）
all_nan_cols = [c for c in num_feats.columns if num_feats[c].notna().sum() == 0]
if all_nan_cols:
    st.warning(f"这些特征在当前数据里全缺失，已忽略：{all_nan_cols}")
    num_feats = num_feats.drop(columns=all_nan_cols)

# mission one-hot（不保留原始 mission 列）
if "mission" in df_lab.columns and df_lab["mission"].nunique() > 1:
    df_lab = pd.get_dummies(df_lab, columns=["mission"], drop_first=True)
    mission_cols = [c for c in df_lab.columns if c.startswith("mission_")]
else:
    df_lab = df_lab.drop(columns=[c for c in ["mission"] if c in df_lab.columns])
    mission_cols = []

# 组装 X / y（只有物理特征 + mission one-hot，不含任何距离列）
X = pd.concat([num_feats, df_lab[mission_cols]], axis=1).select_dtypes(include=["number"])
y = df_lab["label"].astype(int)

# 基本检查
if X.shape[1] == 0:
    st.error("可用的物理特征列为 0。请检查上传数据或减少必需特征。")
    st.stop()
if len(X) < 2:
    st.error(f"清洗后可训练样本数不足（{len(X)} 条）。")
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
# 先缺失值插补，再标准化，再随机森林
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
y_proba = model.predict_proba(X_te)[:, 1]

# F1 最优阈值（在 hold-out 上先给一个参考值）
prec0, rec0, thr0 = precision_recall_curve(y_te, y_proba)
f10 = 2 * prec0 * rec0 / (prec0 + rec0 + 1e-9)
best_idx0 = int(np.nanargmax(f10))
best_thr0 = float(thr0[max(min(best_idx0, len(thr0) - 1), 0)]) if len(thr0) else 0.5

# ---------------- 评估模式 ----------------
st.subheader("3) Thresholds and indicators")
mode = st.radio(
    "Evaluate on",
    ["Hold-out test set", "All labeled (in-sample)", "5-fold CV (out-of-fold)"],
    index=0, horizontal=True
)

use_mode = mode
if mode == "5-fold CV (out-of-fold)" and (len(X) < 5 or y.nunique() < 2):
    st.warning("样本不足以做 5-fold CV 或只有一个类别，自动改为：All labeled (in-sample)")
    use_mode = "All labeled (in-sample)"

if use_mode == "Hold-out test set":
    y_eval = y_te
    proba_eval = y_proba
elif use_mode == "All labeled (in-sample)":
    y_eval = y
    proba_eval = model.predict_proba(X)[:, 1]
else:  # 5-fold CV
    base_pipe = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=rs, n_jobs=-1
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs) if y.nunique() > 1 else None
    if skf is None:
        # 保险兜底：类别为 1 时退回 in-sample
        y_eval = y
        proba_eval = model.predict_proba(X)[:, 1]
        st.warning("只有一个类别，CV 退回 in-sample 评估。")
    else:
        proba_eval = cross_val_predict(base_pipe, X, y, cv=skf, method="predict_proba")[:, 1]
        y_eval = y

# 用 proba_eval + 阈值做所有指标/曲线/混淆矩阵
prec, rec, thr = precision_recall_curve(y_eval, proba_eval)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = int(np.nanargmax(f1))
best_thr = float(thr[max(min(best_idx, len(thr) - 1), 0)]) if len(thr) else best_thr0

sel_thr = st.slider("Decision threshold", 0.0, 1.0, best_thr, 0.01)
y_pred_eval = (proba_eval >= sel_thr).astype(int)

# 混淆矩阵固定 labels=[0,1]，防止 1x1 的情况 ravel 失败
cm = confusion_matrix(y_eval, y_pred_eval, labels=[0, 1])
TN, FP, FN, TP = cm.ravel()
acc = (TP + TN) / cm.sum()
P = TP / (TP + FP + 1e-9)
R = TP / (TP + FN + 1e-9)
F1 = 2 * P * R / (P + R + 1e-9)

# 指标卡片
kpi_grid({
    "Accuracy": f"{acc:.3f}",
    "Precision": f"{P:.3f}",
    "Recall": f"{R:.3f}",
    "F1": f"{F1:.3f}",
})

# 四联图（传入：y_eval/y_pred_eval/proba_eval）
fig = build_dashboard_fig(rf, X.columns.tolist(), y_eval, y_pred_eval, proba_eval)
st.pyplot(fig, clear_figure=True)


# ---------------- 结果浏览 / 下载 ----------------
st.subheader("4) Browse / Download Results")
# 这里仍然展示 hold-out 的统计（更贴近“部署后预测浏览”）
pred_df = pd.DataFrame({
    "y_true": y_te.values,
    "proba": y_proba,
    "pred": (y_proba >= sel_thr).astype(int)
}, index=y_te.index).join(X_te)

mission_cols = [c for c in pred_df.columns if c.startswith("mission_")]
if mission_cols:
    ms = st.multiselect("Filter by mission", mission_cols, default=mission_cols)
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

