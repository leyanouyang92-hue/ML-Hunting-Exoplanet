# app.py  —— Streamlit 前端，复用 src/ 内模块
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.harmonize import harmonize_table
from src.config import CANON_ORDER
from src.visualize import build_dashboard_fig

from src.ui import inject_css, title_bar, upload_card, kpi_grid

st.set_page_config(page_title="Exoplanet ML Explorer", layout="wide")
inject_css()
title_bar()

# ---------- 读上传文件 ----------
def read_uploaded(file):
    if file is None:
        return None
    name = file.name.lower()
    data = file.read()
    if name.endswith((".xls",".xlsx")):
        return pd.read_excel(io.BytesIO(data))
    # CSV/TSV（带 # 注释也兼容）
    try:
        return pd.read_csv(io.BytesIO(data), comment="#", low_memory=False)
    except Exception:
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python", comment="#", low_memory=False)

with st.sidebar:
    st.header("1) Data Upload")
    koi_up  = upload_card("Kepler cumulative", key="koi",
                          types=["csv","tsv","txt","xlsx","xls"])
    tess_up = upload_card("TESS TOI",         key="tess",
                          types=["csv","tsv","txt","xlsx","xls"])
    k2_up   = upload_card("K2 P&C",           key="k2",
                          types=["csv","tsv","txt","xlsx","xls"])

    st.header("2) Training parameters")
    test_size    = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("RF trees", 100, 1000, 400, 50)
    rs           = st.number_input("Random state", 0, 9999, 42, step=1)

# ---------- 统一与合并 ----------
dfs = []
for up, mission in [(koi_up,"Kepler"), (tess_up,"TESS"), (k2_up,"K2")]:
    raw = read_uploaded(up)
    if raw is not None:
        h = harmonize_table(raw, mission)
        st.sidebar.write(f"{mission}: {len(h)} row | with label: {h['label'].notna().sum()}")
        dfs.append(h)

if not dfs:
    st.info("👉 Upload at least one data file on the left（Kepler/TESS/K2）。")
    st.stop()

df_all = pd.concat(dfs, ignore_index=True)

# 训练用数据
need_cols = [c for c in CANON_ORDER if c in df_all.columns]
df_train = df_all.dropna(subset=need_cols + ["label"]).copy()
if df_train["mission"].nunique() > 1:
    df_train = pd.get_dummies(df_train, columns=["mission"], drop_first=True)
else:
    df_train = df_train.drop(columns=["mission"])

st.success(f"合并={len(df_all)}；可训练={len(df_train)}；特征={need_cols + [c for c in df_train.columns if c.startswith('mission_')]}")

X = df_train.drop(columns=["label","label_raw"])
y = df_train["label"].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=rs, n_jobs=-1)
).fit(X_tr, y_tr)

rf = model.named_steps["randomforestclassifier"]
y_proba = model.predict_proba(X_te)[:,1]

# F1 最优阈值
prec, rec, thr = precision_recall_curve(y_te, y_proba)
f1 = 2*prec*rec/(prec+rec+1e-9)
best_idx = int(np.nanargmax(f1))
best_thr = float(thr[max(min(best_idx, len(thr)-1), 0)]) if len(thr) else 0.5

st.subheader("3) Thresholds and indicators")
mode = st.radio("Evaluate on",
                ["Hold-out test set", "All labeled (in-sample)", "5-fold CV (out-of-fold)"],
                index=0, horizontal=True)

if mode == "Hold-out test set":
    y_eval = y_te
    proba_eval = y_proba

elif mode == "All labeled (in-sample)":
    y_eval = y
    proba_eval = model.predict_proba(X)[:, 1]  # 训练集上评估（偏乐观）

else:  # 5-fold CV
    base_pipe = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=rs, n_jobs=-1
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    proba_eval = cross_val_predict(base_pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    y_eval = y

# 用 proba_eval + 阈值做所有指标/曲线/混淆矩阵
prec, rec, thr = precision_recall_curve(y_eval, proba_eval)
f1 = 2*prec*rec/(prec+rec+1e-9)
best_idx = int(np.nanargmax(f1))
best_thr = float(thr[max(min(best_idx, len(thr)-1), 0)]) if len(thr) else 0.5

sel_thr = st.slider("Decision threshold", 0.0, 1.0, best_thr, 0.01)
y_pred_eval = (proba_eval >= sel_thr).astype(int)

cm = confusion_matrix(y_eval, y_pred_eval)
TN, FP, FN, TP = cm.ravel()
acc = (TP+TN)/cm.sum()
P = TP/(TP+FP+1e-9); R = TP/(TP+FN+1e-9); F1 = 2*P*R/(P+R+1e-9)

st.metric("Accuracy", f"{acc:.3f}")
st.metric("Precision", f"{P:.3f}")
st.metric("Recall", f"{R:.3f}")
st.metric("F1", f"{F1:.3f}")

# 你的四联图函数里，用 y_eval / y_pred_eval / proba_eval 作为输入
fig = build_dashboard_fig(rf, X.columns.tolist(), y_eval, y_pred_eval, proba_eval)
st.pyplot(fig, clear_figure=True)

# 结果浏览 & 下载
st.subheader("4) Browse / Download Results")
pred_df = pd.DataFrame({"y_true": y_te.values, "proba": y_proba, "pred": (y_proba>=sel_thr).astype(int)}, index=y_te.index).join(X_te)
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
