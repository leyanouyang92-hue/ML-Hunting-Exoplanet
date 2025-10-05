# app.py —— Streamlit front-end, reusing modules in src/
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

# ---------------- Read uploaded files --------------
def read_uploaded(file):
    if file is None:
        return None
    name = file.name.lower()
    data = file.read()
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(data))
    # CSV/TSV (also supports lines commented with '#')
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

# ---------------- Harmonize & merge --------------
dfs = []
for up, mission in [(koi_up, "Kepler"), (tess_up, "TESS"), (k2_up, "K2")]:
    raw = read_uploaded(up)
    if raw is not None:
        h = harmonize_table(raw, mission)  # tries to keep ra/dec/dist_raw internally
        st.sidebar.write(f"{mission}: {len(h)} row | with label: {h['label'].notna().sum()}")
        dfs.append(h)

if not dfs:
    st.info("👉 At least one file need to be uploaded（Kepler/TESS/K2）。")
    st.stop()

df_all = pd.concat(dfs, ignore_index=True)

# 3D universe view (uses merged full dataset; _to_xyz prefers RA/DEC)
html(render_universe_html(df_all, color_by="label", max_points=6000, point_size=1.8), height=560)

# ---------------- Training data prep (only physical features; do not use distance as a feature) ----------------
# Keep only samples with label 0/1
df_lab = df_all[df_all["label"].isin([0, 1])].copy()
if df_lab.empty:
    st.error("Examples with no available labels (label is 0/1).")
    st.stop()

# —— After df_lab is defined and before feature selection ——
BASE_LEAKY = {
    "prad", "koi_prad", "planet_radius",  # explicit planetary-radius columns
    "pl_rade", "pl_radj", "pl_rads",      # NASA confirmed radii
    "pl_orbper", "koi_prad",              # period tied to confirmed planets (different from KOI/TCE period)
}

# Auto blacklist: any column containing 'prad' (but not 'srad'), or ending with '_radius',
# or 'pl_' columns that include radius/mass/orbit information
auto_leaky = set()
for c in df_lab.columns:
    cl = c.lower()
    if ("prad" in cl and not cl.startswith("srad")):   # catches prad, prad_est, koi_prad... but keeps srad
        auto_leaky.add(c)
    if cl.endswith("_radius"):
        auto_leaky.add(c)
    if cl.startswith("pl_") and any(k in cl for k in ["rad", "mass", "orb"]):
        auto_leaky.add(c)

LEAKY = BASE_LEAKY | auto_leaky

# Pick only from CANON_ORDER and exclude the blacklist
feature_cols = [c for c in CANON_ORDER if c in df_lab.columns and c not in LEAKY]
num_feats = df_lab[feature_cols].apply(pd.to_numeric, errors="coerce")

# —— Optional: normalize depth to a fraction (0.01 means 1%) to handle %, ppm, etc. ——
def depth_to_fraction(s):
    d = pd.to_numeric(s, errors="coerce")
    out = d.copy()
    # 0–1 as fraction; 1–100 as percent; >100 as ppm
    out[(d > 1) & (d <= 100)] = d[(d > 1) & (d <= 100)] / 100.0
    out[d > 100] = d[d > 100] / 1e6
    return out

# Engineered radius: prad_est = srad * sqrt(depth_fraction)
if {"srad", "depth"}.issubset(df_lab.columns):
    srad  = pd.to_numeric(df_lab["srad"], errors="coerce")
    depth = depth_to_fraction(df_lab["depth"])
    df_lab["prad_est"] = srad * np.sqrt(depth)

# Build numeric feature matrix (now may include prad_est, but not prad)
num_feats = df_lab[feature_cols].apply(pd.to_numeric, errors="coerce")

# Drop features that are entirely missing to avoid imputer failure
all_nan_cols = [c for c in num_feats.columns if num_feats[c].notna().sum() == 0]
if all_nan_cols:
    st.warning(f"These features are all missing in the current data and have been ignored：{all_nan_cols}")
    num_feats = num_feats.drop(columns=all_nan_cols)

# mission one-hot (do not keep the original 'mission' column)
if "mission" in df_lab.columns and df_lab["mission"].nunique() > 1:
    df_lab = pd.get_dummies(df_lab, columns=["mission"], drop_first=True)
    mission_cols = [c for c in df_lab.columns if c.startswith("mission_")]
else:
    df_lab = df_lab.drop(columns=[c for c in ["mission"] if c in df_lab.columns])
    mission_cols = []

# Assemble X / y (physical features + mission one-hot only)
X = pd.concat([num_feats, df_lab[mission_cols]], axis=1).select_dtypes(include=["number"])
y = df_lab["label"].astype(int)

# Basic checks
if X.shape[1] == 0:
    st.error("Available physical features are listed as 0. Please check your upload data or reduce the required features.")
    st.stop()
if len(X) < 2:
    st.error(f"Insufficient number of trainable samples after cleaning ({len(X)}).")
    st.stop()

# Use stratify only when there are at least two classes
stratify_y = y if y.nunique() > 1 else None

# Compute a feasible test_size: at least 1 train + 1 test
n = len(X)
n_test = max(1, int(round(n * test_size)))
n_test = min(n - 1, n_test)
test_size_eff = n_test / n

# Split; if stratified split fails, fall back to non-stratified
try:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size_eff, random_state=rs, stratify=stratify_y
    )
except ValueError:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size_eff, random_state=rs, stratify=None
    )

# ---------------- Train ----------------
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

# Hold-out probabilities and a “reference threshold”
y_proba = model.predict_proba(X_te)[:, 1]
prec0, rec0, thr0 = precision_recall_curve(y_te, y_proba)
f10 = 2 * prec0 * rec0 / (prec0 + rec0 + 1e-9)
best_idx0 = int(np.nanargmax(f10)) if len(thr0) else 0
best_thr0 = float(thr0[max(min(best_idx0, len(thr0) - 1), 0)]) if len(thr0) else 0.5

# Unified threshold in session state (so table above and slider below stay in sync)
st.session_state.setdefault("thr", float(best_thr0))
thr_now = float(st.session_state["thr"])

# ---------------- Browse / Download results (put first; use current threshold) ----------------
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

# ---------------- Evaluation mode & metrics (placed later; slider updates global threshold) ----------------
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
else:  # 5-fold CV (pipeline matches training, including imputation)
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

# Use proba_eval + threshold to compute metrics/curves/confusion matrix
prec, rec, thr = precision_recall_curve(y_eval, proba_eval)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = int(np.nanargmax(f1)) if len(thr) else 0
best_thr = float(thr[max(min(best_idx, len(thr) - 1), 0)]) if len(thr) else best_thr0

# Threshold slider (sync with st.session_state['thr'])
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

# 1) Use session_state to remember whether normalization is on
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

    # 2) Put the button next to the note above: two columns; text on the left, button on the right
    msg_col, btn_col = st.columns([4, 2])
    with msg_col:
        st.write("")  # spacer so the two columns align
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

# Brief notes (under the two line charts)
st.markdown(
    """
**What they mean:**
- **ROC curve**（TPR vs FPR）Shows the overall recognition ability from low to high thresholds; the larger the **AUC**, the better the overall ranking.
- **Precision–Recall**（P vs R）It is more sensitive to **positive example scarcity/class imbalance**: the closer the curve is to the upper right corner, the better; **AP** is its area metric.
"""
)