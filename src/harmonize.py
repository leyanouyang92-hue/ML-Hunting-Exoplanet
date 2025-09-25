import re
import pandas as pd
from .config import CANON_FEATURES, DISP_CANDS, LABEL_MAP

# 新增：RA/DEC/距离候选列名（按你手头表格再补都行）
RA_CANDS  = ["ra", "ra_deg", "raj2000", "rastr", "ra_str", "ra_j2000", "ra [deg]"]
DEC_CANDS = ["dec", "dec_deg", "dej2000", "decstr", "dec_str", "dec_j2000", "decl", "declination", "dec [deg]"]
DIST_CANDS = ["st_dist", "sy_dist", "dist", "distance", "st_dist", "plx", "parallax", "sy_plx"]

def _clean_columns(df: pd.DataFrame) -> None:
    df.columns = (df.columns
                  .str.replace("\ufeff", "", regex=False)
                  .str.strip())

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowers = {c.lower(): c for c in df.columns}
    for name in candidates:              # 精确（不区分大小写）
        if name.lower() in lowers:
            return lowers[name.lower()]
    for name in candidates:              # 包含匹配
        key = name.lower().replace("_", " ").strip()
        for c in df.columns:
            if key in c.lower():
                return c
    tail = re.split(r"[_\s]", candidates[0])[-1].lower()  # 兜底：最后一个词
    for c in df.columns:
        if tail in c.lower():
            return c
    return None

def _pick_disposition(df: pd.DataFrame) -> str | None:
    # 1) 明确优先：TESS 的 tfopwg_disp / tfopwg_disposition
    for c in df.columns:
        lc = c.lower()
        if "tfopwg" in lc and ("disposition" in lc or "disp" in lc):
            return c
    # 2) 其他常见命名
    for key in ["koi_disposition","toi_disposition","final_disposition",
                "archive_disposition","disposition"]:
        for c in df.columns:
            if key in c.lower():
                return c
    return None

def _drop_all_nan_features_only(df: pd.DataFrame) -> pd.DataFrame:
    # 修改：把 ra/dec/距离列加入保留
    keep = {"label","label_raw","mission","ra","dec","dist_raw"}
    drop_cols = [c for c in df.columns if c not in keep and df[c].isna().all()]
    return df.drop(columns=drop_cols)

def harmonize_table(df: pd.DataFrame, mission: str) -> pd.DataFrame:
    """映射到统一 schema，并创建二分类标签 label（未知保持 NaN）"""
    _clean_columns(df)

    out = pd.DataFrame()
    # 特征映射（period/duration/depth/prad/srad/steff）
    for canon, alias in CANON_FEATURES.items():
        col = _pick_col(df, alias)
        out[canon] = df[col] if col else pd.NA

    # 处置列
    disp_col = _pick_disposition(df)
    out["label_raw"] = df[disp_col] if disp_col else pd.NA

    # 统一标签
    raw = out["label_raw"].astype(str).str.upper().str.strip()
    raw = raw.replace({
        "CONFIRMED PLANET": "CONFIRMED",
        "VALIDATED PLANET": "CONFIRMED",
    })
    out["label"] = raw.map(LABEL_MAP)  # PC / APC 等保持 NaN

    # ★ 新增：把 RA/DEC/距离带出来（保持原始值，解析在 ui._to_xyz 里做）
    ra_col  = _pick_col(df, RA_CANDS)
    dec_col = _pick_col(df, DEC_CANDS)
    dist_col = _pick_col(df, DIST_CANDS)

    out["ra"]  = df[ra_col]  if ra_col  else pd.NA
    out["dec"] = df[dec_col] if dec_col else pd.NA
    if dist_col:
        out["dist_raw"] = df[dist_col]
    else:
        out["dist_raw"] = pd.NA

    out["mission"] = mission
    out = _drop_all_nan_features_only(out)
    return out
