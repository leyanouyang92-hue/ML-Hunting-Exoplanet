import re
import pandas as pd
from .config import CANON_FEATURES, DISP_CANDS, LABEL_MAP

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
    keep = {"label","label_raw","mission"}
    drop_cols = [c for c in df.columns if c not in keep and df[c].isna().all()]
    return df.drop(columns=drop_cols)

def harmonize_table(df: pd.DataFrame, mission: str) -> pd.DataFrame:
    """映射到统一 schema，并创建二分类标签 label（未知保持 NaN）"""
    _clean_columns(df)

    out = pd.DataFrame()
    # 特征映射
    for canon, alias in CANON_FEATURES.items():
        col = _pick_col(df, alias)
        out[canon] = df[col] if col else pd.NA

    # 处置列
    disp_col = _pick_disposition(df)
    out["label_raw"] = df[disp_col] if disp_col else pd.NA

    # 统一标签
    raw = out["label_raw"].astype(str).str.upper().str.strip()
    raw = raw.replace({  # 规范化别名
        "CONFIRMED PLANET": "CONFIRMED",
        "VALIDATED PLANET": "CONFIRMED",
    })
    out["label"] = raw.map(LABEL_MAP)  # PC / APC 等保持 NaN

    out["mission"] = mission
    out = _drop_all_nan_features_only(out)
    return out