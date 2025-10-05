import re
import pandas as pd
from .config import CANON_FEATURES, DISP_CANDS, LABEL_MAP

# Added: RA/DEC/distance candidate column names (you can add more according to your current table)
RA_CANDS  = ["ra", "ra_deg", "raj2000", "rastr", "ra_str", "ra_j2000", "ra [deg]"]
DEC_CANDS = ["dec", "dec_deg", "dej2000", "decstr", "dec_str", "dec_j2000", "decl", "declination", "dec [deg]"]
DIST_CANDS = ["st_dist", "sy_dist", "dist", "distance", "st_dist", "plx", "parallax", "sy_plx"]

def _clean_columns(df: pd.DataFrame) -> None:
    df.columns = (df.columns
                  .str.replace("\ufeff", "", regex=False)
                  .str.strip())

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowers = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lowers:
            return lowers[name.lower()]
    for name in candidates:
        key = name.lower().replace("_", " ").strip()
        for c in df.columns:
            if key in c.lower():
                return c
    tail = re.split(r"[_\s]", candidates[0])[-1].lower()
    for c in df.columns:
        if tail in c.lower():
            return c
    return None

def _pick_disposition(df: pd.DataFrame) -> str | None:
    # 1) Clear priority: TESS tfopwg_disp / tfopwg_disposition
    for c in df.columns:
        lc = c.lower()
        if "tfopwg" in lc and ("disposition" in lc or "disp" in lc):
            return c
    # 2) Other common names
    for key in ["koi_disposition","toi_disposition","final_disposition",
                "archive_disposition","disposition"]:
        for c in df.columns:
            if key in c.lower():
                return c
    return None

def _drop_all_nan_features_only(df: pd.DataFrame) -> pd.DataFrame:
    # Modification: Add the ra/dec/distance columns to retention
    keep = {"label","label_raw","mission","ra","dec","dist_raw"}
    drop_cols = [c for c in df.columns if c not in keep and df[c].isna().all()]
    return df.drop(columns=drop_cols)

def harmonize_table(df: pd.DataFrame, mission: str) -> pd.DataFrame:
    """Map to a unified schema and create a binary classification label (unknown values remain NaN)"""
    _clean_columns(df)

    out = pd.DataFrame()
    # Feature mapping (period/duration/depth/prad/srad/steff)
    for canon, alias in CANON_FEATURES.items():
        col = _pick_col(df, alias)
        out[canon] = df[col] if col else pd.NA

    # Set column
    disp_col = _pick_disposition(df)
    out["label_raw"] = df[disp_col] if disp_col else pd.NA

    # Unified label
    raw = out["label_raw"].astype(str).str.upper().str.strip()
    raw = raw.replace({
        "CONFIRMED PLANET": "CONFIRMED",
        "VALIDATED PLANET": "CONFIRMED",
    })
    out["label"] = raw.map(LABEL_MAP)  # PC / APC etc. remain NaN

    # ★ New: Bring out RA/DEC/Distance (keep the original values, parsing is done in ui._to_xyz)
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
