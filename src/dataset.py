import pandas as pd
from .io_utils import read_nasa_table
from .harmonize import harmonize_table
from .config import CANON_ORDER 

def _label_count_safe(d: pd.DataFrame) -> int:
    return d["label"].notna().sum() if "label" in d.columns else 0

def load_all(koi_csv: str | None, toi_csv: str | None, k2_csv: str | None) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    if koi_csv:
        koi = read_nasa_table(koi_csv)
        koi_h = harmonize_table(koi, "Kepler")
        print("Kepler raw rows:", len(koi_h), "with label:", _label_count_safe(koi_h))
        dfs.append(koi_h)

    if toi_csv:
        toi = read_nasa_table(toi_csv)
        toi_h = harmonize_table(toi, "TESS")
        print("TESS   raw rows:", len(toi_h), "with label:", _label_count_safe(toi_h))
        dfs.append(toi_h)

    if k2_csv:
        k2 = read_nasa_table(k2_csv)
        k2_h = harmonize_table(k2, "K2")
        print("K2     raw rows:", len(k2_h), "with label:", _label_count_safe(k2_h))
        dfs.append(k2_h)

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print("Concat rows:", len(df_all))

    # Only use features that actually exist
    need_cols = [c for c in CANON_ORDER if c in df_all.columns]
    if "label" in df_all.columns:
        df_all = df_all.dropna(subset=need_cols + ["label"])
    else:
        raise RuntimeError("After merging, there is still no label column. Please check whether the treatment column is recognised successfully (for example tfopwg_disp)")

    print("After dropna rows:", len(df_all))
    df_all = pd.get_dummies(df_all, columns=["mission"], drop_first=False)
    return df_all