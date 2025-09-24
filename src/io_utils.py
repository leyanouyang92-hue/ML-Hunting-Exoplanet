from pathlib import Path
import pandas as pd

def read_nasa_table(path: str | Path) -> pd.DataFrame:
    """兼容 NASA CSV（含 # 注释）和 Excel。自动推断分隔符。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(p)

    # CSV/TSV：优先 ','，失败再自动推断
    try:
        return pd.read_csv(p, comment="#", low_memory=False)
    except Exception:
        return pd.read_csv(p, sep=None, engine="python", comment="#", low_memory=False)
