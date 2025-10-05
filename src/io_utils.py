from pathlib import Path
import pandas as pd

def read_nasa_table(path: str | Path) -> pd.DataFrame:
    """Compatible with NASA CSV (including # comments) and Excel. Automatically infers delimiters."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(p)

    # CSV/TSV: Prefer ',' first, fallback to auto detection if that fails
    try:
        return pd.read_csv(p, comment="#", low_memory=False)
    except Exception:
        return pd.read_csv(p, sep=None, engine="python", comment="#", low_memory=False)
