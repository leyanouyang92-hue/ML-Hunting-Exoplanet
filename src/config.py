# 统一后的特征名与候选列名
CANON_FEATURES = {
    "period":   ["koi_period","pl_orbper","orbital period","orbital_period","period","per"],
    "duration": ["koi_duration","transit duration","transit_duration","tran_dur","duration","dur"],
    "depth":    ["koi_depth","transit depth","transit_depth","tran_depth","depth","depth ppm","depth [ppm]"],
    "prad":     ["koi_prad","pl_rade","planetary radius","planet radius","planet_radius","prad"],
    "srad":     ["koi_srad","st_rad","stellar radius","stellar_radius","srad"],
    "steff":    ["koi_steff","st_teff","stellar teff","stellar_teff","stellar effective temperature","teff"],
}
CANON_ORDER = ["period","duration","depth","prad","srad","steff"]

# 处置列候选
DISP_CANDS = [
    "koi_disposition",               # Kepler
    "tfopwg_disposition", "tfopwg disp",
    "archive_disposition", "disposition", "final_disposition", "toi_disposition",
]

# 标签映射（统一到 0/1）
LABEL_MAP = {
    # 正类
    "CONFIRMED": 1, "CONFIRMED PLANET": 1, "VALIDATED PLANET": 1, "TRUE POSITIVE": 1,
    "CP": 1, "KP": 1,   # TESS: Confirmed Planet / Known Planet
    # 负类
    "FALSE POSITIVE": 0, "FALSE-POSITIVE": 0, "FP": 0, "RETRACTED": 0,
}
