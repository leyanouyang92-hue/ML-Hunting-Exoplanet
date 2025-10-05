# Canonical feature names and their alias candidates
CANON_FEATURES = {
    "period":   ["koi_period","pl_orbper","orbital period","orbital_period","period","per"],
    "duration": ["koi_duration","transit duration","transit_duration","tran_dur","duration","dur"],
    "depth":    ["koi_depth","transit depth","transit_depth","tran_depth","depth","depth ppm","depth [ppm]"],
    "prad":     ["koi_prad","pl_rade","planetary radius","planet radius","planet_radius","prad"],
    "srad":     ["koi_srad","st_rad","stellar radius","stellar_radius","srad"],
    "steff":    ["koi_steff","st_teff","stellar teff","stellar_teff","stellar effective temperature","teff"],
}
CANON_ORDER = ["period", "duration", "depth", "srad", "steff"]

# Disposition column candidates
DISP_CANDS = [
    "koi_disposition",               # Kepler
    "tfopwg_disposition", "tfopwg disp",
    "archive_disposition", "disposition", "final_disposition", "toi_disposition",
]

# Label mapping (normalized to 0/1)
LABEL_MAP = {
    # Positive class
    "CONFIRMED": 1, "CONFIRMED PLANET": 1, "VALIDATED PLANET": 1, "TRUE POSITIVE": 1,
    "CP": 1, "KP": 1,   # TESS: Confirmed Planet / Known Planet
    # Negative class
    "FALSE POSITIVE": 0, "FALSE-POSITIVE": 0, "FP": 0, "RETRACTED": 0,
}