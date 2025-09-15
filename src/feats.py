# src/feats.py
import numpy as np
import pandas as pd

def add_engineered_block(X: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquecimiento ligero:
    - métricas por minuto (si hay columnas *Total y Minutes)
    - diferenciales Red/Blue
    - recencia / layoff si existen
    - fortaleza de oponentes (si existen columnas *_OppWinRate/Elo/FinishRate)
    """
    X = X.copy()

    # Per-minute
    for num_col in list(X.columns):
        if num_col.endswith("Total") and "Minutes" in X.columns:
            rate_col = num_col.replace("Total", "PerMin")
            with np.errstate(divide="ignore", invalid="ignore"):
                X[rate_col] = (pd.to_numeric(X[num_col], errors="coerce") /
                               pd.to_numeric(X["Minutes"], errors="coerce")).fillna(0.0)

    # Diferenciales Red-Blue
    for pr, pb in [("Red", "Blue"), ("R", "B")]:
        reds = [c for c in X.columns if c.startswith(pr)]
        for rc in reds:
            bc = rc.replace(pr, pb, 1)
            if bc in X.columns:
                dif = rc.replace(pr, "")
                X[f"{dif}Dif"] = pd.to_numeric(X[rc], errors="coerce").fillna(0.0) - \
                                 pd.to_numeric(X[bc], errors="coerce").fillna(0.0)

    # Recencia / Layoff
    for c in ["DaysSinceLastFightRed","DaysSinceLastFightBlue","LayoffDaysRed","LayoffDaysBlue"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    if "DaysSinceLastFightRed" in X.columns and "DaysSinceLastFightBlue" in X.columns:
        X["LayoffDif"] = X["DaysSinceLastFightRed"] - X["DaysSinceLastFightBlue"]

    # Fortaleza de oponentes (si existen)
    for side in ["Red","Blue"]:
        for k in ["OppWinRate","OppAvgElo","OppAvgFinishRate"]:
            col = f"{side}{k}"
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    if "RedOppWinRate" in X.columns and "BlueOppWinRate" in X.columns:
        X["OppWinRateDif"] = X["RedOppWinRate"] - X["BlueOppWinRate"]

    return X
