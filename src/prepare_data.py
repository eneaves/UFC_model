"""
prepare_data.py
Prepara el dataset UFC para un modelo desde cero, evitando fuga de información (odds/resultado).
- Lee ufc_clean.csv
- Elimina columnas de apuestas y resultado
- Genera algunas features manuales
- Divide en train/val/test de forma estratificada
- Guarda CSVs y features.json con el orden de columnas
Uso (ejemplo):
  python -m src.prepare_data
  python -m src.prepare_data --csv data/ufc_clean.csv --outdir data
"""
import argparse, json, math, os, random
from typing import List, Tuple
import pandas as pd
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ODDS_COL_FRAGMENTS = ["Odds", "ExpectedValue"]
LEAKY_COLS = ["Winner","Finish","FinishDetails","FinishRound","FinishRoundTime","TotalFightTimeSecs"]
ID_COLS = ["RedFighter","BlueFighter","Date","Location","Country"]

def find_columns(df: pd.DataFrame, fragments: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        for frag in fragments:
            if frag in c:
                cols.append(c)
                break
    return cols

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Asegurar tipos numéricos
    for c in ["BlueWins","RedWins","BlueLosses","RedLosses","BlueDraws","RedDraws",
              "BlueTotalRoundsFought","RedTotalRoundsFought",
              "BlueCurrentWinStreak","RedCurrentWinStreak",
              "BlueAge","RedAge"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Total de peleas
    df["BlueTotalFights"] = df.get("BlueWins",0) + df.get("BlueLosses",0) + df.get("BlueDraws",0)
    df["RedTotalFights"]  = df.get("RedWins",0)  + df.get("RedLosses",0)  + df.get("RedDraws",0)
    df["TotalFightsDif"]  = df["RedTotalFights"] - df["BlueTotalFights"]

    # Win rate (evitar división entre cero)
    bw_den = (df.get("BlueWins",0) + df.get("BlueLosses",0)).replace(0, np.nan)
    rw_den = (df.get("RedWins",0)  + df.get("RedLosses",0)).replace(0, np.nan)
    df["BlueWinRate"] = (df.get("BlueWins",0) / bw_den).fillna(0.5)
    df["RedWinRate"]  = (df.get("RedWins",0)  / rw_den).fillna(0.5)
    df["WinRateDif"]  = df["RedWinRate"] - df["BlueWinRate"]

    # Finish rate (KO+Sub sobre wins)
    bw_den2 = df.get("BlueWins",0).replace(0, np.nan)
    rw_den2 = df.get("RedWins",0).replace(0, np.nan)
    df["BlueFinishRate"] = ((df.get("BlueWinsByKO",0) + df.get("BlueWinsBySubmission",0)) / bw_den2).fillna(0.0)
    df["RedFinishRate"]  = ((df.get("RedWinsByKO",0)  + df.get("RedWinsBySubmission",0))  / rw_den2).fillna(0.0)
    df["FinishRateDif"]  = df["RedFinishRate"] - df["BlueFinishRate"]

    # Momentum (ya existe WinStreakDif, agregamos otra variante con pérdidas)
    df["MomentumDif"] = (df.get("RedCurrentWinStreak",0) - df.get("BlueCurrentWinStreak",0)) \
                        - (df.get("RedCurrentLoseStreak",0) - df.get("BlueCurrentLoseStreak",0))

    # Edad en "prime" (~27-33). Indicadores y diferencia.
    def in_prime(age):
        return 1.0 if (age >= 27) and (age <= 33) else 0.0
    df["RedPrime"]  = df.get("RedAge",0).map(in_prime)
    df["BluePrime"] = df.get("BlueAge",0).map(in_prime)
    df["PrimeDif"]  = df["RedPrime"] - df["BluePrime"]

    return df

def stratified_split(X: pd.DataFrame, y: pd.Series,
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                     seed=RANDOM_SEED) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    assert abs(train_ratio + val_ratio - (1.0 - test_ratio)) < 1e-6, "Ratios deben sumar 1.0"
    classes = sorted(y.unique())
    idx_train, idx_val, idx_test = [], [], []
    rng = np.random.RandomState(seed)
    for c in classes:
        idx = np.where(y.values == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        idx_train.extend(idx[:n_train])
        idx_val.extend(idx[n_train:n_train+n_val])
        idx_test.extend(idx[n_train+n_val:])
    # Mezclar cada split para evitar orden por clases
    rng.shuffle(idx_train); rng.shuffle(idx_val); rng.shuffle(idx_test)
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/ufc_clean.csv", help="Ruta al CSV limpio original")
    parser.add_argument("--outdir", type=str, default="data", help="Carpeta de salida (data/)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Definir target
    if "Winner" not in df.columns:
        raise ValueError("La columna 'Winner' no existe en el CSV.")
    y = df["Winner"].astype(int)

    # Quitar columnas con odds/esperado + columnas con fuga + IDs
    odds_cols = [c for c in df.columns if any(frag in c for frag in ODDS_COL_FRAGMENTS)]
    drop_cols = list(set(odds_cols + LEAKY_COLS + ID_COLS))
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    # Features manuales
    X = add_engineered_features(X)

    # Asegurar numérico y sin NaNs
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    # Guardar orden de features
    features = list(X.columns)
    with open(os.path.join(args.outdir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    # División estratificada
    idx_train, idx_val, idx_test = stratified_split(X, y, 0.7, 0.15, 0.15, seed=RANDOM_SEED)
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_val,   y_val   = X.iloc[idx_val],   y.iloc[idx_val]
    X_test,  y_test  = X.iloc[idx_test],  y.iloc[idx_test]

    # Guardar CSVs (incluyendo target)
    train_df = X_train.copy(); train_df["Winner"] = y_train.values
    val_df   = X_val.copy();   val_df["Winner"]   = y_val.values
    test_df  = X_test.copy();  test_df["Winner"]  = y_test.values

    train_df.to_csv(os.path.join(args.outdir, "ufc_rf_train.csv"), index=False)
    val_df.to_csv(os.path.join(args.outdir, "ufc_rf_val.csv"), index=False)
    test_df.to_csv(os.path.join(args.outdir, "ufc_rf_test.csv"), index=False)

    # Baseline (clase mayoritaria)
    majority = int(y.mean() >= 0.5)
    baseline_acc = float((y == majority).mean())

    print("=== PREPARACIÓN COMPLETA ===")
    print(f"Filas totales: {len(df)}  |  Features usadas: {len(features)}")
    print(f"Drop cols ({len(drop_cols)}): {sorted(drop_cols)}")
    print(f"Splits -> train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")
    print(f"Baseline (mayoría={majority}) accuracy: {baseline_acc:.3f}")

if __name__ == "__main__":
    main()
