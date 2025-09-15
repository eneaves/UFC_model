# src/predict_cli.py
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from .utils import ensure_dirs, load_features_list, save_json
from .feats import add_engineered_block  # necesario para deserializar el modelo

UMBRAL_MODOS = {"recall":0.385, "balanced":0.50, "precision":0.67}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="outputs/models/rf_framework.joblib")
    ap.add_argument("--features", type=str, default="data/features.json")
    ap.add_argument("--input_csv", type=str, required=True, help="CSV con peleas futuras (sin odds)")
    ap.add_argument("--out_csv", type=str, default="outputs/reports/preds_future.csv")
    ap.add_argument("--threshold_mode", type=str, default="balanced", choices=list(UMBRAL_MODOS.keys()))
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    thr = float(args.threshold) if args.threshold is not None else float(UMBRAL_MODOS[args.threshold_mode])

    feats = load_features_list(Path(args.features))
    df = pd.read_csv(args.input_csv)

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en input para inferencia: {missing[:10]}...")

    X = df[feats]

    model = joblib.load(args.model)
    proba = model.predict_proba(X)[:,1]
    pred  = (proba >= thr).astype(int)

    out = df.copy()
    out["win_prob"] = proba
    out["pred"] = pred
    ensure_dirs(Path(args.out_csv).parent)
    out.to_csv(args.out_csv, index=False)

    meta = {"model_path": args.model, "threshold": thr, "threshold_mode": args.threshold_mode, "n_rows": len(out)}
    save_json(meta, Path(args.out_csv).with_suffix(".meta.json"))
    print(f"✅ Predicciones guardadas en {args.out_csv} (umbral {thr:.3f}, modo={args.threshold_mode}).")

if __name__ == "__main__":
    main()
