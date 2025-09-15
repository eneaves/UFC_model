# eval_sklearn.py
import argparse, json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from .feats import add_engineered_block

from .utils import ensure_dirs, save_json, plot_roc, plot_pr

UMBRAL_MODOS = {"recall":0.385, "balanced":0.50, "precision":0.67}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", type=str, default="data")
    ap.add_argument("--model", type=str, default="outputs/models/rf_framework.joblib")
    ap.add_argument("--figdir", type=str, default="outputs/figures")
    ap.add_argument("--repodir", type=str, default="outputs/reports")
    ap.add_argument("--threshold_mode", type=str, default="balanced", choices=list(UMBRAL_MODOS.keys()))
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    ensure_dirs(args.figdir, args.repodir)
    thr = float(args.threshold) if args.threshold is not None else float(UMBRAL_MODOS[args.threshold_mode])

    test = pd.read_csv(Path(args.datadir) / "ufc_rf_test.csv")
    target = "Winner"
    feats = [c for c in test.columns if c != target]
    X, y = test[feats], test[target]

    model = joblib.load(args.model)
    proba = model.predict_proba(X)[:,1]
    pred = (proba >= thr).astype(int)

    cm = confusion_matrix(y, pred)
    rep = classification_report(y, pred, output_dict=True)

    roc_auc = plot_roc(y, proba, Path(args.figdir) / "roc_curve_test_framework.png", label="RF(sklearn)")
    ap = plot_pr(y, proba, Path(args.figdir) / "pr_curve_test_framework.png", label="RF(sklearn)")

    save_json({
        "threshold_used": thr,
        "threshold_mode": args.threshold_mode,
        "confusion_matrix": cm.tolist(),
        "classification_report": rep,
        "curves": {"roc_auc": roc_auc, "average_precision": ap}
    }, Path(args.repodir) / "test_report_framework.json")

    print(f"Evaluación TEST lista. Umbral {thr:.3f}. Reportes/figuras en outputs/.")
if __name__ == "__main__":
    main()
