import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dirs(*dirs: Path):
    for d in dirs:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)

def detect_target(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No se encontró columna objetivo. Prueba con {candidates} o ajusta en config.py.")

def load_features_list(path: Path) -> List[str]:
    with open(path, "r") as f:
        feats = json.load(f)
    if not isinstance(feats, list) or not len(feats):
        raise ValueError("features.json inválido o vacío.")
    return feats

def plot_roc(y_true, y_scores, fig_path: Path, label="Model"):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{label} AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return roc_auc

def plot_pr(y_true, y_scores, fig_path: Path, label="Model"):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f"{label} AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return ap

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def feature_names_after_pre(pre, n_importances: int) -> List[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(n_importances)]

def apply_threshold(proba: np.ndarray, thr: float) -> np.ndarray:
    return (proba >= thr).astype(int)

def plot_feature_importances_barh(features, importances, fig_path: Path, topk: int = 20):
    import numpy as np
    import matplotlib.pyplot as plt

    imp = np.array(importances)
    order = np.argsort(imp)[::-1][:topk]
    top_feats = [features[i] for i in order]
    top_vals  = imp[order]

    plt.figure(figsize=(10, max(4, int(topk*0.35))))
    plt.barh(range(len(top_feats)), top_vals)
    plt.yticks(range(len(top_feats)), top_feats)
    plt.gca().invert_yaxis()
    plt.xlabel("Importancia (ganancia Gini normalizada)")
    plt.title(f"Top {len(top_feats)} Feature Importances (RF scratch)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

def plot_acc_vs_trees(n_list, train_accs, val_accs, oobs, fig_path: Path):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))
    plt.plot(n_list, train_accs, marker="o", label="Train acc")
    plt.plot(n_list, val_accs, marker="o", label="Val acc")
    if any(o is not None for o in oobs):
        plt.plot(n_list, [o if o is not None else np.nan for o in oobs], marker="o", label="OOB acc")
    plt.xlabel("# Árboles")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Número de Árboles (RF scratch)")
    plt.legend()
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=160)
    plt.close()
