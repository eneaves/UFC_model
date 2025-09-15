"""
threshold_perm.py
- Re-entrena el Random Forest (scratch) con hyperparams dados
- Barrido de umbrales en Validación y selección de:
    * Umbral por máximo F1
    * (Opcional) Umbral por precision objetivo (--target_precision)
- Genera CSV/PNG con Precision/Recall/F1 vs threshold y guarda el mejor umbral
- Calcula Importancia por Permutación en Validación con el umbral elegido

Uso típico:
  python -m src.threshold_perm --n_estimators 200 --max_depth 12 --min_samples_leaf 10 --k_thresh 16 --m_try 9 --oob \
      --metric f1 --n_repeats 5 --target_precision 0.70
"""
import argparse, os, math, json, time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

# === Utils compartidos ===
from .utils import (
    ensure_dirs, save_json, plot_roc, plot_pr, plot_feature_importances_barh
)

# ========================
# Árbol de decisión scratch
# ========================
@dataclass
class Node:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    prediction: Optional[int] = None
    depth: int = 0

def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p1 = y.mean()
    return 2.0 * p1 * (1.0 - p1)

def best_split_for_feature(X_col: np.ndarray, y: np.ndarray, k_thresh=16):
    uniq = np.unique(X_col)
    if len(uniq) <= 1:
        return 0.0, None
    qs = np.linspace(0.0, 1.0, num=k_thresh+2)[1:-1]
    cand = np.quantile(X_col, qs, method="linear")
    cand = np.unique(cand)
    parent_gini = gini_impurity(y)
    best_gain, best_thr = 0.0, None
    for thr in cand:
        left = X_col <= thr
        right = ~left
        if left.sum() == 0 or right.sum() == 0:
            continue
        g_left  = gini_impurity(y[left])
        g_right = gini_impurity(y[right])
        w_left  = left.mean()
        w_right = right.mean()
        child   = w_left*g_left + w_right*g_right
        gain = parent_gini - child
        if gain > best_gain:
            best_gain, best_thr = gain, thr
    return best_gain, best_thr

class DecisionTreeScratch:
    def __init__(self, max_depth=12, min_samples_leaf=10, m_try=None, k_thresh=16, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.m_try = m_try
        self.k_thresh = k_thresh
        self.random_state = np.random.RandomState(random_state)
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build(X, y, 0, X.shape[1])
        return self

    def _build(self, X, y, depth, n_features):
        node = Node(depth=depth)
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) <= 2*self.min_samples_leaf:
            node.prediction = int(round(y.mean()))
            return node
        m = self.m_try or int(np.sqrt(n_features)) or 1
        feats = self.random_state.choice(n_features, size=m, replace=False)
        best = None; best_gain = 0.0
        for j in feats:
            gain, thr = best_split_for_feature(X[:, j], y, self.k_thresh)
            if thr is None:
                continue
            left = X[:, j] <= thr
            if left.sum() < self.min_samples_leaf or (~left).sum() < self.min_samples_leaf:
                continue
            if gain > best_gain:
                best_gain = gain; best = (j, thr, left)
        if best is None:
            node.prediction = int(round(y.mean()))
            return node
        j, thr, left = best
        node.feature_idx = j; node.threshold = thr
        node.left  = self._build(X[left],  y[left],  depth+1, n_features)
        node.right = self._build(X[~left], y[~left], depth+1, n_features)
        return node

    def _pred_row(self, x):
        node = self.root
        while node and node.prediction is None:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.prediction if node and node.prediction is not None else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._pred_row(x) for x in X], dtype=int)

# ========================
# Random Forest scratch
# ========================
class RandomForestScratch:
    def __init__(self, n_estimators=200, max_depth=12, min_samples_leaf=10,
                 m_try=None, k_thresh=16, bootstrap=True, oob=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.m_try = m_try
        self.k_thresh = k_thresh
        self.bootstrap = bootstrap
        self.oob = oob
        self.random_state = np.random.RandomState(random_state)
        self.trees = []
        self.boot_indices = []
        self.oob_score_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        self.trees = []; self.boot_indices = []; self.oob_score_ = None
        for _ in range(self.n_estimators):
            idx = self.random_state.randint(0, n, size=n) if self.bootstrap else self.random_state.permutation(n)
            self.boot_indices.append(idx)
            tree = DecisionTreeScratch(self.max_depth, self.min_samples_leaf, self.m_try, self.k_thresh,
                                       self.random_state.randint(0, 1_000_000))
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

        # OOB
        if self.oob and self.bootstrap:
            votes = [[] for _ in range(n)]
            for t, idx in enumerate(self.boot_indices):
                oob_mask = np.ones(n, dtype=bool); oob_mask[idx] = False
                if not oob_mask.any(): continue
                y_hat = self.trees[t].predict(X[oob_mask])
                pos = np.where(oob_mask)[0]
                for i_local, i_global in enumerate(pos):
                    votes[i_global].append(int(y_hat[i_local]))
            usable = 0; correct = 0
            for i in range(n):
                if len(votes[i]) == 0: continue
                usable += 1
                pred = int(round(np.mean(votes[i])))
                if pred == y[i]: correct += 1
            self.oob_score_ = correct / usable if usable > 0 else None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        all_preds = np.column_stack([t.predict(X) for t in self.trees])
        return (all_preds.mean(axis=1) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_preds = np.column_stack([t.predict(X) for t in self.trees])
        return all_preds.mean(axis=1)  # prob(1) ~ promedio de votos

# ========================
# Utilidades métricas e IO
# ========================
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    TP = np.sum((y_true==1) & (y_pred==1))
    TN = np.sum((y_true==0) & (y_pred==0))
    FP = np.sum((y_true==0) & (y_pred==1))
    FN = np.sum((y_true==1) & (y_pred==0))
    return np.array([[TN, FP],[FN, TP]])

def metrics_from_cm(cm):
    TN, FP, FN, TP = cm.ravel()
    precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
    recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    acc       = (TP+TN)/cm.sum() if cm.sum()>0 else 0.0
    return precision, recall, f1, acc

def load_split(datadir: str):
    train = pd.read_csv(os.path.join(datadir, "ufc_rf_train.csv"))
    val   = pd.read_csv(os.path.join(datadir, "ufc_rf_val.csv"))
    test  = pd.read_csv(os.path.join(datadir, "ufc_rf_test.csv"))
    features = [c for c in train.columns if c != "Winner"]
    X_train, y_train = train[features].values.astype(float), train["Winner"].values.astype(int)
    X_val,   y_val   = val[features].values.astype(float),   val["Winner"].values.astype(int)
    X_test,  y_test  = test[features].values.astype(float),  test["Winner"].values.astype(int)
    return features, X_train, y_train, X_val, y_val, X_test, y_test

def sweep_thresholds(y_true, y_scores, start=0.0, stop=1.0, num=201):
    thrs = np.linspace(start, stop, num)
    rows = []
    for t in thrs:
        y_pred = (y_scores >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        precision, recall, f1, acc = metrics_from_cm(cm)
        rows.append({
            "threshold": float(t), "precision": precision, "recall": recall, "f1": f1, "accuracy": acc,
            "TN": int(cm[0,0]), "FP": int(cm[0,1]), "FN": int(cm[1,0]), "TP": int(cm[1,1])
        })
    return pd.DataFrame(rows)

def permutation_importance(rf, X_val, y_val, threshold, n_repeats=5, metric="accuracy", random_state=42):
    rng = np.random.RandomState(random_state)
    base_pred = (rf.predict_proba(X_val) >= threshold).astype(int)
    cm = confusion_matrix(y_val, base_pred)
    precision, recall, f1_base, acc_base = metrics_from_cm(cm)
    base = acc_base if metric=="accuracy" else f1_base

    n_features = X_val.shape[1]
    importances = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        drops = []
        for _ in range(n_repeats):
            Xp = X_val.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pred = (rf.predict_proba(Xp) >= threshold).astype(int)
            cm_p = confusion_matrix(y_val, pred)
            _, _, f1_p, acc_p = metrics_from_cm(cm_p)
            score = acc_p if metric=="accuracy" else f1_p
            drops.append(max(0.0, base - score))
        importances[j] = float(np.mean(drops))
    return importances, base

# ========================
# Main
# ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", type=str, default="data", help="Carpeta con ufc_rf_{train,val,test}.csv")
    ap.add_argument("--repodir", type=str, default="outputs/reports", help="Carpeta para CSV/JSON de resultados")
    ap.add_argument("--figdir",  type=str, default="outputs/figures", help="Carpeta para PNGs")

    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--min_samples_leaf", type=int, default=10)
    ap.add_argument("--k_thresh", type=int, default=16)
    ap.add_argument("--m_try", type=int, default=None)

    # Flags booleanos seguros:
    ap.add_argument("--bootstrap", dest="bootstrap", action="store_true")
    ap.add_argument("--no-bootstrap", dest="bootstrap", action="store_false")
    ap.set_defaults(bootstrap=True)

    ap.add_argument("--oob", dest="oob", action="store_true")
    ap.add_argument("--no-oob", dest="oob", action="store_false")
    ap.set_defaults(oob=True)

    ap.add_argument("--metric", type=str, default="f1", choices=["f1","accuracy"])
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--target_precision", type=float, default=None,
                    help="Si se define, busca el umbral con precision>=target que maximiza el recall")
    args = ap.parse_args()

    # Crear carpetas de salida
    ensure_dirs(args.repodir, args.figdir)

    features, X_train, y_train, X_val, y_val, X_test, y_test = load_split(args.datadir)

    rf = RandomForestScratch(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        m_try=args.m_try,
        k_thresh=args.k_thresh,
        bootstrap=args.bootstrap,
        oob=args.oob,
        random_state=42
    ).fit(X_train, y_train)

    # --- Barrido de umbrales en Validación ---
    scores_val = rf.predict_proba(X_val)
    df_thr = sweep_thresholds(y_val, scores_val, 0.0, 1.0, 201)
    df_thr.to_csv(Path(args.repodir) / "threshold_sweep.csv", index=False)

    # Mejor por F1
    idx_best_f1 = int(df_thr["f1"].values.argmax())
    best_row = df_thr.iloc[idx_best_f1].to_dict()

    # Alternativa: precision objetivo
    best_row_prec = None
    if args.target_precision is not None:
        sub = df_thr[df_thr["precision"] >= args.target_precision]
        if len(sub) > 0:
            # máximo recall entre los que cumplen la precision mínima
            best_row_prec = sub.iloc[sub["recall"].values.argmax()].to_dict()

    # Evaluaciones rápidas en test para umbrales típicos
    for thr in [0.385, 0.50, 0.670]:
        probs_te = rf.predict_proba(X_test)
        yhat_te  = (probs_te >= thr).astype(int)
        cm = confusion_matrix(y_test, yhat_te)
        p,r,f1,acc = metrics_from_cm(cm)
        print(f"\n== Test @ thr={thr:.3f} ==")
        print("Confusion:\n", cm)
        print(f"Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}  Acc={acc:.3f}")

    # Plots de curvas (usando scores de validación)
    # 1) PR
    ap_val = plot_pr(y_val, scores_val, Path(args.figdir) / "pr_curve.png", label="RF scratch")
    # 2) ROC
    roc_auc_val = plot_roc(y_val, scores_val, Path(args.figdir) / "roc_curve.png", label="RF scratch")

    # Plot sweep P/R/F1 vs threshold
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(df_thr["threshold"], df_thr["precision"], label="Precision")
    plt.plot(df_thr["threshold"], df_thr["recall"],    label="Recall")
    plt.plot(df_thr["threshold"], df_thr["f1"],        label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs Threshold (Validación)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(args.figdir) / "threshold_sweep.png", dpi=160)
    plt.close()

    # Guardar mejor(es) umbral(es)
    out_thr = {
        "best_by_f1": best_row,
        "best_by_precision_target": best_row_prec,
        "oob": rf.oob_score_,
        "roc_auc_val": roc_auc_val,
        "average_precision_val": ap_val
    }
    save_json(out_thr, Path(args.repodir) / "best_threshold.json")

    print("=== Threshold sweep ===")
    print(f"Best F1 at thr={best_row['threshold']:.3f} -> F1={best_row['f1']:.3f}, "
          f"P={best_row['precision']:.3f}, R={best_row['recall']:.3f}, Acc={best_row['accuracy']:.3f}")
    if best_row_prec is not None:
        print(f"Best Recall with Precision>={args.target_precision:.2f} at thr={best_row_prec['threshold']:.3f} "
              f"-> P={best_row_prec['precision']:.3f}, R={best_row_prec['recall']:.3f}, F1={best_row_prec['f1']:.3f}")

    chosen_thr = float(best_row["threshold"]) if best_row_prec is None else float(best_row_prec["threshold"])

    # --- Permutation Importance (en Validación) con el umbral elegido ---
    importances, base_score = permutation_importance(
        rf, X_val, y_val, threshold=chosen_thr, n_repeats=args.n_repeats, metric=args.metric, random_state=42
    )
    df_imp = pd.DataFrame({"feature": features, "perm_importance": importances}) \
             .sort_values("perm_importance", ascending=False)
    df_imp.to_csv(Path(args.repodir) / "perm_importance.csv", index=False)

    # Plot top-k importances
    plot_feature_importances_barh(
        features, importances,
        Path(args.figdir) / "perm_importance.png",
        topk=min(20, len(features))
    )

    print("Guardado en outputs/:")
    print(" - reports: threshold_sweep.csv, best_threshold.json, perm_importance.csv")
    print(" - figures: pr_curve.png, roc_curve.png, threshold_sweep.png, perm_importance.png")

if __name__ == "__main__":
    main()
