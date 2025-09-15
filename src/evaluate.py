# src/evaluate.py
"""
evaluate.py
Evalúa el Random Forest (scratch) en el conjunto de TEST aplicando un umbral óptimo.
- Reentrena RF scratch con los hiperparámetros dados (sobre train).
- Carga el umbral desde outputs/reports/best_threshold.json generado por threshold_perm.py
  (por defecto usa 'best_by_f1'; con --prefer_precision_target usa ese si existe).
- Calcula métricas en TEST, guarda JSON y figuras ROC/PR en outputs/.

Uso:
  python -m src.evaluate --datadir data --repodir outputs/reports --figdir outputs/figures \
    --n_estimators 200 --max_depth 12 --min_samples_leaf 10 --k_thresh 16 --m_try 9 --oob \
    --prefer_precision_target
"""
import argparse, os, json, math, time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from .utils import (
    ensure_dirs, save_json, plot_roc, plot_pr
)

# ======== Árbol y Bosque (scratch) ========
class Node:
    __slots__ = ("feature_idx","threshold","left","right","prediction","depth")
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, prediction=None, depth=0):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.depth = depth

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
        self.rng = np.random.RandomState(random_state)
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build(X, y, 0, X.shape[1])
        return self

    def _build(self, X, y, depth, n_features) -> Node:
        node = Node(depth=depth)
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) <= 2*self.min_samples_leaf:
            node.prediction = int(round(y.mean()))
            return node
        m = self.m_try or int(np.sqrt(n_features)) or 1
        feats = self.rng.choice(n_features, size=m, replace=False)
        best_gain, best = 0.0, None
        for j in feats:
            gain, thr = best_split_for_feature(X[:, j], y, self.k_thresh)
            if thr is None:
                continue
            left = X[:, j] <= thr
            if left.sum() < self.min_samples_leaf or (~left).sum() < self.min_samples_leaf:
                continue
            if gain > best_gain:
                best_gain, best = gain, (j, thr, left)
        if best is None:
            node.prediction = int(round(y.mean()))
            return node
        j, thr, left = best
        node.feature_idx, node.threshold = j, thr
        node.left  = self._build(X[left],  y[left],  depth+1, n_features)
        node.right = self._build(X[~left], y[~left], depth+1, n_features)
        return node

    def _pred_row(self, x):
        n = self.root
        while n and n.prediction is None:
            n = n.left if x[n.feature_idx] <= n.threshold else n.right
        return n.prediction if n and n.prediction is not None else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._pred_row(x) for x in X], dtype=int)

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
        self.rng = np.random.RandomState(random_state)
        self.trees: List[DecisionTreeScratch] = []
        self.boot_indices: List[np.ndarray] = []
        self.oob_score_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        self.trees, self.boot_indices = [], []
        for _ in range(self.n_estimators):
            idx = self.rng.randint(0, n, size=n) if self.bootstrap else self.rng.permutation(n)
            self.boot_indices.append(idx)
            tree = DecisionTreeScratch(self.max_depth, self.min_samples_leaf, self.m_try, self.k_thresh,
                                       self.rng.randint(0, 1_000_000)).fit(X[idx], y[idx])
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
                if not votes[i]: continue
                usable += 1
                pred = int(round(np.mean(votes[i])))
                if pred == y[i]: correct += 1
            self.oob_score_ = correct/usable if usable>0 else None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        all_preds = np.column_stack([t.predict(X) for t in self.trees])
        return (all_preds.mean(axis=1) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_preds = np.column_stack([t.predict(X) for t in self.trees])
        return all_preds.mean(axis=1)  # prob(1) ~ promedio de votos

# ======== Métricas & IO ========
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
    Xt, yt = train[features].values.astype(float), train["Winner"].values.astype(int)
    Xv, yv = val[features].values.astype(float),   val["Winner"].values.astype(int)
    Xs, ys = test[features].values.astype(float),  test["Winner"].values.astype(int)
    return features, Xt, yt, Xv, yv, Xs, ys

# ======== Main ========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", type=str, default="data", help="Carpeta con ufc_rf_{train,val,test}.csv")
    ap.add_argument("--repodir", type=str, default="outputs/reports", help="Carpeta para JSON/CSV")
    ap.add_argument("--figdir",  type=str, default="outputs/figures", help="Carpeta para PNGs")

    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--min_samples_leaf", type=int, default=10)
    ap.add_argument("--k_thresh", type=int, default=16)
    ap.add_argument("--m_try", type=int, default=None)

    ap.add_argument("--bootstrap", dest="bootstrap", action="store_true")
    ap.add_argument("--no-bootstrap", dest="bootstrap", action="store_false")
    ap.set_defaults(bootstrap=True)

    ap.add_argument("--oob", dest="oob", action="store_true")
    ap.add_argument("--no-oob", dest="oob", action="store_false")
    ap.set_defaults(oob=True)

    # Umbral
    ap.add_argument("--prefer_precision_target", action="store_true",
                    help="Si existe, usa el umbral de 'best_by_precision_target' del JSON.")
    ap.add_argument("--threshold_override", type=float, default=None,
                    help="Si se define, ignora el JSON y usa este threshold manual.")

    args = ap.parse_args()

    ensure_dirs(args.repodir, args.figdir)

    # 1) Cargar splits
    features, X_train, y_train, X_val, y_val, X_test, y_test = load_split(args.datadir)

    # 2) Entrenar RF scratch (sobre TRAIN)
    rf = RandomForestScratch(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, m_try=args.m_try,
        k_thresh=args.k_thresh, bootstrap=args.bootstrap, oob=args.oob, random_state=42
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    t1 = time.time()

    # 3) Cargar threshold
    thr = 0.5
    best_json_path = Path(args.repodir) / "best_threshold.json"
    if args.threshold_override is not None:
        thr = float(args.threshold_override)
        src = "override"
    else:
        if best_json_path.exists():
            with open(best_json_path, "r") as f:
                obj = json.load(f)
            if args.prefer_precision_target and obj.get("best_by_precision_target"):
                thr = float(obj["best_by_precision_target"]["threshold"])
                src = "best_by_precision_target"
            else:
                thr = float(obj["best_by_f1"]["threshold"])
                src = "best_by_f1"
        else:
            src = "default_0.5"
    print(f"Threshold usado para TEST: {thr:.3f} (source: {src})")

    # 4) Predicción y métricas en TEST
    y_scores_test = rf.predict_proba(X_test)
    y_pred_test   = (y_scores_test >= thr).astype(int)
    cm_test = confusion_matrix(y_test, y_pred_test)
    p, r, f1, acc = metrics_from_cm(cm_test)

    # Figuras en TEST (con los scores)
    roc_auc = plot_roc(y_test, y_scores_test, Path(args.figdir) / "roc_curve_test.png", label="RF scratch")
    ap_val  = plot_pr (y_test, y_scores_test, Path(args.figdir) / "pr_curve_test.png",  label="RF scratch")

    # 5) Guardar reporte
    save_json({
        "model": "RandomForestScratch",
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "m_try": args.m_try or int(math.sqrt(len(features))),
            "k_thresh": args.k_thresh,
            "bootstrap": args.bootstrap,
            "oob": args.oob
        },
        "threshold_used": thr,
        "threshold_source": src,
        "timing_train_sec": round(t1 - t0, 3),
        "oob_score": rf.oob_score_,
        "test_confusion_matrix": cm_test.tolist(),
        "test_metrics": {"precision": p, "recall": r, "f1": f1, "accuracy": acc},
        "test_curves": {"roc_auc": roc_auc, "average_precision": ap_val}
    }, Path(args.repodir) / "test_report.json")

    print("=== Evaluación TEST ===")
    print("Confusion:\n", cm_test)
    print(f"Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}  Acc={acc:.3f}")
    print("Artefactos guardados en outputs/reports (test_report.json) y outputs/figures (roc_curve_test.png, pr_curve_test.png)")

if __name__ == "__main__":
    main()
