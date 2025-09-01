"""
random_forest.py
Random Forest desde cero para clasificación binaria.
- Sin sklearn ni frameworks de ML.
- Incluye:
  * Árbol de decisión básico (Gini, percentiles como thresholds, m_try por nodo)
  * Bootstrap sampling por árbol
  * Votación mayoritaria en predicción
  * (Opcional) OOB accuracy (estimación fuera de bolsa)
  * (Opcional) Importancia de features basada en ganancia de Gini acumulada

Uso (ejemplo):
  python step3_random_forest.py --datadir out_rf --n_estimators 200 \
    --max_depth 12 --min_samples_leaf 10 --k_thresh 16 --m_try 9 --oob True
"""
import argparse, math, os, random, time, json
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ========================
# Árbol de decisión básico
# ========================
@dataclass
class Node:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    prediction: Optional[int] = None  # 0/1 en hojas
    depth: int = 0

def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p1 = y.mean()
    return 2.0 * p1 * (1.0 - p1)  # binaria

def best_split_for_feature(X_col: np.ndarray, y: np.ndarray, k_thresh=16):
    """
    Devuelve (mejora_gini, mejor_umbral). Si no mejora, retorna (0.0, None).
    Umbrales candidatos: percentiles equiespaciados.
    """
    uniq = np.unique(X_col)
    if len(uniq) <= 1:
        return 0.0, None
    qs = np.linspace(0.0, 1.0, num=k_thresh+2)[1:-1]  # evitar extremos exactos
    cand = np.quantile(X_col, qs, method="linear")
    cand = np.unique(cand)
    parent_gini = gini_impurity(y)
    best_gain = 0.0
    best_thr = None
    for thr in cand:
        left_mask = X_col <= thr
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue
        g_left  = gini_impurity(y[left_mask])
        g_right = gini_impurity(y[right_mask])
        w_left  = left_mask.mean()
        w_right = right_mask.mean()
        child_gini = w_left * g_left + w_right * g_right
        gain = parent_gini - child_gini
        if gain > best_gain:
            best_gain = gain
            best_thr = thr
    return best_gain, best_thr

class DecisionTreeScratch:
    def __init__(self, max_depth=10, min_samples_leaf=5, m_try=None, k_thresh=16, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.m_try = m_try  # si None, sqrt(p)
        self.k_thresh = k_thresh
        self.root: Optional[Node] = None
        self.random_state = np.random.RandomState(random_state)
        # Para importancia de features (suma de ganancias)
        self.feature_importance_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        self.feature_importance_ = np.zeros(n_features, dtype=float)
        self.root = self._build_tree(X, y, depth=0, n_features=n_features)
        return self

    def _build_tree(self, X, y, depth, n_features) -> Node:
        node = Node(depth=depth)
        # Criterios de paro
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) <= 2*self.min_samples_leaf:
            node.prediction = int(round(y.mean()))
            return node

        # Seleccionar subconjunto de features
        m_try = self.m_try or int(math.sqrt(n_features)) or 1
        feat_indices = self.random_state.choice(n_features, size=m_try, replace=False)

        # Buscar mejor split
        parent_gini = gini_impurity(y)
        best_gain = 0.0
        best_feat = None
        best_thr  = None
        best_left_mask = None

        for j in feat_indices:
            gain, thr = best_split_for_feature(X[:, j], y, k_thresh=self.k_thresh)
            if thr is None:
                continue
            left_mask = X[:, j] <= thr
            if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
                continue
            if gain > best_gain:
                best_gain = gain
                best_feat = j
                best_thr = thr
                best_left_mask = left_mask

        if best_feat is None:
            node.prediction = int(round(y.mean()))
            return node

        # Registrar ganancia en importancia de feature
        self.feature_importance_[best_feat] += best_gain

        node.feature_idx = best_feat
        node.threshold   = best_thr

        X_left, y_left   = X[best_left_mask], y[best_left_mask]
        X_right, y_right = X[~best_left_mask], y[~best_left_mask]

        node.left  = self._build_tree(X_left,  y_left,  depth+1, n_features)
        node.right = self._build_tree(X_right, y_right, depth+1, n_features)
        return node

    def _predict_row(self, x: np.ndarray) -> int:
        node = self.root
        while node and node.prediction is None:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction if (node and node.prediction is not None) else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(x) for x in X], dtype=int)

# ========================
# Random Forest
# ========================
class RandomForestScratch:
    def __init__(self, n_estimators=100, max_depth=12, min_samples_leaf=10,
                 m_try=None, k_thresh=16, bootstrap=True, oob=False, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.m_try = m_try
        self.k_thresh = k_thresh
        self.bootstrap = bootstrap
        self.oob = oob
        self.random_state = np.random.RandomState(random_state)
        self.trees: List[DecisionTreeScratch] = []
        self.boot_indices: List[np.ndarray] = []
        self.oob_score_ = None
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        self.trees = []
        self.boot_indices = []
        fi_accum = np.zeros(p, dtype=float)

        for t in range(self.n_estimators):
            # Bootstrap indices
            if self.bootstrap:
                idx = self.random_state.randint(0, n, size=n)
            else:
                # Sin bootstrap: muestreo aleatorio sin reemplazo (mismo tamaño)
                idx = self.random_state.permutation(n)
            self.boot_indices.append(idx)

            Xb, yb = X[idx], y[idx]
            tree = DecisionTreeScratch(max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf,
                                       m_try=self.m_try,
                                       k_thresh=self.k_thresh,
                                       random_state=self.random_state.randint(0, 1_000_000))
            tree.fit(Xb, yb)
            self.trees.append(tree)
            if tree.feature_importance_ is not None:
                fi_accum += tree.feature_importance_

        # Normalizar importancias
        if fi_accum.sum() > 0:
            self.feature_importances_ = fi_accum / fi_accum.sum()
        else:
            self.feature_importances_ = np.zeros(p, dtype=float)

        # OOB score
        if self.oob and self.bootstrap:
            # Para cada muestra, juntar predicciones de árboles donde NO fue incluida
            votes = [[] for _ in range(n)]
            for t, idx in enumerate(self.boot_indices):
                oob_mask = np.ones(n, dtype=bool)
                oob_mask[idx] = False  # fuera de bolsa = no muestreada
                if not oob_mask.any():
                    continue
                y_hat_oob = self.trees[t].predict(X[oob_mask])
                # Guardar votos
                pos = np.where(oob_mask)[0]
                for i_local, i_global in enumerate(pos):
                    votes[i_global].append(int(y_hat_oob[i_local]))
            # Agregar votación mayoritaria OOB
            usable = 0
            correct = 0
            for i in range(n):
                if len(votes[i]) == 0:
                    continue
                usable += 1
                pred = int(round(np.mean(votes[i])))  # mayoría
                if pred == y[i]:
                    correct += 1
            self.oob_score_ = correct / usable if usable > 0 else None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Votación mayoritaria
        all_preds = np.column_stack([tree.predict(X) for tree in self.trees])
        maj = (all_preds.mean(axis=1) >= 0.5).astype(int)
        return maj
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve la probabilidad estimada de clase 1 como promedio
        de predicciones de todos los árboles.
        """
        all_preds = np.column_stack([tree.predict(X) for tree in self.trees])
        # promedio de votos = prob de clase 1
        probs = all_preds.mean(axis=1)
        return probs


# ========================
# Utilidades
# ========================
def load_split(datadir: str):
    train = pd.read_csv(os.path.join(datadir, "ufc_rf_train.csv"))
    val   = pd.read_csv(os.path.join(datadir, "ufc_rf_val.csv"))
    test  = pd.read_csv(os.path.join(datadir, "ufc_rf_test.csv"))
    features = [c for c in train.columns if c != "Winner"]
    X_train, y_train = train[features].values.astype(float), train["Winner"].values.astype(int)
    X_val,   y_val   = val[features].values.astype(float),   val["Winner"].values.astype(int)
    X_test,  y_test  = test[features].values.astype(float),  test["Winner"].values.astype(int)
    return (features, X_train, y_train, X_val, y_val, X_test, y_test)

def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())

def confusion_matrix(y_true, y_pred):
    # Asegurar arrays numpy enteros
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    # Calcular entradas
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TN, FP],
                     [FN, TP]])

def classification_metrics(cm):
    """
    Calcula precisión, recall y F1 para cada clase usando la matriz de confusión 2x2.
    cm = [[TN, FP],
          [FN, TP]]
    """
    TN, FP, FN, TP = cm.ravel()

    # Para clase 1 (ganador = 1)
    precision_1 = TP / (TP + FP) if (TP+FP) > 0 else 0.0
    recall_1    = TP / (TP + FN) if (TP+FN) > 0 else 0.0
    f1_1        = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1+recall_1) > 0 else 0.0

    # Para clase 0 (ganador = 0)
    precision_0 = TN / (TN + FN) if (TN+FN) > 0 else 0.0
    recall_0    = TN / (TN + FP) if (TN+FP) > 0 else 0.0
    f1_0        = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0+recall_0) > 0 else 0.0

    return {
        "class_0": {"precision": precision_0, "recall": recall_0, "f1": f1_0},
        "class_1": {"precision": precision_1, "recall": recall_1, "f1": f1_1}
    }

def roc_curve_points(y_true, y_scores, n_thresh=100):
    """
    Calcula pares (FPR, TPR) para distintos umbrales.
    """
    thresholds = np.linspace(0, 1, n_thresh)
    tpr_list, fpr_list = [], []
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TPR = TP / (TP+FN) if (TP+FN)>0 else 0
        FPR = FP / (FP+TN) if (FP+TN)>0 else 0
        tpr_list.append(TPR)
        fpr_list.append(FPR)
    return np.array(fpr_list), np.array(tpr_list)

def pr_curve_points(y_true, y_scores, n_thresh=100):
    """
    Calcula pares (Recall, Precision) para distintos umbrales.
    """
    thresholds = np.linspace(0, 1, n_thresh)
    prec_list, rec_list = [], []
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        precision = TP/(TP+FP) if (TP+FP)>0 else 1.0
        recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
        prec_list.append(precision)
        rec_list.append(recall)
    return np.array(rec_list), np.array(prec_list)

def plot_curves(y_true, y_scores, outdir="out_rf"):
    # ROC
    fpr, tpr = roc_curve_points(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0,1],[0,1],'k--',label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{outdir}/roc_curve.png")
    plt.close()
    # PR
    rec, prec = pr_curve_points(y_true, y_scores)
    plt.figure()
    plt.plot(rec, prec, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"{outdir}/pr_curve.png")
    plt.close()

def plot_top_importances(features, importances, outdir="out_rf", topk=20):
    os.makedirs(outdir, exist_ok=True)
    imp = np.array(importances)
    order = np.argsort(imp)[::-1][:topk]
    top_feats = [features[i] for i in order]
    top_vals  = imp[order]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_feats)), top_vals)
    plt.yticks(range(len(top_feats)), top_feats)
    plt.gca().invert_yaxis()
    plt.xlabel("Importancia (ganancia Gini normalizada)")
    plt.title(f"Top {topk} Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_importances.png"))
    plt.close()
def run_sweep_estimators(n_list, X_train, y_train, X_val, y_val,
                         max_depth=12, min_samples_leaf=10, m_try=None, k_thresh=16,
                         bootstrap=True, oob=False, random_state=42):
    val_accs = []
    train_accs = []
    oobs = []
    for n in n_list:
        rf_tmp = RandomForestScratch(
            n_estimators=n,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            m_try=m_try,
            k_thresh=k_thresh,
            bootstrap=bootstrap,
            oob=oob,
            random_state=random_state
        )
        t0 = time.time()
        rf_tmp.fit(X_train, y_train)
        t1 = time.time()

        yhat_tr = rf_tmp.predict(X_train)
        yhat_va = rf_tmp.predict(X_val)
        tr_acc = float((y_train == yhat_tr).mean())
        va_acc = float((y_val   == yhat_va).mean())
        val_accs.append(va_acc)
        train_accs.append(tr_acc)
        oobs.append(rf_tmp.oob_score_ if oob else None)

        print(f"n={n:4d} | train={tr_acc:.3f} | val={va_acc:.3f} | oob={oobs[-1] if oob else 'n/a'} | {t1-t0:.1f}s")
    return train_accs, val_accs, oobs
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", type=str, default="out_rf")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--min_samples_leaf", type=int, default=10)
    ap.add_argument("--k_thresh", type=int, default=16)
    ap.add_argument("--m_try", type=int, default=None)
    ap.add_argument("--bootstrap", type=bool, default=True)
    ap.add_argument("--oob", type=bool, default=True)
    args = ap.parse_args()

    features, X_train, y_train, X_val, y_val, X_test, y_test = load_split(args.datadir)

    rf = RandomForestScratch(n_estimators=args.n_estimators,
                             max_depth=args.max_depth,
                             min_samples_leaf=args.min_samples_leaf,
                             m_try=args.m_try,
                             k_thresh=args.k_thresh,
                             bootstrap=args.bootstrap,
                             oob=args.oob,
                             random_state=42)

    t0 = time.time()
    rf.fit(X_train, y_train)
    t1 = time.time()

    yhat_tr = rf.predict(X_train)
    yhat_va = rf.predict(X_val)
    yhat_te = rf.predict(X_test)

    print("=== Random Forest (desde cero) ===")
    print(f"Arboles: {args.n_estimators} | max_depth: {args.max_depth} | min_leaf: {args.min_samples_leaf} | m_try: {args.m_try or int(math.sqrt(len(features)))} | k_thresh: {args.k_thresh}")
    print(f"Entrenamiento: {t1 - t0:.2f}s | OOB: {rf.oob_score_ if rf.oob_score_ is not None else 'n/a'}")
    print(f"Accuracy  Train: {accuracy(y_train, yhat_tr):.3f}")
    print(f"Accuracy  Val  : {accuracy(y_val,   yhat_va):.3f}")
    print(f"Accuracy  Test : {accuracy(y_test,  yhat_te):.3f}")

    yhat_val = rf.predict(X_val)
    cm_val = confusion_matrix(y_val, yhat_val)
    print("\nMatriz de confusión (Val):\n", cm_val)

    metrics_val = classification_metrics(cm_val)
    print("\nMétricas en Validación:")
    for cls, vals in metrics_val.items():
        print(f"{cls}: Precision={vals['precision']:.3f} | Recall={vals['recall']:.3f} | F1={vals['f1']:.3f}")

    # Top importances
    if rf.feature_importances_ is not None:
        order = np.argsort(rf.feature_importances_)[::-1][:15]
        print("\nTop 15 features (importancia por ganancia Gini acumulada):")
        for rank, j in enumerate(order, start=1):
            print(f"{rank:2d}. {features[j]}  ->  {rf.feature_importances_[j]:.4f}")

    # Guardar importancias
    out = {"features": features, "importances": (rf.feature_importances_.tolist() if rf.feature_importances_ is not None else [])}
    with open(os.path.join(args.datadir, "rf_importances.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    # Obtener probabilidades
    y_scores_val = rf.predict_proba(X_val)

    # Graficar curvas
    plot_curves(y_val, y_scores_val, outdir="out_rf")
    print("Curvas guardadas en out_rf/roc_curve.png y out_rf/pr_curve.png")

    #Graficar importnaces
    plot_top_importances(features, rf.feature_importances_, outdir="out_rf", topk=20)
    print("Guardado: out_rf/feature_importances.png")

    # Define el barrido (ajusta si quieres más puntos)
    n_list = [10, 20, 40, 80, 120, 160, 200]
    train_accs, val_accs, oobs = run_sweep_estimators(
        n_list,
        X_train, y_train, X_val, y_val,
        max_depth=12, min_samples_leaf=10, m_try=9, k_thresh=16,
        bootstrap=True, oob=True, random_state=42
    )

    # Plot
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
    plt.savefig("out_rf/acc_vs_trees.png")
    plt.close()

    print("Guardado: out_rf/acc_vs_trees.png")

if __name__ == "__main__":
    main()
