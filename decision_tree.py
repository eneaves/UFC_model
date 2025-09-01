"""
decision_tree.py
Implementación manual de un árbol de decisión para clasificación binaria (Gini).
- Sin sklearn ni frameworks de ML.
- Soporta solo features numéricas (nuestro preprocesamiento ya dejó todo en numérico).
- Usa selección aleatoria de features por nodo (m_try) y umbrales candidatos por percentiles.
Uso (ejemplo):
  python decision_tree.py --datadir ./out_rf --max_depth 12 --min_samples_leaf 5 --k_thresh 16
"""
import argparse, json, math, os, random, time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
    return 2.0 * p1 * (1.0 - p1)  # equivalente a 1 - sum(p_k^2) en binaria

def best_split_for_feature(X_col: np.ndarray, y: np.ndarray, k_thresh=16) -> Tuple[float,float]:
    """
    Devuelve (mejora_gini, mejor_umbral). Si no mejora, retorna (0.0, None).
    Umbrales candidatos: percentiles equiespaciados entre valores min y max (basado en valores únicos).
    """
    uniq = np.unique(X_col)
    if len(uniq) <= 1:
        return 0.0, None
    # percentiles candidatos
    qs = np.linspace(0.0, 1.0, num=k_thresh+2)[1:-1]  # evitar extremos exactos
    cand = np.quantile(X_col, qs, method="linear")
    cand = np.unique(cand)  # quitar repetidos
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
        self.m_try = m_try  # si None, se usa sqrt(n_features) al entrenar
        self.k_thresh = k_thresh
        self.root: Optional[Node] = None
        self.random_state = np.random.RandomState(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0, n_features=n_features)

    def _build_tree(self, X, y, depth, n_features) -> Node:
        node = Node(depth=depth)
        # Criterios de parada
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) <= 2*self.min_samples_leaf:
            node.prediction = int(round(y.mean()))
            return node

        # Seleccionar subconjunto de features (m_try)
        m_try = self.m_try or int(math.sqrt(n_features)) or 1
        feat_indices = self.random_state.choice(n_features, size=m_try, replace=False)

        # Buscar el mejor split entre ese subconjunto
        best_gain = 0.0
        best_feat = None
        best_thr  = None
        for j in feat_indices:
            gain, thr = best_split_for_feature(X[:, j], y, k_thresh=self.k_thresh)
            if thr is None:
                continue
            # Validar tamaño mínimo de hojas
            left_mask = X[:, j] <= thr
            if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
                continue
            if gain > best_gain:
                best_gain, best_feat, best_thr = gain, j, thr

        if best_feat is None:
            node.prediction = int(round(y.mean()))
            return node

        node.feature_idx = best_feat
        node.threshold   = best_thr

        left_mask = X[:, best_feat] <= best_thr
        X_left, y_left   = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

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
        # Si por alguna razón node es None, fallback a 0
        return node.prediction if (node and node.prediction is not None) else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(x) for x in X], dtype=int)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="out_rf")
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--k_thresh", type=int, default=16, help="Umbrales candidatos por feature")
    parser.add_argument("--m_try", type=int, default=None, help="#features por nodo (default sqrt(p))")
    args = parser.parse_args()

    features, X_train, y_train, X_val, y_val, X_test, y_test = load_split(args.datadir)

    tree = DecisionTreeScratch(max_depth=args.max_depth,
                               min_samples_leaf=args.min_samples_leaf,
                               m_try=args.m_try,
                               k_thresh=args.k_thresh,
                               random_state=42)
    t0 = time.time()
    tree.fit(X_train, y_train)
    t1 = time.time()
    yhat_tr = tree.predict(X_train)
    yhat_va = tree.predict(X_val)
    print("=== Árbol de Decisión (desde cero) ===")
    print(f"features: {len(features)} | max_depth: {args.max_depth} | min_leaf: {args.min_samples_leaf} | m_try: {args.m_try or int(math.sqrt(len(features)))} | k_thresh: {args.k_thresh}")
    print(f"Entrenamiento en {t1-t0:.2f}s")
    print(f"Accuracy  Train: {accuracy(y_train, yhat_tr):.3f}")
    print(f"Accuracy  Val  : {accuracy(y_val,   yhat_va):.3f}")
    # (opcional) evaluar en test
    yhat_te = tree.predict(X_test)
    print(f"Accuracy  Test : {accuracy(y_test,  yhat_te):.3f}")

if __name__ == "__main__":
    main()
