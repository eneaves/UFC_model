# src/rf_sklearn.py
from __future__ import annotations
"""
Random Forest con scikit-learn incorporando aprendizajes del scratch:
- Umbral operativo configurable (recall≈0.385, balanced=0.50, precision≈0.67…0.70)
- Tuning ligero con GridSearchCV (n_estimators 150–200, max_depth 8–12, min_samples_leaf 10–25,
  max_features 7–11 o 'sqrt'); OOB activado
- Opcional calibración de probabilidades (sigmoid/isotonic)
- Importancias (Gini) + Permutation Importance; curvas ROC/PR
- Artefactos reproducibles en outputs/{reports,figures,models}

Uso típico:
  python -m src.rf_sklearn --oob --threshold_mode precision
"""

import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

# utilidades locales (asegúrate de tener estos helpers en src/utils.py)
from .utils import (
    ensure_dirs, save_json, load_features_list,
    plot_roc, plot_pr, plot_feature_importances_barh
)
# bloque de features derivadas (centralizado para compatibilidad con joblib)
from .feats import add_engineered_block

# =========================
# Constantes y configuración
# =========================
RANDOM_STATE = 42
DEFAULT_SCORING = "f1"     # puedes usar 'roc_auc', 'average_precision', etc.

# Modos de umbral operativo
UMBRAL_MODOS = {
    "recall":   0.385,  # priorizar recall/F1 (scouting)
    "balanced": 0.50,   # equilibrio general
    "precision":0.67    # alta precisión, menos FP (puedes subir a 0.70)
}

# Columnas típicamente categóricas que queremos OHE aunque en TRAIN no vengan como 'object'
KNOWN_CATEGORICAL = {"WeightClass", "BlueStance", "RedStance", "Gender"}

# =========================
# Helpers
# =========================
def _make_ohe():
    """
    Crea OneHotEncoder compatible con sklearn >=1.2 (sparse_output)
    y versiones previas (sparse). Evita crashear por versión.
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_pipeline(num_feats: List[str], cat_feats: List[str], oob: bool, n_jobs: int = -1) -> Pipeline:
    """
    Pipeline: [enrich engineered] -> [preprocesamiento(num + OHE cat)] -> [RandomForest]
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_feats),
            ("cat", _make_ohe(), cat_feats),
        ],
        remainder="drop"
    )

    enrich = FunctionTransformer(add_engineered_block, validate=False)

    rf = RandomForestClassifier(
        n_estimators=175,  # punto medio 150–200
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        oob_score=oob,
        bootstrap=True
    )

    pipe = Pipeline([
        ("enrich", enrich),
        ("pre", pre),
        ("clf", rf),
    ])
    return pipe

def parse_max_features_grid(values: List[Union[int, str]], p_dim: int) -> List[Union[int, str]]:
    """
    Acepta enteros (7..11) y/o strings ('sqrt', 'log2', None); recorta enteros al rango [1, p_dim].
    """
    cleaned: List[Union[int, str]] = []
    for v in values:
        if isinstance(v, int):
            cleaned.append(min(max(1, v), max(1, p_dim)))
        else:
            cleaned.append(v)
    return cleaned

def get_feature_names_after_pre(pre: ColumnTransformer, fallback: List[str]) -> List[str]:
    """
    Intenta obtener los nombres de salida tras el ColumnTransformer; si falla, devuelve `fallback`.
    """
    try:
        names = list(pre.get_feature_names_out())
        # Limpieza de prefijos "num__" / "cat__" (opcional)
        names = [n.replace("num__", "").replace("cat__", "") for n in names]
        return names
    except Exception:
        return fallback

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    # Rutas
    ap.add_argument("--datadir", type=str, default="data", help="Carpeta con ufc_rf_{train,val,test}.csv y features.json")
    ap.add_argument("--repodir", type=str, default="outputs/reports", help="Salida para JSON/CSV")
    ap.add_argument("--figdir",  type=str, default="outputs/figures", help="Salida para figuras PNG")
    ap.add_argument("--models",  type=str, default="outputs/models", help="Salida para el modelo .joblib")

    # Tuning ligero guiado por OOB/val
    ap.add_argument("--oob", action="store_true", default=True)
    ap.add_argument("--no-oob", dest="oob", action="store_false")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--scoring", type=str, default=DEFAULT_SCORING)  # 'f1', 'roc_auc', 'average_precision', etc.

    # Rejilla de búsqueda
    ap.add_argument("--leaf_min", type=int, nargs="+", default=[10, 15, 20, 25])
    ap.add_argument("--depth",    type=int, nargs="+", default=[8, 10, 12])
    ap.add_argument("--mtry",     nargs="+", default=[7, 9, 11, "sqrt"])

    # Calibración opcional
    ap.add_argument("--calibrate", type=str, default=None, choices=[None, "sigmoid", "isotonic"])

    # Umbral operativo
    ap.add_argument("--threshold_mode", type=str, default="balanced", choices=list(UMBRAL_MODOS.keys()))
    ap.add_argument("--threshold", type=float, default=None, help="Sobrescribe threshold_mode si se especifica")

    args = ap.parse_args()
    ensure_dirs(args.repodir, args.figdir, args.models)

    # Carga datos y features
    datadir = Path(args.datadir)
    train_df = pd.read_csv(datadir / "ufc_rf_train.csv")
    val_df   = pd.read_csv(datadir / "ufc_rf_val.csv")
    feats    = load_features_list(datadir / "features.json")
    target   = "Winner" if "Winner" in train_df.columns else "target"

    X_train, y_train = train_df[feats], train_df[target]
    X_val,   y_val   = val_df[feats],   val_df[target]

    # Detectar categóricas/numéricas con lista forzada
    cat_feats = [c for c in feats if (c in KNOWN_CATEGORICAL) or (X_train[c].dtype == 'object')]
    num_feats = [c for c in feats if c not in cat_feats]

    # Persistir roles para reproducibilidad
    save_json({"num_feats": num_feats, "cat_feats": cat_feats},
              Path(args.repodir) / "feature_roles_framework.json")

    # Construir pipeline con esas listas
    pipe = build_pipeline(num_feats, cat_feats, oob=args.oob, n_jobs=args.n_jobs)

    # Construir grid usando las numéricas como dimensión base
    p_dim = len(num_feats) + 0  # las OHE se expanden internamente (desconocidas a priori)
    max_features_grid = parse_max_features_grid(args.mtry, max(1, p_dim))

    # Rejilla (tuning ligero)
    param_grid = {
        "clf__n_estimators": [150, 175, 200],
        "clf__max_depth": args.depth,
        "clf__min_samples_leaf": args.leaf_min,
        "clf__max_features": max_features_grid,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring=args.scoring,
        cv=cv, n_jobs=args.n_jobs, verbose=1
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    save_json(
        {"best_params": gs.best_params_, "scoring": args.scoring},
        Path(args.repodir) / "best_params_framework.json"
    )

    # Calibración (opcional) sobre VAL para no sobreajustar TRAIN
    calibrated = False
    if args.calibrate:
        cal = CalibratedClassifierCV(best, method=args.calibrate, cv="prefit")
        cal.fit(X_val, y_val)
        model_to_use = cal
        calibrated = True
    else:
        model_to_use = best

    # Probabilidades y umbral operativo
    y_val_proba = model_to_use.predict_proba(X_val)[:, 1]
    thr = float(args.threshold) if args.threshold is not None else float(UMBRAL_MODOS[args.threshold_mode])
    y_val_pred  = (y_val_proba >= thr).astype(int)

    # Reportes de validación
    report = classification_report(y_val, y_val_pred, output_dict=True)
    save_json(report, Path(args.repodir) / "val_classification_report_framework.json")

    auc_roc = plot_roc(y_val, y_val_proba, Path(args.figdir) / "roc_curve_val_framework.png", label="RF(sklearn)")
    ap_val  = plot_pr (y_val, y_val_proba, Path(args.figdir) / "pr_curve_val_framework.png", label="RF(sklearn)")
    save_json(
        {"roc_auc_val": auc_roc, "average_precision_val": ap_val,
         "threshold_used": thr, "threshold_mode": args.threshold_mode,
         "calibrated": calibrated},
        Path(args.repodir) / "val_curve_scores_framework.json"
    )

    # Importancias (Gini) desde el RF dentro del Pipeline/Calibrated
    if isinstance(model_to_use, Pipeline):
        pipe_used = model_to_use
    elif hasattr(model_to_use, "base_estimator"):
        pipe_used = model_to_use.base_estimator  # CalibratedClassifierCV -> Pipeline
    else:
        pipe_used = best

    rf_step = pipe_used.named_steps.get("clf", None)
    pre     = pipe_used.named_steps.get("pre", None)
    if rf_step is None:
        raise RuntimeError("No se encontró el paso 'clf' (RandomForest) dentro del Pipeline.")

    out_names = get_feature_names_after_pre(pre, fallback=feats)
    plot_feature_importances_barh(
        out_names, rf_step.feature_importances_,
        Path(args.figdir) / "feature_importances_framework.png", topk=30
    )
    save_json(
        {"features_out": out_names, "importances": list(map(float, rf_step.feature_importances_))},
        Path(args.repodir) / "rf_importances_framework.json"
    )

    # Permutation importance (en validación)
    perm = permutation_importance(best, X_val, y_val, n_repeats=10, random_state=RANDOM_STATE, n_jobs=args.n_jobs)
    perm_df = pd.DataFrame({
        "feature": out_names[:perm.importances_mean.shape[0]],
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std
    }).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(Path(args.repodir) / "perm_importance_framework.csv", index=False)

    # Guardar modelo
    import joblib
    ensure_dirs(args.models)
    model_path = Path(args.models) / ("rf_framework_calibrated.joblib" if calibrated else "rf_framework.joblib")
    joblib.dump(model_to_use, model_path)

    print(f"Listo ✅ Modelo: {model_path}")
    print(f"Umbral usado en VAL: {thr:.3f} (modo={args.threshold_mode}, calibrado={calibrated})")
    print("Reportes en outputs/reports y figuras en outputs/figures.")

if __name__ == "__main__":
    main()
