# 🥊 UFC Model – Predicting Fight Outcomes

Este proyecto implementa modelos de **Árbol de Decisión** y **Random Forest** para predecir resultados de peleas de UFC.
Incluye versiones **from scratch** y con **scikit-learn**, lo que permite comparar desempeño, bias y varianza.

---

## 🔧 Requisitos

* Python 3.10+ (probado en 3.12)
* Librerías principales: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`
* Make (si quieres usar el `Makefile`)

Instalación rápida:

```bash
conda create -n ufcmodel python=3.12 -y
conda activate ufcmodel
pip install -r requirements.txt
```

---

## ⚡ Ejecución

### Opción A – Paso a paso

1️⃣ **Preparar datos**
Genera splits `train/val/test` a partir del CSV limpio:

```bash
python -m src.prepare_data --csv ufc_clean.csv --outdir data
```

2️⃣ **Árbol de decisión (from scratch)**
Baseline con un solo árbol:

```bash
python -m src.decision_tree --datadir data --out_reports outputs/reports \
  --max_depth 12 --min_samples_leaf 5 --k_thresh 32
```

3️⃣ **Random Forest (from scratch)**
Configuración recomendada (200 árboles, profundidad 12):

```bash
python -m src.random_forest --datadir data --figdir outputs/figures --repodir outputs/reports \
  --n_estimators 200 --max_depth 12 --min_samples_leaf 10 --k_thresh 16
```

(Opcional) Variante profunda (400 árboles, depth=20):

```bash
python -m src.random_forest --datadir data --figdir outputs/figures --repodir outputs/reports \
  --n_estimators 400 --max_depth 20 --min_samples_leaf 5 --k_thresh 32
```

4️⃣ **Random Forest (scikit-learn)**
Entrena con GridSearch + OOB:

```bash
python -m src.rf_sklearn --datadir data --repodir outputs/reports --figdir outputs/figures \
  --models outputs/models --oob --n_jobs -1 --scoring balanced_accuracy \
  --leaf_min 5 --depth 12 --threshold_mode balanced
```

5️⃣ **Evaluar modelo sklearn en TEST**

```bash
python -m src.eval_sklearn --datadir data --model outputs/models/rf_framework.joblib \
  --repodir outputs/reports --figdir outputs/figures --threshold_mode balanced
```

---

### Opción B – Todo de una vez con Makefile

El proyecto incluye un **Makefile** con targets predefinidos.

Ver opciones:

```bash
make help
```

Pipeline completo:

```bash
make all
```

Ejecutar componentes específicos:

```bash
make prepare         # genera splits
make tree            # árbol de decisión baseline
make rf_scratch      # random forest from scratch (200 árboles, depth 12)
make rf_scratch_deep # variante profunda
make rf_sklearn      # sklearn + GridSearch
make eval_sklearn    # evaluación en TEST
```

---

## 📊 Resultados esperados

| Modelo                  | Train Acc | Val Acc | Test Acc | ROC-AUC (Val) | AP (Val) |
| ----------------------- | --------- | ------- | -------- | ------------- | -------- |
| Árbol scratch           | 0.746     | 0.555   | 0.543    | —             | —        |
| RF scratch (200, d=12)  | 0.871     | 0.627   | 0.621    | 0.627         | 0.668    |
| RF sklearn (grid tuned) | —         | 0.614   | 0.611    | 0.623         | 0.673    |

👉 Patrón: recall alto en **clase 1 (\~0.83)** y bajo en **clase 0 (\~0.31)**.

---

## 📈 Artefactos generados

* `outputs/reports/` → JSON/CSV con métricas, reportes de validación/test, importancias
* `outputs/figures/` → curvas ROC/PR, importancias de features, sweep de árboles
* `outputs/models/` → modelos guardados (.joblib)

---
