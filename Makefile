# ===========================
# UFCMODEL — Makefile
# Ejecutar desde la raíz del repo (ufcmodel/)
# ===========================

# ---------- Rutas ----------
PYTHON ?= python
DATADIR := data
FIGDIR  := outputs/figures
REPDIR  := outputs/reports
MODELD  := outputs/models

CSV     := ufc_clean.csv
MODEL_SK := $(MODELD)/rf_framework.joblib

# ---------- Hiperparámetros recomendados ----------
# Scratch (mejor trade-off observado)
RF_EST  := 200
RF_DEPTH:= 12
RF_LEAF := 10
RF_KT   := 16

# Variante profunda (comparativa bias/varianza)
RF_EST_DEEP   := 400
RF_DEPTH_DEEP := 20
RF_LEAF_DEEP  := 5
RF_KT_DEEP    := 32

# Sklearn (grid + OOB)
SK_SCORING := balanced_accuracy
SK_LEAF    := 5
SK_DEPTH   := 12
SK_THR_MODE:= balanced
SK_JOBS    := -1

# ===========================
# Targets
# ===========================
.PHONY: help dirs prepare tree rf_scratch rf_scratch_deep rf_sklearn eval_sklearn all clean

help:
	@echo "Targets disponibles:"
	@echo "  make prepare           -> Genera splits (train/val/test) en ./data"
	@echo "  make tree              -> Árbol de decisión (from scratch) baseline"
	@echo "  make rf_scratch        -> Random Forest (from scratch) configuración recomendada"
	@echo "  make rf_scratch_deep   -> RF scratch profundo (comparativa bias/varianza)"
	@echo "  make rf_sklearn        -> RF sklearn (GridSearch + OOB), guarda modelo y reportes"
	@echo "  make eval_sklearn      -> Evalúa el modelo sklearn en TEST"
	@echo "  make all               -> Ejecuta todo el pipeline (prepare -> tree -> rf_scratch -> rf_sklearn -> eval_sklearn)"
	@echo "  make clean             -> Limpia ./outputs (figures, reports, models)"

dirs:
	@mkdir -p $(DATADIR) $(FIGDIR) $(REPDIR) $(MODELD)

prepare: dirs
	@echo ">> Preparando datos en $(DATADIR)"
	$(PYTHON) -m src.prepare_data --csv $(CSV) --outdir $(DATADIR)

tree: dirs
	@echo ">> Árbol (from scratch) baseline"
	$(PYTHON) -m src.decision_tree \
		--datadir $(DATADIR) \
		--out_reports $(REPDIR) \
		--max_depth 12 \
		--min_samples_leaf 5 \
		--k_thresh 32

rf_scratch: dirs
	@echo ">> Random Forest (from scratch) — configuración recomendada"
	$(PYTHON) -m src.random_forest \
		--datadir $(DATADIR) \
		--figdir $(FIGDIR) \
		--repodir $(REPDIR) \
		--n_estimators $(RF_EST) \
		--max_depth $(RF_DEPTH) \
		--min_samples_leaf $(RF_LEAF) \
		--k_thresh $(RF_KT)

rf_scratch_deep: dirs
	@echo ">> Random Forest (from scratch) — variante profunda (comparativa)"
	$(PYTHON) -m src.random_forest \
		--datadir $(DATADIR) \
		--figdir $(FIGDIR) \
		--repodir $(REPDIR) \
		--n_estimators $(RF_EST_DEEP) \
		--max_depth $(RF_DEPTH_DEEP) \
		--min_samples_leaf $(RF_LEAF_DEEP) \
		--k_thresh $(RF_KT_DEEP)

rf_sklearn: dirs
	@echo ">> Random Forest (sklearn) — GridSearch + OOB"
	$(PYTHON) -m src.rf_sklearn \
		--datadir $(DATADIR) \
		--repodir $(REPDIR) \
		--figdir $(FIGDIR) \
		--models $(MODELD) \
		--oob \
		--n_jobs $(SK_JOBS) \
		--scoring $(SK_SCORING) \
		--leaf_min $(SK_LEAF) \
		--depth $(SK_DEPTH) \
		--threshold_mode $(SK_THR_MODE)

eval_sklearn: dirs
	@echo ">> Evaluando modelo sklearn en TEST"
	$(PYTHON) -m src.eval_sklearn \
		--datadir $(DATADIR) \
		--model $(MODEL_SK) \
		--repodir $(REPDIR) \
		--figdir $(FIGDIR) \
		--threshold_mode $(SK_THR_MODE)

all: prepare tree rf_scratch rf_sklearn eval_sklearn
	@echo ">> Pipeline completo finalizado."

clean:
	@echo ">> Limpiando outputs/"
	@rm -rf outputs/figures/* outputs/reports/* outputs/models/*
