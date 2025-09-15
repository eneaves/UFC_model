from pathlib import Path

# Raíz del repo (este archivo vive en src/, así que retrocede un nivel)
ROOT = Path(__file__).resolve().parents[1]

# Carpetas
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
REPORTS = OUTPUTS / "reports"

# (Opcional) carpeta de “presentación”
RESULTS_WITH_FW = ROOT / "results_withFrameWork"

# Archivos de datos
CSV_CLEAN = DATA / "ufc_clean.csv"
CSV_TRAIN = DATA / "ufc_rf_train.csv"
CSV_VAL   = DATA / "ufc_rf_val.csv"
CSV_TEST  = DATA / "ufc_rf_test.csv"
FEATURES_JSON = DATA / "features.json"

# Configuración ML
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE  = 0.15          # del remanente de train

# Nombre(s) posibles del target (ajusta si ya sabes el exacto)
TARGET_CANDIDATES = ["target", "label", "winner", "Winner", "RedWin", "blue_won", "result"]

# Scoring por defecto (ajusta a tu objetivo: 'f1', 'roc_auc', 'average_precision', etc.)
DEFAULT_SCORING = "f1"
