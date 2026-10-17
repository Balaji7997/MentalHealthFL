"""Project paths and shared constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ==================================================
# DIRECTORIES
# ==================================================

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

SHAP_DIR = OUTPUTS_DIR / "shap"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# ==================================================
# MODEL PATHS
# ==================================================

CENTRALIZED_MODEL_DIR = MODELS_DIR / "severity_model"
FEDERATED_MODEL_DIR = MODELS_DIR / "federated_global"
CENTRALIZED_LORA_DIR = MODELS_DIR / "centralized_lora"

# ==================================================
# DATA FILES
# ==================================================

SEVERITY_CSV = DATA_DIR / "severity_encoded.csv"
FL_TEST_CSV = DATA_DIR / "fl_test.csv"

# ==================================================
# ROBERTA CONFIGURATION
# ==================================================

BASE_MODEL_NAME = "FacebookAI/roberta-base"

NUM_LABELS = 4
MAX_LENGTH = 128
NUM_CLIENTS = 4

# ==================================================
# LABEL MAPPING
# ==================================================

ID2LABEL = {
    0: "Minimal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
}

LABEL2ID = {
    "Minimal": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
}

# ==================================================
# LORA CONFIGURATION
# ==================================================

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# RoBERTa Attention Layers
LORA_TARGET_MODULES = [
    "query",
    "key",
    "value",
]

# ==================================================
# TRAINING SETTINGS
# ==================================================

LEARNING_RATE = 2e-4
BATCH_SIZE = 8
LOCAL_EPOCHS = 2

# ==================================================
# RANDOM SEED
# ==================================================

SEED = 42