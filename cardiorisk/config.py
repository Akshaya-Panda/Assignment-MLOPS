"""Central configuration: paths, dataset schema and reproducibility constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Paths -----------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "raw" / "processed.cleveland.data"
CLEAN_FILE = DATA_DIR / "processed" / "heart_cleveland_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

# --- Data source (UCI ML Repository, dataset id 45, CC BY 4.0) ---------------
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
UCI_ZIP_MEMBER = "processed.cleveland.data"
RAW_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

# --- Schema ------------------------------------------------------------------
RAW_COLUMNS = [
    "age",  # years
    "sex",  # 1 = male, 0 = female
    "cp",  # chest pain type: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
    "trestbps",  # resting systolic blood pressure (mm Hg)
    "chol",  # serum cholesterol (mg/dl)
    "fbs",  # fasting blood sugar > 120 mg/dl
    "restecg",  # resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "thalach",  # maximum heart rate achieved in the exercise test
    "exang",  # exercise-induced angina
    "oldpeak",  # ST depression induced by exercise relative to rest
    "slope",  # slope of peak exercise ST segment: 1 up, 2 flat, 3 down
    "ca",  # number of major vessels coloured by fluoroscopy (0-3)
    "thal",  # thallium stress test: 3 normal, 6 fixed defect, 7 reversible defect
    "num",  # diagnosis: 0 = no disease, 1-4 = disease severity
]
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# --- Reproducibility ---------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Experiment tracking -----------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "cardiorisk-heart-disease"
REGISTERED_MODEL_NAME = "cardiorisk-classifier"
