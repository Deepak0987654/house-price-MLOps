import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts", "best_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "artifacts", "features.pkl")
DEFAULTS_PATH = os.path.join(BASE_DIR, "..", "artifacts", "defaults.pkl")
MODEL_VERSION_PATH = os.path.join(BASE_DIR, "..", "artifacts", "model_version.pkl")

CURRENT_YEAR = 2026