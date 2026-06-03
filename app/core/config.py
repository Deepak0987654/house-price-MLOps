from pathlib import Path

# Go to project root (/app)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "best_model.pkl"
FEATURES_PATH = BASE_DIR / "artifacts" / "features.pkl"
DEFAULTS_PATH = BASE_DIR / "artifacts" / "defaults.pkl"
MODEL_VERSION_PATH = BASE_DIR / "artifacts" / "model_version.pkl"

print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", MODEL_PATH.exists())

CURRENT_YEAR = 2026