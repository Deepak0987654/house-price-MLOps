import joblib
from app.core.config import MODEL_PATH, FEATURES_PATH, DEFAULTS_PATH, MODEL_VERSION_PATH

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
defaults = joblib.load(DEFAULTS_PATH)
model_version = joblib.load(MODEL_VERSION_PATH)