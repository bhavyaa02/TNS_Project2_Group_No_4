from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw" / "heart_disease_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "src" / "models"

# Target
TARGET_COL = "heart_disease"

# Exact feature headers as in your CSV
FEATURES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_induced_angina",
    "st_depression",
    "st_slope",
    "num_major_vessels",
    "thalassemia"
]

# Reproducibility and split
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Scale only continuous numeric features (helps LR/SVM)
SCALING_COLS = [
    "age",
    "resting_blood_pressure",
    "cholesterol",
    "max_heart_rate",
    "st_depression"
]
