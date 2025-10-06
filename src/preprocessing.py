import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import (
    DATA_RAW, PROCESSED_DIR, TARGET_COL, FEATURES,
    TEST_SIZE, RANDOM_STATE, SCALING_COLS
)
from .utils import save_artifact

def load_data() -> pd.DataFrame:
    # Load and validate headers
    df = pd.read_csv(DATA_RAW)
    missing = set(FEATURES + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    # Force numeric and drop rows with coercion failures
    df = df[FEATURES + [TARGET_COL]].apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        df = df.dropna().reset_index(drop=True)
    return df

def split_and_scale(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET_COL].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    X_train_s[SCALING_COLS] = scaler.fit_transform(X_train[SCALING_COLS])
    X_test_s[SCALING_COLS] = scaler.transform(X_test[SCALING_COLS])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_artifact(scaler, PROCESSED_DIR / "scaler.pkl")
    X_train_s.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test_s.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    pd.Series(y_train).to_csv(PROCESSED_DIR / "y_train.csv", index=False, header=False)
    pd.Series(y_test).to_csv(PROCESSED_DIR / "y_test.csv", index=False, header=False)

if __name__ == "__main__":
    df = load_data()
    split_and_scale(df)
    print("Preprocessing complete for your CSV headers.")
