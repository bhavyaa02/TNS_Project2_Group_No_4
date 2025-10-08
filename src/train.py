import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .config import PROCESSED_DIR, MODELS_DIR, FEATURES, RANDOM_STATE
from .metrics import compute_all_metrics
from .utils import save_artifact, save_json

def _load_split():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv", header=None).values.ravel()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv", header=None).values.ravel()
    return X_train, X_test, y_train, y_test

def _eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        import numpy as np
        s = model.decision_function(X_test)
        y_prob = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return compute_all_metrics(y_test, y_pred, y_prob)

def train_compare_and_tune():
    X_train, X_test, y_train, y_test = _load_split()

    candidates = {
        "logreg": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "dt": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE)
    }

    rows = []
    best_key, best_model, best_tuple = None, None, (-1, -1, -1)
    for key, model in candidates.items():
        model.fit(X_train, y_train)
        m = _eval(model, X_test, y_test)
        m["model"] = key
        rows.append(m)
        t = (m["f1"], m["recall"], m["roc_auc"])
        if t > best_tuple:
            best_tuple, best_key, best_model = t, key, model

    Path("reports").mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(Path("reports") / "model_comparison.csv", index=False)

    grid_map = {
        "logreg": {"C": [0.1, 1, 3], "penalty": ["l2"], "solver": ["lbfgs"]},
        "svm": {"C": [0.5, 1, 2], "gamma": ["scale", "auto"]},
        "dt": {"max_depth": [3, 5, 7, None], "min_samples_split": [2, 5, 10]},
        "rf": {"n_estimators": [400, 700], "max_depth": [None, 10, 14]}
    }
    gs = GridSearchCV(best_model, grid_map[best_key], cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned = gs.best_estimator_

    tuned_m = _eval(tuned, X_test, y_test)
    tuned_m["model"] = f"{best_key}_tuned"
    cmp = pd.concat([pd.DataFrame(rows), pd.DataFrame([tuned_m])])
    cmp.to_csv(Path("reports") / "model_comparison.csv", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_artifact(tuned, MODELS_DIR / "best_model.pkl")
    save_json({"selected": tuned.__class__.__name__, "best_params": gs.best_params_, "metrics": tuned_m, "features": FEATURES}, MODELS_DIR / "model_card.json")
    print("Saved best_model.pkl and model_card.json")

if __name__ == "__main__":
    train_compare_and_tune()
