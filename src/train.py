import json
import glob
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

RANDOM_STATE = 42

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "metadata.json"


def find_csv() -> Path:
    matches = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not matches:
        raise FileNotFoundError("Aucun CSV trouvé dans data/raw. Vérifie le téléchargement Kaggle.")
    return Path(matches[0])


def main():
    csv_path = find_csv()
    df = pd.read_csv(csv_path)

    target_col = "Outcome"
    if target_col not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_col}' introuvable. Colonnes: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)

    metadata = {
        "dataset_csv": str(csv_path),
        "model_type": "RandomForestClassifier",
        "threshold": 0.5,
        "features": list(X.columns),
        "metrics": {
            "accuracy": acc,
            "roc_auc": auc,
            "confusion_matrix": cm.tolist()
        }
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved model:", MODEL_PATH)
    print("Saved metadata:", META_PATH)
    print("\nAccuracy:", acc)
    print("ROC AUC:", auc)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, pred))


if __name__ == "__main__":
    main()
