"""
train_model.py — EcoSort AI
Trains a TF-IDF + Logistic Regression text classifier on waste_samples.csv
and saves the trained model to model.pkl for use by app.py.

Run this script once before starting the Streamlit app:
    python train_model.py
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "waste_samples.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

VALID_CATEGORIES = {"Organic", "Recyclable", "General Waste"}


def load_and_validate(path: str) -> pd.DataFrame:
    """Load CSV and check for required columns and known labels."""
    df = pd.read_csv(path)

    required = {"item", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Drop rows with empty values
    initial = len(df)
    df = df.dropna(subset=["item", "category"])
    dropped = initial - len(df)
    if dropped:
        print(f"  [!] Dropped {dropped} rows with missing values.")

    # Warn about unknown labels
    unknown = set(df["category"].unique()) - VALID_CATEGORIES
    if unknown:
        print(f"  [!] Unknown categories found and will be excluded: {unknown}")
        df = df[df["category"].isin(VALID_CATEGORIES)]

    print(f"  Loaded {len(df)} samples across {df['category'].nunique()} categories.")
    print(f"  Category distribution:\n{df['category'].value_counts().to_string()}\n")
    return df


def build_pipeline() -> Pipeline:
    """Return a scikit-learn Pipeline with TF-IDF and Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams improve phrase matching
            lowercase=True,
            # No stop_words removal: words like "used", "clean", "dirty"
            # carry category signal for small waste-item descriptions.
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # corrects for unequal category sizes
            random_state=42
        ))
    ])


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Split data, train the pipeline, evaluate on held-out test set, return model."""
    X = df["item"].to_numpy(dtype=str)
    y = df["category"].to_numpy(dtype=str)

    # Stratify to keep class balance in both train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}\n")

    model = build_pipeline()
    model.fit(X_train, y_train)

    # Evaluate on held-out test set (no training data leakage)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 50)
    print(f"  Test Accuracy: {acc * 100:.1f}%")
    print("=" * 50)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    return model


def save_model(model: Pipeline, path: str) -> None:
    """Persist the trained pipeline with pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {path}")


def main():
    print("\n EcoSort AI — Model Training\n" + "-" * 35)

    print("Step 1: Loading dataset ...")
    df = load_and_validate(DATA_PATH)

    print("Step 2: Training and evaluating model ...")
    model = train_and_evaluate(df)

    print("Step 3: Saving model ...")
    save_model(model, MODEL_PATH)

    print("\n Training complete. Run the app with:\n  streamlit run app.py\n")


if __name__ == "__main__":
    main()
