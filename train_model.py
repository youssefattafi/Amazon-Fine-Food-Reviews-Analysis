"""Train and save a review sentiment classifier from the Kaggle SQLite dataset."""
import os
import sqlite3
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
DB = Path(os.environ.get("AMAZON_REVIEWS_DB", str(ROOT / "database.sqlite"))).expanduser()
OUT = ROOT / "models" / "sentiment_pipeline.joblib"

def main():
    if not DB.is_file():
        raise FileNotFoundError(f"Missing {DB}. Download database.sqlite from Kaggle.")
    with sqlite3.connect(DB) as conn:
        df = pd.read_sql_query("SELECT Text, Score FROM Reviews", conn)
    df = df.dropna(subset=["Text", "Score"]).copy()
    df["Text"] = df["Text"].astype(str).str.strip()
    df = df.loc[df["Text"].str.len().gt(0) & df["Score"].isin([1, 2, 4, 5])].copy()
    df = df.drop_duplicates(subset=["Text"])
    if len(df) > 100_000:
        df = df.sample(100_000, random_state=42)
    y = (df["Score"] >= 4).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["Text"], y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_features=50_000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved trained model to {OUT}")

if __name__ == "__main__":
    main()
