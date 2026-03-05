"""Task: Train a logistic regression classifier and produce predictions.

Mirrors the original pipeline structure (StandardScaler + LogisticRegression)
but operates on local parquet sample data instead of S3/zarr sources.
"""

import logging
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.parquet")


def create_lr_pipeline(C=0.01, solver="liblinear", max_iter=1000):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=C, solver=solver, max_iter=max_iter)),
        ]
    )


def load_data():
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    logger.debug(
        "Loaded data — train: %d samples, test: %d samples",
        len(y_train),
        len(y_test),
    )
    return X_train, y_train, X_test, y_test


def run_task():
    X_train, y_train, X_test, y_test = load_data()

    pipeline = create_lr_pipeline(C=0.1, solver="liblinear", max_iter=800)
    logger.debug("Fitting logistic regression pipeline …")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    results_df = pd.DataFrame({"y_true": y_test, "y_pred": predictions})
    results_df.to_parquet(PREDICTIONS_PATH, index=False)
    logger.info("Saved %d predictions to %s", len(predictions), PREDICTIONS_PATH)

    return results_df


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(name)s — %(message)s")
    run_task()


if __name__ == "__main__":
    main()
