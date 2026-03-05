"""Generate synthetic sample data mimicking a binary classification task.

Creates train/test splits of feature matrices and binary labels,
saved as parquet files in the data/ directory.
"""

import os

import numpy as np
import pandas as pd

SEED = 42
N_TRAIN = 500
N_TEST = 150
N_FEATURES = 20


def generate_classification_data(n_samples, n_features, weights, rng):
    """Generate linearly separable data with some noise using shared weights."""
    X = rng.standard_normal((n_samples, n_features))
    logits = X @ weights + rng.standard_normal(n_samples) * 0.5
    y = (logits > 0).astype(int)
    return X, y


def to_dataframe(X, y):
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["label"] = y
    return df


def main():
    rng = np.random.default_rng(SEED)
    data_dir = os.path.dirname(os.path.abspath(__file__))

    weights = rng.standard_normal(N_FEATURES)
    X_train, y_train = generate_classification_data(N_TRAIN, N_FEATURES, weights, rng)
    X_test, y_test = generate_classification_data(N_TEST, N_FEATURES, weights, rng)

    train_df = to_dataframe(X_train, y_train)
    test_df = to_dataframe(X_test, y_test)

    train_df.to_parquet(os.path.join(data_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(data_dir, "test.parquet"), index=False)

    print(f"Generated training data:  {train_df.shape} "
          f"(positive rate: {y_train.mean():.2f})")
    print(f"Generated test data:      {test_df.shape} "
          f"(positive rate: {y_test.mean():.2f})")
    print(f"Saved to: {data_dir}")


if __name__ == "__main__":
    main()
