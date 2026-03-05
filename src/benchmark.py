"""Benchmark: Run the task and evaluate prediction performance."""

import logging

from sklearn.metrics import accuracy_score, classification_report

from src.task import run_task

logger = logging.getLogger(__name__)


def evaluate():
    logger.info("Running task …")
    results_df = run_task()

    y_true = results_df["y_true"].values
    y_pred = results_df["y_pred"].values

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=["class_0", "class_1"]
    )

    logger.info("Task performance — accuracy: %.4f", accuracy)
    logger.info("Detailed classification report:\n%s", report)

    return accuracy


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(name)s — %(message)s")
    evaluate()


if __name__ == "__main__":
    main()
