"""Benchmark: Run the task, evaluate performance, and log to Weights & Biases."""

import logging
import uuid

import wandb
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.task import run_task

load_dotenv()

logger = logging.getLogger(__name__)


def evaluate():
    run_hash = uuid.uuid4().hex[:12]
    logger.info("Run hash: %s", run_hash)

    wandb.init(
        project="artemis-intelligence",
        job_type="benchmark",
    )

    logger.info("Running task …")
    results_df = run_task()

    y_true = results_df["y_true"].values
    y_pred = results_df["y_pred"].values

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    wandb.log({
        "run_hash": run_hash,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    })

    report = classification_report(
        y_true, y_pred, target_names=["class_0", "class_1"]
    )

    logger.info("Task performance — accuracy: %.4f", accuracy)
    logger.info("Task performance — precision: %.4f, recall: %.4f, f1: %.4f",
                precision, recall, f1)
    logger.info("Detailed classification report:\n%s", report)

    wandb.finish()
    return accuracy


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(name)s — %(message)s")
    evaluate()


if __name__ == "__main__":
    main()
