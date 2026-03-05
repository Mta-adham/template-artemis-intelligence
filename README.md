# Artemis Intelligence Template

Minimal logistic regression pipeline with Weights & Biases tracking.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your W&B API key:

```
WANDB_API_KEY=your_key_here
```

## Usage

```bash
# Generate sample data
python data/generate_sample_data.py

# Run benchmark (trains model + evaluates + logs to W&B)
python -m src.benchmark
```

## Project Structure

```
data/               Sample train/test parquet files
src/task.py         Logistic regression training + prediction
src/benchmark.py    Runs the task, evaluates accuracy, logs to W&B
```
