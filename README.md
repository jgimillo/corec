<div align="center">
  <h1>Corec</h1>
</div>

<p align="center">
  <!-- Python -->
  <a href="https://www.python.org"><img src="https://badges.aleen42.com/src/python.svg" alt="Python"></a>
  <!-- Version -->
  <a href="https://pypi.org/project/corec/"><img src="https://img.shields.io/pypi/v/corec?color=light-green" alt="PyPI version"></a>
  <a href="https://amenra.github.io/ranx"><img src="https://img.shields.io/badge/docs-passing-<COLOR>.svg" alt="Documentation Status"></a>
  <!-- Ruff -->
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <!-- License -->
  <a href="https://lbesson.mit-license.org/"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>


Corec is a flexible and configurable framework designed for context-aware recommenders. It consists of two independent modules: one for generating recommendations and the other for evaluating recommendation metrics.

The recommendation module supports both **Elliot-based models** and **contextual heuristic models** to generate predictions. The evaluation module helps you compute various metrics from prediction files (either generated with the recommendation module or externally). It includes support for metrics from the **Ranx** library, as well as custom Corec metrics, such as **context satisfaction accumulation** or **mean context satisfaction**.

## Features

- **Recommendation Module**: 
  - Generate predictions using Elliot-based models or context-aware heuristic models.
  - Fully configurable based on dataset structure, prediction format, and system resources.
  - Supports parallel processing with flexible chunk sizes and multiple processors.
  
- **Evaluation Module**:
  - Evaluate recommendation quality by generating CSV files with metrics.
  - Supports **Ranx** metrics and Corec's custom metrics like context satisfaction.
  - Easily configurable to match prediction file formats and dataset structures.

## Installation

To install Corec, simply use `pip`:

```bash
pip install corec
```

## Usage

### Recommendation Module Examples

Here’s an example of how to use the **Elliot Recommendation Module** to generate predictions based on the library Elliot:

```python
from corec.recommenders import ElliotRec

# Instantiate the Elliot recommender
elliot_rec = ElliotRec(
    preds_path_template="preds/{model}/{model}.tsv.gzip",
    train_path="dataset/train.inter",
    test_path="dataset/test.inter",
    preds_score_col_name="score",
    elliot_work_dir="elliot_work_dir",
)

# Setup the model parameters according to the official docs from [Elliot](https://elliot.readthedocs.io/en/latest/guide/recommenders.html)
models_config = {
    "ItemKNN": {
        "implementation": "classic",
        "neighbors": 40,
        "similarity": "cosine",
    },
    "FM": {
        "epochs": 10,
        "batch_size": 512,
        "factors": 10,
        "lr": 0.001,
        "reg": 0.1,
    }
}

# You are ready to run the experiment
elliot_rec.run_elliot_experiment(
    models_config,
    K=50,
    clean_elliot_work_dir=True,
    clean_temp_dataset_files=True,
)
```

And here is shown an example of usage of **Heuristic Recommendation Module**:

```python
from corec.recommenders import ContextRandomRec

# Instantiate the context-aware recommender
cp_rec = ContextPopRec(
    train_path="dataset/train.inter",
    test_path="dataset/test.inter",
    valid_path="dataset/valid.inter"
    logs_path="ContextPop.log",
    chunk_size=100,
)

cp_rec.compute_predictions(
    output_path="preds/ContextPop.tsv.gzip",
    K=5,
)
```

### Evaluation Module Example

After generating predictions, you can evaluate them with the **Evaluation Module**. Here's an example of how to use the module to compute metrics:

```python

```

### Dataset Structure

Corec assumes the following structure for the input datasets:

- **Training Set**: A file containing training data for recommender model training.
- **Test Set**: A file containing test data for generating recommendations.
- **Optional Validation Set**: An optional file for validation during model training or evaluation.

Each dataset should have the following columns:
- **User ID**: A column representing the unique identifier for each user (either `str` or `int`).
- **Item ID**: A column representing the unique identifier for each item (either `str` or `int`).
- **Rating ID**: A column representing the rating given by the user (usually a `float`).
- **Context Columns**: Additional columns representing the context for each recommendation (all `int`).

Example:

| User ID | Item ID | Rating ID | Context 1 | Context 2 | ... |
|---------|---------|-----------|-----------|-----------|-----|
| 1       | 101     | 4.5       | 1         | 0         | ... |
| 2       | 102     | 3.8       | 0         | 1         | ... |
