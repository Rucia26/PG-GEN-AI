# Decision Tree & Ensemble Learning — Telco Customer Churn

A learning-focused, end-to-end machine learning project covering decision
trees, entropy/Gini/information gain, overfitting, cross-validation,
hyperparameter tuning, and ensemble learning (bagging, random forests,
boosting), built on the Telco Customer Churn dataset.

## Project Overview

This project predicts customer churn while teaching the reasoning behind
every step — not just the syntax. Every notebook explains a concept
(what it is, why it's needed, the math, a worked example) before
implementing it, first from scratch where practical, then with
scikit-learn, comparing the two.

## Business Problem

A telecom company wants to know which customers are likely to cancel their
service, so retention efforts can be targeted before the customer leaves.

## Dataset

The [Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
(IBM sample dataset, 7,043 customers, 21 columns), downloaded into
`data/raw/telco_churn.csv`.

## Objectives

- Understand classification, decision trees, and the theory behind splits
  (entropy, Gini impurity, information gain)
- Practice proper preprocessing without data leakage
- Understand and demonstrate overfitting/underfitting with real experiments
- Evaluate models correctly on an imbalanced target (not accuracy alone)
- Tune hyperparameters with cross-validation
- Understand and compare bagging (Random Forest) vs. boosting (Gradient Boosting)
- Practice model selection as a business decision, not a leaderboard

## Project Structure

```
decision-tree-learning/
├── config.yaml              # Central configuration (paths, seeds, logging)
├── requirements.txt
├── data/
│   ├── raw/                 # Original downloaded dataset
│   └── processed/           # Cleaned + feature-engineered dataset
├── notebooks/                # The learning path — see below
├── src/
│   ├── config/               # Typed settings loaded from config.yaml
│   ├── data/                 # Loading + validating the raw dataset
│   ├── preprocessing/        # Cleaning, feature engineering, encoding/split
│   ├── models/                # DecisionTree / RandomForest / GradientBoosting wrappers
│   ├── evaluation/            # Shared classification metrics
│   ├── visualization/         # Reusable plotting functions
│   └── utils/logger.py        # Console + file logging setup
├── tests/                    # Unit tests for data/preprocessing/models/evaluation
├── scripts/                   # One-off scripts that generate the notebooks
├── logs/ml_project.log        # Runtime log output
└── reports/
    ├── figures/
    └── model_comparison.csv   # Final metrics table across all models
```

Each `src/` package has exactly one responsibility (loading vs. validating
vs. cleaning vs. encoding, etc.) so a bug in one step can't be confused with
a bug in another — this separation is explained in each notebook as it's
used.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## How to Run

Open the notebooks in order inside VS Code or Jupyter:

```bash
jupyter lab notebooks/
```

Or, since the notebooks were generated programmatically, regenerate and
re-execute any of them from the command line:

```bash
python scripts/build_notebook_01.py   # regenerates the notebook file
python -m pytest tests/ -v            # run the test suite
```

## Learning Path

| Notebook | Covers |
|---|---|
| `01_data_understanding.ipynb` | Shape, dtypes, hidden missing values, class balance |
| `02_eda.ipynb` | Univariate/bivariate/multivariate analysis, each answering a specific question |
| `03_preprocessing.ipynb` | Cleaning, feature engineering, leakage-safe encoding & split |
| `04_decision_tree.ipynb` | Entropy/Gini/information gain from scratch, tree visualization, overfitting experiment |
| `05_model_evaluation.ipynb` | Baseline vs. real model, precision/recall/F1, confusion matrix, error analysis |
| `06_hyperparameter_tuning.ipynb` | Cross-validation, GridSearchCV, what it does internally |
| `07_ensemble_learning.ipynb` | Bagging, Random Forest, boosting, Gradient Boosting, final comparison |

## ML Pipeline

```
Raw Dataset → Data Understanding → EDA → Cleaning → Feature Engineering
  → Encoding → Train/Test Split → Decision Tree → Evaluation
  → Hyperparameter Tuning → Random Forest / Gradient Boosting
  → Model Comparison → Error Analysis
```

## Models

- **Decision Tree** (`src/models/decision_tree_model.py`) — the base model,
  fully interpretable, visualized directly.
- **Random Forest** (`src/models/random_forest_model.py`) — bagging +
  random feature selection, reduces variance.
- **Gradient Boosting** (`src/models/gradient_boosting_model.py`) —
  sequential error-correcting ensemble.

## Evaluation Metrics

Accuracy, Precision, Recall, F1-score, ROC-AUC, and a confusion matrix —
see `src/evaluation/model_evaluator.py`. Notebook 05 explains why accuracy
alone is misleading on this dataset's ~26.5% / 73.5% class imbalance.

## Results

See `reports/model_comparison.csv` for the final metrics table (regenerate
by re-running `07_ensemble_learning.ipynb`). As of the last run:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.735 | 0.000 | 0.000 | 0.000 | — |
| Decision Tree | 0.798 | 0.635 | 0.567 | 0.599 | 0.830 |
| Random Forest | 0.798 | 0.650 | 0.521 | 0.579 | 0.839 |
| Gradient Boosting | 0.803 | 0.668 | 0.516 | 0.582 | 0.842 |

## Key Findings

- `Contract` type and `tenure` are the strongest churn signals — month-to-
  month customers churn at ~42.7% vs. ~2.8% for two-year contracts.
- The dataset has a ~26.5% churn rate — accuracy alone is a misleading
  metric here (a naive baseline hits 73.5% "accuracy" while catching zero
  churners).
- Ensemble models improve ranking quality (ROC-AUC) over a single tree,
  even where raw accuracy is similar.

## Future Improvements

- Add `AdaBoostClassifier` as a second boosting example
- Try SMOTE or class-weighting to address class imbalance directly during
  training, not just at evaluation
- Add a `RandomizedSearchCV` example for larger hyperparameter spaces
