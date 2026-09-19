"""
One-off script that builds notebooks/01_data_understanding.ipynb via nbformat.

Not part of the learning pipeline itself — this just generates the notebook
file. You can delete this script later; it's kept in scripts/ (not src/) so
it's clearly separate from the reusable project code.
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))

# --- Title -------------------------------------------------------------
md("""# 01 — Data Understanding

Telco Customer Churn — Decision Trees & Ensemble Learning project""")

# --- 1. Learning Objectives --------------------------------------------
md("""## 1. Learning Objectives

By the end of this notebook you will be able to:

- Load a dataset through project code instead of ad-hoc `pd.read_csv`
- Read a DataFrame's shape, dtypes, and memory usage and explain what each tells you
- Distinguish numerical from categorical features in a real dataset
- Find missing values **and** "hidden" missing values that aren't `NaN`
- Check for duplicate records
- Compute basic descriptive statistics and explain what they mean
- Examine the target variable's class balance and explain why it matters later""")

# --- 2. Business Problem -------------------------------------------------
md("""## 2. Business Problem

A telecom company wants to know **which customers are likely to cancel their
service (churn)** so it can intervene — offer a discount, a call from
retention, a plan change — before the customer leaves.

Every row in this dataset is one customer. The target column, `Churn`, is
`"Yes"` if that customer left and `"No"` if they stayed. Everything else is a
potential predictor: how long they've been a customer, what services they
use, how they pay, and so on.

We are not building a model yet. Before you can trust any model, you have to
know what you're actually feeding it — that's what this notebook is for.""")

# --- 3. Concept Explanation ----------------------------------------------
md("""## 3. Concept Explanation — "Data Understanding" as a distinct step

### What is it?
The systematic first look at a dataset: its size, structure, types, and
obvious quality issues — before any cleaning, visualization, or modeling.

### Why do we need it?
Every later step assumes things about the data (which columns exist, which
are numeric, whether the target has exactly two classes). If those
assumptions are wrong, bugs show up much later — inside a model training
call — where they are far harder to diagnose than right here.

### What problem does it solve?
It turns "I think this is a customer churn dataset" into a precise, verified
description: exact shape, exact column types, exact missing-value counts.

### What could go wrong if you skip it?
- You one-hot encode a column you thought was categorical but is actually a
  free-text ID field → thousands of useless columns.
- You assume a column is numeric (like `TotalCharges` here) when it's
  actually stored as text with some blank values → a confusing
  `could not convert string to float` error during training, far from the
  actual cause.
- You don't notice severe class imbalance → you trust a 73% "accurate" model
  that never predicts churn at all (we'll prove this in notebook 05).

### How do we verify it worked?
We'll end this notebook with a short written summary of the dataset that
you could hand to a colleague who has never seen it.""")

# --- 4. Imports -----------------------------------------------------------
md("""## 4. Imports

We import our own `load_dataset` / `validate_dataset` functions instead of
calling `pd.read_csv` directly in the notebook. That way the *same* loading
and validation logic runs no matter which notebook or script uses it — if we
ever need to change how the file is found, we change it in one place
(`src/data/data_loader.py`), not in seven notebooks.""")

code("""import sys
from pathlib import Path

# Allow "from src...." imports when Jupyter's working directory is
# notebooks/ rather than the project root.
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_loader import load_dataset
from src.data.data_validator import validate_dataset

# Notebooks are for exploration, so we display full column output
# rather than pandas' default truncation.
pd.set_option("display.max_columns", None)""")

# --- 5. Load Data -----------------------------------------------------------
md("""## 5. Load Data

`load_dataset()` reads `data/raw/telco_churn.csv` (the path comes from
`config.yaml`, not a hard-coded string) and logs what it did.
`validate_dataset()` then checks the result against the schema we expect and
reports anything unusual — without changing the data.""")

code("""customer_data = load_dataset()
validation_result = validate_dataset(customer_data)

print("Valid:", validation_result.is_valid)
for warning in validation_result.warnings:
    print("Warning:", warning)""")

# --- 6. Explore Data ---------------------------------------------------------
md("""## 6. Explore Data

### `shape`
Returns `(number_of_rows, number_of_columns)`. This is the first sanity
check — does the row count match what you expect for this dataset?""")

code("""customer_data.shape""")

md("""### `.head()`
Shows the first few rows so you can visually confirm the data looks the way
you expect — right columns, sensible values, no obvious encoding issues.""")

code("""customer_data.head()""")

md("""### `.info()`
Shows, per column: how many non-null values it has and its dtype (`int64`,
`float64`, `object`). `object` usually means text/categorical — **but watch
`TotalCharges` closely below.**""")

code("""customer_data.info()""")

md("""### Interpretation — `TotalCharges` is `object`, not `float64`

Notice `TotalCharges` looks like a number in `.head()` above but pandas
loaded it as `object` (text). That's your signal something is off with this
column — we'll find out exactly what next.""")

code("""# pd.to_numeric with errors="coerce" turns anything it can't parse into NaN,
# without crashing. Comparing the coerced version to the original count
# tells us exactly how many values are "secretly" non-numeric.
non_numeric_mask = pd.to_numeric(customer_data["TotalCharges"], errors="coerce").isnull()
print("Non-numeric TotalCharges rows:", non_numeric_mask.sum())
customer_data.loc[non_numeric_mask, ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]]""")

md("""### What are we looking at?
Every one of these rows has `tenure == 0` — a customer who just signed up
and hasn't been billed yet, so `TotalCharges` is an **empty string**, not a
missing marker pandas recognizes.

### Why does this matter?
`customer_data["TotalCharges"].isnull().sum()` below will report **zero**
missing values, even though 11 rows are not usable as numbers yet. This is
exactly the kind of "hidden missing value" that a checklist-only approach
(just run `.isnull()` and move on) would miss entirely.""")

code("""# This UNDERSTATES missingness for TotalCharges — that's the point of this cell.
customer_data.isnull().sum()""")

md("""### `.nunique()` — how many distinct values per column

Useful for telling numeric-looking columns apart from categorical ones, and
for spotting ID-like columns (nearly as many unique values as rows).""")

code("""customer_data.nunique().sort_values()""")

md("""### `.duplicated()` — exact duplicate rows

We also check `customerID` specifically, since two rows could differ by a
typo but share the same customer ID (a data integrity problem worth knowing
about even if it doesn't happen here).""")

code("""print("Fully duplicated rows:", customer_data.duplicated().sum())
print("Duplicate customerIDs:", customer_data["customerID"].duplicated().sum())""")

md("""### Numerical vs. categorical features

Splitting columns into these two groups up front makes every later notebook
(EDA, encoding, feature engineering) simpler — each function can just ask
"is this column numerical or categorical?" instead of re-deriving it.""")

code("""numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    column for column in customer_data.columns
    if column not in numerical_features + ["customerID", "Churn"]
]

print("Numerical features:", numerical_features)
print()
print("Categorical features:", categorical_features)""")

md("""### `.describe()` — descriptive statistics for numerical columns

`count`, `mean`, `std`, `min`, `25%`/`50%`/`75%` (quartiles), `max`. Note
`TotalCharges` is excluded here since pandas still sees it as text.""")

code("""customer_data[numerical_features].describe()""")

md("""### `.value_counts()` — class distribution of the target

This is the single most important cell in this notebook. If one class
dominates, a model can get high accuracy by (almost) always predicting the
majority class — which would be useless for the business goal of catching
churners.""")

code("""churn_counts = customer_data["Churn"].value_counts()
churn_percentages = customer_data["Churn"].value_counts(normalize=True) * 100

print(churn_counts)
print()
print(churn_percentages.round(1))""")

md("""### What pattern do we see?

About 73% of customers did **not** churn and about 27% did. This is a
moderate class imbalance — not extreme (like 99/1 fraud data), but enough
that accuracy alone will be a misleading metric later (notebook 05 proves
this with a baseline model).""")

# --- 7. Practical Implementation note --------------------------------------
md("""## 7. Practical Implementation

There's no additional implementation in this notebook beyond what's above —
"practical implementation" here *is* the exploration itself. Preprocessing
(actually fixing `TotalCharges`, encoding categoricals) happens in
`03_preprocessing.ipynb`, deliberately kept separate so this notebook stays
about *understanding*, not *changing*, the data.""")

# --- 8/9 Results & Interpretation -----------------------------------------
md("""## 8. Results & 9. Interpretation — Summary

| Fact | Value | Why it matters |
|---|---|---|
| Rows | 7,043 | Sample size for training/testing |
| Columns | 21 | 1 ID, 19 features, 1 target |
| Missing values (`.isnull()`) | 0 reported | **Misleading** — see next row |
| Hidden missing values | 11 (`TotalCharges`, blank strings, all `tenure == 0`) | Needs explicit handling in preprocessing |
| Duplicate rows / IDs | 0 | No deduplication needed |
| Target balance | 73.5% No / 26.5% Yes | Moderate imbalance — accuracy alone will mislead |
| Numerical features | 3 (`tenure`, `MonthlyCharges`, `TotalCharges`) | Will need scaling considerations for some models |
| Categorical features | 16 | Will need encoding before modeling |""")

# --- 10. Common Mistakes -----------------------------------------------------
md("""## 10. Common Mistakes

- **Trusting `.isnull().sum() == 0` as proof of "no missing data."** As shown
  above, blank strings, placeholder values like `"Unknown"` or `-1`, and
  whitespace-only strings all slip past `.isnull()`.
- **Treating `SeniorCitizen` as purely numerical** because it's `int64`. It's
  actually a binary categorical flag (0/1) — averaging it or feeding it into
  a distance-based model without treating it as a category can be misleading.
- **Not checking `.nunique()` for ID-like columns before modeling.**
  `customerID` has 7,043 unique values — if accidentally included as a
  feature, a tree could "memorize" it and appear to perform well while
  learning nothing generalizable.
- **Skipping the target's class balance check.** It's one line of code and
  changes how you should interpret every metric later.""")

# --- 11. What We Learned -----------------------------------------------------
md("""## 11. What We Learned

- The dataset has 7,043 customers and 21 columns, no exact duplicates.
- `TotalCharges` is stored as text and has 11 rows that are blank (new
  customers with `tenure == 0`) — invisible to a plain `.isnull()` check.
- The target class `Churn` is imbalanced (~73.5% / 26.5%), which we must
  account for during evaluation, not just training.
- We now have an explicit list of numerical vs. categorical features to
  carry into EDA and preprocessing.""")

# --- 12. Exercises -----------------------------------------------------------
md("""## 12. Exercises

1. `SeniorCitizen` is stored as `int64` (0 or 1). Use `.unique()` to confirm
   it only ever takes those two values.
2. Find how many customers have `InternetService == "No"`. What do you think
   happens to columns like `OnlineSecurity` for those customers? (Hint: look
   at their `.value_counts()` — there's a third category besides Yes/No.)
3. Compute the churn rate (`Churn == "Yes"` percentage) *separately* for
   customers with `Contract == "Month-to-month"` vs. `Contract == "Two year"`.
   Are they similar? Keep your answer in mind for notebook 02 (EDA).
4. `validate_dataset()` currently treats the blank `TotalCharges` values as a
   warning, not an error. Read `src/data/data_validator.py` and explain, in
   your own words, why that's the right design choice here rather than
   raising an exception.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

output_path = Path(__file__).resolve().parent.parent / "notebooks" / "01_data_understanding.ipynb"
output_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, output_path)
print(f"Wrote {output_path}")
