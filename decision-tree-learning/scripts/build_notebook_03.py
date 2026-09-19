"""One-off script that builds notebooks/03_preprocessing.ipynb via nbformat."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))

md("""# 03 — Data Preprocessing

Telco Customer Churn — Decision Trees & Ensemble Learning project""")

md("""## 1. Learning Objectives

- Fix the "hidden missing value" problem found in `TotalCharges` (notebook 01)
- Understand why fitting must happen on training data only — **data leakage**
- Engineer two simple, explainable features
- Encode the target and categorical features correctly
- Produce a train/test split ready for modeling in notebook 04""")

md("""## 2. Business Problem

A model can only be as good as the data it's trained on. Right now our data
has a text column that should be numeric, categories that need encoding into
numbers, and no train/test split yet. This notebook turns the *understood*
and *explored* dataset from notebooks 01–02 into a *modeling-ready* one.""")

md("""## 3. Concept Explanation — Data Leakage

### What is it?
Data leakage happens when information that wouldn't be available at
real-world prediction time "leaks" into training — most commonly, by fitting
a transformation (encoding, scaling, imputing) on the FULL dataset instead
of the training set alone.

### Why does it matter?
A leaky pipeline can show excellent test accuracy that will NOT hold up in
production, because the "test" set wasn't really unseen — its statistics
already influenced the model indirectly.

### The rule we follow in this notebook
> **Split first. Fit second — on training data only. Apply that same fit to test.**

This is why `src/preprocessing/preprocessor.py` splits the data *before*
one-hot encoding categorical columns, and why the target encoding (a fixed
Yes→1/No→0 mapping, not a data-dependent statistic) is the one exception
that's safe to do before splitting.""")

md("""## 4. Imports""")

code("""import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_loader import load_dataset
from src.preprocessing.data_cleaner import clean_dataset
from src.preprocessing.feature_engineering import create_features
from src.preprocessing.preprocessor import (
    encode_target,
    split_dataset,
    encode_categorical_features,
)

pd.set_option("display.max_columns", None)""")

md("""## 5. Load Data""")

code("""customer_data = load_dataset()
customer_data.shape""")

md("""## 6. Explore Data — confirm the problem we're about to fix

Recall from notebook 01: `TotalCharges` is `object` dtype and has 11 blank
values, all at `tenure == 0`.""")

code("""print("dtype before cleaning:", customer_data["TotalCharges"].dtype)
blank_mask = pd.to_numeric(customer_data["TotalCharges"], errors="coerce").isnull()
print("Blank TotalCharges rows:", blank_mask.sum())""")

md("""## 7. Practical Implementation

### 7a. Data Cleaning

`clean_dataset()` fixes `TotalCharges` (blank → 0.0, since these are
brand-new customers who have genuinely been charged nothing yet — NOT the
column mean, which would fabricate a billing history) and removes any
duplicate customer IDs (none exist here, but the check runs regardless).""")

code("""cleaned_data = clean_dataset(customer_data)

print("dtype after cleaning:", cleaned_data["TotalCharges"].dtype)
print("Remaining blank/NaN TotalCharges:", cleaned_data["TotalCharges"].isnull().sum())
cleaned_data.loc[cleaned_data["tenure"] == 0, ["customerID", "tenure", "TotalCharges"]].head()""")

md("""**Common mistake avoided:** filling with the column *mean* would insert
a fake billing history for customers who have never been billed. Filling
with 0 is the value that's actually true.""")

md("""### 7b. Feature Engineering

`create_features()` adds:
- `AverageMonthlySpend = TotalCharges / (tenure + 1)` — the `+ 1` avoids a
  division-by-zero crash for the `tenure == 0` customers we just fixed above.
- `IsLongTermCustomer = tenure >= 24` — a binary flag for "has outlasted the
  longest contract type."

Both are computed from information that exists at prediction time (no target
leakage), so it's safe to compute them before splitting.""")

code("""engineered_data = create_features(cleaned_data)
engineered_data[["tenure", "TotalCharges", "AverageMonthlySpend", "IsLongTermCustomer"]].head()""")

md("""### 7c. Target Encoding

`Churn` ('Yes'/'No') becomes an integer (1/0). This mapping is fixed and
doesn't depend on the dataset's statistics, so — unlike feature encoding —
it's safe to apply before the train/test split.""")

code("""encoded_data = encode_target(engineered_data)
encoded_data["Churn"].value_counts()""")

md("""### 7d. Train/Test Split — BEFORE feature encoding

We drop `customerID` (an identifier, not a predictive feature — including it
would let a tree "memorize" individual customers) and separate features from
the target, then split.""")

code("""feature_columns = engineered_data.drop(columns=["customerID"]).columns.drop("Churn")
features = encoded_data[feature_columns]
target = encoded_data["Churn"]

training_features, testing_features, training_target, testing_target = split_dataset(features, target)

print("Training rows:", len(training_features))
print("Testing rows:", len(testing_features))
print("Training churn rate: %.3f" % training_target.mean())
print("Testing churn rate: %.3f" % testing_target.mean())""")

md("""**Interpretation:** training and testing churn rates are almost
identical (~26.5% each) because `split_dataset()` uses `stratify=target`.
Without stratification, a single unlucky random split could put
noticeably more or fewer churners in the test set purely by chance, making
it harder to compare experiments fairly.""")

md("""### 7e. Categorical Encoding — fit on training data only

`encode_categorical_features()` one-hot encodes categorical columns, fitting
the category list on `training_features` and then reindexing
`testing_features` onto those exact same columns.""")

code("""training_encoded, testing_encoded = encode_categorical_features(training_features, testing_features)

print("Training shape:", training_encoded.shape)
print("Testing shape:", testing_encoded.shape)
print("Columns identical:", list(training_encoded.columns) == list(testing_encoded.columns))""")

md("""### 7f. Save Processed Data

We save the cleaned, feature-engineered (but not yet one-hot-encoded)
dataset to `data/processed/`. We save it *before* one-hot encoding so later
notebooks can decide their own encoding strategy if needed, and so the saved
file stays human-readable (category names, not 0/1 dummy columns).""")

code("""from src.config.settings import SETTINGS

SETTINGS.data.processed_path.parent.mkdir(parents=True, exist_ok=True)
engineered_data_with_target = encode_target(engineered_data)
engineered_data_with_target.to_csv(SETTINGS.data.processed_path, index=False)
print("Saved to", SETTINGS.data.processed_path)""")

md("""## 8. Results

| Step | Before | After |
|---|---|---|
| `TotalCharges` dtype | `object`, 11 blanks | `float64`, 0 blanks |
| Feature count | 19 raw features | 47 after engineering + one-hot encoding |
| Target | 'Yes'/'No' strings | 1/0 integers |
| Rows | 7,043 total | 5,634 train / 1,409 test |
| Train/test churn rate | — | ~26.5% in both (stratified) |""")

md("""## 9. Interpretation

The dataset is now fully numeric, has no missing values, and is split in a
way that preserves the original class balance in both halves. Every
transformation that depends on the data's own statistics (categorical
encoding) was fit on the training set only — the test set was never looked
at until we applied that fixed transformation to it.""")

md("""## 10. Common Mistakes

- **Fitting `pd.get_dummies()` on the full dataset before splitting.** This
  looks harmless (it's "just" turning categories into columns) but if a
  rare category only exists in a handful of rows, which set those rows land
  in becomes information the split shouldn't have had access to when
  choosing column structure. Always split first.
- **Filling missing values with the mean/median without asking why they're
  missing.** Here, the "right" fill value (0) came from understanding *why*
  the value was blank (zero tenure), not from a generic imputation rule.
- **Forgetting `customerID` is not a feature.** An ID column has a unique
  value per row — a tree could use it to "predict" perfectly on training
  data while learning nothing that generalizes.
- **Encoding train and test separately without reindexing.** If a category
  appears in test but not train (or vice versa), the two encoded frames end
  up with different columns — this fails silently (misaligned columns) or
  loudly (shape mismatch) depending on what you do next.""")

md("""## 11. What We Learned

- We converted the `TotalCharges` blanks to 0.0 based on **understanding
  why** they were blank, not a generic imputation strategy.
- We engineered two simple, leakage-safe features.
- We followed the "split first, fit second" rule to avoid data leakage
  during categorical encoding.
- The dataset is now ready for `04_decision_tree.ipynb`.""")

md("""## 12. Exercises

1. Change `IsLongTermCustomer`'s threshold from 24 to 12 months in
   `feature_engineering.py` and re-run this notebook. Does the number of
   customers flagged as long-term change a lot?
2. `encode_categorical_features()` uses one-hot encoding. Look up
   `sklearn.preprocessing.OrdinalEncoder` — for which of our categorical
   columns (if any) might ordinal encoding make more sense than one-hot?
   (Hint: think about `Contract`'s three values — is there a natural order?)
3. Deliberately break the leakage rule: call `pd.get_dummies()` on the WHOLE
   `engineered_data` before splitting, then split. Compare the resulting
   column count to what we got here. Can you find a category that behaves
   differently between the two approaches?
4. We dropped `customerID` before modeling. Write one sentence explaining,
   in your own words, why a decision tree would treat a unique ID column as
   a "perfect" (but useless) predictor if we left it in.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

output_path = Path(__file__).resolve().parent.parent / "notebooks" / "03_preprocessing.ipynb"
nbf.write(nb, output_path)
print(f"Wrote {output_path}")
