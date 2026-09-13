# Hands-on Session: Data Preprocessing with AutoMPG Dataset

## Topics Covered
- Introduction to the AutoMPG dataset and its features
- Python libraries for data preprocessing and visualization
- Loading and initial exploration of the dataset
- Identifying categorical vs. numerical features
- Missing value analysis and artificial missing value insertion
- Imputation strategies: mean, median, mode, class-wise mode
- Principal Component Analysis (PCA) for visualization
- Comparing imputation strategies via PCA variance and scatter plots
- Exploratory Data Analysis (EDA): count plots, pair plots, correlation heatmap
- Univariate analysis: histograms, KDE plots, box plots, violin plots
- Class-wise separation in visualizations
- Interpretation of plots and statistical outputs

---

## Detailed Notes by Topic

### 1. Session Overview and Setup
- **Instructor**: Durga Toshniwal; **Hands-on lead**: Ashish Kumar (Senior PhD Scholar, IIT Roorkee)
- **Lab materials**: Available on LMS and chat link (Python notebook, CSV dataset, pre-run output PDF)
- **Environment**: Google Colab notebooks; students upload CSV file to session storage
- **Dataset**: AutoMPG (benchmark dataset from 1970s–80s car models)
  - Target: `mpg_high` (binary: 1 = high fuel efficiency, 0 = low)
  - Features:
    - `cylinders`: number of engine cylinders (categorical)
    - `displacement`: engine size (continuous)
    - `horsepower`: engine power (continuous)
    - `weight`: vehicle weight (continuous)
    - `acceleration`: acceleration (continuous)
    - `model_year`: categorical (1970–1982)
    - `origin`: country of origin (1=USA, 2=Europe, 3=Japan) – categorical but stored as integers
    - `car_name`: string (categorical)
- **Goal**: Hands-on practice of data preprocessing steps: missing value handling, imputation, visualization, EDA.

### 2. Libraries Used
| Library | Purpose |
|---------|---------|
| `warnings` | Suppress unwanted warnings in output |
| `pandas` | Load CSV, data frame operations, `df.info()`, `df.shape`, `df.corr()` |
| `numpy` | Efficient mathematical operations on arrays; random seed for reproducibility |
| `sklearn.preprocessing.SimpleImputer` | Mean, median, most_frequent (mode) imputation |
| `sklearn.decomposition.PCA` | Principal Component Analysis for dimensionality reduction |
| `sklearn.preprocessing.StandardScaler` | Standardize features to zero mean, unit variance before PCA |
| `matplotlib` | Base plotting |
| `seaborn` | Enhanced statistical visualizations (countplot, pairplot, heatmap, boxplot, violinplot) |

> **Note**: All libraries pre-installed in Colab; local users may need `pip install`.

### 3. Loading and Initial Exploration
```python
import pandas as pd
df = pd.read_csv('AutoMPG.csv')
print(df.shape)  # (398, 9)
df.info()
```
- **Output**: 398 rows, 9 columns; all columns show non-null counts = 398 → **no missing values originally**.
- **Data types**: `object` for `car_name` (string), `int64`/`float64` for others.
- **Critical observation**: `origin` and `mpg_high` are categorical but stored as integers → Pandas `df.info()` mislabels them as numeric. **Cannot rely solely on dtype**; must check unique values.

#### Identifying Categorical vs. Numerical Features
```python
for col in df.columns:
    print(col, df[col].nunique())
```
- **Numerical (high unique counts)**: `displacement` (82), `horsepower` (94), `weight` (351), `acceleration` (95)
- **Categorical (low unique counts)**: `cylinders` (5), `model_year` (13), `origin` (3), `mpg_high` (2), `car_name` (many but string)
- **Decision**: Treat the four high-cardinality columns as numerical for imputation and PCA; rest as categorical.

### 4. Missing Value Analysis (Artificial Insertion)
Since the dataset has no missing values, they are **artificially inserted** to demonstrate imputation.

#### Random Seed for Reproducibility
```python
import numpy as np
np.random.seed(42)  # Any integer; 42 used for consistency across runs
```
- **Purpose**: Fix randomness so that the same rows get NaN values every run → consistent results for comparison.
- **Mechanism**: `np.random.seed` sets the internal state of the random number generator; same seed → same sequence of random indices.

#### Inserting 10% Missing Values per Numerical Column
```python
df_10 = df.copy()
numeric_cols = ['displacement', 'horsepower', 'weight', 'acceleration']
for col in numeric_cols:
    n_missing = int(0.1 * len(df_10))
    missing_idx = np.random.choice(df_10.index, n_missing, replace=False)
    df_10.loc[missing_idx, col] = np.nan
```
- **Result**: `df_10.info()` shows ~359 non-null for each numerical column (10% of 398 ≈ 39 missing per column).
- **Later**: Same process repeated with 30% missing (`df_30`) to study impact of higher missingness.

### 5. Imputation Strategies
All imputations applied **column-wise** on the four numerical columns only.

#### 5.1 Mean Imputation
```python
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy='mean')
df_mean_10 = df_10.copy()
df_mean_10[numeric_cols] = imp_mean.fit_transform(df_10[numeric_cols])
```
- `fit_transform`: **fit** computes column means (ignoring NaN); **transform** replaces NaN with those means.

#### 5.2 Median Imputation
```python
imp_median = SimpleImputer(strategy='median')
df_median_10 = df_10.copy()
df_median_10[numeric_cols] = imp_median.fit_transform(df_10[numeric_cols])
```

#### 5.3 Mode (Most Frequent) Imputation
```python
imp_mode = SimpleImputer(strategy='most_frequent')
df_mode_10 = df_10.copy()
df_mode_10[numeric_cols] = imp_mode.fit_transform(df_10[numeric_cols])
```
- Works for both numerical and categorical columns.

#### 5.4 Class-wise Mode Imputation (Custom Logic)
- **Concept**: Compute mode **separately within each class** (mpg_high = 0 and 1) and fill missing values using the class-specific mode.
- **Why**: Preserves class-conditional distributions.
```python
df_class_mode_10 = df_10.copy()
for cls in df_10['mpg_high'].unique():
    mask = df_10['mpg_high'] == cls
    for col in numeric_cols:
        class_mode = df_10.loc[mask, col].mode()[0]
        df_class_mode_10.loc[mask & df_10[col].isna(), col] = class_mode
```
- **Loop**: Over each class label → create boolean mask → compute mode for that subset → fill NaNs only in that subset.

> **Key point**: `SimpleImputer` does not have built-in class-wise strategy; custom code required.

### 6. Principal Component Analysis (PCA) for Visualization
- **Theory not yet covered**; used here as a **visualization tool** to project 4D numerical data onto 2D (PC1, PC2).
- **Steps in code**:
  1. Select numerical columns → `X`
  2. Standardize: `StandardScaler().fit_transform(X)` → zero mean, unit variance per column.
  3. Apply PCA: `PCA(n_components=2).fit_transform(X_scaled)` → get PC1, PC2.
  4. Plot scatter of PC1 vs PC2, colored by `mpg_high` (class 0 = blue, class 1 = orange).
  5. Also compute **explained variance ratio** and **loadings matrix** (correlation of original features with PCs).

#### Explained Variance (Original Data)
| Component | Explained Variance |
|-----------|-------------------|
| PC1       | 80.07%            |
| PC2       | 16.00%            |
| PC3       | ~3%               |
| PC4       | ~1%               |
- **Interpretation**: PC1 captures most variance; PC1+PC2 ≈ 96% → good 2D representation.

#### Loadings Matrix (Original Data)
| Feature       | PC1    | PC2    | PC3     | PC4     |
|---------------|--------|--------|---------|---------|
| displacement  | 0.53   | 0.25   | -0.45   | -0.66   |
| horsepower    | 0.51   | 0.18   | -0.58   | 0.60    |
| weight        | 0.52   | 0.12   | 0.66    | 0.45    |
| acceleration  | -0.39  | -0.95  | -0.12   | -0.15   |
- **Interpretation**: PC1 positively correlated with displacement, horsepower, weight; negatively with acceleration. PC2 strongly negatively correlated with acceleration.

### 7. Comparing Imputation Strategies via PCA Visualization
For each imputed dataset (mean, median, mode, class-wise mode), the same PCA pipeline is run and scatter plots are placed side-by-side with original.

#### 10% Missing Values Results
| Strategy          | PC1 Variance | PC2 Variance | Visual Similarity to Original |
|-------------------|--------------|--------------|-------------------------------|
| Original          | 80.07%       | 16.00%       | —                             |
| Mean              | 73.0%        | 17.0%        | Very similar                  |
| Median            | 72.98%       | ~17%         | Very similar                  |
| Mode              | 70.0%        | ~18%         | Slightly more class mixing    |
| **Class-wise Mode** | **76.0%**    | **~16%**     | **Most similar**              |

- **Observation**: Class-wise mode preserves variance best (closest to original 80%). Mean/median close; mode worst among the four.

#### 30% Missing Values Results (Higher Missingness → Larger Differences)
| Strategy          | PC1 Variance | PC2 Variance | Visual Similarity |
|-------------------|--------------|--------------|-------------------|
| Original          | 80.07%       | 16.00%       | —                 |
| Mean              | 62.0%        | 19.0%        | Noticeable shrinkage |
| Median            | ~62%         | ~19%         | Similar to mean   |
| Mode              | 54.0%        | ~20%         | Severe shrinkage  |
| **Class-wise Mode** | **71.0%**    | **~17%**     | **Best preservation** |

- **Conclusion**: **Class-wise mode imputation outperforms others**, especially at higher missingness, because it respects class-conditional distributions. The dataset has clear class separation (high vs low MPG), making class-wise strategies effective.

> **Note**: Probabilistic imputation (estimating P(feature|class)) was discussed in theory but not implemented here; could be explored later.

### 8. Exploratory Data Analysis (EDA) on Original Data (No Missing Values)
All EDA uses the original `df` (no artificial NaNs).

#### 8.1 Count Plots for Categorical Features
```python
import seaborn as sns
sns.countplot(data=df, x='cylinders')   # 4 cylinders dominant
sns.countplot(data=df, x='model_year')  # 1973 peak
sns.countplot(data=df, x='origin')      # USA > Japan ≈ Europe
```
- **Purpose**: Show frequency of each category.
- **Special handling for `origin`**: Map 1→'USA', 2→'Europe', 3→'Japan' for readability.

#### 8.2 Pair Plots for Numerical Features
```python
sns.pairplot(df, vars=numeric_cols, diag_kind='kde')
```
- **Diagonal**: Kernel Density Estimate (KDE) of each feature.
- **Off-diagonal**: Scatter plots for each pair.
- **Observations**:
  - `displacement` vs `horsepower`: Strong positive linear relationship (r ≈ 0.89).
  - `displacement` vs `weight`: Strong positive linear (r ≈ 0.93).
  - `horsepower` vs `acceleration`: Negative relationship (higher horsepower → lower acceleration).
  - `acceleration` KDE: Near-normal (bell-shaped); others right-skewed.

#### 8.3 Class-Colored Pair Plot
```python
sns.pairplot(df, vars=numeric_cols, hue='mpg_high', diag_kind='kde')
```
- **Insight**: Class separation visible in `displacement`/`horsepower`/`weight` space:
  - Class 1 (high MPG): Lower displacement, horsepower, weight.
  - Class 0 (low MPG): Higher values.
- **Multimodality resolved**: Unimodal KDEs in univariate view split into two class-conditional modes.

#### 8.4 Correlation Heatmap
```python
corr_matrix = df[numeric_cols + ['mpg_high']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```
|               | displacement | horsepower | weight | acceleration | mpg_high |
|---------------|--------------|------------|--------|--------------|----------|
| displacement  | 1.00         | 0.89       | 0.93   | -0.50        | -0.78    |
| horsepower    | 0.89         | 1.00       | 0.86   | -0.69        | -0.73    |
| weight        | 0.93         | 0.86       | 1.00   | -0.42        | -0.83    |
| acceleration  | -0.50        | -0.69      | -0.42  | 1.00         | 0.30     |
| mpg_high      | -0.78        | -0.73      | -0.83  | 0.30         | 1.00     |
- **Interpretation**:
  - Engine size (displacement, horsepower, weight) **negatively correlated** with fuel efficiency.
  - Acceleration **positively correlated** with efficiency (mild, 0.30).
  - High multicollinearity among displacement, horsepower, weight.

#### 8.5 Univariate Analysis: Histogram, KDE, Box Plot per Numerical Feature
```python
for col in numeric_cols:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    sns.histplot(df[col], ax=axes[0], kde=False)
    sns.kdeplot(df[col], ax=axes[1])
    sns.boxplot(y=df[col], ax=axes[2])
```
- **Displacement**: Multimodal, right-skewed; no outliers in overall boxplot.
- **Horsepower**: Multimodal, right-skewed; **outliers present** (points above upper whisker).
- **Weight**: Right-skewed; no outliers.
- **Acceleration**: Near-normal; **outliers present** (both low and high extremes).

#### 8.6 Class-Separated Box Plots
```python
for col in numeric_cols:
    sns.boxplot(x='mpg_high', y=col, data=df)
```
- **Displacement**: Class 1 has lower median, smaller IQR; Class 0 higher median, larger spread. **Outliers appear in Class 1** when separated (not visible in combined boxplot).
- **Horsepower/Weight**: Similar pattern.
- **Acceleration**: Class 1 higher median, more symmetric; Class 0 lower median, outliers on both ends.

#### 8.7 Violin Plots (Combined & Class-Separated)
```python
sns.violinplot(data=df[numeric_cols])           # Combined
sns.violinplot(x='mpg_high', y=col, data=df)    # Class-separated
```
- **Violin plot** = Box plot + mirrored KDE (shows density shape).
- **Interpretation**:
  - Width = density (wider = more points at that value).
  - Central line = median; box = IQR; whiskers = 1.5×IQR.
  - Pointy ends = low-density regions (potential outliers).
- **Class-separated violins**: Reveal bimodal densities within each class (e.g., displacement for Class 0 has two peaks).

---

## Definitions

| Term | Definition |
|------|------------|
| **Data Preprocessing** | Steps to clean, transform, and prepare raw data for modeling (handling missing values, encoding, scaling, etc.). |
| **Missing Value Imputation** | Replacing missing (NaN) values with substituted estimates (mean, median, mode, model-based, etc.). |
| **SimpleImputer (sklearn)** | Transformer for univariate imputation; strategies: `mean`, `median`, `most_frequent`, `constant`. |
| **Class-wise Imputation** | Computing imputation statistics (mean/mode) separately per target class to preserve class-conditional distributions. |
| **Principal Component Analysis (PCA)** | Linear dimensionality reduction technique that projects data onto orthogonal axes (principal components) capturing maximum variance. |
| **StandardScaler** | Standardizes features by removing mean and scaling to unit variance: \( z = (x - \mu) / \sigma \). |
| **Explained Variance Ratio** | Proportion of total dataset variance captured by each principal component. |
| **Loadings Matrix** | Correlation coefficients between original features and principal components; shows feature contribution to each PC. |
| **Pair Plot** | Matrix of scatter plots for each pair of numerical features; diagonals show univariate distributions (hist/KDE). |
| **Correlation Heatmap** | Color-coded matrix of pairwise Pearson correlation coefficients. |
| **Box Plot** | Displays median, quartiles (Q1, Q3), whiskers (1.5×IQR), and outliers (points beyond whiskers). |
| **Violin Plot** | Hybrid of box plot and kernel density estimate; shows full distribution shape, symmetry, multimodality. |
| **Random Seed** | Integer initializing pseudorandom number generator to ensure reproducible random operations (e.g., sampling, splitting). |

---

## Examples

### Example 1: Artificial Missing Value Insertion with Fixed Seed
```python
np.random.seed(42)
df_miss = df.copy()
for col in numeric_cols:
    n = int(0.1 * len(df_miss))
    idx = np.random.choice(df_miss.index, n, replace=False)
    df_miss.loc[idx, col] = np.nan
```
- Running this twice with `seed=42` produces **identical NaN positions**.

### Example 2: Class-wise Mode Imputation
```python
df_imp = df_miss.copy()
for cls in df_miss['mpg_high'].unique():
    mask = df_miss['mpg_high'] == cls
    for col in numeric_cols:
        mode_val = df_miss.loc[mask, col].mode()[0]
        df_imp.loc[mask & df_miss[col].isna(), col] = mode_val
```
- For `mpg_high=0`, missing `horsepower` filled with mode of `horsepower` in Class 0.
- For `mpg_high=1`, filled with mode in Class 1.

### Example 3: PCA Visualization Pipeline
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

X = df[numeric_cols]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['mpg_high'], palette={0:'blue', 1:'orange'})
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
```

### Example 4: Correlation Heatmap with Target
```python
corr = df[numeric_cols + ['mpg_high']].corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
```

---

## Questions & Answers

### Q1: Why use `np.random.seed(42)` when inserting missing values?
**A**: To fix the random indices chosen for NaN insertion so that every run (by any user) produces the **exact same missing pattern**, enabling consistent comparison of imputation results. The seed value itself is arbitrary (could be 0, 100, 2024); consistency matters.

### Q2: The dataset originally has no missing values. Why artificially create them?
**A**: To **demonstrate and practice imputation techniques** in a controlled way. The exercise requires comparing strategies at different missingness levels (10%, 30%). Real-world datasets often have missing values; this simulates that scenario.

### Q3: Why does `df.info()` show `origin` and `mpg_high` as integers instead of categorical?
**A**: Pandas infers dtype from storage format. Since `origin` (1,2,3) and `mpg_high` (0,1) are stored as integers, Pandas labels them `int64`. **Always check unique value counts** to identify true categorical features.

### Q4: What is the difference between `fit`, `transform`, and `fit_transform` in sklearn imputers?
**A**:
- `fit`: Learns statistics (e.g., column means) from data (ignoring NaNs).
- `transform`: Applies learned statistics to replace NaNs.
- `fit_transform`: Does both in one step (used when imputing the same dataset used to fit).

### Q5: Why standardize data before PCA?
**A**: PCA is sensitive to feature scales. Features with larger variance (e.g., `weight` ~ 2000–5000) would dominate PCs. `StandardScaler` gives each feature zero mean and unit variance so all contribute equally.

### Q6: What do PC1 and PC2 represent?
**A**: They are **linear combinations of the original 4 numerical features** that capture the directions of maximum variance in the data. PC1 captures 80% of variance; PC2 captures 16%. They are orthogonal (uncorrelated).

### Q7: How to interpret the loadings matrix?
**A**: Each entry is the correlation between an original feature and a principal component. E.g., `displacement` has 0.53 on PC1 → strong positive contribution. `acceleration` has -0.39 on PC1 → negative contribution. Helps understand what each PC "means" in terms of original features.

### Q8: Why does class-wise mode preserve variance better than global mode?
**A**: Global mode ignores class structure. If classes have different typical values (e.g., Class 0: high horsepower; Class 1: low horsepower), global mode may fill with a value not representative of either class. Class-wise mode uses the appropriate class-specific mode, preserving within-class distributions and overall variance.

### Q9: At 30% missingness, why does mode imputation perform worst?
**A**: Mode replaces many values with a single frequent value, **reducing variance** (spread) of the feature. With 30% replaced by the same number, the feature becomes artificially concentrated → PCA captures less variance. Mean/median preserve spread better; class-wise mode adapts to class subpopulations.

### Q10: What does a violin plot show that a box plot does not?
**A**: Violin plot shows the **full density shape** (via mirrored KDE), revealing multimodality, skewness, and concentration regions. Box plot only shows quartiles, median, whiskers, and outliers. Violin = box plot + distribution shape.

### Q11: In the class-separated box plot for displacement, why do outliers appear for Class 1 but not in the combined box plot?
**A**: Combined box plot computes quartiles over **all data**, masking class-specific extremes. When separated, Class 1's own IQR is smaller, so some of its points fall outside 1.5×IQR *within that class* → flagged as outliers. This reveals class-conditional anomalies.

### Q12: Can we use mean/median imputation class-wise?
**A**: Yes, and it would likely outperform global mean/median. The notebook only implemented class-wise **mode**, but the same logic applies: compute class-specific mean/median and fill within each class. Suggested as a self-exercise.

### Q13: What is the purpose of the `diag_kind='kde'` in `pairplot`?
**A**: It plots **Kernel Density Estimates** on the diagonal (instead of histograms), giving a smooth continuous view of each feature's univariate distribution.

### Q14: Why is `acceleration` the only feature with a near-normal distribution?
**A**: Domain-specific: acceleration times (0–60 mph) for 1970s–80s cars may naturally cluster around a typical value with symmetric variation, unlike engine size or weight which are constrained by design choices and technology limits, causing skew.

### Q15: How does the correlation heatmap handle the binary target `mpg_high`?
**A**: Pearson correlation with a binary variable (0/1) is equivalent to **point-biserial correlation**. Pandas computes it directly. It measures linear association between the continuous feature and the binary class.

---

## Action Items
- [ ] **All students**: Load lab materials from LMS/chat link; upload `AutoMPG.csv` to Colab session storage.
- [ ] **If file name has `(1)` suffix**: Rename to `AutoMPG.csv` to match notebook path.
- [ ] **If upload issues persist**: Use the pre-run output PDF (contains all code outputs) for reference; fix environment offline.
- [ ] **Review** the hands-on notebook step-by-step; re-run cells to internalize code flow.
- [ ] **Self-exercise**: Implement class-wise mean and median imputation; compare PCA variance with class-wise mode.
- [ ] **Recap theory**: PCA, correlation, box/violin plot interpretation, imputation strategies.
- [ ] **Prepare for next session**: Theory on PCA, probabilistic imputation, feature engineering.

---

## Additional Notes

### Key Takeaways from the Session
1. **Always verify feature types** via unique counts, not just `df.info()`.
2. **Missing value imputation is not one-size-fits-all**; class-wise strategies can significantly outperform global ones when class separation exists.
3. **PCA is a powerful visualization tool** for high-dimensional data; explained variance and loadings guide interpretation.
4. **EDA must include both univariate and bivariate views**, and **class-separated visualizations** often reveal hidden structure (multimodality, class-conditional outliers).
5. **Correlation heatmaps** quickly expose multicollinearity and target-feature relationships.
6. **Violin plots** combine box plot summaries with density shapes, ideal for comparing distributions across classes.
7. **Reproducibility via random seeds** is essential for consistent experimentation and teaching.

### Instructor Emphasis
- **Durga Toshniwal**: Stressed correlation between theory and hands-on; urged students to recap both sequentially.
- **Ashish Kumar**: Highlighted that most code is reusable functions (e.g., `analyze_pca`, `plot_violin`); focus on understanding parameters and outputs, not syntax memorization.

### Dataset Characteristics Recap
- **Size**: 398 samples, 9 features (4 numerical, 4 categorical, 1 ID-like string).
- **Target**: Binary fuel efficiency (`mpg_high`).
- **Class balance**: Not explicitly stated, but visualizations show both classes well-represented.
- **Feature relationships**: Strong positive correlations among engine-size features; negative with efficiency; acceleration mildly positive with efficiency.

### Code Structure Best Practices Demonstrated
- Modular functions for repeated tasks (PCA visualization, violin plots).
- Consistent naming: `df_10`, `df_mean_10`, `df_class_mode_10`, etc.
- Separation of data copying, imputation, and visualization steps.
- Use of `hue` in seaborn for class-colored plots.
- Looping over columns for univariate plots to avoid repetition.

### Further Exploration Suggestions
- Try **KNN imputation** (`sklearn.impute.KNNImputer`) or **iterative imputation** (`IterativeImputer`).
- Apply **feature scaling** (MinMax, Robust) and compare PCA results.
- Perform **feature selection** based on correlation heatmap (e.g., drop one of displacement/horsepower/weight).
- Build a baseline classifier (Logistic Regression, Random Forest) on original vs. imputed data to measure downstream impact.