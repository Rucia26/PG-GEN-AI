# Data Preprocessing and Curation – Class Notes

## Summary
This session continued the data preprocessing module, covering handling missing values, noisy data treatment (binning, clustering, regression), data transformation (normalization), dimensionality reduction techniques (feature selection, feature construction, numerosity reduction, sampling, discretization, concept hierarchies), and an introduction to similarity/dissimilarity measures. The instructor also addressed student questions on binning strategies, normalization edge cases, and course logistics.

## Topics Covered
- **Recap of Previous Session**: AI evolution, cognitive science, AI/ML/DL/Generative AI/Agentic AI relationships, supervised/unsupervised learning, discriminative vs. generative models.
- **Handling Missing Values**: Removal, default value, mean/median/mode, class-conditional mean, probabilistic imputation.
- **Handling Noisy Data**:
  - Binning (equi-width, equi-frequency, smoothing by bin means, smoothing by bin boundaries)
  - Clustering for outlier removal
  - Regression for noise filtering
- **Data Transformation**:
  - Min-max normalization
  - Z-score normalization
- **Dimensionality Reduction**:
  - Feature selection (e.g., decision tree–based)
  - Feature construction/extraction
  - Numerosity reduction (parametric: model-based; non-parametric: clustering, histograms)
  - Sampling (simple random, stratified)
  - Data discretization (binarization, multi-level discretization)
  - Concept hierarchies (e.g., location: street → city → state → country)
- **Similarity and Dissimilarity**: Definitions, proximity, range [0,1], distinction from correlation.

## Definitions
- **Cognitive Science**: Interdisciplinary study of mind and its processes, spanning neuroscience, psychology, philosophy, computer science, and AI.
- **Supervised Learning**: Learning a mapping function from input attributes to a known class label using labeled training data.
- **Unsupervised Learning**: Grouping data into natural clusters using similarity measures without class labels.
- **Discriminative Model**: Learns decision boundaries to classify and predict class labels (e.g., logistic regression, SVM).
- **Generative Model**: Learns data distribution to generate new samples (e.g., GPT, BERT, GANs).
- **Agentic AI**: Multiple autonomous agents that collaborate, plan, and execute goals with minimal human intervention.
- **Binning**: Partitioning sorted data into bins (containers) to smooth noise; retains bin-level information.
- **Equi-width Binning**: Divides attribute range into N bins of equal width: width = (max − min) / N.
- **Equi-frequency (Equi-depth) Binning**: Partitions data so each bin contains approximately the same number of samples.
- **Smoothing by Bin Means**: Replaces each value in a bin with the bin’s mean.
- **Smoothing by Bin Boundaries**: Replaces each value with the nearest bin boundary (min or max of the bin).
- **Min-max Normalization**: Maps original range [min_A, max_A] to new range [new_min_A, new_max_A] via linear scaling.
- **Z-score Normalization**: Transforms value v to (v − mean) / std_dev, centering at zero with unit variance.
- **Feature Selection**: Choosing a relevant subset of original attributes (e.g., via decision tree importance).
- **Feature Construction/Extraction**: Deriving new features from original data (e.g., keyword sets from text).
- **Numerosity Reduction**: Reducing data volume via parametric (model-based) or non-parametric (clustering, histograms) methods.
- **Stratified Sampling**: Sampling that preserves class distribution by drawing from each stratum proportionally.
- **Data Discretization**: Converting continuous attributes into discrete intervals (e.g., age → bins labeled 0–9).
- **Concept Hierarchy**: Implicit multi-level representation of an attribute (e.g., location: street < city < state < country).
- **Similarity**: Measure of alikeness between points, typically in [0,1] (1 = identical).
- **Dissimilarity**: Measure of difference; 0 = identical, 1 = maximally different.
- **Proximity**: Generic term for similarity or dissimilarity (distance/closeness).

## Examples
### Missing Value Handling
- **Scenario**: 30% missing values in a dataset with classes C1, C2. Removing rows may eliminate minority class or skew distribution.
- **Imputation Methods Tested**:
  - Default constant (e.g., 100) → creates artificial class if missing % high.
  - Global mean/median/mode → may create points far from true distribution.
  - Class-conditional mean → better, respects class distribution.
  - Probabilistic imputation → best among discussed, uses distribution to sample plausible values.

### Equi-width Binning (Age Example)
- **Data**: 50 rows, age range 20–70, 5 bins.
- **Bin width** = (70−20)/5 = 10.
- **Bins**: [20,30), [30,40), [40,50), [50,60), [60,70].
- **Assignment**: Age 20–29 → bin 0, 30–39 → bin 1, etc.
- **Observation**: Bins 3 and 4 more populated; distribution roughly uniform.

### Equi-width Binning Failure (Screen Time vs. Sleep)
- **Data**: 52 points, screen time range split into 6 bins.
- **Result**: Most points concentrated in bins 4 and 5; bins 0–3 nearly empty.
- **Conclusion**: Equi-width binning fails when data is skewed; equi-frequency preferred.

### Equi-frequency Binning (Same Screen Time Data)
- **5 bins**, ~10 points each.
- **Result**: All bins equally populated; boundaries determined by sorted data percentiles.

### Smoothing by Bin Means (Price Data)
- **Data**: 12 sorted prices, 3 bins of 4 points each.
- **Operation**: Replace each bin’s values with bin mean (e.g., bin 1 mean = 9 → all become 9).
- **Effect**: Removes jitter, retains trend.

### Smoothing by Bin Boundaries
- **Bin boundaries**: 4 and 15.
- **Value 8** → closer to 4 → replaced by 4.
- **Value 9** → |9−4|=5, |15−9|=6 → closer to 4 → replaced by 4.

### Min-max Normalization (X, Y Coordinates)
- **Original**: X ∈ [1,20], Y ∈ [1,25].
- **Target**: X ∈ [5,25], Y ∈ [10,40].
- **Mapping**: Point (1,1) → (5,10); (20,25) → (25,40).
- **Formula**: v' = (v − min) / (max − min) × (new_max − new_min) + new_min.

### Z-score Normalization
- **Use case**: Comparing scores from different normal distributions (e.g., exam marks across subjects).
- **Effect**: Centers data at 0, scales by standard deviation.

### Feature Selection via Decision Tree
- **Attributes**: A1–A6.
- **Tree uses**: A1, A4, A6 → retains 3/6 attributes; A2, A3, A5 ignored as redundant.

### Histogram for Numerosity Reduction
- **Attribute**: First letter of first name.
- **Result**: ‘S’ highest frequency, followed by ‘A’, ‘M’, ‘R’.

### Stratified Sampling
- **Original**: 80% class C1, 20% class C2 (1