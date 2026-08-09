# Data Preprocessing and Curation – Class Notes  
**Instructor:** Durga Toshniwal  
**Session Date:** Sunday (exact date not specified)  
**Course:** Generative AI Batch 2  

---

## Topics Covered
1. Recap of Previous Session (AI Foundations, ML Taxonomy, Generative & Agentic AI)  
2. Data Cleaning  
   - Handling Missing Values  
   - Handling Noisy Data (Binning, Clustering, Regression)  
3. Data Transformation  
   - Normalization (Min‑Max, Z‑Score)  
4. Dimensionality Reduction  
   - Feature Selection  
   - Feature Construction / Extraction  
   - Numerosity Reduction (Parametric & Non‑Parametric)  
   - Histograms  
   - Sampling (Simple Random, Stratified, Clustering‑Based)  
5. Data Discretization & Concept Hierarchy  
6. Introduction to Similarity & Dissimilarity Measures  
7. Administrative Updates (Slides, Hands‑On Sessions, Batch Manager Concerns)  

---

## Detailed Notes by Topic  

### 1. Recap of Previous Session  
- **Cognitive Science**: Interdisciplinary field (neuroscience, psychology, philosophy, computer science, AI) studying mind and its processes.  
- **Artificial Intelligence (AI)**: Emulating human‑like intelligence via computer systems; uses artificial neurons inspired by biological neurons.  
- **AI Taxonomy**:  
  - AI (superset) → Machine Learning (subset) → Deep Learning (subset) → Generative AI & Agentic AI.  
  - NLP also falls under AI.  
- **Learning Task Focus**: Machines learn patterns from historical data to generate new facts, distinguish situations/environments.  
- **Major AI Techniques**: Supervised learning, unsupervised learning, regression, anomaly detection, deep learning, generative AI, agentic AI.  
- **Supervised Learning**: Training data includes attribute values **and** class labels; goal is to learn a mapping function *f(attributes) → class label*.  
- **Discriminative Models** (e.g., logistic regression, SVM): Classify and predict class labels by finding decision boundaries.  
- **Unsupervised Learning**: No class labels; group data via similarity measures (minimize intra‑cluster distance, maximize inter‑cluster distance).  
- **Regression & Forecasting**: Predict continuous values; part of predictive analytics.  
- **Anomaly Detection**: Identify abnormal behavior/outliers using supervised or unsupervised methods.  
- **Deep Learning**: Neural networks (CNN, RNN, LSTM) for complex data (text, images, audio). Applications: computer vision, NLP, speech, healthcare, finance.  
- **Generative AI**: Creates new content (text, audio, video, images, code) by learning patterns from massive data; understands context, style, structure. Contrasts with discriminative models (which only classify). Examples: GPT, BERT, GANs.  
- **Agentic AI**: Multiple autonomous agents collaborate, plan, coordinate, learn goals, take feedback, improve; act as collaborative assistants with minimal human intervention.  

---

### 2. Data Cleaning  

#### 2.1 Handling Missing Values  
- **Impact of Missing‑Value Percentage**:  
  - Low percentage (e.g., <5‑10%): Removing records (listwise deletion) may be acceptable.  
  - High percentage (e.g., ~30%): Removing rows can distort class distribution – majority class may become uniform, minority class may be lost or become majority.  
- **Imputation Strategies Discussed**:  
  1. **Ignore/Remove Tuple** – Simple but loses information; risky when missingness is high.  
  2. **Fill with Global Constant** (e.g., 0, 999, “Unknown”) – Can create a new artificial class if missing percentage is high.  
  3. **Fill with Attribute Mean** – May produce values far from actual distribution; creates artificial points.  
  4. **Fill with Attribute Median** – Better than mean for skewed data, but still creates non‑representative points.  
  5. **Fill with Attribute Mode** – Works for categorical data; sometimes good, sometimes not.  
  6. **Class‑Conditional Mean/Median/Mode** – Compute statistic per class label; **much better** because respects class‑specific distribution.  
  7. **Probabilistic Methods** (e.g., regression imputation, EM algorithm, multiple imputation) – **Best solution discussed**; uses relationships among attributes to estimate missing values.  
- **Time‑Series Dependency**: If data has temporal ordering (e.g., sales over time), forward‑fill (previous value) or backward‑fill (next value) is logical. **Not valid** for independent cross‑sectional data (e.g., sales across countries).  

#### 2.2 Handling Noisy Data  
- **Definition**: Noise = random errors, outliers, jitter in measurements.  
- **Three Main Approaches**:  
  1. **Binning (Data Smoothing)**  
     - Sort values, partition into bins (buckets), replace values in each bin by a representative (bin mean, median, boundary).  
     - **Equi‑Width Binning**:  
       - Bin width = (max − min) / *N* (number of bins).  
       - Example: Age 20–70, 5 bins → width = 10 → bins: [20‑30), [30‑40), … [60‑70].  
       - **Issue**: If data is skewed, some bins empty, others overcrowded (e.g., screen‑time example: most points in bins 4‑5, bins 0‑3 nearly empty).  
     - **Equi‑Depth (Equi‑Frequency) Binning**:  
       - Sort data, put ≈ equal number of points per bin.  
       - Bin boundaries determined by frequency, not value range.  
       - Example: 52 points, 5 bins → ~10 points/bin. All bins uniformly populated.  
       - **Preferred when distribution is skewed**.  
     - **Smoothing by Bin Means**: Replace each value in a bin by the bin’s mean.  
     - **Smoothing by Bin Boundaries**: Replace each value by the closer bin boundary (min or max of that bin).  
  2. **Clustering**  
     - Group similar points; points not belonging to any cluster (or far from cluster centroids) treated as noise/outliers and removed.  
     - **Difference from Binning**: Clustering uses point‑to‑point similarity; binning uses only attribute value ranges (static partitions).  
  3. **Regression**  
     - Fit a curve (linear, logarithmic, exponential, etc.) to data.  
     - Points with large residuals (far from fitted curve) considered noise and discarded.  
     - Goal: Minimize sum of distances (errors) between points and model.  

---

### 3. Data Transformation  

#### 3.1 Normalization  
- **Purpose**: Map attributes to a common scale so no single attribute dominates due to larger numeric range.  
- **Min‑Max Normalization**:  
  - Formula:  
    \[
    v' = \frac{v - \min_A}{\max_A - \min_A} \times (\text{new\_max}_A - \text{new\_min}_A) + \text{new\_min}_A
    \]  
  - Maps original range \([\min_A, \max_A]\) to new range \([\text{new\_min}_A, \text{new\_max}_A]\).  
  - Example: X ∈ [1,20] → new range [5,25]; Y ∈ [1,25] → new range [10,40].  
  - **New range chosen by user** based on analysis needs (e.g., all attributes to [0,1] or [0,100]).  
- **Z‑Score Normalization (Standardization)**:  
  - Formula:  
    \[
    v' = \frac{v - \mu_A}{\sigma_A}
    \]  
  - Centers data at 0, scales by standard deviation.  
  - Useful for comparing scores from different normal distributions (e.g., exam marks across subjects).  
  - Resulting values typically in range ≈ [-3, 3] for Gaussian data.  

---

### 4. Dimensionality Reduction  

#### 4.1 Feature Selection  
- **Definition**: Choose a subset of original attributes that are most relevant; discard redundant/irrelevant ones.  
- **Example**: Decision‑tree‑based selection – attributes used in the tree are retained (e.g., A1, A4, A6 out of A1–A6).  
- **Goal**: Reduce dimensionality while preserving predictive power.  

#### 4.2 Feature Construction / Extraction  
- **Definition**: Derive **new** features from original data (not just selecting existing ones).  
- **Examples**:  
  - Text → keyword vectors, TF‑IDF, embeddings.  
  - Mapping points to a new coordinate space (e.g., PCA, though not explicitly named).  
- **Result**: New feature set may have different dimensionality (often lower).  

#### 4.3 Numerosity Reduction (Volume Reduction)  
- **Parametric Methods**:  
  - Fit a model (e.g., linear regression \(Y = M X + C\)) to entire dataset.  
  - Estimate model parameters (slope \(M\), intercept \(C\)).  
  - **Retain only the model**, discard raw points.  
  - Assumes model is a good representation.  
- **Non‑Parametric Methods**:  
  - No model parameters estimated.  
  - **Clustering**: Represent data by cluster centroids + membership.  
  - **Histograms**: Bin data (equi‑width or equi‑depth), store bin counts & boundaries.  
  - Example: Histogram of first‑name initials (S highest, then A, M, etc.).  

#### 4.4 Sampling  
- **Simple Random Sampling (SRS)**: Randomly pick *k* points from *N*.  
  - **Problem**: Skewed distributions → rare classes may be omitted, altering distribution.  
- **Stratified Sampling**:  
  - Partition data by class (strata).  
  - Sample from each stratum **proportionally** to its size in original data.  
  - Example: 80% Class C1, 20% Class C2 → for 50% reduction, draw 40 from C1, 10 from C2.  
  - Preserves class distribution.  
- **Clustering‑Based Sampling**:  
  - Cluster data first, then sample proportionally from each cluster (similar to stratified but clusters may not align with class labels).  

---

### 5. Data Discretization & Concept Hierarchy  

#### 5.1 Data Discretization  
- **Definition**: Convert continuous attribute into finite set of intervals (bins), each labeled with a discrete value (e.g., 0, 1, 2 or “low”, “medium”, “high”).  
- **Binarization**: Special case – two bins (e.g., 0‑5 → 0, >5‑10 → 1).  
- **General Discretization**: *k* bins (e.g., age 1‑100 → 10 bins of width 10).  
- **Purpose**: Simplify handling, enable algorithms that require categorical input.  

#### 5.2 Concept Hierarchy  
- **Definition**: Implicit multi‑level hierarchy in an attribute (e.g., Location: Country → State → City → Street).  
- **Usage**: Choose granularity per analysis need:  
  - Fine‑grained → Street level (many distinct values).  
  - Coarse → Country level (few distinct values).  
- **Benefit**: Allows roll‑up / drill‑down in OLAP‑style analysis.  

---

### 6. Introduction to Similarity & Dissimilarity  
- **Similarity**: Degree of alikeness between two data objects.  
  - Range typically [0, 1]; 0 = no similarity, 1 = identical.  
- **Dissimilarity (Distance)**: Degree of difference.  
  - Range typically [0, ∞) or normalized [0, 1]; 0 = identical, 1 = maximally different.  
- **Proximity**: Generic term covering both similarity and dissimilarity.  
- **Next Session**: Detailed measures (Euclidean, Manhattan, Cosine, Jaccard, etc.) and computation.  

---

### 7. Administrative Updates & Action Items  

#### Slides & Materials  
- Yesterday’s slides uploaded to LMS this morning.  
- Today’s slides will be shared with Futurance team this afternoon; expected on LMS soon (possibly tomorrow if batch manager is away).  

#### Hands‑On Sessions  
- **Every theory session followed by hands‑on** (Google Colab).  
- Colab link to be posted on LMS; students can also use Colab independently.  
- Hands‑on likely starting next week or week after; schedule to be announced on LMS.  

#### Batch Manager Concerns  
- Student Aditya Banda raised unresolved emails/DMs to batch manager (Simran, currently away).  
- Instructor will follow up with Simran; students advised to email Simran directly.  
- Attendance/LMS issues also directed to Simran.  

#### Study Recommendations  
- Review material after each class; concepts build sequentially.  
- Use Futurance study material, web resources for unfamiliar topics.  
- Detailed week‑by‑week syllabus (with dates) to be released by next week.  

#### Upcoming Exercise Questions  
- Instructor plans to share ungraded multiple‑choice practice questions after future classes.  

---

## Definitions  

| Term | Definition |
|------|------------|
| **Cognitive Science** | Interdisciplinary study of mind and its processes (neuroscience, psychology, philosophy, CS, AI). |
| **Artificial Intelligence (AI)** | Computer systems emulating human‑like intelligence via artificial neurons. |
| **Supervised Learning** | Learning a mapping *f(attributes) → class label* from labeled training data. |
| **Discriminative Model** | Model that learns decision boundaries to classify/predict class labels (e.g., logistic regression, SVM). |
| **Unsupervised Learning** | Grouping data without class labels using similarity measures. |
| **Regression** | Predicting continuous numeric values; used for forecasting. |
| **Anomaly Detection** | Identifying abnormal/outlier points via supervised or unsupervised methods. |
| **Deep Learning** | Neural networks (CNN, RNN, LSTM) for complex patterns in text, images, audio. |
| **Generative AI** | Models that create new content (text, audio, video, images, code) by learning data patterns. |
| **Agentic AI** | Multiple autonomous agents that collaborate, plan, learn goals, and act as assistants. |
| **Missing Value Imputation** | Replacing missing entries with estimated values (constant, mean, median, mode, class‑conditional, probabilistic). |
| **Binning (Data Smoothing)** | Partitioning sorted continuous values into bins; replacing values by bin representative (mean, median, boundary). |
| **Equi‑Width Binning** | Bins have equal value range: width = (max − min) / *N*. |
| **Equi‑Depth (Equi‑Frequency) Binning** | Bins contain ≈ equal number of points; boundaries set by frequency. |
| **Clustering (for Noise Removal)** | Grouping similar points; points not in any cluster treated as noise. |
| **Regression (for Noise Removal)** | Fitting a curve; points with large residuals removed as noise. |
| **Min‑Max Normalization** | Linear mapping of attribute range \([\min, \max]\) to new range \([\text{new\_min}, \text{new\_max}]\). |
| **Z‑Score Normalization** | Standardization: \( (v - \mu) / \sigma \); centers at 0, scales by standard deviation. |
| **Feature Selection** | Choosing a subset of original attributes for analysis. |
| **Feature Construction / Extraction** | Deriving new features from original data (e.g., text → embeddings). |
| **Numerosity Reduction** | Reducing data volume via parametric (model‑based) or non‑parametric (clustering, histograms) methods. |
| **Parametric Reduction** | Fit a model, estimate parameters, retain only model. |
| **Non‑Parametric Reduction** | No model parameters; use clustering or histograms. |
| **Simple Random Sampling (SRS)** | Randomly select *k* points from *N* without regard to distribution. |
| **Stratified Sampling** | Sample proportionally from each class/stratum to preserve distribution. |
| **Data Discretization** | Converting continuous attribute into discrete intervals with symbolic labels. |
| **Binarization** | Discretization into exactly two bins (0/1). |
| **Concept Hierarchy** | Multi‑level granularity in an attribute (e.g., Country → State → City → Street). |
| **Similarity** | Measure of alikeness, typically in [0,1]. |
| **Dissimilarity** | Measure of difference, typically in [0,∞) or normalized [0,1]. |
| **Proximity** | Generic term for similarity or dissimilarity. |

---

## Examples  

### Equi‑Width Binning (Age)  
- Data: 50 rows, age min=20, max=70, 5 bins.  
- Bin width = (70‑20)/5 = 10.  
- Bins: Bin0 [20‑30), Bin1 [30‑40), Bin2 [40‑50), Bin3 [50‑60), Bin4 [60‑70].  
- Age 20 → Bin0; Age 35 → Bin1; etc.  

### Equi‑Depth Binning (Screen Time vs. Sleep Duration)  
- 52 points, 5 bins → ~10 points/bin.  
- Sort by sleep duration, assign first 10 to Bin0, next 10 to Bin1, …  
- All bins uniformly populated; avoids empty bins seen in equi‑width.  

### Min‑Max Normalization (X, Y Coordinates)  
- Original X ∈ [1,20], Y ∈ [1,25].  
- New X ∈ [5,25], New Y ∈ [10,40].  
- Point (1,1) → (5,10); Point (20,25) → (25,40).  
- Formula applied per coordinate.  

### Z‑Score Normalization  
- Same data centered at mean, scaled by std. dev.  
- Resulting spread roughly [-3,3] for Gaussian‑like data.  

### Stratified Sampling  
- 100 samples: 80 Class C1, 20 Class C2.  
- Reduce to 50 samples → draw 40 from C1, 10 from C2 (preserves 80/20 ratio).  

### Concept Hierarchy (Location)  
- Levels: Country → State → City → Street.  
- Analysis at City level: fewer distinct values than Street level.  

---

## Questions & Answers  

### Missing Values & Imputation  
**Q (Dilip S Biradar):** When to use previous/next record value (forward/backward fill)?  
**A:** Only when attribute has **temporal dependency** (e.g., sales over time). Not valid for independent cross‑sectional data (e.g., sales across countries).  

**Q (Laxmi Sahu):** What if no time dependency?  
**A:** Use other methods: remove record, global constant, mean/median/mode, class‑conditional statistics, probabilistic imputation.  

### Binning  
**Q (Gunjan Bhaiya):** Equi‑width binning forms equivalent clusters?  
**A:** Bins are static partitions based on value range, not point‑to‑point similarity. They categorize but are not true clusters.  

**Q (Aditya Shrivastava):** Distance between points in same bin?  
**A:** Binning does not consider inter‑point distances; only value ranges.  

**Q (Neeraj Kumar):** Is binning categorizing data?  
**A:** Yes, binning converts continuous range into finite categories (bins).  

**Q (Deepak Katara):** Relationship between binning and clustering?  
**A:** Binning creates static groups by value range; clustering groups by similarity. Bins are not clusters.  

**Q (Sunil Saini):** What is the noise being removed in binning?  
**A:** Jitter – fine‑grained variations. Binning replaces individual values with bin IDs, smoothing out noise.  

**Q (Jithu Tagore):** When data distribution large, binning analyzes only that range?  
**A:** Binning retains bin information for further analysis; individual values discarded.  

**Q (Pallavi Chakravarty):** Equi‑width binning on skewed screen‑time data – bins 4‑5 crowded, others empty.  
**A:** Correct – equi‑width fails when distribution skewed; equi‑depth preferred.  

**Q (Deepak Katara):** Should we ignore binning step for such data?  
**A:** No – we still need binning, but must use equi‑depth or other method.  

**Q (Shivansh Sharma):** Min‑max normalization yields fractional values – acceptable?  
**A:** Yes, fractions are fine unless algorithm requires integers. New range chosen by user.  

**Q (Shivansh Sharma):** Equi‑frequency binning with many repeated values at boundaries?  
**A:** Repeated values count toward frequency; they stay in same bin. Bin boundaries adjust to keep equal counts.  

**Q (Shivansh Sharma):** Can binning handle negative values?  
**A:** Yes – range calculation works with negatives (max − min handles sign).  

### Normalization  
**Q (Aditya Banda):** What is *v* in min‑max formula?  
**A:** *v* is the original attribute value being normalized.  

**Q (Chandrasekhar Sahu):** Example for normalization?  
**A:** Shown X/Y coordinate mapping and Z‑score centering.  

**Q (Rashmi Tribhuwan):** How to decide new min/max?  
**A:** User‑defined based on analysis needs (e.g., bring all attributes to [0,1] or [0,100]).  

### Dimensionality Reduction  
**Q (Shriram P):** What is parametric reduction?  
**A:** Fit a model (e.g., linear regression), estimate parameters (slope, intercept), retain only model.  

**Q (Ankit Sood):** Why reduce data if more data improves ML?  
**A:** Reduces compute time, enables EDA, keeps distribution intact (via stratified sampling). Accuracy preserved if distribution unchanged.  

**Q (Ankit Sood):** Prioritizing time‑to‑result over accuracy?  
**A:** Time and EDA are key reasons; distribution must stay same – hence stratified sampling.  

### Similarity vs. Correlation  
**Q (Laxmi Sahu):** Are similarity/dissimilarity same as correlation in heatmaps?  
**A:** No. Similarity = closeness/alikeness (e.g., two fabrics). Correlation = co‑occurrence/relationship (e.g., shirt & trouser).  

### Administrative  
**Q (Aditya Banda):** Batch manager unresponsive to emails/DMs.  
**A:** Instructor will contact Simran (batch manager); students also advised to email Simran directly.  

**Q (Rashmi Tribhuwan):** Attendance on LMS?  
**A:** Direct to Simran next session.  

**Q (Jithu Tagore):** Brief explanation of feature selection?  
**A:** Select relevant subset of attributes (e.g., via decision tree); discard redundant ones.  

---

## Action Items  

| Item | Responsible | Deadline / Status |
|------|-------------|-------------------|
| Upload yesterday’s slides to LMS | Futurance team (Simran) | Uploaded this morning |
| Upload today’s slides to LMS | Futurance team (Simran) | Shared with team this afternoon; expected on LMS today/tomorrow |
| Announce hands‑on schedule (Colab links) | Instructor / Futurance | Next week or week after; via LMS |
| Provide detailed week‑by‑week syllabus | Instructor / Futurance | By next week |
| Follow up on batch manager (Simran) responsiveness | Instructor | Immediate (text sent during class) |
| Students to review class material & prerequisites | All students | Ongoing |
| Students to attempt ungraded practice questions (when posted) | All students | After future classes |
| Resolve attendance/LMS issues | Simran (batch manager) | Next session |

---

## Additional Notes  

- **Sequential Dependency**: Concepts build cumulatively – data preprocessing used throughout course.  
- **Environment**: Google Colab standard for hands‑on; ensure access.  
- **Study Resources**: Futurance material, web searches for unfamiliar topics.  
- **Next Session**: Detailed similarity/dissimilarity measures (distance metrics, computation).  
- **Instructor’s Pedagogical Approach**: Theory → hands‑on → practice questions; emphasizes understanding over memorization.  
- **Class Dynamics**: Interactive – multiple student questions clarified binning, normalization, sampling, dimensionality reduction.  
- **Technical Issues**: Instructor’s digital pen intermittent; verbal explanations compensated.  

---  

*End of Notes*