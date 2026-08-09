# Class Notes: Introduction to AI, Machine Learning, and Data Preprocessing

**Instructor:** Durga Toshniwal  
**Date:** Not specified (transcript timestamps present)  
**Participants:** Deepan Kanagaraj, Aditya Shrivastava, Aditya Banda, Gunjan Bhaiya, Muni Prakash Ganji, Sacheen Adavinavar, Shivansh Sharma, Hitesh Suryawanshi, Abhishek Dagar, Dwarakesh T P, Sunil Saini, Deepesh Verma, Pallavi Chakravarty, Swagat Kumar Pattnaik, Nikhil Chinta

---

## Topics Covered

1. Cognitive Science and Artificial Intelligence Foundations
2. Relationship Between AI, ML, DL, NLP, Generative AI, and Agentic AI
3. AI Tasks Overview
4. Supervised Learning: Classification and Discriminative Models
5. Unsupervised Learning: Clustering
6. Regression and Forecasting
7. Anomaly Detection
8. Deep Learning
9. Generative AI
10. Agentic AI
11. Data Preprocessing: Data Cleaning and Handling Missing Values

---

## Detailed Notes by Topic

### 1. Cognitive Science and Artificial Intelligence Foundations

**Cognitive Science**  
- Interdisciplinary field studying human intelligence: perception, understanding, knowledge acquisition.  
- Draws from computer science, philosophy, linguistics, neuroscience, and other fields.  
- Focus: how the human nervous system represents, processes, and transforms information.

**Artificial Intelligence (AI)**  
- Part of cognitive science.  
- Multiple perspectives:  
  - Making computers think.  
  - Automating activities associated with human thinking (e.g., decision-making).  
  - Creating machines that perform functions requiring human-like intelligence.  
  - Field that tries to emulate human intelligence and automation.  
- **Definition breakdown:**  
  - *Artificial*: created by explicit human effort, not naturally occurring.  
  - *Intelligence*: ability to acquire knowledge and utilize it.  
- AI deals with the part of computer science focused on designing computer systems that exhibit human-like intelligence in multiple ways.  
- Two primary aspects:  
  1. Study human intelligence.  
  2. Represent it via techniques/actions using computers.  
- AI is multidisciplinary: draws from computer science, statistics, mathematics, sociology, etc.

### 2. Relationship Between AI, ML, DL, NLP, Generative AI, and Agentic AI

**Hierarchy (superset to subset):**  
- **Artificial Intelligence (AI)** – superset.  
- **Machine Learning (ML)** – subset of AI.  
- **Deep Learning (DL)** – subset of ML.  
- **Generative AI** – subset of DL.  
- **Agentic AI** – subset of Generative AI.  
- **Natural Language Processing (NLP)** – subset of AI; intersects with ML, DL, Generative AI, Agentic AI (draws concepts from all).

**Purpose of diagram:** Clarify common confusion about terminology relationships.

### 3. AI Tasks Overview

Major AI tasks include:  
- Knowledge representation and reasoning  
- Learning (primary focus of course)  
- Natural language processing  
- Searching  
- Path planning  
- Others  

Course will discuss learning at length, involving NLP concepts. Expert systems also mentioned.

### 4. Supervised Learning: Classification and Discriminative Models

**Supervised Learning**  
- Called "supervised" because training data includes class labels (group information).  
- Two main types covered: classification and regression (though regression later discussed separately).  
- Classification based on **discriminative models**.

**Classification**  
- **Training dataset**: special data with records (tuples/instances/rows) and attributes (fields/columns).  
- One attribute is special: **classifying attribute / class label** (given in training data).  
- Task: learn a mapping from input attributes (X) to output class label (Y).  
- Model: Y = f(X), where f is the function to be learned.  
- After building model on training data, evaluate on **test data** (also has attributes and class labels).  
- If performance acceptable, use model to predict class labels of **unseen data** (class label unknown).  
- Primary goal: make predictions on future unseen data.

**Discriminative Models**  
- Learn decision boundaries to categorize different classes.  
- Goal: classify/predict class labels from input data.  
- Method: build decision boundary separating classes in data space.  
- Output: probability for each class label (e.g., P(yes), P(no)).  
- Examples: logistic regression, support vector machines (SVM), decision trees, random forest, k-nearest neighbors (k-NN), etc.  
- Each will be covered in detail later.

**Applications of Classification**  
1. **Targeted marketing**: predict which customers will buy a new product based on attributes (salary, lifestyle, demographics). Binary classification: buy / don't buy.  
2. **Fraud detection**: identify fraudulent credit card transactions. Classifier learns from historical data; flags anomalies like unusually high amounts, foreign currency transactions, unfamiliar websites.

### 5. Unsupervised Learning: Clustering

**Unsupervised Learning**  
- No class labels or group information in training data.  
- Group information discovered at end of process.  
- Also called **clustering**.

**Clustering**  
- Given: set of data points with attributes.  
- Use similarity measure (e.g., Euclidean distance) to identify natural groupings (clusters).  
- **Intra-cluster distance**: distance between points within same cluster → **minimize**.  
- **Inter-cluster distance**: distance between points across different clusters → **maximize**.  
- Points in same cluster are highly similar; points in different clusters are dissimilar.

**Difference from Classification**  
- Classification: group information (class label) given upfront.  
- Clustering: no group information; groups formed and identified at end.

**Example: Clustering on First Names**  
- Histogram of first-name initials in class: highest frequency 'S', then 'A', 'M', 'V', 'R', 'P', 'D', etc.  
- Earlier example: location clusters on map of India → predict number of participants per location or per course.

**Applications of Clustering**  
1. **Market segmentation**: divide customers into subsets (budget buyers, high-spending niche buyers) based on buying habits, geography, lifestyle.  
2. **Document clustering**: group documents by topic; new document assigned to cluster for quick topic-based retrieval. Analogous to manual folder organization.  
3. **Stock behavior clustering**: identify clusters of stocks moving similarly (up/down).

### 6. Regression and Forecasting

**Forecasting**  
- Predicting continuous values (not class labels).  
- Regression used for forecasting → regression performs forecasting task.

**Predictive Analytics**  
- Analyze current/historical data to predict future unseen data.  
- Applications: weather forecast, stock price, performance prediction, medical diagnosis.

### 7. Anomaly Detection

**Definition**  
- Identify behavior very different from normal (anomalies/outliers).  
- Critical in applications where abnormal behavior is more important than normal (e.g., medical).

**Methods**  
- **Unsupervised**: cluster data; anomalies lie far from normal clusters.  
- **Supervised**: use labeled anomaly/normal data to train classifier.

### 8. Deep Learning

**Definition**  
- Subset of ML using neural networks organized in layers (depth).  
- Useful for complex data where patterns/relationships hard to identify manually (text, images, audio).

**Neural Networks**  
- Composed of neurons (nodes) inspired by human brain.  
- Key difference from traditional ML: **automatic feature extraction** (not handcrafted).

**Applications**  
- Natural language processing, computer vision, speech recognition, healthcare (diagnosis, drug discovery), finance (algorithmic trading, fraud, credit risk), autonomous vehicles (360° cameras, driver drowsiness detection, automatic braking), voice assistants (Alexa), transcription, AI summarization.

### 9. Generative AI

**Definition**  
- Category of AI models/algorithms that **generate new content** (text, images, code, audio) similar to real-world examples.  
- Understands context, style, structure of input data to produce human-like output.

**Contrast with Traditional AI**  
- Traditional AI: discriminative (classify/predict).  
- Generative AI: generative (create new outputs).  
- Learns patterns from vast data; uses patterns to generate novel outputs.

**Models**  
- BERT, GPT, T5, GANs, LLMs, etc. (to be covered in course).

### 10. Agentic AI

**Definition**  
- Evolution beyond generative AI: autonomous (partially/fully) agents that not only generate but **make decisions, learn from feedback, adapt, plan, set goals, act in dynamic environments** with minimal human intervention.

**Mechanism**  
- Agents take actions → receive rewards/penalties (performance improvement/degradation) → take corrective actions, plan tasks, set goals, adapt.

**Example: Gaming**  
- Avatars/players take actions; environment changes dynamically; rewards/penalties guide learning.

**Applications**  
- Autonomous robotics (hazardous environments, manufacturing, cleaning).  
- Software development.  
- Personal AI assistants (calendar, email, reports, bookings, preference adaptation).  
- Scientific discovery (run experiments, interpret data, refine experiments, make inferences).

### 11. Data Preprocessing: Data Cleaning and Handling Missing Values

**Why Preprocessing?**  
- Data not useful until in suitable condition.  
- Quality results require quality data.

**Key Preprocessing Tasks**  
1. Data cleaning (missing values, noise smoothing, inconsistency resolution).  
2. Data integration (multiple sources → single warehouse/structure).  
3. Data transformation (operations to bring data to desired quality).  
4. Data reduction (handle huge data volumes under compute constraints).  
5. Data discretization (binning continuous data).

**Data Cleaning: Handling Missing Values**  
- Example dataset: 10 records, 6 class C2, 3 class C1, 1 unknown class; 3 cells missing attribute values (~5% missing).  
- **Approaches discussed:**  
  - **Mean/average of column** (for numerical attributes).  
  - **Median** (robust to outliers).  
  - **Mode/most frequent value** (for categorical or class label).  
  - **Rule-based inference** (e.g., derive class from other attribute thresholds).  
  - **Clustering-based imputation** (group similar records, impute from cluster).  
  - **Replace with constant** (e.g., 0, -1) – mentioned but not recommended generally.  
- Instructor noted: many ways exist; details covered in later lectures.

---

## Definitions

| Term | Definition |
|------|------------|
| Cognitive Science | Interdisciplinary study of human intelligence: perception, understanding, knowledge acquisition; draws from computer science, philosophy, linguistics, neuroscience. |
| Artificial Intelligence (AI) | Field of computer science designing systems exhibiting human-like intelligence; created by human effort (artificial) + ability to acquire/use knowledge (intelligence). |
| Machine Learning (ML) | Subset of AI focused on systems that learn from experience/data. |
| Deep Learning (DL) | Subset of ML using multi-layer neural networks for automatic feature extraction on complex data. |
| Generative AI | Subset of DL; models that generate new content (text, image, code, audio) resembling real-world data. |
| Agentic AI | Subset of Generative AI; autonomous agents that plan, decide, act, learn from feedback, adapt in dynamic environments. |
| Natural Language Processing (NLP) | Subset of AI handling human spoken/written language; intersects with ML, DL, Generative AI, Agentic AI. |
| Supervised Learning | Learning with labeled training data (class labels given); includes classification and regression. |
| Classification | Supervised task: learn mapping from input attributes to discrete class label; uses discriminative models. |
| Discriminative Model | Model that learns decision boundaries to distinguish classes; outputs class probabilities. |
| Unsupervised Learning | Learning without class labels; discovers group structure (clustering). |
| Clustering | Grouping data points by similarity; minimize intra-cluster distance, maximize inter-cluster distance. |
| Regression | Predicting continuous values; used for forecasting. |
| Forecasting | Value prediction for future unseen data using historical/current data. |
| Anomaly Detection | Identifying rare/abnormal behavior (outliers) deviating from normal patterns. |
| Data Preprocessing | Transforming raw data into clean, usable form for analysis/modeling. |
| Data Cleaning | Handling missing values, smoothing noise, resolving inconsistencies. |
| Data Integration | Combining data from multiple sources into unified structure. |
| Data Transformation | Applying operations to bring data to desired quality/format. |
| Data Reduction | Reducing data volume while preserving essential information. |
| Data Discretization | Converting continuous attributes into discrete bins/intervals. |

---

## Examples

1. **Classification – Targeted Marketing**  
   - Input: customer attributes (salary, lifestyle, demographics).  
   - Output: binary class label (will buy / will not buy).  
   - Goal: predict purchase likelihood for new customers.

2. **Classification – Fraud Detection**  
   - Input: transaction attributes (amount, currency, merchant, location).  
   - Output: fraud / not fraud.  
   - Classifier learns patterns: high amount, foreign currency, new website → flag.

3. **Clustering – First Name Initials**  
   - Histogram of class participants' first-name initials: S (highest), A, M, V, R, P, D.

4. **Clustering – Location-Based**  
   - Map of India with participant locations → predict participant count per location or per course.

5. **Clustering – Market Segmentation**  
   - Groups: budget buyers, high-spending niche buyers; based on spending habits, geography, lifestyle.

6. **Clustering – Document Grouping**  
   - Thousands of documents → clusters by topic; new document assigned to cluster for quick retrieval.

7. **Clustering – Stock Behavior**  
   - Clusters of stocks with similar movement patterns (up, down, volatile).

8. **Deep Learning – Autonomous Vehicles**  
   - 360° cameras, driver drowsiness detection, automatic braking via computer vision.

9. **Generative AI – Content Creation**  
   - Generate text, images, code, audio mimicking human style.

10. **Agentic AI – Travel Planning**  
    - Task: travel A→B minimizing time and cost.  
    - Agent evaluates transport modes, permutations, constraints → outputs optimal itinerary.

11. **Missing Value Handling**  
    - Dataset with 10 records, 3 missing cells.  
    - Proposed imputations: column mean, median, mode, rule-based class inference, clustering-based.

---

## Questions & Answers

### Q1: Gunjan Bhaiya – Classification Types
**Q:** Is classification only binary? Can it be multi-class?  
**A:** Yes, two main types:  
- **Binary classification**: two class labels (yes/no, buy/don't buy).  
- **Multi-class classification**: more than two classes (e.g., 5 fraud types).  
- **One-vs-Rest**: multi-class problem but interest in one class; that class vs. all others combined.

### Q2: Sacheen Adavinavar – Intra-cluster vs Inter-cluster Distance
**Q:** Explain intra-cluster and inter-cluster distance.  
**A:**  
- **Intra-cluster distance**: distance between points *within* same cluster → minimize.  
- **Inter-cluster distance**: distance between points *across* different clusters → maximize.  
- Process: compute pairwise distances between all points (n×n matrix), group close points, separate far points. Similarity measures (Euclidean, etc.) used.

### Q3: Shivansh Sharma – Clustering vs Classification Training
**Q:** In clustering, no labels – what does the model learn?  
**A:**  
- Clustering: only attribute values given; no group info. Model finds similar points via attributes and groups them.  
- Classification: training data includes class label Y; learn function f: X → Y.  
- **Combined use**: if no labels, first cluster → study clusters → assign labels → then train classifier.

### Q4: Hitesh Suryawanshi – Why Maximize Inter-cluster Distance?
**Q:** Why maximize distance between clusters?  
**A:**  
- Clustering aims to find *natural* groupings.  
- Similar points close → intra-cluster distance small.  
- Dissimilar points far → inter-cluster distance large.  
- Example: points (1,1), (2,2), (10,10) → first two similar (close), third far → natural clusters.

### Q5: Abhishek Dagar – Distance Cutoff for Clustering
**Q:** How to set distance cutoff when distances between clusters are similar?  
**A:** Covered in later lectures; many algorithms/techniques exist for determining clusters and thresholds.

### Q6: Dwarakesh T P – Data Not Forming Clear Clusters
**Q:** What if raw data doesn't form clear clusters?  
**A:** High-dimensional data often sparse; apply transformations (dimensionality reduction, feature engineering) before clustering. Covered later.

### Q7: Sunil Saini – Agent vs LLM
**Q:** Fundamental difference between AI agent and LLM? Can't we just prompt LLM to act as agent?  
**A:** LLMs *can* be used as agents in Agentic AI. They are not limited to NLP; can serve as decision-making agents. Agentic AI can also use non-LLM models (e.g., vision models).

### Q8: Sunil Saini – Agentic AI Without LLM
**Q:** Can we have Agentic AI without LLMs?  
**A:** Yes, Agentic AI can use other generative models (e.g., vision-based).

### Q9: Gunjan Bhaiya – AI Agent vs Agentic AI
**Q:** Difference between traditional AI agent and Agentic AI? Both need triggers.  
**A:**  
- **Traditional AI agent**: executes specific task on demand/schedule (e.g., book train ticket). No autonomy, no planning.  
- **Agentic AI**: autonomous; given high-level goal (travel A→B min time/cost), it plans, evaluates options, optimizes constraints, adapts dynamically. Works without step-by-step human instruction.

### Q10: Deepesh Verma – Rule-based vs LLM Agent
**Q:** Simple agent = rule-based; LLM agent = non-rule-based, flexible?  
**A:** Correct.

### Q11: Pallavi Chakravarty – Agentic AI Without LLM Example
**Q:** How does Agentic AI work without LLM?  
**A:** Can use vision models or other generative models. Agentic AI builds on generative AI (not necessarily LLM). Prompts/instructions can be dynamic.

### Q12: Sacheen Adavinavar – Agentic AI Models Beyond LLM/VLM
**Q:** What other models can Agentic AI use?  
**A:** Course will cover in depth; too early for full answer. Patience requested.

### Q13: Deepesh Verma – Data Preprocessing = Unstructured to Structured?
**Q:** Preprocessing means converting unstructured to structured?  
**A:** Not necessarily. Preprocessing brings data "not in good shape" to usable condition; may be structured but messy (missing values, noise, etc.).

### Q14: Gunjan Bhaiya – Missing Value Imputation Ideas
**Q:** Suggestions for handling missing values in example?  
**A:**  
- Column mean/average for numerical attributes.  
- Rule-based class prediction from other attributes.  
- Clustering-based imputation.

### Q15: Swagat Kumar Pattnaik – Replace with 0/-1?
**Q:** Can we replace missing with 0 or -1?  
**A:** (Not directly answered; instructor collected multiple suggestions.)

### Q16: Nikhil Chinta – Median and Mode
**Q:** Use median for attributes (range 1-6), mode for class label.  
**A:** (Acknowledged as valid approach.)

### Q17: Hitesh Suryawanshi – Most Frequent Value
**Q:** Replace with most frequent value (mode).  
**A:** (Acknowledged.)

---

## Action Items

1. **Instructor** will share slides via email after class.  
2. **Participants** to review introductory concepts; upcoming sessions will cover each topic in detail.  
3. **Deep learning, generative AI, agentic AI** will have multiple dedicated sessions.  
4. **Data preprocessing** (cleaning, integration, transformation, reduction, discretization) started today; continue in next class.  
5. **Clustering algorithms, distance measures, cutoff determination** to be covered in future lectures.  
6. **Classification algorithms** (logistic regression, SVM, decision trees, random forest, k-NN) to be covered in detail.  
7. **Regression and forecasting** techniques to be discussed.  
8. **Anomaly detection methods** (supervised/unsupervised) to be elaborated.

---

## Additional Notes

- **Terminology**: Records = tuples = instances = rows. Attributes = fields = columns. Class label = classifying attribute.  
- **Discriminative models** also called classifiers.  
- **Clustering** often used as preprocessing for classification when labels absent.  
- **Generative AI** is discriminative? No – generative (creates), traditional AI discriminative (classifies).  
- **Agentic AI** requires generative AI as foundation; can use LLMs, VLMs, or other generative models.  
- **Data preprocessing** is critical for quality results; "garbage in, garbage out."  
- **Missing value handling** has no single best method; depends on data type, distribution, domain knowledge.  
- **Course structure**: introductory session → detailed sessions per topic → hands-on/applications.  
- **Participant engagement**: questions encouraged throughout; instructor responsive to clarifications.  
- **Next session**: continue data preprocessing (integration, transformation, reduction, discretization).