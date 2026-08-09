# Introduction to AI, Machine Learning, and Data Preprocessing

## Summary
This session covers foundational concepts in artificial intelligence, including cognitive science, AI/ML/DL relationships, supervised and unsupervised learning techniques, generative and agentic AI, and an introduction to data preprocessing with focus on handling missing values.

## Topics Covered
- Cognitive science and AI definitions
- Hierarchy: AI → Machine Learning → Deep Learning → Generative AI → Agentic AI
- NLP as intersecting field
- Supervised learning: Classification (discriminative models)
- Unsupervised learning: Clustering
- Regression and forecasting
- Anomaly detection
- Deep learning and neural networks
- Generative AI
- Agentic AI
- Data preprocessing: cleaning, integration, transformation, reduction, discretization
- Handling missing values in data cleaning

---

## Definitions

### Cognitive Science
- Interdisciplinary study of human intelligence, perception, understanding, and knowledge acquisition
- Draws from computer science, philosophy, linguistics, neuroscience
- Focuses on how the human nervous system represents, processes, and transforms information

### Artificial Intelligence (AI)
- **Artificial**: Created by explicit human effort, not naturally occurring
- **Intelligence**: Ability to acquire knowledge and utilize it
- Field of computer science focused on designing systems that exhibit human-like intelligence
- Multidisciplinary: draws from computer science, statistics, mathematics, sociology

### Machine Learning (ML)
- Subset of AI focused on systems that understand situations and change actions based on experience
- Involves generating new facts from old ones, new concepts, or distinguishing environments
- Making machines intelligent through learning

### Deep Learning
- Subset of ML using neural networks organized in layers
- Useful for complex data where patterns are difficult to identify manually
- Automatic feature extraction (vs. manual in traditional ML)
- Applications: NLP, computer vision, speech recognition, healthcare, finance

### Generative AI
- Category of AI models that generate new content (text, images, code, audio)
- Learns patterns from vast data to produce human-like outputs
- Generative (creates) rather than discriminative (classifies/predicts)
- Models: BERT, GPT, T5, GANs, LLMs

### Agentic AI
- Autonomous agents that plan, execute, refine tasks, set goals, adapt to dynamic environments
- Learn from feedback (rewards/penalties) and take corrective actions
- Not just generation—decision-making and action in environments
- Applications: autonomous robotics, software development, personal AI assistants, scientific discovery

### Supervised Learning
- Learning with labeled training data (attributes + class labels)
- Includes classification and regression
- Goal: build model to predict class labels or values for unseen data

### Classification
- Mapping input attributes to discrete class labels
- Training data: records (rows) with attributes (columns) and a classifying attribute (class label)
- Model learns function Y = f(X) where X = attributes, Y = class label
- Test data evaluates performance before predicting on unseen data

### Discriminative Models
- Learn decision boundaries to categorize classes
- Model probability of class label given input: P(Y|X)
- Examples: logistic regression, SVM, decision trees, random forest, k-NN

### Unsupervised Learning / Clustering
- No class labels or group information provided
- Groups similar data points into clusters using similarity measures (e.g., Euclidean distance)
- Maximize inter-cluster distances, minimize intra-cluster distances
- Group information discovered at end of process

### Regression / Forecasting
- Predicting continuous values (vs. discrete classes)
- Used in predictive analytics: weather, stock prices, performance forecasting

### Anomaly Detection
- Identifying abnormal behavior (outliers) different from normal patterns
- Critical in medical, fraud detection, safety systems
- Can use supervised or unsupervised methods

### Data Preprocessing
- Transforming raw data into useful form for quality results
- Tasks: cleaning, integration, transformation, reduction, discretization

---

## Examples

### Classification Applications
| Application | Description |
|-------------|-------------|
| Consumer Targeting | Classify customers as likely/unlikely to buy based on salary, lifestyle, demographics |
| Fraud Detection | Identify fraudulent credit card transactions using high amount, foreign currency, new website patterns |

### Classification Types
- **Binary**: Two classes (buy/don't buy, fraud/non-fraud)
- **Multi-class**: Multiple categories (e.g., 5 types of fraud)
- **One-vs-Rest**: Focus on one class vs. all others combined

### Clustering Applications
| Application | Description |
|-------------|-------------|
| Market Segmentation | Group customers by spending habits, geography, lifestyle (budget buyers, high-spenders) |
| Document Clustering | Group documents by topic for quick access |
| Stock Behavior | Cluster stocks by movement patterns (rising, falling) |

### Clustering Example from Class
- First-name histogram clusters: S (highest), A, M, V, R, P, D
- Location clusters on India map for participant analysis

### Deep Learning Applications
- Computer vision: autonomous vehicles, 360° cameras, driver sleep detection, automatic braking
- NLP: voice assistants (Alexa), transcription, AI summarization
- Healthcare: diagnosis, drug discovery, treatment planning
- Finance: algorithmic trading, fraud detection, credit risk modeling

### Agentic AI Examples
- Autonomous robotics in hazardous environments
- Personal AI assistants managing calendars, emails, bookings
- Scientific discovery: running experiments, interpreting data, refining hypotheses

### Missing Value Handling (Class Exercise)
**Dataset**: 10 records, 6 class C2, 3 class C1, 1 unknown class; ~5% missing attribute values

**Proposed Solutions**:
- Column mean/average for numerical attributes
- Column median for numerical attributes
- Mode (most frequent) for class labels
- Rule-based inference from other attributes
- Clustering-based class prediction

---

## Key Takeaways

### AI Hierarchy
```
Artificial Intelligence
├── Machine Learning
│   ├── Supervised Learning (Classification, Regression)
│   ├── Unsupervised Learning (Clustering)
│   └── Deep Learning
│       └── Generative AI
│           └── Agentic AI
└── NLP (intersects all)
```

### Classification vs. Clustering
| Aspect | Classification | Clustering |
|--------|---------------|------------|
| Labels | Given in training data | Not given; discovered |
| Goal | Predict class for new data | Find natural groupings |
| Type | Supervised | Unsupervised |

### Clustering Principle
- **Intra-cluster distance**: Minimize (points within cluster similar)
- **Inter-cluster distance**: Maximize (clusters well separated)

### Discriminative vs. Generative
- **Discriminative**: Classify/predict (P(Y|X)), learn decision boundaries
- **Generative**: Create new content, learn data distribution

### Agentic AI vs. Simple AI Agents
| Simple AI Agents | Agentic AI |
|------------------|------------|
| Rule-based, single task | Autonomous, multi-step planning |
| Execute and finish | Adapt, refine, set goals |
| No dynamic decision-making | Learn from feedback, optimize |

---

## Questions Raised

### Classification
- **Q**: Is classification only binary? **A**: No—binary, multi-class, one-vs-rest
- **Q**: How does one-vs-rest work? **A**: Treat one class as positive, all others as negative

### Clustering
- **Q**: Why maximize inter-cluster distance? **A**: Ensure dissimilar points aren't grouped together; find natural groupings
- **Q**: How to set distance cutoff? **A**: Covered in later lectures (various algorithms)
- **Q**: What if data doesn't form clusters naturally? **A**: Transform data (e.g., high-dimensional data)

### Agentic AI
- **Q**: Difference between AI agent and Agentic AI? **A**: Agentic AI autonomously plans, executes, refines; simple agents do single tasks
- **Q**: Can Agentic AI work