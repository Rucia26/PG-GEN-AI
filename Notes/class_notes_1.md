# Introductory Concepts in Artificial Intelligence and Machine Learning

## Summary
This session provides foundational concepts in cognitive science, artificial intelligence (AI), machine learning (ML), and their subfields. It covers the relationships between AI, ML, deep learning, NLP, generative AI, and agentic AI. The lecture then focuses on supervised learning (classification with discriminative models), unsupervised learning (clustering), and regression/forecasting, with practical applications and a Q&A on classification types.

## Topics Covered
- Cognitive science as an interdisciplinary field
- Definitions and scope of Artificial Intelligence
- Hierarchy of AI subfields: AI → ML → Deep Learning → Generative AI → Agentic AI; NLP as intersecting subset
- AI tasks: knowledge representation, reasoning, learning, NLP, search, path planning
- Supervised learning: classification, discriminative models, decision boundaries
- Unsupervised learning: clustering, similarity measures, intra/inter-cluster distances
- Regression and forecasting for value prediction
- Applications: customer targeting, fraud detection, market segmentation, document clustering, stock behavior analysis

## Definitions
- **Cognitive Science**: Interdisciplinary study of human intelligence, perception, understanding, and knowledge acquisition, drawing from computer science, philosophy, linguistics, neuroscience, etc.
- **Artificial Intelligence (AI)**: Field of computer science focused on designing systems that exhibit human-like intelligence; "artificial" = created by human effort, "intelligence" = ability to acquire and utilize knowledge.
- **Machine Learning (ML)**: Subset of AI concerned with systems that learn from experience, generate new facts/concepts, and adapt to environments.
- **Deep Learning**: Subset of ML using neural networks with multiple layers.
- **Generative AI**: Subset of deep learning focused on generating new content.
- **Agentic AI**: Subset of generative AI involving autonomous agents.
- **Natural Language Processing (NLP)**: Subset of AI intersecting with ML, deep learning, generative AI, and agentic AI for language understanding/generation.
- **Supervised Learning**: Learning from labeled training data (attributes + class labels) to predict labels for unseen data.
- **Classification**: Supervised learning task mapping input attributes to discrete class labels; finds function *Y = f(X)* where *Y* is class label.
- **Discriminative Models**: Models that learn decision boundaries to distinguish classes; output class probabilities (e.g., logistic regression, SVM, decision trees, random forest, naïve Bayes).
- **Unsupervised Learning / Clustering**: Learning without class labels; groups similar data points into clusters using similarity measures (e.g., Euclidean distance); maximizes inter-cluster distance, minimizes intra-cluster distance.
- **Regression / Forecasting**: Predicting continuous numerical values rather than discrete classes.

## Examples
### Classification Applications
- **Customer Targeting**: Predict whether a customer will buy a new product based on attributes (salary, lifestyle, demographics).
- **Fraud Detection**: Identify fraudulent credit card transactions using features like unusually high amount, foreign currency, unfamiliar website.

### Clustering Applications
- **Market Segmentation**: Group customers by spending habits (budget vs. high-spending), geography, lifestyle.
- **Document Clustering**: Organize documents by topic for quick retrieval; e.g., grouping news articles or AI papers.
- **Stock Behavior Analysis**: Cluster stocks with similar movement patterns (e.g., rising, falling).

### Illustrative Clustering Demo
- Histogram of first-name initials in class: highest frequency for 'S', then 'A', 'M', 'V', 'R', 'P', 'D'.
- Geographic clusters of participants on India map used to estimate course participation by location.

## Questions & Answers
**Q**: Is classification only binary (two classes) or can it be multi-class?  
**A**: Classification can be:
- **Binary**: Two class labels (e.g., buy/don't buy, fraud/non-fraud).
- **Multi-class**: More than two labels (e.g., five types of fraudulent activity).
- **One-vs-Rest**: Treat one class of interest vs. all others combined into a single "rest" class.

## Additional Notes
- **Training vs. Test Data**: Training data includes attributes and class labels; test data also has labels for evaluation; unseen data lacks labels for prediction.
- **Model Building Process**: Train on training data → evaluate on test data → if performance acceptable, deploy to predict on unseen data.
- **Similarity Measures**: Euclidean distance mentioned; others to be covered in later sessions.
- **Discriminative vs. Generative**: Discriminative models learn decision boundaries (P(y|x)); generative models learn data distribution (P(x|y)) – not detailed here but implied.
- **Upcoming Sessions**: Each technique (logistic regression, SVM, decision trees, random forest, naïve Bayes, clustering algorithms, regression methods) will be covered in depth.