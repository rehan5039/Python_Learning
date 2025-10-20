# Chapter 4: Classification

## Overview
Classification is a supervised learning technique where the goal is to predict the categorical class labels of new instances based on past observations. This chapter covers fundamental classification algorithms, evaluation metrics, and practical implementation techniques.

## Topics Covered
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Naive Bayes
- Evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Cross-validation techniques
- Hyperparameter tuning

## Learning Objectives
By the end of this chapter, you should be able to:
- Understand the principles of classification algorithms
- Implement various classification techniques in Python
- Evaluate model performance using appropriate metrics
- Handle imbalanced datasets
- Apply cross-validation for robust model assessment
- Tune hyperparameters for optimal performance
- Select appropriate algorithms for different problem types

## Prerequisites
- Understanding of supervised learning concepts
- Basic knowledge of Python and scikit-learn
- Familiarity with data preprocessing techniques
- Understanding of linear algebra and statistics

## Content Files
- [logistic_regression.py](logistic_regression.py) - Logistic Regression implementation
- [knn.py](knn.py) - K-Nearest Neighbors implementation
- [decision_trees.py](decision_trees.py) - Decision Trees and Random Forest
- [svm.py](svm.py) - Support Vector Machines
- [naive_bayes.py](naive_bayes.py) - Naive Bayes classifiers
- [evaluation_metrics.py](evaluation_metrics.py) - Model evaluation techniques
- [practice_problems/](practice_problems/) - Exercises to reinforce learning
  - [problems.py](practice_problems/problems.py) - Practice problems with solutions
  - [README.md](practice_problems/README.md) - Practice problem descriptions

## Real-World Applications
- **Email Spam Detection**: Classifying emails as spam or not spam
- **Medical Diagnosis**: Predicting diseases based on symptoms
- **Image Recognition**: Identifying objects in images
- **Credit Risk Assessment**: Determining loan approval
- **Customer Churn Prediction**: Identifying customers likely to leave
- **Sentiment Analysis**: Classifying text as positive, negative, or neutral

## Example: Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate sample data
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

## Next Chapter
[Chapter 5: Clustering](../05_clustering/)