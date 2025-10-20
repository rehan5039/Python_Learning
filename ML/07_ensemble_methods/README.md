# Chapter 7: Ensemble Methods

## Overview
Ensemble methods combine multiple machine learning models to create a stronger predictor than any individual model alone. This chapter covers fundamental ensemble techniques, their theoretical foundations, and practical implementations for both classification and regression tasks.

## Topics Covered
- Bagging (Bootstrap Aggregating)
- Random Forest
- Boosting algorithms (AdaBoost, Gradient Boosting, XGBoost, LightGBM)
- Stacking and blending
- Voting classifiers and regressors
- Ensemble diversity and error decomposition
- Hyperparameter tuning for ensemble methods
- Handling overfitting in ensembles

## Learning Objectives
By the end of this chapter, you should be able to:
- Understand the theory behind ensemble methods
- Implement various ensemble techniques from scratch
- Apply bagging and boosting algorithms to real problems
- Combine different models effectively
- Evaluate ensemble performance and diversity
- Tune hyperparameters for optimal ensemble performance
- Handle overfitting in ensemble models
- Choose appropriate ensemble methods for different scenarios

## Prerequisites
- Understanding of basic machine learning concepts
- Experience with classification and regression algorithms
- Knowledge of decision trees
- Familiarity with Python and scikit-learn
- Basic understanding of statistics and probability

## Content Files
- [bagging.py](bagging.py) - Bagging and Random Forest implementation
- [boosting.py](boosting.py) - Boosting algorithms (AdaBoost, Gradient Boosting)
- [stacking.py](stacking.py) - Stacking and blending techniques
- [voting.py](voting.py) - Voting classifiers and regressors
- [ensemble_evaluation.py](ensemble_evaluation.py) - Ensemble evaluation and comparison
- [practice_problems/](practice_problems/) - Exercises to reinforce learning
  - [problems.py](practice_problems/problems.py) - Practice problems with solutions
  - [README.md](practice_problems/README.md) - Practice problem descriptions

## Real-World Applications
- **Financial Risk Assessment**: Combining multiple models for credit scoring
- **Medical Diagnosis**: Ensemble of specialists for disease prediction
- **Recommendation Systems**: Combining collaborative and content-based filtering
- **Image Recognition**: Ensemble of CNNs for improved accuracy
- **Natural Language Processing**: Combining different text classifiers
- **Fraud Detection**: Multiple models to identify suspicious transactions
- **Stock Market Prediction**: Ensemble of time series models
- **Customer Churn Prediction**: Combining behavioral and demographic models

## Example: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Feature importance
feature_importance = rf.feature_importances_
print("Feature Importance:", feature_importance)
```

## Next Chapter
[Chapter 8: Neural Networks](../08_neural_networks/)