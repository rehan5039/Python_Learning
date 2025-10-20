"""
Practice Problems Solutions: Classification

This module contains example solutions for the practice problems in Chapter 4.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.datasets import make_classification
from sklearn.utils import resample
from scipy import stats

# Generate sample datasets for problems
def generate_medical_data():
    """Generate synthetic medical diagnosis dataset"""
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                              n_redundant=2, n_clusters_per_class=1, 
                              weights=[0.7, 0.3], random_state=42)
    return X, y

def generate_text_data():
    """Generate synthetic text classification dataset"""
    X, y = make_classification(n_samples=2000, n_features=1000, n_informative=50,
                              n_redundant=50, n_clusters_per_class=2, 
                              random_state=42)
    return X, y

def generate_fraud_data():
    """Generate synthetic fraud detection dataset"""
    # Create imbalanced dataset
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=15,
                              n_redundant=5, n_clusters_per_class=1,
                              weights=[0.99, 0.01], random_state=42)
    return X, y

def generate_customer_data():
    """Generate synthetic customer churn dataset"""
    X, y = make_classification(n_samples=5000, n_features=15, n_informative=12,
                              n_redundant=3, n_clusters_per_class=1,
                              random_state=42)
    return X, y

# Problem 1: Algorithm Comparison
def problem_1_solution():
    """Solution for Problem 1: Algorithm Comparison"""
    print("Problem 1: Algorithm Comparison")
    print("=" * 30)
    
    # Generate data
    X, y = generate_medical_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Train and evaluate classifiers
    results = {}
    for name, clf in classifiers.items():
        # Train
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
    
    # Display results
    print("Performance Comparison:")
    print(f"{'Algorithm':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc']:<10.4f}")
    
    # Analysis
    best_algorithm = max(results, key=lambda x: results[x]['f1'])
    print(f"\nBest performing algorithm based on F1-score: {best_algorithm}")
    print("Analysis:")
    print("1. Logistic Regression: Good baseline, interpretable coefficients")
    print("2. KNN: Non-parametric, good for local patterns")
    print("3. Decision Tree: Interpretable, handles non-linear relationships")

# Problem 2: Hyperparameter Tuning
def problem_2_solution():
    """Solution for Problem 2: Hyperparameter Tuning"""
    print("\n\nProblem 2: Hyperparameter Tuning")
    print("=" * 30)
    
    # Generate data
    X, y = generate_text_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Before tuning
    svm_base = SVC(random_state=42)
    svm_base.fit(X_train, y_train)
    y_pred_base = svm_base.predict(X_test)
    accuracy_base = accuracy_score(y_test, y_pred_base)
    print(f"Base SVM Accuracy: {accuracy_base:.4f}")
    
    # Grid search for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # After tuning
    best_svm = grid_search.best_estimator_
    y_pred_tuned = best_svm.predict(X_test)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    
    print(f"Tuned SVM Accuracy: {accuracy_tuned:.4f}")
    print(f"Improvement: {accuracy_tuned - accuracy_base:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Compare kernel functions
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_results = {}
    
    for kernel in kernels:
        svm_kernel = SVC(kernel=kernel, random_state=42)
        scores = cross_val_score(svm_kernel, X_train, y_train, cv=5)
        kernel_results[kernel] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    print("\nKernel Comparison (Cross-Validation):")
    for kernel, results in kernel_results.items():
        print(f"{kernel:<10}: {results['mean']:.4f} (+/- {results['std']*2:.4f})")

# Problem 3: Handling Imbalanced Data
def problem_3_solution():
    """Solution for Problem 3: Handling Imbalanced Data"""
    print("\n\nProblem 3: Handling Imbalanced Data")
    print("=" * 30)
    
    # Generate imbalanced data
    X, y = generate_fraud_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Class distribution in training set:")
    print(f"  Class 0 (Normal): {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")
    print(f"  Class 1 (Fraud):  {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
    
    # Without handling imbalance
    clf_base = LogisticRegression(random_state=42)
    clf_base.fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)
    
    print(f"\nWithout handling imbalance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_base):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred_base):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred_base):.4f}")
    
    # With class weights
    clf_weighted = LogisticRegression(class_weight='balanced', random_state=42)
    clf_weighted.fit(X_train, y_train)
    y_pred_weighted = clf_weighted.predict(X_test)
    
    print(f"\nWith class weights:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_weighted):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_weighted):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred_weighted):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred_weighted):.4f}")
    
    # Analysis
    print("\nAnalysis:")
    print("1. Accuracy paradox: High accuracy doesn't mean good fraud detection")
    print("2. Class weights help improve recall for minority class")
    print("3. F1-score is better metric than accuracy for imbalanced data")

# Problem 4: Model Evaluation and Selection
def problem_4_solution():
    """Solution for Problem 4: Model Evaluation and Selection"""
    print("\n\nProblem 4: Model Evaluation and Selection")
    print("=" * 35)
    
    # Generate data
    X, y = generate_customer_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Cross-validation
    cv_results = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    print("Cross-Validation Results (ROC-AUC):")
    for name, results in cv_results.items():
        print(f"{name:<20}: {results['mean']:.4f} (+/- {results['std']*2:.4f})")
    
    # Statistical significance testing
    print("\nStatistical Significance Testing:")
    names = list(classifiers.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            # Paired t-test (simplified approach)
            clf1_scores = cross_val_score(classifiers[names[i]], X_train, y_train, cv=5, scoring='roc_auc')
            clf2_scores = cross_val_score(classifiers[names[j]], X_train, y_train, cv=5, scoring='roc_auc')
            
            t_stat, p_value = stats.ttest_rel(clf1_scores, clf2_scores)
            significant = "Yes" if p_value < 0.05 else "No"
            print(f"{names[i]} vs {names[j]}: p-value = {p_value:.4f} ({significant})")
    
    # ROC curves
    plt.figure(figsize=(10, 6))
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Problem 5: Ensemble Methods Implementation
def problem_5_solution():
    """Solution for Problem 5: Ensemble Methods Implementation"""
    print("\n\nProblem 5: Ensemble Methods Implementation")
    print("=" * 38)
    
    # Generate data
    X, y = generate_customer_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Individual classifiers
    base_classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Ensemble methods
    ensembles = {
        'Bagging (Random Forest)': RandomForestClassifier(n_estimators=100, random_state=42),
        'Boosting (AdaBoost)': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate
    results = {}
    
    # Base classifiers
    for name, clf in base_classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
    
    # Ensemble methods
    for name, clf in ensembles.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
    
    # Display results
    print("Performance Comparison:")
    print(f"{'Method':<25} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 45)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f}")
    
    # Ensemble size impact
    ensemble_sizes = [10, 50, 100, 200]
    size_results = []
    
    for n_estimators in ensemble_sizes:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
        size_results.append({
            'size': n_estimators,
            'mean_f1': scores.mean(),
            'std_f1': scores.std()
        })
    
    print("\nEnsemble Size Impact:")
    for result in size_results:
        print(f"Size {result['size']:<4}: F1 = {result['mean_f1']:.4f} (+/- {result['std_f1']*2:.4f})")
    
    print("\nAnalysis:")
    print("1. Ensemble methods generally outperform individual classifiers")
    print("2. Bagging reduces variance, boosting reduces bias")
    print("3. There's a trade-off between ensemble size and computational cost")
    print("4. Diminishing returns as ensemble size increases")

# Run all solutions
if __name__ == "__main__":
    problem_1_solution()
    problem_2_solution()
    problem_3_solution()
    problem_4_solution()
    problem_5_solution()