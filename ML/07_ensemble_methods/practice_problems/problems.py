"""
Chapter 7: Ensemble Methods - Practice Problems
=====================================

This file contains practice problems for ensemble methods with solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


# Problem 1: Random Forest Implementation from Scratch
def problem_1_random_forest_implementation():
    """
    Problem: Implement Random Forest from scratch and compare with scikit-learn.
    
    Requirements:
    - Implement Random Forest without using scikit-learn's RandomForest
    - Compare results with scikit-learn implementation
    - Analyze feature importance
    """
    print("Problem 1: Random Forest Implementation from Scratch")
    print("=" * 55)
    
    # Load sample data
    X, y = load_iris(return_X_y=True)
    feature_names = load_iris().feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Custom Random Forest implementation
    class CustomRandomForestClassifier:
        def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.max_features = max_features
            self.random_state = random_state
            
        def fit(self, X, y):
            if self.random_state:
                np.random.seed(self.random_state)
                
            self.estimators = []
            self.classes = np.unique(y)
            
            # Calculate max_features
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == 'log2':
                max_features = int(np.log2(X.shape[1]))
            else:
                max_features = X.shape[1]
            
            for _ in range(self.n_estimators):
                # Bootstrap sample
                n_samples = X.shape[0]
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
                
                # Random feature selection
                feature_indices = np.random.choice(X.shape[1], max_features, replace=False)
                X_subset = X_bootstrap[:, feature_indices]
                
                # Train decision tree
                tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
                tree.fit(X_subset, y_bootstrap)
                
                self.estimators.append((tree, feature_indices))
            
            return self
        
        def predict(self, X):
            predictions = []
            for tree, feature_indices in self.estimators:
                X_subset = X[:, feature_indices]
                pred = tree.predict(X_subset)
                predictions.append(pred)
            
            predictions = np.array(predictions).T
            final_predictions = []
            for sample_predictions in predictions:
                unique, counts = np.unique(sample_predictions, return_counts=True)
                final_predictions.append(unique[np.argmax(counts)])
            
            return np.array(final_predictions)
    
    # Apply custom implementation
    custom_rf = CustomRandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    custom_rf.fit(X_train, y_train)
    y_pred_custom = custom_rf.predict(X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)
    
    # Apply scikit-learn implementation
    sklearn_rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    sklearn_rf.fit(X_train, y_train)
    y_pred_sklearn = sklearn_rf.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    # Compare results
    print(f"Custom Random Forest Accuracy: {acc_custom:.4f}")
    print(f"Scikit-learn Random Forest Accuracy: {acc_sklearn:.4f}")
    
    # Feature importance comparison
    print("\nFeature Importance Comparison:")
    print("Scikit-learn:", sklearn_rf.feature_importances_)
    
    return custom_rf, sklearn_rf


# Problem 2: Voting Classifier Implementation
def problem_2_voting_classifier():
    """
    Problem: Build a voting classifier combining different algorithms.
    
    Requirements:
    - Implement hard and soft voting classifiers
    - Compare performance with individual classifiers
    - Analyze voting behavior
    """
    print("\nProblem 2: Voting Classifier Implementation")
    print("=" * 45)
    
    # Load sample data
    X, y = load_wine(return_X_y=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Individual classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train individual classifiers and collect predictions
    predictions = {}
    accuracies = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        predictions[name] = y_pred
        accuracies[name] = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracies[name]:.4f}")
    
    # Hard voting
    hard_votes = np.array(list(predictions.values())).T
    hard_predictions = []
    for votes in hard_votes:
        unique, counts = np.unique(votes, return_counts=True)
        hard_predictions.append(unique[np.argmax(counts)])
    
    hard_accuracy = accuracy_score(y_test, hard_predictions)
    print(f"Hard Voting Accuracy: {hard_accuracy:.4f}")
    
    # Soft voting
    probabilities = {}
    for name, clf in classifiers.items():
        proba = clf.predict_proba(X_test)
        probabilities[name] = proba
    
    soft_probabilities = np.mean(list(probabilities.values()), axis=0)
    soft_predictions = np.argmax(soft_probabilities, axis=1)
    soft_accuracy = accuracy_score(y_test, soft_predictions)
    print(f"Soft Voting Accuracy: {soft_accuracy:.4f}")
    
    # Compare all results
    print("\nComparison:")
    for name, acc in accuracies.items():
        print(f"  {name}: {acc:.4f}")
    print(f"  Hard Voting: {hard_accuracy:.4f}")
    print(f"  Soft Voting: {soft_accuracy:.4f}")
    
    return classifiers, hard_accuracy, soft_accuracy


# Problem 3: Ensemble Visualization
def problem_3_ensemble_visualization():
    """
    Problem: Visualize decision boundaries of ensemble methods.
    
    Requirements:
    - Create 2D dataset for visualization
    - Visualize decision boundaries of different ensemble methods
    - Compare ensemble methods visually
    """
    print("\nProblem 3: Ensemble Visualization")
    print("=" * 35)
    
    # Create a simple 2D dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define ensemble methods
    methods = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Voting': None  # Will create separately
    }
    
    # Create voting classifier
    from sklearn.ensemble import VotingClassifier
    methods['Voting'] = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=25, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=25, random_state=42))
        ],
        voting='soft'
    )
    
    # Train methods and visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (name, method) in enumerate(methods.items()):
        # Train method
        method.fit(X_train, y_train)
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Predict on mesh
        Z = method.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        axes[i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        scatter = axes[i].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='black')
        axes[i].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, method.predict(X_test)):.3f}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return methods


# Problem 4: AdaBoost Implementation from Scratch
def problem_4_adaboost_implementation():
    """
    Problem: Implement AdaBoost algorithm from scratch.
    
    Requirements:
    - Implement AdaBoost without using scikit-learn's AdaBoost
    - Compare results with scikit-learn implementation
    - Analyze weight updates and error rates
    """
    print("\nProblem 4: AdaBoost Implementation from Scratch")
    print("=" * 50)
    
    # Load sample data (binary classification)
    X, y = load_iris(return_X_y=True)
    # Convert to binary classification
    binary_mask = y != 2
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    # Convert labels to -1, 1
    y_binary = np.where(y_binary == 0, -1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)
    
    # Custom AdaBoost implementation
    class CustomAdaBoostClassifier:
        def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.random_state = random_state
            
        def fit(self, X, y):
            if self.random_state:
                np.random.seed(self.random_state)
                
            n_samples = X.shape[0]
            # Initialize weights
            sample_weights = np.full(n_samples, (1 / n_samples))
            
            self.estimators = []
            self.estimator_weights = []
            self.estimator_errors = []
            
            for _ in range(self.n_estimators):
                # Train weak learner
                estimator = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
                estimator.fit(X, y, sample_weight=sample_weights)
                
                # Calculate error
                y_pred = estimator.predict(X)
                incorrect = (y_pred != y)
                estimator_error = np.average(incorrect, weights=sample_weights)
                
                # Check if estimator is better than random guessing
                if estimator_error <= 0 or estimator_error >= 0.5:
                    break
                
                # Calculate estimator weight
                estimator_weight = self.learning_rate * 0.5 * np.log((1 - estimator_error) / estimator_error)
                
                # Update sample weights
                sample_weights *= np.exp(-estimator_weight * y * y_pred)
                sample_weights /= np.sum(sample_weights)
                
                # Store estimator and its weight
                self.estimators.append(estimator)
                self.estimator_weights.append(estimator_weight)
                self.estimator_errors.append(estimator_error)
            
            return self
        
        def predict(self, X):
            # Get predictions from all estimators
            predictions = np.array([estimator.predict(X) for estimator in self.estimators])
            
            # Weighted voting
            weighted_predictions = np.dot(self.estimator_weights, predictions)
            return np.sign(weighted_predictions)
    
    # Apply custom implementation
    custom_adaboost = CustomAdaBoostClassifier(n_estimators=50, random_state=42)
    custom_adaboost.fit(X_train, y_train)
    y_pred_custom = custom_adaboost.predict(X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)
    
    # Apply scikit-learn implementation
    sklearn_adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
    sklearn_adaboost.fit(X_train, y_train)
    y_pred_sklearn = sklearn_adaboost.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    # Compare results
    print(f"Custom AdaBoost Accuracy: {acc_custom:.4f}")
    print(f"Scikit-learn AdaBoost Accuracy: {acc_sklearn:.4f}")
    print(f"Number of estimators used (custom): {len(custom_adaboost.estimators)}")
    print(f"Number of estimators used (sklearn): {sklearn_adaboost.n_estimators}")
    
    # Plot error rates
    plt.figure(figsize=(10, 6))
    plt.plot(custom_adaboost.estimator_errors, 'b-', label='Custom AdaBoost Errors')
    plt.plot(sklearn_adaboost.estimator_errors_, 'r-', label='Scikit-learn AdaBoost Errors')
    plt.xlabel('Estimator')
    plt.ylabel('Error Rate')
    plt.title('AdaBoost Estimator Error Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return custom_adaboost, sklearn_adaboost


# Problem 5: Stacking Ensemble Implementation
def problem_5_stacking_ensemble():
    """
    Problem: Build a stacking ensemble with custom meta-learner.
    
    Requirements:
    - Implement stacking ensemble from scratch
    - Use different base estimators
    - Train custom meta-learner
    - Compare with individual estimators
    """
    print("\nProblem 5: Stacking Ensemble Implementation")
    print("=" * 45)
    
    # Load sample data
    X, y = load_wine(return_X_y=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Base estimators
    base_estimators = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        SVC(probability=True, random_state=42),
        KNeighborsClassifier(n_neighbors=5)
    ]
    
    # Train base estimators and get predictions
    base_predictions_train = []
    base_predictions_test = []
    
    for estimator in base_estimators:
        estimator.fit(X_train, y_train)
        # Get probability predictions for training set (for stacking)
        pred_proba_train = estimator.predict_proba(X_train)
        pred_proba_test = estimator.predict_proba(X_test)
        base_predictions_train.append(pred_proba_train)
        base_predictions_test.append(pred_proba_test)
    
    # Stack predictions
    X_meta_train = np.hstack(base_predictions_train)
    X_meta_test = np.hstack(base_predictions_test)
    
    # Train meta-learner
    meta_learner = LogisticRegression(random_state=42)
    meta_learner.fit(X_meta_train, y_train)
    
    # Make final predictions
    y_pred_final = meta_learner.predict(X_meta_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)
    
    # Compare with individual estimators
    individual_accuracies = []
    for estimator in base_estimators:
        y_pred = estimator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        individual_accuracies.append(acc)
    
    # Print results
    print("Individual Estimator Accuracies:")
    estimator_names = ['Random Forest', 'SVM', 'KNN']
    for name, acc in zip(estimator_names, individual_accuracies):
        print(f"  {name}: {acc:.4f}")
    print(f"Stacking Ensemble Accuracy: {final_accuracy:.4f}")
    
    # Feature importance in meta-learner
    print(f"\nMeta-learner coefficients shape: {meta_learner.coef_.shape}")
    
    return base_estimators, meta_learner, final_accuracy


# Main execution
if __name__ == "__main__":
    print("Chapter 7: Ensemble Methods - Practice Problems")
    print("=============================================")
    
    # Run all problems
    try:
        # Problem 1: Random Forest Implementation
        custom_rf, sklearn_rf = problem_1_random_forest_implementation()
        
        # Problem 2: Voting Classifier
        classifiers, hard_acc, soft_acc = problem_2_voting_classifier()
        
        # Problem 3: Ensemble Visualization
        ensemble_methods = problem_3_ensemble_visualization()
        
        # Problem 4: AdaBoost Implementation
        custom_ada, sklearn_ada = problem_4_adaboost_implementation()
        
        # Problem 5: Stacking Ensemble
        base_estimators, meta_learner, stacking_acc = problem_5_stacking_ensemble()
        
        print("\n🎉 All practice problems completed successfully!")
        print("\n📋 Summary of what you've learned:")
        print("  • Implemented Random Forest from scratch and compared with scikit-learn")
        print("  • Built voting classifiers with hard and soft voting")
        print("  • Visualized decision boundaries of ensemble methods")
        print("  • Implemented AdaBoost algorithm from scratch")
        print("  • Built stacking ensemble with custom meta-learner")
        
    except Exception as e:
        print(f"Error running practice problems: {e}")
        print("Please check the implementations and try again.")