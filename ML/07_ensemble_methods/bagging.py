"""
Bagging and Random Forest Implementation
==================================

This module provides comprehensive implementations of bagging and Random Forest
ensemble methods with detailed explanations and practical examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class BaggingClassifier:
    """
    Bagging Classifier implementation.
    
    Parameters:
    -----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : float, default=1.0
        The fraction of samples to draw from X to train each base estimator.
    max_features : float, default=1.0
        The fraction of features to draw from X to train each base estimator.
    bootstrap : bool, default=True
        Whether samples are drawn with replacement.
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes:
    -----------
    estimators : list
        The collection of fitted sub-estimators.
    """
    
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, random_state=None):
        self.base_estimator = base_estimator or DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample of data."""
        n_samples = X.shape[0]
        n_bootstrap = int(self.max_samples * n_samples)
        
        if self.bootstrap:
            indices = np.random.choice(n_samples, n_bootstrap, replace=True)
        else:
            indices = np.random.choice(n_samples, n_bootstrap, replace=False)
        
        return X[indices], y[indices]
    
    def _feature_sample(self, X):
        """Select random subset of features."""
        n_features = X.shape[1]
        n_feature_subset = int(self.max_features * n_features)
        feature_indices = np.random.choice(n_features, n_feature_subset, replace=False)
        return feature_indices
    
    def fit(self, X, y):
        """
        Build a Bagging ensemble of estimators from the training set.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : BaggingClassifier
            Fitted estimator.
        """
        self.estimators = []
        self.feature_indices = []
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # Select random features
            feature_idx = self._feature_sample(X_bootstrap)
            X_subset = X_bootstrap[:, feature_idx]
            
            # Fit estimator
            estimator = type(self.base_estimator)(**self.base_estimator.get_params())
            estimator.fit(X_subset, y_bootstrap)
            
            self.estimators.append(estimator)
            self.feature_indices.append(feature_idx)
        
        return self
    
    def predict(self, X):
        """
        Predict class for X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted classes.
        """
        # Get predictions from all estimators
        predictions = []
        for estimator, feature_idx in zip(self.estimators, self.feature_indices):
            X_subset = X[:, feature_idx]
            pred = estimator.predict(X_subset)
            predictions.append(pred)
        
        # Aggregate predictions (majority voting)
        predictions = np.array(predictions).T
        y_pred = []
        for sample_predictions in predictions:
            unique, counts = np.unique(sample_predictions, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])
        
        return np.array(y_pred)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        p : array, shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Get probability predictions from all estimators
        probas = []
        for estimator, feature_idx in zip(self.estimators, self.feature_indices):
            X_subset = X[:, feature_idx]
            proba = estimator.predict_proba(X_subset)
            probas.append(proba)
        
        # Average probabilities
        return np.mean(probas, axis=0)


class RandomForestClassifierCustom:
    """
    Random Forest Classifier implementation.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : str, default='sqrt'
        The number of features to consider when looking for the best split.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Build a forest of trees from the training set.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : RandomForestClassifierCustom
            Fitted estimator.
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # Convert max_features to numeric value
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(X.shape[1]))
        elif self.max_features == 'log2':
            max_features = int(np.log2(X.shape[1]))
        else:
            max_features = int(self.max_features * X.shape[1])
        
        # Create bagging classifier with decision trees
        base_estimator = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        self.bagging = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            max_features=max_features / X.shape[1],
            random_state=self.random_state
        )
        
        self.bagging.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class for X."""
        return self.bagging.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        return self.bagging.predict_proba(X)


def compare_bagging_implementations(X, y, n_estimators=10):
    """
    Compare custom bagging implementation with scikit-learn's implementation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    n_estimators : int, default=10
        Number of estimators.
        
    Returns:
    --------
    results : dict
        Dictionary containing results from both implementations.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Custom implementation
    custom_bagging = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n_estimators,
        random_state=42
    )
    custom_bagging.fit(X_train, y_train)
    y_pred_custom = custom_bagging.predict(X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)
    
    # Scikit-learn implementation
    sklearn_bagging = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )
    sklearn_bagging.fit(X_train, y_train)
    y_pred_sklearn = sklearn_bagging.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    results = {
        'custom': {
            'accuracy': acc_custom,
            'predictions': y_pred_custom
        },
        'sklearn': {
            'accuracy': acc_sklearn,
            'predictions': y_pred_sklearn
        }
    }
    
    return results


def plot_feature_importance(rf_model, feature_names=None, title="Feature Importance"):
    """
    Plot feature importance from Random Forest model.
    
    Parameters:
    -----------
    rf_model : RandomForestClassifier or RandomForestRegressor
        Fitted Random Forest model.
    feature_names : list, default=None
        Names of features.
    title : str, default="Feature Importance"
        Title for the plot.
    """
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices])
    if feature_names:
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    else:
        plt.xticks(range(len(importances)), indices)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()


def out_of_bag_error(X, y, n_estimators_range=range(10, 201, 10)):
    """
    Calculate out-of-bag error for different numbers of estimators.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    n_estimators_range : range, default=range(10, 201, 10)
        Range of n_estimators to test.
        
    Returns:
    --------
    oob_errors : list
        Out-of-bag errors for each n_estimator.
    """
    oob_errors = []
    
    for n_estimators in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=True,
            random_state=42
        )
        rf.fit(X, y)
        oob_errors.append(1 - rf.oob_score_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, oob_errors, 'bo-')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Out-of-Bag Error')
    plt.title('Out-of-Bag Error vs Number of Estimators')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return oob_errors


# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data
    X, y = load_iris(return_X_y=True)
    feature_names = load_iris().feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Apply Random Forest
    rf = RandomForestClassifierCustom(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Compare implementations
    print("\nComparing implementations:")
    comparison_results = compare_bagging_implementations(X, y, n_estimators=50)
    print(f"Custom implementation accuracy: {comparison_results['custom']['accuracy']:.4f}")
    print(f"Scikit-learn implementation accuracy: {comparison_results['sklearn']['accuracy']:.4f}")
    
    # Plot feature importance
    sklearn_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    sklearn_rf.fit(X_train, y_train)
    plot_feature_importance(sklearn_rf, feature_names, "Iris Dataset Feature Importance")
    
    # Out-of-bag error analysis
    print("\nAnalyzing out-of-bag error...")
    oob_errors = out_of_bag_error(X_train, y_train)
    
    # Demonstrate with larger dataset
    print("\nWine Dataset Example:")
    from sklearn.datasets import load_wine
    X_wine, y_wine = load_wine(return_X_y=True)
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        X_wine, y_wine, test_size=0.3, random_state=42
    )
    
    rf_wine = RandomForestClassifierCustom(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    rf_wine.fit(X_train_wine, y_train_wine)
    y_pred_wine = rf_wine.predict(X_test_wine)
    acc_wine = accuracy_score(y_test_wine, y_pred_wine)
    print(f"Wine dataset accuracy: {acc_wine:.4f}")