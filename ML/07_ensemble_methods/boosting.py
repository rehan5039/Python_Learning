"""
Boosting Algorithms Implementation
============================

This module provides comprehensive implementations of boosting algorithms
including AdaBoost, Gradient Boosting, and their practical applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


class AdaBoostClassifierCustom:
    """
    AdaBoost Classifier implementation.
    
    Parameters:
    -----------
    base_estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration.
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes:
    -----------
    estimators : list
        The collection of fitted sub-estimators.
    estimator_weights : array
        Weights for each estimator in the ensemble.
    estimator_errors : array
        Classification error for each estimator in the ensemble.
    """
    
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, 
                 random_state=None):
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
    
    def fit(self, X, y):
        """
        Build a boosted classifier from the training set.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : AdaBoostClassifierCustom
            Fitted estimator.
        """
        # Initialize weights uniformly
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, (1 / n_samples))
        
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        
        for _ in range(self.n_estimators):
            # Fit base estimator
            estimator = type(self.base_estimator)(**self.base_estimator.get_params())
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Predict and calculate error
            y_pred = estimator.predict(X)
            incorrect = (y_pred != y)
            estimator_error = np.average(incorrect, weights=sample_weights)
            
            # Check if estimator is better than random guessing
            if estimator_error <= 0 or estimator_error >= 0.5:
                break
            
            # Calculate estimator weight
            estimator_weight = self.learning_rate * np.log((1 - estimator_error) / estimator_error)
            
            # Update sample weights
            sample_weights *= np.exp(estimator_weight * incorrect)
            sample_weights /= np.sum(sample_weights)
            
            # Store estimator and its weight
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
            self.estimator_errors.append(estimator_error)
        
        return self
    
    def predict(self, X):
        """
        Predict classes for X.
        
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
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        
        # Weighted voting
        class_predictions = []
        for i in range(X.shape[0]):
            class_votes = {}
            for j, estimator_pred in enumerate(predictions[:, i]):
                weight = self.estimator_weights[j]
                if estimator_pred not in class_votes:
                    class_votes[estimator_pred] = 0
                class_votes[estimator_pred] += weight
            
            # Predict class with highest weighted votes
            predicted_class = max(class_votes, key=class_votes.get)
            class_predictions.append(predicted_class)
        
        return np.array(class_predictions)
    
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
        # Get class predictions from all estimators
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        
        # Calculate weighted probabilities
        n_samples = X.shape[0]
        unique_classes = np.unique([pred for pred_list in predictions for pred in pred_list])
        n_classes = len(unique_classes)
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        probas = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for j, estimator_pred in enumerate(predictions[:, i]):
                weight = self.estimator_weights[j]
                class_idx = class_to_idx[estimator_pred]
                probas[i, class_idx] += weight
        
        # Normalize probabilities
        probas = np.exp(probas)
        probas /= np.sum(probas, axis=1, keepdims=True)
        
        return probas


class GradientBoostingRegressorCustom:
    """
    Gradient Boosting Regressor implementation.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Build a gradient boosting regressor from the training set.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : GradientBoostingRegressorCustom
            Fitted estimator.
        """
        # Initialize with mean of target values
        self.initial_prediction = np.mean(y)
        self.estimators = []
        
        # Initialize residuals
        residuals = y - self.initial_prediction
        
        for _ in range(self.n_estimators):
            # Fit weak learner to residuals
            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
            estimator.fit(X, residuals)
            
            # Update predictions
            predictions = self.learning_rate * estimator.predict(X)
            
            # Update residuals
            residuals -= predictions
            
            # Store estimator
            self.estimators.append(estimator)
        
        return self
    
    def predict(self, X):
        """
        Predict regression target for X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values.
        """
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        # Add predictions from all estimators
        for estimator in self.estimators:
            predictions += self.learning_rate * estimator.predict(X)
        
        return predictions


def compare_adaboost_implementations(X, y, n_estimators=50):
    """
    Compare custom AdaBoost implementation with scikit-learn's implementation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    n_estimators : int, default=50
        Number of estimators.
        
    Returns:
    --------
    results : dict
        Dictionary containing results from both implementations.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Custom implementation
    custom_adaboost = AdaBoostClassifierCustom(
        n_estimators=n_estimators,
        random_state=42
    )
    custom_adaboost.fit(X_train, y_train)
    y_pred_custom = custom_adaboost.predict(X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)
    
    # Scikit-learn implementation
    sklearn_adaboost = AdaBoostClassifier(
        n_estimators=n_estimators,
        random_state=42
    )
    sklearn_adaboost.fit(X_train, y_train)
    y_pred_sklearn = sklearn_adaboost.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    results = {
        'custom': {
            'accuracy': acc_custom,
            'predictions': y_pred_custom,
            'n_estimators': len(custom_adaboost.estimators)
        },
        'sklearn': {
            'accuracy': acc_sklearn,
            'predictions': y_pred_sklearn,
            'n_estimators': sklearn_adaboost.n_estimators
        }
    }
    
    return results


def plot_boosting_performance(X, y, n_estimators_range=range(10, 201, 20)):
    """
    Plot boosting performance for different numbers of estimators.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    n_estimators_range : range, default=range(10, 201, 20)
        Range of n_estimators to test.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    train_errors = []
    test_errors = []
    
    for n_estimators in n_estimators_range:
        # AdaBoost
        adaboost = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
        adaboost.fit(X_train, y_train)
        
        # Training error
        y_train_pred = adaboost.predict(X_train)
        train_error = 1 - accuracy_score(y_train, y_train_pred)
        train_errors.append(train_error)
        
        # Test error
        y_test_pred = adaboost.predict(X_test)
        test_error = 1 - accuracy_score(y_test, y_test_pred)
        test_errors.append(test_error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_errors, 'b-', label='Training Error')
    plt.plot(n_estimators_range, test_errors, 'r-', label='Test Error')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error Rate')
    plt.title('AdaBoost Performance vs Number of Estimators')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data
    X, y = load_iris(return_X_y=True)
    
    # For binary classification, use only two classes
    binary_mask = y != 2
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)
    
    # Apply AdaBoost
    adaboost = AdaBoostClassifierCustom(n_estimators=50, random_state=42)
    adaboost.fit(X_train, y_train)
    y_pred = adaboost.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("AdaBoost Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of estimators used: {len(adaboost.estimators)}")
    
    # Compare implementations
    print("\nComparing implementations:")
    comparison_results = compare_adaboost_implementations(X_binary, y_binary, n_estimators=50)
    print(f"Custom implementation accuracy: {comparison_results['custom']['accuracy']:.4f}")
    print(f"Scikit-learn implementation accuracy: {comparison_results['sklearn']['accuracy']:.4f}")
    print(f"Custom estimators: {comparison_results['custom']['n_estimators']}")
    print(f"Scikit-learn estimators: {comparison_results['sklearn']['n_estimators']}")
    
    # Plot boosting performance
    print("\nAnalyzing boosting performance...")
    plot_boosting_performance(X_binary, y_binary)
    
    # Demonstrate with regression
    print("\nGradient Boosting Regression Example:")
    try:
        from sklearn.datasets import load_boston
        X_reg, y_reg = load_boston(return_X_y=True)
        
        # Split data
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        # Custom implementation
        gb_custom = GradientBoostingRegressorCustom(n_estimators=100, learning_rate=0.1, random_state=42)
        gb_custom.fit(X_train_reg, y_train_reg)
        y_pred_reg_custom = gb_custom.predict(X_test_reg)
        mse_custom = mean_squared_error(y_test_reg, y_pred_reg_custom)
        
        # Scikit-learn implementation
        from sklearn.ensemble import GradientBoostingRegressor
        gb_sklearn = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        gb_sklearn.fit(X_train_reg, y_train_reg)
        y_pred_reg_sklearn = gb_sklearn.predict(X_test_reg)
        mse_sklearn = mean_squared_error(y_test_reg, y_pred_reg_sklearn)
        
        print(f"Custom Gradient Boosting MSE: {mse_custom:.2f}")
        print(f"Scikit-learn Gradient Boosting MSE: {mse_sklearn:.2f}")
        
    except ImportError:
        print("Boston housing dataset not available. Skipping regression example.")
    
    print("\nKey Points about Boosting:")
    print("• AdaBoost focuses on difficult examples by adjusting sample weights")
    print("• Gradient Boosting builds trees to minimize residuals")
    print("• Learning rate controls the contribution of each estimator")
    print("• More estimators can lead to overfitting if not properly regularized")
    print("• Boosting often achieves high accuracy but can be prone to overfitting")