"""
Voting Classifiers and Regressors Implementation
=========================================

This module provides comprehensive implementations of voting ensemble methods
with detailed explanations and practical examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')


class VotingClassifier:
    """
    Voting Classifier implementation.
    
    Parameters:
    -----------
    estimators : list, default=None
        List of (name, estimator) tuples.
    voting : str, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        If 'soft', predicts the class label based on the argmax of the sums of the predicted probabilities.
    weights : array-like, default=None
        Sequence of weights for each estimator.
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes:
    -----------
    estimators : list
        Fitted estimators.
    """
    
    def __init__(self, estimators=None, voting='hard', weights=None, random_state=None):
        self.estimators = estimators or [
            ('lr', LogisticRegression(random_state=random_state)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=random_state)),
            ('svc', SVC(probability=True, random_state=random_state))
        ]
        self.voting = voting
        self.weights = weights
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit the voting classifier.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : VotingClassifier
            Fitted estimator.
        """
        self.classes = np.unique(y)
        self.estimators_ = []
        
        for name, estimator in self.estimators:
            estimator_copy = type(estimator)(**estimator.get_params())
            estimator_copy.fit(X, y)
            self.estimators_.append((name, estimator_copy))
        
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
        if self.voting == 'hard':
            return self._predict_hard(X)
        else:
            return self._predict_soft(X)
    
    def _predict_hard(self, X):
        """Predict using hard voting."""
        predictions = []
        for name, estimator in self.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions).T
        
        # Apply weights if provided
        if self.weights is not None:
            weighted_predictions = []
            for i, pred in enumerate(predictions):
                # Repeat predictions according to weights
                for j in range(int(self.weights[i])):
                    weighted_predictions.append(pred)
            predictions = np.array(weighted_predictions).T
        
        # Majority voting
        y_pred = []
        for sample_predictions in predictions:
            unique, counts = np.unique(sample_predictions, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])
        
        return np.array(y_pred)
    
    def _predict_soft(self, X):
        """Predict using soft voting."""
        probas = []
        for name, estimator in self.estimators_:
            proba = estimator.predict_proba(X)
            probas.append(proba)
        
        probas = np.array(probas)
        
        # Apply weights if provided
        if self.weights is not None:
            probas = probas * np.array(self.weights).reshape(-1, 1, 1)
        
        # Average probabilities
        avg_probas = np.mean(probas, axis=0)
        
        # Predict class with highest probability
        return self.classes[np.argmax(avg_probas, axis=1)]
    
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
        if self.voting != 'soft':
            raise ValueError("predict_proba is only available when voting='soft'")
        
        probas = []
        for name, estimator in self.estimators_:
            proba = estimator.predict_proba(X)
            probas.append(proba)
        
        probas = np.array(probas)
        
        # Apply weights if provided
        if self.weights is not None:
            probas = probas * np.array(self.weights).reshape(-1, 1, 1)
        
        # Average probabilities
        return np.mean(probas, axis=0)


class VotingRegressor:
    """
    Voting Regressor implementation.
    
    Parameters:
    -----------
    estimators : list, default=None
        List of (name, estimator) tuples.
    weights : array-like, default=None
        Sequence of weights for each estimator.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, estimators=None, weights=None, random_state=None):
        self.estimators = estimators or [
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=50, random_state=random_state)),
            ('svr', SVR())
        ]
        self.weights = weights
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit the voting regressor.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : VotingRegressor
            Fitted estimator.
        """
        self.estimators_ = []
        
        for name, estimator in self.estimators:
            estimator_copy = type(estimator)(**estimator.get_params())
            estimator_copy.fit(X, y)
            self.estimators_.append((name, estimator_copy))
        
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
        predictions = []
        for name, estimator in self.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Apply weights if provided
        if self.weights is not None:
            predictions = predictions * np.array(self.weights).reshape(-1, 1)
        
        # Average predictions
        return np.mean(predictions, axis=0)


def compare_voting_implementations(X, y, task='classification'):
    """
    Compare custom voting implementation with scikit-learn's implementation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    task : str, default='classification'
        Task type ('classification' or 'regression').
        
    Returns:
    --------
    results : dict
        Dictionary containing results from both implementations.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if task == 'classification':
        # Custom implementation
        custom_voting = VotingClassifier(random_state=42)
        custom_voting.fit(X_train, y_train)
        y_pred_custom = custom_voting.predict(X_test)
        acc_custom = accuracy_score(y_test, y_pred_custom)
        
        # Scikit-learn implementation
        try:
            from sklearn.ensemble import VotingClassifier as SklearnVotingClassifier
            sklearn_voting = SklearnVotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('svc', SVC(probability=True, random_state=42))
                ],
                voting='hard'
            )
            sklearn_voting.fit(X_train, y_train)
            y_pred_sklearn = sklearn_voting.predict(X_test)
            acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
        except ImportError:
            sklearn_voting = None
            acc_sklearn = 0
        
        results = {
            'custom': {
                'accuracy': acc_custom,
                'predictions': y_pred_custom
            },
            'sklearn': {
                'accuracy': acc_sklearn,
                'predictions': y_pred_sklearn if sklearn_voting else None
            }
        }
    
    else:  # regression
        # Custom implementation
        custom_voting = VotingRegressor(random_state=42)
        custom_voting.fit(X_train, y_train)
        y_pred_custom = custom_voting.predict(X_test)
        mse_custom = mean_squared_error(y_test, y_pred_custom)
        
        # Scikit-learn implementation
        try:
            from sklearn.ensemble import VotingRegressor as SklearnVotingRegressor
            sklearn_voting = SklearnVotingRegressor(
                estimators=[
                    ('lr', LinearRegression()),
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                    ('svr', SVR())
                ]
            )
            sklearn_voting.fit(X_train, y_train)
            y_pred_sklearn = sklearn_voting.predict(X_test)
            mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
        except ImportError:
            sklearn_voting = None
            mse_sklearn = float('inf')
        
        results = {
            'custom': {
                'mse': mse_custom,
                'predictions': y_pred_custom
            },
            'sklearn': {
                'mse': mse_sklearn,
                'predictions': y_pred_sklearn if sklearn_voting else None
            }
        }
    
    return results


def plot_voting_comparison(X, y):
    """
    Compare hard voting vs soft voting performance.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Hard voting
    hard_voting = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('nb', GaussianNB())
        ],
        voting='hard'
    )
    hard_voting.fit(X_train, y_train)
    y_pred_hard = hard_voting.predict(X_test)
    acc_hard = accuracy_score(y_test, y_pred_hard)
    
    # Soft voting
    soft_voting = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('nb', GaussianNB())
        ],
        voting='soft'
    )
    soft_voting.fit(X_train, y_train)
    y_pred_soft = soft_voting.predict(X_test)
    acc_soft = accuracy_score(y_test, y_pred_soft)
    
    # Individual estimators
    individual_scores = []
    estimator_names = []
    for name, estimator in hard_voting.estimators:
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        individual_scores.append(acc)
        estimator_names.append(name)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(estimator_names) + 2)
    scores = individual_scores + [acc_hard, acc_soft]
    labels = estimator_names + ['Hard Voting', 'Soft Voting']
    
    bars = plt.bar(x_pos, scores, color=['skyblue'] * len(estimator_names) + ['orange', 'red'])
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy')
    plt.title('Voting Methods Comparison')
    plt.xticks(x_pos, labels, rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'hard_voting_accuracy': acc_hard,
        'soft_voting_accuracy': acc_soft,
        'individual_scores': dict(zip(estimator_names, individual_scores))
    }


def weighted_voting_example(X, y):
    """
    Demonstrate weighted voting with different weights.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define weights
    weights_list = [
        [1, 1, 1],  # Equal weights
        [2, 1, 1],  # Logistic Regression has more weight
        [1, 2, 1],  # Random Forest has more weight
        [1, 1, 2]   # SVM has more weight
    ]
    
    results = []
    for weights in weights_list:
        voting = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('svc', SVC(probability=True, random_state=42))
            ],
            voting='soft',
            weights=weights
        )
        voting.fit(X_train, y_train)
        y_pred = voting.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append(acc)
        
        print(f"Weights {weights}: Accuracy = {acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(weights_list))
    bars = plt.bar(x_pos, results, color='lightgreen')
    plt.xlabel('Weight Configurations')
    plt.ylabel('Accuracy')
    plt.title('Weighted Voting Performance')
    plt.xticks(x_pos, [str(w) for w in weights_list], rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, results)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results


# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data
    X, y = load_iris(return_X_y=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Apply Voting Classifier
    voting = VotingClassifier(random_state=42)
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("Voting Classifier Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Compare implementations
    print("\nComparing implementations:")
    comparison_results = compare_voting_implementations(X, y, task='classification')
    print(f"Custom voting accuracy: {comparison_results['custom']['accuracy']:.4f}")
    if comparison_results['sklearn']['accuracy'] > 0:
        print(f"Scikit-learn voting accuracy: {comparison_results['sklearn']['accuracy']:.4f}")
    else:
        print("Scikit-learn voting not available")
    
    # Compare hard vs soft voting
    print("\nComparing hard vs soft voting:")
    voting_comparison = plot_voting_comparison(X, y)
    print(f"Hard voting accuracy: {voting_comparison['hard_voting_accuracy']:.4f}")
    print(f"Soft voting accuracy: {voting_comparison['soft_voting_accuracy']:.4f}")
    
    # Demonstrate weighted voting
    print("\nWeighted voting example:")
    weighted_results = weighted_voting_example(X, y)
    
    # Demonstrate with regression
    print("\nVoting Regressor Example:")
    try:
        from sklearn.datasets import load_boston
        X_reg, y_reg = load_boston(return_X_y=True)
        
        # Split data
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        # Custom implementation
        voting_reg = VotingRegressor(random_state=42)
        voting_reg.fit(X_train_reg, y_train_reg)
        y_pred_reg = voting_reg.predict(X_test_reg)
        mse_custom = mean_squared_error(y_test_reg, y_pred_reg)
        
        # Scikit-learn implementation
        try:
            from sklearn.ensemble import VotingRegressor as SklearnVotingRegressor
            sklearn_voting_reg = SklearnVotingRegressor(
                estimators=[
                    ('lr', LinearRegression()),
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                    ('svr', SVR())
                ]
            )
            sklearn_voting_reg.fit(X_train_reg, y_train_reg)
            y_pred_sklearn_reg = sklearn_voting_reg.predict(X_test_reg)
            mse_sklearn = mean_squared_error(y_test_reg, y_pred_sklearn_reg)
            
            print(f"Custom voting regressor MSE: {mse_custom:.2f}")
            print(f"Scikit-learn voting regressor MSE: {mse_sklearn:.2f}")
        except ImportError:
            print(f"Custom voting regressor MSE: {mse_custom:.2f}")
            print("Scikit-learn voting regressor not available")
        
    except ImportError:
        print("Boston housing dataset not available. Skipping regression example.")
    
    print("\nKey Points about Voting Methods:")
    print("• Hard voting uses predicted class labels for majority rule")
    print("• Soft voting averages predicted probabilities for final prediction")
    print("• Soft voting generally performs better when base estimators are well-calibrated")
    print("• Weighted voting allows emphasizing better performing estimators")
    print("• Voting combines diverse models to improve overall performance")