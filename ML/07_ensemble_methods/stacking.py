"""
Stacking and Blending Implementation
==============================

This module provides comprehensive implementations of stacking and blending
ensemble methods with detailed explanations and practical examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')


class StackingClassifier:
    """
    Stacking Classifier implementation.
    
    Parameters:
    -----------
    base_estimators : list, default=None
        List of base estimators. If None, uses default estimators.
    meta_estimator : object, default=None
        Meta estimator to combine base estimators. If None, uses LogisticRegression.
    cv : int, default=5
        Number of folds for cross-validation.
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes:
    -----------
    base_estimators : list
        Fitted base estimators.
    meta_estimator : object
        Fitted meta estimator.
    """
    
    def __init__(self, base_estimators=None, meta_estimator=None, cv=5, random_state=None):
        self.base_estimators = base_estimators or [
            RandomForestClassifier(n_estimators=50, random_state=random_state),
            SVC(probability=True, random_state=random_state),
            KNeighborsClassifier(n_neighbors=5)
        ]
        self.meta_estimator = meta_estimator or LogisticRegression(random_state=random_state)
        self.cv = cv
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit the stacking classifier.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : StackingClassifier
            Fitted estimator.
        """
        # Split data for stacking
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Fit base estimators on training set
        self.base_estimators_ = []
        for estimator in self.base_estimators:
            estimator_copy = type(estimator)(**estimator.get_params())
            estimator_copy.fit(X_train, y_train)
            self.base_estimators_.append(estimator_copy)
        
        # Generate predictions for blending set
        blend_predictions = []
        for estimator in self.base_estimators_:
            pred_proba = estimator.predict_proba(X_blend)
            blend_predictions.append(pred_proba)
        
        # Stack predictions as new features
        X_meta = np.hstack(blend_predictions)
        
        # Fit meta estimator
        self.meta_estimator.fit(X_meta, y_blend)
        
        # Refit base estimators on full dataset
        self.base_estimators_final = []
        for estimator in self.base_estimators:
            estimator_copy = type(estimator)(**estimator.get_params())
            estimator_copy.fit(X, y)
            self.base_estimators_final.append(estimator_copy)
        
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
        # Get predictions from base estimators
        base_predictions = []
        for estimator in self.base_estimators_final:
            pred_proba = estimator.predict_proba(X)
            base_predictions.append(pred_proba)
        
        # Stack predictions
        X_meta = np.hstack(base_predictions)
        
        # Predict with meta estimator
        return self.meta_estimator.predict(X_meta)
    
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
        # Get predictions from base estimators
        base_predictions = []
        for estimator in self.base_estimators_final:
            pred_proba = estimator.predict_proba(X)
            base_predictions.append(pred_proba)
        
        # Stack predictions
        X_meta = np.hstack(base_predictions)
        
        # Predict probabilities with meta estimator
        return self.meta_estimator.predict_proba(X_meta)


class StackingRegressor:
    """
    Stacking Regressor implementation.
    
    Parameters:
    -----------
    base_estimators : list, default=None
        List of base estimators. If None, uses default estimators.
    meta_estimator : object, default=None
        Meta estimator to combine base estimators. If None, uses LinearRegression.
    cv : int, default=5
        Number of folds for cross-validation.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, base_estimators=None, meta_estimator=None, cv=5, random_state=None):
        self.base_estimators = base_estimators or [
            RandomForestRegressor(n_estimators=50, random_state=random_state),
            SVR(),
            KNeighborsRegressor(n_neighbors=5)
        ]
        self.meta_estimator = meta_estimator or LinearRegression()
        self.cv = cv
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit the stacking regressor.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : StackingRegressor
            Fitted estimator.
        """
        # Split data for stacking
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Fit base estimators on training set
        self.base_estimators_ = []
        for estimator in self.base_estimators:
            estimator_copy = type(estimator)(**estimator.get_params())
            estimator_copy.fit(X_train, y_train)
            self.base_estimators_.append(estimator_copy)
        
        # Generate predictions for blending set
        blend_predictions = []
        for estimator in self.base_estimators_:
            pred = estimator.predict(X_blend).reshape(-1, 1)
            blend_predictions.append(pred)
        
        # Stack predictions as new features
        X_meta = np.hstack(blend_predictions)
        
        # Fit meta estimator
        self.meta_estimator.fit(X_meta, y_blend)
        
        # Refit base estimators on full dataset
        self.base_estimators_final = []
        for estimator in self.base_estimators:
            estimator_copy = type(estimator)(**estimator.get_params())
            estimator_copy.fit(X, y)
            self.base_estimators_final.append(estimator_copy)
        
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
        # Get predictions from base estimators
        base_predictions = []
        for estimator in self.base_estimators_final:
            pred = estimator.predict(X).reshape(-1, 1)
            base_predictions.append(pred)
        
        # Stack predictions
        X_meta = np.hstack(base_predictions)
        
        # Predict with meta estimator
        return self.meta_estimator.predict(X_meta)


class BlendingClassifier:
    """
    Blending Classifier implementation.
    
    Parameters:
    -----------
    base_estimators : list, default=None
        List of base estimators.
    meta_estimator : object, default=None
        Meta estimator to combine base estimators.
    test_size : float, default=0.2
        Proportion of dataset for blending.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, base_estimators=None, meta_estimator=None, test_size=0.2, random_state=None):
        self.base_estimators = base_estimators or [
            RandomForestClassifier(n_estimators=50, random_state=random_state),
            SVC(probability=True, random_state=random_state),
            KNeighborsClassifier(n_neighbors=5)
        ]
        self.meta_estimator = meta_estimator or LogisticRegression(random_state=random_state)
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit the blending classifier.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : BlendingClassifier
            Fitted estimator.
        """
        # Split data for blending
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Fit base estimators on training set
        self.base_estimators_ = []
        for estimator in self.base_estimators:
            estimator_copy = type(estimator)(**estimator.get_params())
            estimator_copy.fit(X_train, y_train)
            self.base_estimators_.append(estimator_copy)
        
        # Generate predictions for blending set
        blend_predictions = []
        for estimator in self.base_estimators_:
            pred_proba = estimator.predict_proba(X_blend)
            blend_predictions.append(pred_proba)
        
        # Stack predictions as new features
        X_meta = np.hstack(blend_predictions)
        
        # Fit meta estimator
        self.meta_estimator.fit(X_meta, y_blend)
        
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
        # Get predictions from base estimators
        base_predictions = []
        for estimator in self.base_estimators_:
            pred_proba = estimator.predict_proba(X)
            base_predictions.append(pred_proba)
        
        # Stack predictions
        X_meta = np.hstack(base_predictions)
        
        # Predict with meta estimator
        return self.meta_estimator.predict(X_meta)


def compare_stacking_implementations(X, y, task='classification'):
    """
    Compare custom stacking implementation with scikit-learn's implementation.
    
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
        custom_stacking = StackingClassifier(random_state=42)
        custom_stacking.fit(X_train, y_train)
        y_pred_custom = custom_stacking.predict(X_test)
        acc_custom = accuracy_score(y_test, y_pred_custom)
        
        # Scikit-learn implementation
        try:
            from sklearn.ensemble import StackingClassifier as SklearnStackingClassifier
            sklearn_stacking = SklearnStackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('svc', SVC(probability=True, random_state=42)),
                    ('knn', KNeighborsClassifier(n_neighbors=5))
                ],
                final_estimator=LogisticRegression(random_state=42)
            )
            sklearn_stacking.fit(X_train, y_train)
            y_pred_sklearn = sklearn_stacking.predict(X_test)
            acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
        except ImportError:
            sklearn_stacking = None
            acc_sklearn = 0
        
        results = {
            'custom': {
                'accuracy': acc_custom,
                'predictions': y_pred_custom
            },
            'sklearn': {
                'accuracy': acc_sklearn,
                'predictions': y_pred_sklearn if sklearn_stacking else None
            }
        }
    
    else:  # regression
        # Custom implementation
        custom_stacking = StackingRegressor(random_state=42)
        custom_stacking.fit(X_train, y_train)
        y_pred_custom = custom_stacking.predict(X_test)
        mse_custom = mean_squared_error(y_test, y_pred_custom)
        
        # Scikit-learn implementation
        try:
            from sklearn.ensemble import StackingRegressor as SklearnStackingRegressor
            sklearn_stacking = SklearnStackingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                    ('svr', SVR()),
                    ('knn', KNeighborsRegressor(n_neighbors=5))
                ],
                final_estimator=LinearRegression()
            )
            sklearn_stacking.fit(X_train, y_train)
            y_pred_sklearn = sklearn_stacking.predict(X_test)
            mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
        except ImportError:
            sklearn_stacking = None
            mse_sklearn = float('inf')
        
        results = {
            'custom': {
                'mse': mse_custom,
                'predictions': y_pred_custom
            },
            'sklearn': {
                'mse': mse_sklearn,
                'predictions': y_pred_sklearn if sklearn_stacking else None
            }
        }
    
    return results


def plot_ensemble_comparison(X, y, n_estimators_range=range(10, 101, 10)):
    """
    Compare performance of different ensemble methods.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    n_estimators_range : range, default=range(10, 101, 10)
        Range of n_estimators to test.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf_scores = []
    stack_scores = []
    
    for n_estimators in n_estimators_range:
        # Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_scores.append(accuracy_score(y_test, y_pred_rf))
        
        # Simple stacking (RF + SVM)
        rf_base = RandomForestClassifier(n_estimators=n_estimators//2, random_state=42)
        svm_base = SVC(probability=True, random_state=42)
        
        rf_base.fit(X_train, y_train)
        svm_base.fit(X_train, y_train)
        
        rf_pred = rf_base.predict_proba(X_test)
        svm_pred = svm_base.predict_proba(X_test)
        
        X_meta = np.hstack([rf_pred, svm_pred])
        meta = LogisticRegression(random_state=42)
        meta.fit(X_meta, y_train)
        
        y_pred_stack = meta.predict(X_meta)
        stack_scores.append(accuracy_score(y_test, y_pred_stack))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, rf_scores, 'b-', label='Random Forest')
    plt.plot(n_estimators_range, stack_scores, 'r-', label='Simple Stacking')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Ensemble Method Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data
    X, y = load_iris(return_X_y=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Apply Stacking Classifier
    stacking = StackingClassifier(random_state=42)
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("Stacking Classifier Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Compare implementations
    print("\nComparing implementations:")
    comparison_results = compare_stacking_implementations(X, y, task='classification')
    print(f"Custom stacking accuracy: {comparison_results['custom']['accuracy']:.4f}")
    if comparison_results['sklearn']['accuracy'] > 0:
        print(f"Scikit-learn stacking accuracy: {comparison_results['sklearn']['accuracy']:.4f}")
    else:
        print("Scikit-learn stacking not available")
    
    # Demonstrate with regression
    print("\nStacking Regressor Example:")
    try:
        from sklearn.datasets import load_boston
        X_reg, y_reg = load_boston(return_X_y=True)
        
        stacking_reg = StackingRegressor(random_state=42)
        stacking_reg.fit(X_train, y_train)
        y_pred_reg = stacking_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_reg)
        print(f"Stacking regressor MSE: {mse:.2f}")
        
    except ImportError:
        print("Boston housing dataset not available. Skipping regression example.")
    
    # Compare ensemble methods
    print("\nComparing ensemble methods...")
    plot_ensemble_comparison(X, y)
    
    # Demonstrate blending
    print("\nBlending Classifier Example:")
    blending = BlendingClassifier(random_state=42)
    blending.fit(X_train, y_train)
    y_pred_blend = blending.predict(X_test)
    acc_blend = accuracy_score(y_test, y_pred_blend)
    print(f"Blending classifier accuracy: {acc_blend:.4f}")
    
    print("\nKey Points about Stacking and Blending:")
    print("• Stacking uses cross-validation to generate meta-features, blending uses a holdout set")
    print("• Meta-estimator learns to combine base estimators optimally")
    print("• Stacking often provides better performance than individual models")
    print("• Requires careful validation to avoid overfitting")
    print("• Computational cost is higher than simple ensembles")