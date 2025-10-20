"""
Ensemble Evaluation and Comparison
============================

This module provides comprehensive tools for evaluating and comparing
ensemble methods with detailed metrics and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')


def evaluate_ensemble_diversity(X, y, estimators, cv=5):
    """
    Evaluate diversity among ensemble members.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    estimators : list
        List of fitted estimators.
    cv : int, default=5
        Number of cross-validation folds.
        
    Returns:
    --------
    diversity_metrics : dict
        Dictionary containing diversity metrics.
    """
    # Calculate pairwise correlations between predictions
    predictions = []
    for estimator in estimators:
        pred = estimator.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Pairwise correlation matrix
    correlation_matrix = np.corrcoef(predictions)
    
    # Average pairwise correlation
    n_estimators = len(estimators)
    sum_corr = np.sum(correlation_matrix) - n_estimators  # Subtract diagonal
    avg_correlation = sum_corr / (n_estimators * (n_estimators - 1))
    
    # Q-statistic (measure of disagreement)
    q_statistics = []
    for i in range(n_estimators):
        for j in range(i+1, n_estimators):
            pred_i = predictions[i]
            pred_j = predictions[j]
            
            # Calculate Q-statistic
            n00 = np.sum((pred_i != y) & (pred_j != y))
            n01 = np.sum((pred_i != y) & (pred_j == y))
            n10 = np.sum((pred_i == y) & (pred_j != y))
            n11 = np.sum((pred_i == y) & (pred_j == y))
            
            if (n00 + n01) * (n10 + n11) * (n00 + n10) * (n01 + n11) != 0:
                q = (n11 * n00 - n01 * n10) / ((n00 + n01) * (n10 + n11) * (n00 + n10) * (n01 + n11))**0.5
                q_statistics.append(q)
    
    avg_q_statistic = np.mean(q_statistics) if q_statistics else 0
    
    return {
        'correlation_matrix': correlation_matrix,
        'average_correlation': avg_correlation,
        'average_q_statistic': avg_q_statistic,
        'n_estimators': n_estimators
    }


def plot_ensemble_diversity(diversity_metrics, estimator_names=None):
    """
    Plot ensemble diversity metrics.
    
    Parameters:
    -----------
    diversity_metrics : dict
        Dictionary containing diversity metrics.
    estimator_names : list, default=None
        Names of estimators.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Correlation matrix heatmap
    sns.heatmap(diversity_metrics['correlation_matrix'], 
                annot=True, cmap='coolwarm', center=0, ax=axes[0])
    axes[0].set_title('Pairwise Correlation Matrix')
    if estimator_names:
        axes[0].set_xticklabels(estimator_names, rotation=45)
        axes[0].set_yticklabels(estimator_names, rotation=0)
    
    # Diversity metrics bar plot
    metrics = ['Average Correlation', 'Average Q-Statistic']
    values = [diversity_metrics['average_correlation'], diversity_metrics['average_q_statistic']]
    
    bars = axes[1].bar(metrics, values, color=['skyblue', 'lightcoral'])
    axes[1].set_ylabel('Value')
    axes[1].set_title('Ensemble Diversity Metrics')
    axes[1].set_ylim(-1, 1)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if value >= 0 else -0.1), 
                    f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top')
    
    plt.tight_layout()
    plt.show()


def compare_ensemble_methods(X, y, task='classification', cv=5):
    """
    Compare different ensemble methods.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    task : str, default='classification'
        Task type ('classification' or 'regression').
    cv : int, default=5
        Number of cross-validation folds.
        
    Returns:
    --------
    comparison_results : dict
        Dictionary containing comparison results.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if task == 'classification':
        # Define ensemble methods
        methods = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Voting (Hard)': None,  # Will create separately
            'Voting (Soft)': None   # Will create separately
        }
        
        # Create voting classifiers
        from sklearn.ensemble import VotingClassifier
        methods['Voting (Hard)'] = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
            ],
            voting='hard'
        )
        
        methods['Voting (Soft)'] = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
            ],
            voting='soft'
        )
        
        # Evaluate methods
        results = {}
        for name, method in methods.items():
            # Cross-validation scores
            cv_scores = cross_val_score(method, X_train, y_train, cv=cv)
            
            # Fit and predict
            method.fit(X_train, y_train)
            y_pred = method.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'test_accuracy': accuracy,
                'model': method
            }
    
    else:  # regression
        # Define ensemble methods
        methods = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Voting': None  # Will create separately
        }
        
        # Create voting regressor
        from sklearn.ensemble import VotingRegressor
        methods['Voting'] = VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                ('ada', AdaBoostRegressor(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
            ]
        )
        
        # Evaluate methods
        results = {}
        for name, method in methods.items():
            # Cross-validation scores
            cv_scores = -cross_val_score(method, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            
            # Fit and predict
            method.fit(X_train, y_train)
            y_pred = method.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            
            results[name] = {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'test_mse': mse,
                'model': method
            }
    
    return results


def plot_ensemble_comparison(comparison_results, task='classification'):
    """
    Plot ensemble method comparison.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary containing comparison results.
    task : str, default='classification'
        Task type ('classification' or 'regression').
    """
    methods = list(comparison_results.keys())
    cv_means = [comparison_results[method]['cv_mean'] for method in methods]
    cv_stds = [comparison_results[method]['cv_std'] for method in methods]
    
    if task == 'classification':
        test_scores = [comparison_results[method]['test_accuracy'] for method in methods]
        ylabel = 'Accuracy'
        title = 'Ensemble Methods Comparison - Classification'
    else:
        test_scores = [comparison_results[method]['test_mse'] for method in methods]
        ylabel = 'MSE'
        title = 'Ensemble Methods Comparison - Regression'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    # Plot cross-validation scores
    bars1 = ax.bar(x_pos - width/2, cv_means, width, yerr=cv_stds, 
                   label='CV Score', capsize=5, alpha=0.8)
    
    # Plot test scores
    bars2 = ax.bar(x_pos + width/2, test_scores, width, 
                   label=f'Test {ylabel}', alpha=0.8)
    
    ax.set_xlabel('Methods')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def ensemble_bias_variance_analysis(X, y, ensemble_method, n_iterations=100):
    """
    Perform bias-variance analysis for ensemble method.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    ensemble_method : estimator
        Ensemble method to analyze.
    n_iterations : int, default=100
        Number of iterations for analysis.
        
    Returns:
    --------
    bias_variance : dict
        Dictionary containing bias-variance decomposition.
    """
    # Store predictions for each iteration
    all_predictions = []
    
    for i in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(len(X), len(X), replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        
        # Fit model
        model = type(ensemble_method)(**ensemble_method.get_params())
        model.fit(X_bootstrap, y_bootstrap)
        
        # Predict on all data
        predictions = model.predict(X)
        all_predictions.append(predictions)
    
    all_predictions = np.array(all_predictions)
    
    # Calculate mean prediction
    mean_predictions = np.mean(all_predictions, axis=0)
    
    # Calculate bias, variance, and noise
    bias_squared = np.mean((mean_predictions - y) ** 2)
    variance = np.mean(np.var(all_predictions, axis=0))
    noise = np.mean(np.var(all_predictions, axis=0))  # Simplified noise estimation
    
    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'noise': noise,
        'total_error': bias_squared + variance + noise,
        'predictions': all_predictions
    }


def plot_bias_variance_decomposition(bias_variance_results, method_names):
    """
    Plot bias-variance decomposition for different methods.
    
    Parameters:
    -----------
    bias_variance_results : dict
        Dictionary containing bias-variance results for each method.
    method_names : list
        Names of methods.
    """
    methods = list(bias_variance_results.keys())
    bias_squared = [bias_variance_results[method]['bias_squared'] for method in methods]
    variance = [bias_variance_results[method]['variance'] for method in methods]
    noise = [bias_variance_results[method]['noise'] for method in methods]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(methods))
    width = 0.5
    
    # Plot components
    ax.bar(x_pos, bias_squared, width, label='Bias²', color='skyblue')
    ax.bar(x_pos, variance, width, bottom=bias_squared, label='Variance', color='lightcoral')
    ax.bar(x_pos, noise, width, bottom=np.array(bias_squared) + np.array(variance), 
           label='Noise', color='lightgreen')
    
    ax.set_xlabel('Methods')
    ax.set_ylabel('Error Components')
    ax.set_title('Bias-Variance Decomposition')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data
    X, y = load_iris(return_X_y=True)
    
    # Compare ensemble methods
    print("Comparing ensemble methods...")
    comparison_results = compare_ensemble_methods(X, y, task='classification')
    
    print("Classification Results:")
    for method, results in comparison_results.items():
        print(f"{method}:")
        print(f"  CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    
    # Plot comparison
    plot_ensemble_comparison(comparison_results, task='classification')
    
    # Evaluate ensemble diversity
    print("\nEvaluating ensemble diversity...")
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X, y)
    
    diversity_metrics = evaluate_ensemble_diversity(X, y, rf.estimators_)
    print(f"Average correlation: {diversity_metrics['average_correlation']:.4f}")
    print(f"Average Q-statistic: {diversity_metrics['average_q_statistic']:.4f}")
    
    # Plot diversity
    plot_ensemble_diversity(diversity_metrics)
    
    # Demonstrate with regression
    print("\nRegression Example:")
    try:
        from sklearn.datasets import load_boston
        X_reg, y_reg = load_boston(return_X_y=True)
        
        comparison_results_reg = compare_ensemble_methods(X_reg, y_reg, task='regression')
        
        print("Regression Results:")
        for method, results in comparison_results_reg.items():
            print(f"{method}:")
            print(f"  CV MSE: {results['cv_mean']:.2f} (+/- {results['cv_std']*2:.2f})")
            print(f"  Test MSE: {results['test_mse']:.2f}")
        
        # Plot comparison
        plot_ensemble_comparison(comparison_results_reg, task='regression')
        
    except ImportError:
        print("Boston housing dataset not available. Skipping regression example.")
    
    print("\nKey Points about Ensemble Evaluation:")
    print("• Diversity among ensemble members is crucial for performance improvement")
    print("• Bias-variance tradeoff helps understand model behavior")
    print("• Cross-validation provides robust performance estimates")
    print("• Different ensemble methods have different strengths and weaknesses")
    print("• Proper evaluation helps select the best ensemble for your problem")