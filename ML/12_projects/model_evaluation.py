"""
Comprehensive Model Evaluation
============================

This module provides comprehensive model evaluation techniques for machine learning projects.
It covers various metrics, visualization tools, and advanced evaluation methods.

Key Components:
- Classification metrics and visualization
- Regression metrics and visualization
- Cross-validation techniques
- Model comparison tools
- Advanced evaluation methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc,
                           precision_recall_curve, mean_squared_error, mean_absolute_error,
                           r2_score, silhouette_score)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    
    Parameters:
    -----------
    problem_type : str, default='classification'
        Type of ML problem ('classification', 'regression', 'clustering')
    """
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.metrics = {}
        self.plots = {}
        
        print(f"ModelEvaluator initialized for {problem_type} problems")
    
    def evaluate_classification(self, y_true, y_pred, y_prob=None, class_names=None):
        """
        Evaluate classification model performance.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_prob : array-like, optional
            Predicted probabilities
        class_names : list, optional
            Names of classes
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Per-class metrics
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        self.metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = cm
        
        # ROC and AUC (if probabilities provided)
        if y_prob is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            self.metrics['roc_auc'] = roc_auc
            self.metrics['roc_curve'] = (fpr, tpr)
            
            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            self.metrics['pr_curve'] = (precision_curve, recall_curve)
        
        # Print results
        print("Classification Evaluation Results:")
        print("=" * 35)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"ROC AUC:   {roc_auc:.4f}")
        
        return self.metrics
    
    def evaluate_regression(self, y_true, y_pred):
        """
        Evaluate regression model performance.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape
        }
        
        # Print results
        print("Regression Evaluation Results:")
        print("=" * 30)
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return self.metrics
    
    def evaluate_clustering(self, X, labels):
        """
        Evaluate clustering model performance.
        
        Parameters:
        -----------
        X : array-like
            Data points
        labels : array-like
            Cluster labels
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Silhouette score
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            self.metrics['silhouette_score'] = silhouette
            print(f"Silhouette Score: {silhouette:.4f}")
        else:
            print("Cannot compute silhouette score with only one cluster")
        
        # Number of clusters
        n_clusters = len(np.unique(labels))
        self.metrics['n_clusters'] = n_clusters
        print(f"Number of clusters: {n_clusters}")
        
        return self.metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                            normalize=False, title='Confusion Matrix'):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, optional
            Names of classes
        normalize : bool, default=False
            Whether to normalize confusion matrix
        title : str, default='Confusion Matrix'
            Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(cm))]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=class_names, yticklabels=class_names,
                   cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        self.plots['confusion_matrix'] = plt.gcf()
    
    def plot_roc_curve(self, y_true, y_prob, title='ROC Curve'):
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_prob : array-like
            Predicted probabilities
        title : str, default='ROC Curve'
            Plot title
        """
        if len(np.unique(y_true)) != 2:
            print("ROC curve is only available for binary classification")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        self.plots['roc_curve'] = plt.gcf()
    
    def plot_precision_recall_curve(self, y_true, y_prob, title='Precision-Recall Curve'):
        """
        Plot Precision-Recall curve.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_prob : array-like
            Predicted probabilities
        title : str, default='Precision-Recall Curve'
            Plot title
        """
        if len(np.unique(y_true)) != 2:
            print("Precision-Recall curve is only available for binary classification")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        self.plots['pr_curve'] = plt.gcf()
    
    def plot_residuals(self, y_true, y_pred, title='Residuals Plot'):
        """
        Plot residuals for regression models.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        title : str, default='Residuals Plot'
            Plot title
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 4))
        
        # Residuals vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.grid(True)
        
        # Residuals distribution
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        self.plots['residuals'] = plt.gcf()
    
    def cross_validate(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        X : array-like
            Features
        y : array-like
            Target
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='accuracy'
            Scoring metric
            
        Returns:
        --------
        cv_scores : array
            Cross-validation scores
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        print(f"Cross-Validation Results ({cv}-fold):")
        print("=" * 35)
        print(f"Scores: {cv_scores}")
        print(f"Mean:   {cv_scores.mean():.4f}")
        print(f"Std:    {cv_scores.std():.4f}")
        print(f"95% CI: [{cv_scores.mean() - 2*cv_scores.std():.4f}, "
              f"{cv_scores.mean() + 2*cv_scores.std():.4f}]")
        
        self.metrics['cv_scores'] = cv_scores
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()
        
        return cv_scores
    
    def plot_learning_curve(self, model, X, y, cv=5, train_sizes=None,
                          title='Learning Curve'):
        """
        Plot learning curve.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        X : array-like
            Features
        y : array-like
            Target
        cv : int, default=5
            Number of cross-validation folds
        train_sizes : array-like, optional
            Training sizes to evaluate
        title : str, default='Learning Curve'
            Plot title
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        self.plots['learning_curve'] = plt.gcf()
    
    def plot_validation_curve(self, model, X, y, param_name, param_range,
                            cv=5, title='Validation Curve'):
        """
        Plot validation curve.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        X : array-like
            Features
        y : array-like
            Target
        param_name : str
            Parameter name to vary
        param_range : array-like
            Parameter values to test
        cv : int, default=5
            Number of cross-validation folds
        title : str, default='Validation Curve'
            Plot title
        """
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='red')
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        self.plots['validation_curve'] = plt.gcf()


class ModelComparator:
    """
    Compare multiple models using various metrics.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to compare {name: model}
    """
    
    def __init__(self, models):
        self.models = models
        self.results = {}
        
        print(f"ModelComparator initialized with {len(models)} models")
        for name in models.keys():
            print(f"  - {name}")
    
    def compare_models(self, X_train, X_test, y_train, y_test, 
                      problem_type='classification'):
        """
        Compare multiple models.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        X_test : array-like
            Test features
        y_train : array-like
            Training target
        y_test : array-like
            Test target
        problem_type : str, default='classification'
            Type of problem
            
        Returns:
        --------
        comparison_df : pandas.DataFrame
            DataFrame with model comparison results
        """
        results = []
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            if problem_type == 'classification':
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })
                
            else:  # Regression
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2-Score': r2
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Sort by primary metric
        if problem_type == 'classification':
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        else:
            comparison_df = comparison_df.sort_values('R2-Score', ascending=False)
        
        print("\nModel Comparison Results:")
        print("=" * 50)
        print(comparison_df.to_string(index=False))
        
        self.results = comparison_df
        return comparison_df
    
    def plot_comparison(self, metric='Accuracy'):
        """
        Plot model comparison.
        
        Parameters:
        -----------
        metric : str, default='Accuracy'
            Metric to plot
        """
        if self.results.empty:
            print("No results to plot. Run compare_models first.")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.results['Model'], self.results[metric],
                      color=plt.cm.viridis(np.linspace(0, 1, len(self.results))))
        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data for demonstration
    print("Comprehensive Model Evaluation Demonstration")
    print("=" * 45)
    
    # Generate sample classification data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Classification data
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=3, random_state=42
    )
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # Train sample model
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_clf, y_train_clf)
    y_pred_clf = clf_model.predict(X_test_clf)
    y_prob_clf = clf_model.predict_proba(X_test_clf)[:, 1] if len(np.unique(y_clf)) == 2 else None
    
    print("Classification Model Evaluation:")
    print("-" * 30)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(problem_type='classification')
    
    # Evaluate classification model
    clf_metrics = evaluator.evaluate_classification(
        y_test_clf, y_pred_clf, y_prob_clf,
        class_names=['Class 0', 'Class 1', 'Class 2']
    )
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(y_test_clf, y_pred_clf,
                                  class_names=['Class 0', 'Class 1', 'Class 2'])
    
    # Generate sample regression data
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Train sample regression model
    reg_model = LinearRegression()
    reg_model.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg_model.predict(X_test_reg)
    
    print("\nRegression Model Evaluation:")
    print("-" * 25)
    
    # Initialize regression evaluator
    reg_evaluator = ModelEvaluator(problem_type='regression')
    
    # Evaluate regression model
    reg_metrics = reg_evaluator.evaluate_regression(y_test_reg, y_pred_reg)
    
    # Plot residuals
    reg_evaluator.plot_residuals(y_test_reg, y_pred_reg)
    
    # Cross-validation demonstration
    print("\nCross-Validation:")
    print("-" * 15)
    cv_scores = evaluator.cross_validate(clf_model, X_clf, y_clf, cv=5)
    
    # Model comparison demonstration
    print("\nModel Comparison:")
    print("-" * 15)
    
    # Create multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Initialize comparator
    comparator = ModelComparator(models)
    
    # Compare models
    comparison_results = comparator.compare_models(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf
    )
    
    # Plot comparison
    comparator.plot_comparison('Accuracy')
    
    # Advanced evaluation techniques summary
    print("\n" + "="*50)
    print("Advanced Evaluation Techniques Summary")
    print("="*50)
    print("1. Classification Evaluation:")
    print("   - Accuracy, Precision, Recall, F1-Score")
    print("   - Confusion Matrix visualization")
    print("   - ROC Curve and AUC")
    print("   - Precision-Recall Curve")
    print("   - Classification Report")
    
    print("\n2. Regression Evaluation:")
    print("   - MSE, RMSE, MAE, R²")
    print("   - Residuals analysis")
    print("   - Prediction vs Actual plots")
    print("   - Cross-validation scores")
    
    print("\n3. Clustering Evaluation:")
    print("   - Silhouette Score")
    print("   - Calinski-Harabasz Index")
    print("   - Davies-Bouldin Index")
    print("   - Elbow method")
    
    print("\n4. Model Comparison:")
    print("   - Multiple model evaluation")
    print("   - Statistical significance testing")
    print("   - Performance visualization")
    print("   - Trade-off analysis")
    
    print("\n5. Advanced Validation:")
    print("   - Learning curves")
    print("   - Validation curves")
    print("   - Bootstrap validation")
    print("   - Nested cross-validation")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Model Evaluation")
    print("="*50)
    print("1. Comprehensive Metrics:")
    print("   - Use multiple evaluation metrics")
    print("   - Consider domain-specific requirements")
    print("   - Balance different performance aspects")
    print("   - Validate on multiple datasets")
    
    print("\n2. Proper Validation:")
    print("   - Use separate test sets")
    print("   - Implement cross-validation")
    print("   - Avoid data leakage")
    print("   - Monitor for overfitting")
    
    print("\n3. Visualization:")
    print("   - Create informative plots")
    print("   - Show uncertainty and variance")
    print("   - Compare models visually")
    print("   - Document findings clearly")
    
    print("\n4. Reporting:")
    print("   - Provide detailed results")
    print("   - Include limitations and assumptions")
    print("   - Explain methodology clearly")
    print("   - Make results reproducible")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using:")
    print("- Scikit-learn: Comprehensive evaluation metrics")
    print("- Yellowbrick: Visual analysis and evaluation")
    print("- MLflow: Experiment tracking and comparison")
    print("- These provide enterprise-grade evaluation tools")