"""
Regression - Polynomial Regression
============================

This module covers polynomial regression techniques, which extend linear regression
to model non-linear relationships by including polynomial terms of the features.

Topics Covered:
- Polynomial features generation
- Overfitting and underfitting in polynomial regression
- Model selection and validation
- Regularization for polynomial regression
- Implementation with scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_polynomial_data(n_samples=100, degree=3, noise=0.1, random_state=42):
    """
    Generate synthetic data with polynomial relationship
    
    Args:
        n_samples (int): Number of samples
        degree (int): Degree of polynomial relationship
        noise (float): Noise level
        random_state (int): Random seed
    
    Returns:
        tuple: (X, y) where X is feature matrix and y is target vector
    """
    np.random.seed(random_state)
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    
    # Generate polynomial relationship
    y = 0.5 * X.ravel() ** degree + 0.3 * X.ravel() ** (degree-1) - 0.2 * X.ravel() + 1
    
    # Add noise
    y += np.random.normal(0, noise, n_samples)
    
    return X, y

class PolynomialRegression:
    """
    Polynomial Regression implementation using scikit-learn pipeline
    """
    
    def __init__(self, degree=2, regularization=None, alpha=1.0):
        """
        Initialize Polynomial Regression model
        
        Args:
            degree (int): Degree of polynomial features
            regularization (str): Type of regularization ('ridge', 'lasso', None)
            alpha (float): Regularization strength
        """
        self.degree = degree
        self.regularization = regularization
        self.alpha = alpha
        
        # Create pipeline
        steps = [('poly', PolynomialFeatures(degree=degree, include_bias=False))]
        
        if regularization == 'ridge':
            from sklearn.linear_model import Ridge
            steps.append(('regressor', Ridge(alpha=alpha)))
        elif regularization == 'lasso':
            from sklearn.linear_model import Lasso
            steps.append(('regressor', Lasso(alpha=alpha)))
        else:
            steps.append(('regressor', LinearRegression()))
        
        self.pipeline = Pipeline(steps)
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the polynomial regression model
        
        Args:
            X (array-like): Training data
            y (array-like): Target values
        """
        self.pipeline.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X):
        """
        Make predictions using the fitted model
        
        Args:
            X (array-like): Test data
            
        Returns:
            array: Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.pipeline.predict(X)
    
    def score(self, X, y):
        """
        Calculate R² score
        
        Args:
            X (array-like): Test data
            y (array-like): True values
            
        Returns:
            float: R² score
        """
        return self.pipeline.score(X, y)
    
    def get_coefficients(self):
        """
        Get the coefficients of the linear model (after polynomial features)
        
        Returns:
            array: Model coefficients
        """
        return self.pipeline.named_steps['regressor'].coef_

def compare_polynomial_degrees(X, y, degrees=[1, 2, 3, 4, 5, 10]):
    """
    Compare polynomial regression models with different degrees
    
    Args:
        X (array-like): Feature matrix
        y (array-like): Target vector
        degrees (list): List of polynomial degrees to compare
    
    Returns:
        dict: Dictionary containing results for each degree
    """
    results = {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for degree in degrees:
        # Create polynomial regression model
        poly_reg = PolynomialRegression(degree=degree)
        poly_reg.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = poly_reg.predict(X_train)
        y_test_pred = poly_reg.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results[degree] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'model': poly_reg
        }
    
    return results

def polynomial_regression_demo():
    """
    Demonstrate polynomial regression techniques
    """
    print("=== Polynomial Regression Demo ===")
    
    # Generate synthetic data with cubic relationship
    X, y = generate_polynomial_data(n_samples=100, degree=3, noise=0.1)
    
    print("1. Generated Data with Cubic Relationship:")
    print(f"   Data shape: {X.shape}")
    print(f"   X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   y range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Compare different polynomial degrees
    print("\n2. Comparing Polynomial Degrees:")
    results = compare_polynomial_degrees(X, y, degrees=[1, 2, 3, 4, 5, 10])
    
    print(f"{'Degree':<8} {'Train MSE':<12} {'Test MSE':<12} {'Train R²':<10} {'Test R²':<10}")
    print("-" * 60)
    for degree, metrics in results.items():
        print(f"{degree:<8} {metrics['train_mse']:<12.3f} {metrics['test_mse']:<12.3f} "
              f"{metrics['train_r2']:<10.3f} {metrics['test_r2']:<10.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Data and fitted curves
    plt.subplot(1, 3, 1)
    plt.scatter(X, y, alpha=0.6, label='Data points')
    
    X_plot = np.linspace(0, 1, 300).reshape(-1, 1)
    for degree in [1, 3, 5]:
        model = results[degree]['model']
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'Degree {degree}')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Model complexity vs performance
    plt.subplot(1, 3, 2)
    degrees = list(results.keys())
    train_errors = [results[d]['train_mse'] for d in degrees]
    test_errors = [results[d]['test_mse'] for d in degrees]
    
    plt.plot(degrees, train_errors, 'o-', label='Training Error')
    plt.plot(degrees, test_errors, 'o-', label='Testing Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Complexity vs Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: R² scores
    plt.subplot(1, 3, 3)
    train_r2 = [results[d]['train_r2'] for d in degrees]
    test_r2 = [results[d]['test_r2'] for d in degrees]
    
    plt.plot(degrees, train_r2, 'o-', label='Training R²')
    plt.plot(degrees, test_r2, 'o-', label='Testing R²')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('Model Complexity vs R²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def regularization_demo():
    """
    Demonstrate regularization in polynomial regression
    """
    print("\n=== Regularization in Polynomial Regression ===")
    
    # Generate data with high-degree polynomial relationship and noise
    X, y = generate_polynomial_data(n_samples=50, degree=3, noise=0.3)
    
    print("1. Noisy Data with Cubic Relationship:")
    print(f"   Data points: {len(X)}")
    print(f"   Noise level: 0.3")
    
    # Compare regularized vs non-regularized models
    degrees = [10, 15]  # High-degree polynomials
    
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Non-regularized model
        model_no_reg = PolynomialRegression(degree=degree)
        model_no_reg.fit(X_train, y_train)
        
        # Ridge regularization
        model_ridge = PolynomialRegression(degree=degree, regularization='ridge', alpha=0.1)
        model_ridge.fit(X_train, y_train)
        
        # Plot results
        plt.subplot(2, 2, 2*i + 1)
        X_plot = np.linspace(0, 1, 300).reshape(-1, 1)
        
        plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
        plt.scatter(X_test, y_test, alpha=0.6, label='Test data')
        
        y_plot_no_reg = model_no_reg.predict(X_plot)
        y_plot_ridge = model_ridge.predict(X_plot)
        
        plt.plot(X_plot, y_plot_no_reg, label=f'No Regularization (degree {degree})')
        plt.plot(X_plot, y_plot_ridge, label=f'Ridge Regularization (degree {degree})')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Polynomial Degree {degree} - Regularization Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Performance comparison
        plt.subplot(2, 2, 2*i + 2)
        alphas = np.logspace(-6, 2, 50)
        train_scores, test_scores = validation_curve(
            Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                     ('ridge', Ridge())]),
            X_train, y_train, param_name='ridge__alpha', param_range=alphas,
            cv=5, scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.semilogx(alphas, train_mean, 'o-', label='Training score')
        plt.fill_between(alphas, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.semilogx(alphas, test_mean, 'o-', label='Cross-validation score')
        plt.fill_between(alphas, test_mean - test_std, test_mean + test_std, alpha=0.1)
        
        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel('R² Score')
        plt.title(f'Validation Curve - Degree {degree}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print("\n2. Performance Comparison:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for degree in degrees:
        # Non-regularized
        model_no_reg = PolynomialRegression(degree=degree)
        model_no_reg.fit(X_train, y_train)
        test_r2_no_reg = model_no_reg.score(X_test, y_test)
        
        # Ridge regularized
        model_ridge = PolynomialRegression(degree=degree, regularization='ridge', alpha=0.1)
        model_ridge.fit(X_train, y_train)
        test_r2_ridge = model_ridge.score(X_test, y_test)
        
        print(f"Degree {degree}:")
        print(f"   No Regularization R²: {test_r2_no_reg:.3f}")
        print(f"   Ridge Regularization R²: {test_r2_ridge:.3f}")

def real_world_example():
    """
    Real-world example: Modeling non-linear relationships in economic data
    """
    print("\n=== Real-World Example: Economic Growth Modeling ===")
    
    # Create realistic economic data
    np.random.seed(42)
    years = np.arange(1990, 2020)  # 30 years
    n_years = len(years)
    
    # Simulate GDP growth with non-linear pattern
    # Initial slow growth, then rapid growth, then stabilization
    growth_trend = 0.02 + 0.03 * np.sin(2 * np.pi * (years - 1990) / 15)  # Cyclical component
    gdp_base = 1000 * np.exp(np.cumsum(growth_trend))  # Exponential base
    
    # Add polynomial component for technological advancement effect
    tech_impact = 0.5 * ((years - 1990) / 10) ** 2  # Quadratic technological effect
    
    gdp = gdp_base * (1 + tech_impact / 100)
    
    # Add noise and economic shocks
    noise = np.random.normal(0, 0.02 * gdp, n_years)
    shocks = np.zeros(n_years)
    shock_years = [2001, 2008, 2020]  # Economic shocks
    for year in shock_years:
        if year >= 1990 and year < 2020:
            idx = year - 1990
            shocks[idx] = -0.05 * gdp[idx]  # 5% negative shock
    
    gdp = gdp + noise + shocks
    
    # Create DataFrame
    df = pd.DataFrame({
        'year': years,
        'years_since_start': years - 1990,
        'gdp': gdp
    })
    
    print("Economic Data Summary:")
    print(f"Period: {years[0]} - {years[-1]}")
    print(f"Initial GDP: ${gdp[0]:,.0f} billion")
    print(f"Final GDP: ${gdp[-1]:,.0f} billion")
    print(f"Total Growth: {((gdp[-1] / gdp[0]) - 1) * 100:.1f}%")
    
    # Prepare data for modeling
    X = df[['years_since_start']].values
    y = df['gdp'].values
    
    # Split data (use last 5 years for testing)
    split_idx = len(X) - 5
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Compare different models
    models = {
        'Linear': PolynomialRegression(degree=1),
        'Quadratic': PolynomialRegression(degree=2),
        'Cubic': PolynomialRegression(degree=3),
        'Quartic': PolynomialRegression(degree=4),
        'Ridge (degree 4)': PolynomialRegression(degree=4, regularization='ridge', alpha=1000)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'model': model
        }
    
    print(f"\nModel Comparison:")
    print(f"{'Model':<20} {'Train R²':<10} {'Test R²':<10} {'Train MSE':<15} {'Test MSE':<15}")
    print("-" * 75)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['train_r2']:<10.3f} {metrics['test_r2']:<10.3f} "
              f"{metrics['train_mse']:<15.0f} {metrics['test_mse']:<15.0f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Data and model fits
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, alpha=0.6, label='Training Data')
    plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Test Data')
    
    X_plot = np.linspace(0, 29, 300).reshape(-1, 1)
    for name in ['Linear', 'Quadratic', 'Ridge (degree 4)']:
        model = results[name]['model']
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('Years Since 1990')
    plt.ylabel('GDP (Billions $)')
    plt.title('Economic Growth Modeling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals for best model
    plt.subplot(1, 3, 2)
    best_model = results['Ridge (degree 4)']['model']
    y_pred_train = best_model.predict(X_train)
    residuals = y_train - y_pred_train
    
    plt.scatter(y_pred_train, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted GDP')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot - Ridge Model')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Growth rate analysis
    plt.subplot(1, 3, 3)
    growth_rates = np.diff(y) / y[:-1] * 100  # Percentage growth
    years_for_growth = years[1:]
    
    plt.plot(years_for_growth, growth_rates, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Year')
    plt.ylabel('GDP Growth Rate (%)')
    plt.title('Annual GDP Growth Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def overfitting_underfitting():
    """
    Explain and demonstrate overfitting and underfitting in polynomial regression
    """
    print("\n=== Overfitting and Underfitting ===")
    
    # Generate data
    X, y = generate_polynomial_data(n_samples=30, degree=3, noise=0.2)
    
    print("Demonstrating Overfitting and Underfitting:")
    print("- Underfitting: Model too simple to capture patterns")
    print("- Overfitting: Model too complex, captures noise instead of patterns")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Compare models
    degrees = [1, 3, 15]  # Underfit, correct fit, overfit
    models = {}
    
    plt.figure(figsize=(15, 5))
    
    for i, degree in enumerate(degrees):
        # Fit model
        model = PolynomialRegression(degree=degree)
        model.fit(X_train, y_train)
        models[degree] = model
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Plot
        plt.subplot(1, 3, i + 1)
        plt.scatter(X_train, y_train, alpha=0.6, label='Training Data')
        plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Test Data')
        
        X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, 'g-', linewidth=2, label=f'Polynomial (degree {degree})')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Degree {degree}\nTrain R²: {train_r2:.3f}, Test R²: {test_r2:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAnalysis:")
    print(f"Degree 1 (Underfitting): Simple model, low variance, high bias")
    print(f"Degree 3 (Good Fit): Balanced model, captures true relationship")
    print(f"Degree 15 (Overfitting): Complex model, high variance, low bias")

# Example usage and testing
if __name__ == "__main__":
    # Polynomial regression demo
    polynomial_regression_demo()
    print("\n" + "="*50 + "\n")
    
    # Regularization demo
    regularization_demo()
    print("\n" + "="*50 + "\n")
    
    # Real-world example
    real_world_example()
    print("\n" + "="*50 + "\n")
    
    # Overfitting/underfitting demonstration
    overfitting_underfitting()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Polynomial regression for modeling non-linear relationships")
    print("2. Comparison of different polynomial degrees")
    print("3. Regularization techniques to prevent overfitting")
    print("4. Real-world economic growth modeling example")
    print("5. Overfitting and underfitting concepts")
    print("6. Model selection and validation techniques")
    print("\nKey takeaways:")
    print("- Polynomial regression can model complex non-linear relationships")
    print("- Higher degree polynomials are prone to overfitting")
    print("- Regularization helps prevent overfitting in high-degree polynomials")
    print("- Always validate model performance on unseen data")
    print("- Use cross-validation for robust model selection")