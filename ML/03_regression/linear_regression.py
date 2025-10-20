"""
Regression - Linear Regression
========================

This module covers linear regression techniques, from simple linear regression to
multiple linear regression, including implementation details, assumptions, and
practical applications.

Topics Covered:
- Simple Linear Regression
- Multiple Linear Regression
- Assumptions and diagnostics
- Model evaluation metrics
- Implementation from scratch
- Scikit-learn implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SimpleLinearRegression:
    """
    Simple Linear Regression implementation from scratch
    y = β₀ + β₁ * x
    """
    
    def __init__(self):
        self.beta_0 = 0  # Intercept
        self.beta_1 = 0  # Slope
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the simple linear regression model
        
        Args:
            X (array-like): Independent variable (1D array)
            y (array-like): Dependent variable (1D array)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope (β₁)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        self.beta_1 = numerator / denominator
        
        # Calculate intercept (β₀)
        self.beta_0 = y_mean - self.beta_1 * x_mean
        
        self.is_fitted = True
    
    def predict(self, X):
        """
        Make predictions using the fitted model
        
        Args:
            X (array-like): Independent variable values
            
        Returns:
            array: Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        return self.beta_0 + self.beta_1 * X
    
    def score(self, X, y):
        """
        Calculate R² score
        
        Args:
            X (array-like): Independent variable
            y (array-like): Dependent variable
            
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        y = np.array(y)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)

class MultipleLinearRegression:
    """
    Multiple Linear Regression implementation from scratch
    y = β₀ + β₁*x₁ + β₂*x₂ + ... + βₙ*xₙ
    """
    
    def __init__(self):
        self.coefficients = None  # β coefficients
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the multiple linear regression model using normal equation
        
        Args:
            X (array-like): Independent variables (2D array)
            y (array-like): Dependent variable (1D array)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Add bias column (intercept term)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: β = (X^T * X)^(-1) * X^T * y
        try:
            XtX = np.dot(X_with_bias.T, X_with_bias)
            XtX_inv = np.linalg.inv(XtX)
            Xty = np.dot(X_with_bias.T, y)
            self.coefficients = np.dot(XtX_inv, Xty)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            self.coefficients = np.linalg.pinv(X_with_bias).dot(y)
        
        self.is_fitted = True
    
    def predict(self, X):
        """
        Make predictions using the fitted model
        
        Args:
            X (array-like): Independent variables
            
        Returns:
            array: Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        return np.dot(X_with_bias, self.coefficients)
    
    def score(self, X, y):
        """
        Calculate R² score
        
        Args:
            X (array-like): Independent variables
            y (array-like): Dependent variable
            
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        y = np.array(y)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def get_coefficients(self):
        """
        Get the fitted coefficients
        
        Returns:
            array: Coefficients [β₀, β₁, β₂, ...]
        """
        return self.coefficients

def linear_regression_assumptions_check(X, y, model):
    """
    Check linear regression assumptions
    
    Args:
        X (array-like): Independent variables
        y (array-like): Dependent variable
        model: Fitted regression model
    
    Returns:
        dict: Dictionary containing assumption check results
    """
    # Make predictions
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Check assumptions
    assumptions = {}
    
    # 1. Linearity
    assumptions['linearity'] = "Check scatter plots of residuals vs fitted values"
    
    # 2. Independence
    assumptions['independence'] = "Ensure observations are independent"
    
    # 3. Homoscedasticity (constant variance)
    assumptions['homoscedasticity'] = "Check for constant variance in residuals"
    
    # 4. Normality of residuals
    assumptions['normality'] = "Check histogram/Q-Q plot of residuals"
    
    # 5. No multicollinearity (for multiple regression)
    if X.shape[1] > 1:
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        max_corr = np.max(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])))
        assumptions['multicollinearity'] = f"Max correlation: {max_corr:.3f}"
    
    return assumptions

def linear_regression_demo():
    """
    Demonstrate linear regression techniques
    """
    print("=== Linear Regression Demo ===")
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Simple linear regression example
    print("1. Simple Linear Regression:")
    X_simple = np.random.normal(0, 1, n_samples)
    y_simple = 2 * X_simple + 1 + np.random.normal(0, 0.5, n_samples)  # y = 2x + 1 + noise
    
    # Fit custom implementation
    slr_custom = SimpleLinearRegression()
    slr_custom.fit(X_simple, y_simple)
    y_pred_simple = slr_custom.predict(X_simple)
    
    # Fit scikit-learn implementation
    slr_sklearn = LinearRegression()
    slr_sklearn.fit(X_simple.reshape(-1, 1), y_simple)
    y_pred_sklearn = slr_sklearn.predict(X_simple.reshape(-1, 1))
    
    print(f"   Custom Implementation:")
    print(f"     β₀ (intercept): {slr_custom.beta_0:.3f}")
    print(f"     β₁ (slope): {slr_custom.beta_1:.3f}")
    print(f"     R² score: {slr_custom.score(X_simple, y_simple):.3f}")
    
    print(f"   Scikit-learn Implementation:")
    print(f"     β₀ (intercept): {slr_sklearn.intercept_:.3f}")
    print(f"     β₁ (slope): {slr_sklearn.coef_[0]:.3f}")
    print(f"     R² score: {slr_sklearn.score(X_simple.reshape(-1, 1), y_simple):.3f}")
    
    # Multiple linear regression example
    print("\n2. Multiple Linear Regression:")
    X_multi = np.random.normal(0, 1, (n_samples, 3))
    y_multi = (1.5 * X_multi[:, 0] + 
               (-2.0) * X_multi[:, 1] + 
               0.5 * X_multi[:, 2] + 
               1.0 + 
               np.random.normal(0, 0.5, n_samples))
    
    # Fit custom implementation
    mlr_custom = MultipleLinearRegression()
    mlr_custom.fit(X_multi, y_multi)
    y_pred_multi = mlr_custom.predict(X_multi)
    
    # Fit scikit-learn implementation
    mlr_sklearn = LinearRegression()
    mlr_sklearn.fit(X_multi, y_multi)
    y_pred_sklearn_multi = mlr_sklearn.predict(X_multi)
    
    print(f"   Custom Implementation Coefficients:")
    for i, coef in enumerate(mlr_custom.get_coefficients()):
        print(f"     β{i}: {coef:.3f}")
    print(f"     R² score: {mlr_custom.score(X_multi, y_multi):.3f}")
    
    print(f"   Scikit-learn Implementation Coefficients:")
    print(f"     β₀: {mlr_sklearn.intercept_:.3f}")
    for i, coef in enumerate(mlr_sklearn.coef_):
        print(f"     β{i+1}: {coef:.3f}")
    print(f"     R² score: {mlr_sklearn.score(X_multi, y_multi):.3f}")

def real_world_example():
    """
    Real-world example: Predicting house prices
    """
    print("\n=== Real-World Example: House Price Prediction ===")
    
    # Create realistic house price dataset
    np.random.seed(42)
    n_houses = 1000
    
    # Features
    size = np.random.normal(2000, 500, n_houses)  # Square feet
    bedrooms = np.random.randint(1, 6, n_houses)  # Number of bedrooms
    age = np.random.randint(0, 50, n_houses)      # House age in years
    location_score = np.random.uniform(1, 10, n_houses)  # Location quality score
    
    # Create realistic price based on features
    price = (size * 100 +                    # $100 per sq ft
             bedrooms * 10000 +              # $10,000 per bedroom
             -age * 1000 +                   # -$1,000 per year of age
             location_score * 5000 +         # $5,000 per location point
             np.random.normal(0, 20000, n_houses))  # Random noise
    
    # Ensure positive prices
    price = np.maximum(price, 50000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'size': size,
        'bedrooms': bedrooms,
        'age': age,
        'location_score': location_score,
        'price': price
    })
    
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(df.describe())
    
    # Prepare data
    X = df[['size', 'bedrooms', 'age', 'location_score']]
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"Training MSE: ${train_mse:,.2f}")
    print(f"Testing MSE: ${test_mse:,.2f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Testing R²: {test_r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nFeature Importance (by coefficient magnitude):")
    print(feature_importance)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

def linear_regression_metrics():
    """
    Explain and demonstrate regression evaluation metrics
    """
    print("\n=== Regression Evaluation Metrics ===")
    
    # Create sample predictions
    np.random.seed(42)
    y_true = np.random.normal(100, 15, 100)
    y_pred = y_true + np.random.normal(0, 5, 100)  # Add some noise
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Custom R² calculation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_custom = 1 - (ss_res / ss_tot)
    
    print("Common Regression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Custom R² Calculation: {r2_custom:.3f}")
    
    # Metric explanations
    metrics_explanation = {
        "MSE": "Average of squared differences between actual and predicted values. Sensitive to outliers.",
        "RMSE": "Square root of MSE, in same units as target variable. More interpretable than MSE.",
        "MAE": "Average of absolute differences. Less sensitive to outliers than MSE.",
        "R²": "Proportion of variance explained by the model. Range: -∞ to 1. Higher is better."
    }
    
    print("\nMetric Explanations:")
    for metric, explanation in metrics_explanation.items():
        print(f"{metric}: {explanation}")

def when_to_use_linear_regression():
    """
    Guidelines for when to use linear regression
    """
    print("\n=== When to Use Linear Regression ===")
    
    guidelines = {
        "Simple Linear Regression": {
            "When to use": "When you have one independent variable and want to understand the relationship",
            "Assumptions": "Linear relationship, independence, homoscedasticity, normality of residuals",
            "Benefits": "Simple to understand and interpret, fast to train",
            "Drawbacks": "Limited to linear relationships, sensitive to outliers",
            "Best for": "Exploratory analysis, simple predictive models"
        },
        "Multiple Linear Regression": {
            "When to use": "When you have multiple independent variables",
            "Assumptions": "All simple linear regression assumptions plus no multicollinearity",
            "Benefits": "Can model complex relationships, provides feature importance",
            "Drawbacks": "Assumes linear relationships, can overfit with many features",
            "Best for": "When you believe relationships are linear and want interpretable coefficients"
        },
        "Regularized Linear Regression": {
            "When to use": "When you have many features or multicollinearity issues",
            "Assumptions": "Same as multiple regression but more robust to violations",
            "Benefits": "Prevents overfitting, handles multicollinearity",
            "Drawbacks": "Requires hyperparameter tuning, less interpretable coefficients",
            "Best for": "High-dimensional data, when you need to prevent overfitting"
        }
    }
    
    for method, info in guidelines.items():
        print(f"\n{method}:")
        for key, value in info.items():
            print(f"   {key}: {value}")

# Example usage and testing
if __name__ == "__main__":
    # Linear regression demo
    linear_regression_demo()
    print("\n" + "="*50 + "\n")
    
    # Real-world example
    real_world_example()
    print("\n" + "="*50 + "\n")
    
    # Regression metrics
    linear_regression_metrics()
    print("\n" + "="*50 + "\n")
    
    # When to use guidelines
    when_to_use_linear_regression()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Simple and multiple linear regression implementations")
    print("2. Comparison between custom and scikit-learn implementations")
    print("3. Real-world house price prediction example")
    print("4. Regression evaluation metrics and their interpretations")
    print("5. When to use different linear regression approaches")
    print("6. Model evaluation and validation techniques")
    print("\nKey takeaways:")
    print("- Linear regression is interpretable and fast but assumes linear relationships")
    print("- Always check assumptions and validate model performance")
    print("- Use regularization when dealing with many features or multicollinearity")
    print("- Scale features when using regularization or comparing coefficients")
    print("- Cross-validation helps assess model generalization")