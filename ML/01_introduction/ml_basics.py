"""
Machine Learning - Introduction
==============================

This module provides a basic introduction to machine learning concepts
using Python and scikit-learn, focusing on practical implementation.

Topics Covered:
- What is Machine Learning?
- Types of Machine Learning
- Basic workflow
- Simple examples
"""

# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def what_is_ml():
    """
    Explanation of Machine Learning concepts
    """
    print("=== What is Machine Learning? ===")
    print("Machine Learning is a subset of Artificial Intelligence that enables")
    print("computers to learn and make decisions from data without being explicitly programmed.")
    print()
    
    print("=== Types of Machine Learning ===")
    print("1. Supervised Learning: Learning with labeled data")
    print("   - Regression: Predict continuous values")
    print("   - Classification: Predict categories")
    print()
    print("2. Unsupervised Learning: Learning with unlabeled data")
    print("   - Clustering: Group similar data points")
    print("   - Dimensionality Reduction: Reduce data complexity")
    print()
    print("3. Reinforcement Learning: Learning through interaction with environment")
    print("   - Learn optimal actions through rewards/penalties")

def ml_workflow():
    """
    Basic Machine Learning workflow
    """
    print("=== Machine Learning Workflow ===")
    steps = [
        "1. Problem Definition",
        "2. Data Collection",
        "3. Data Preprocessing",
        "4. Feature Selection/Engineering",
        "5. Model Selection",
        "6. Model Training",
        "7. Model Evaluation",
        "8. Model Deployment",
        "9. Model Monitoring"
    ]
    
    for step in steps:
        print(step)

def simple_regression_example():
    """
    Simple Linear Regression example
    """
    print("\n=== Simple Linear Regression Example ===")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 1) * 10  # Feature
    y = 2 * X.squeeze() + 3 + np.random.randn(100) * 2  # Target with noise
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Coefficients: {model.coef_[0]:.2f}")
    print(f"Model Intercept: {model.intercept_:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual')
    plt.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, mse, r2

def data_science_integration():
    """
    Example of ML integration with data science tools
    """
    print("\n=== Data Science Integration Example ===")
    
    # Create sample dataset
    data = {
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.randn(1000)
    }
    
    df = pd.DataFrame(data)
    
    # Basic data exploration
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print(f"\nMissing Values: {df.isnull().sum().sum()}")
    
    # Correlation analysis
    print("\nFeature Correlations:")
    correlation_matrix = df.corr()
    print(correlation_matrix)
    
    return df

def preprocessing_example(df):
    """
    Basic data preprocessing example
    """
    print("\n=== Data Preprocessing Example ===")
    
    # Handle missing values (if any)
    df_cleaned = df.fillna(df.mean())
    
    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    
    features = ['feature1', 'feature2', 'feature3']
    scaler = StandardScaler()
    df_cleaned[features] = scaler.fit_transform(df_cleaned[features])
    
    print("Data after preprocessing:")
    print(df_cleaned.describe())
    
    return df_cleaned

# Example usage and testing
if __name__ == "__main__":
    # Introduction to ML concepts
    what_is_ml()
    print("\n" + "="*50 + "\n")
    
    # ML workflow
    ml_workflow()
    print("\n" + "="*50 + "\n")
    
    # Simple regression example
    model, mse, r2 = simple_regression_example()
    print("\n" + "="*50 + "\n")
    
    # Data science integration
    df = data_science_integration()
    print("\n" + "="*50 + "\n")
    
    # Preprocessing example
    df_processed = preprocessing_example(df)
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Basic ML concepts and workflow")
    print("2. Simple linear regression implementation")
    print("3. Integration with data science tools (pandas, numpy)")
    print("4. Basic data preprocessing techniques")
    print("\nIn the full ML course, you'll learn:")
    print("- Advanced algorithms (Random Forest, SVM, Neural Networks)")
    print("- Deep learning with TensorFlow/PyTorch")
    print("- Model evaluation and validation techniques")
    print("- Deployment and production considerations")