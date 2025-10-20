"""
Regression - Regularization Techniques
================================

This module covers regularization techniques for linear regression models,
including Ridge, Lasso, and Elastic Net regularization to prevent overfitting
and handle multicollinearity.

Topics Covered:
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net Regression (L1 + L2 regularization)
- Regularization parameter tuning
- Feature selection with Lasso
- Comparison of regularization methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class RegularizedRegression:
    """
    Comprehensive regularized regression implementation
    """
    
    def __init__(self, method='ridge', alpha=1.0, l1_ratio=0.5):
        """
        Initialize regularized regression model
        
        Args:
            method (str): Regularization method ('ridge', 'lasso', 'elastic_net')
            alpha (float): Regularization strength
            l1_ratio (float): L1 ratio for Elastic Net (0 = Ridge, 1 = Lasso)
        """
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Create model based on method
        if method == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif method == 'lasso':
            self.model = Lasso(alpha=alpha)
        elif method == 'elastic_net':
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        else:
            raise ValueError("Method must be 'ridge', 'lasso', or 'elastic_net'")
    
    def fit(self, X, y):
        """
        Fit the regularized regression model
        
        Args:
            X (array-like): Training features
            y (array-like): Target values
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X):
        """
        Make predictions using the fitted model
        
        Args:
            X (array-like): Test features
            
        Returns:
            array: Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features using same scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X, y):
        """
        Calculate R² score
        
        Args:
            X (array-like): Test features
            y (array-like): True values
            
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def get_coefficients(self):
        """
        Get model coefficients
        
        Returns:
            array: Model coefficients
        """
        return self.model.coef_
    
    def get_intercept(self):
        """
        Get model intercept
        
        Returns:
            float: Model intercept
        """
        return self.model.intercept_

def generate_multicollinear_data(n_samples=100, n_features=20, noise=0.1, random_state=42):
    """
    Generate data with multicollinearity for regularization demonstration
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        noise (float): Noise level
        random_state (int): Random seed
    
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    np.random.seed(random_state)
    
    # Create correlated features
    X = np.random.normal(0, 1, (n_samples, 5))  # 5 base features
    
    # Create correlated features from base features
    correlated_features = []
    for i in range(5):
        # Create 3 correlated features for each base feature
        for j in range(3):
            noise_feature = np.random.normal(0, 0.1, n_samples)
            correlated_feature = X[:, i] + noise_feature
            correlated_features.append(correlated_feature)
    
    # Add some independent features
    independent_features = np.random.normal(0, 1, (n_samples, 5))
    
    # Combine all features
    X = np.column_stack([X] + correlated_features + [independent_features])
    
    # Create true coefficients (some zero for feature selection)
    true_coef = np.array([2, -1.5, 0, 0.8, -0.5] +  # Base features
                        [1.2, -0.8, 0, 0.6, -0.4, 0, 0.9, -0.7, 0, 0.5, -0.3, 0, 0.4, -0.6, 0] +  # Correlated
                        [0, 0, 0, 0, 0])  # Independent (zero coefficients)
    
    # Generate target with noise
    y = np.dot(X, true_coef) + np.random.normal(0, noise, n_samples)
    
    return X, y

def compare_regularization_methods(X, y, alphas=np.logspace(-4, 2, 50)):
    """
    Compare different regularization methods
    
    Args:
        X (array-like): Feature matrix
        y (array-like): Target vector
        alphas (array): Regularization parameters to test
    
    Returns:
        dict: Results for each regularization method
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    methods = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Elastic Net': ElasticNet(l1_ratio=0.5)
    }
    
    results = {}
    
    for name, model in methods.items():
        if name == 'Linear Regression':
            # No regularization parameter for linear regression
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'best_alpha': None,
                'model': model
            }
        else:
            # Validation curve for regularization methods
            train_scores, test_scores = validation_curve(
                model, X_train, y_train, param_name='alpha', param_range=alphas,
                cv=5, scoring='r2'
            )
            
            # Calculate mean scores
            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            
            # Find best alpha
            best_idx = np.argmax(test_mean)
            best_alpha = alphas[best_idx]
            
            # Fit model with best alpha
            model.set_params(alpha=best_alpha)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'best_alpha': best_alpha,
                'train_scores': train_mean,
                'test_scores': test_mean,
                'model': model
            }
    
    return results, alphas

def regularization_demo():
    """
    Demonstrate regularization techniques
    """
    print("=== Regularization Techniques Demo ===")
    
    # Generate data with multicollinearity
    X, y = generate_multicollinear_data(n_samples=200, n_features=20, noise=0.5)
    
    print("1. Generated Data with Multicollinearity:")
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Feature correlation matrix determinant: {np.linalg.det(np.corrcoef(X.T)):.2e}")
    print(f"   (Low determinant indicates high multicollinearity)")
    
    # Compare regularization methods
    results, alphas = compare_regularization_methods(X, y)
    
    print(f"\n2. Model Comparison:")
    print(f"{'Method':<15} {'Best Alpha':<12} {'Train R²':<10} {'Test R²':<10} {'Train MSE':<12} {'Test MSE':<12}")
    print("-" * 75)
    for name, metrics in results.items():
        best_alpha = metrics['best_alpha'] if metrics['best_alpha'] is not None else "N/A"
        print(f"{name:<15} {str(best_alpha):<12} {metrics['train_r2']:<10.3f} {metrics['test_r2']:<10.3f} "
              f"{metrics['train_mse']:<12.2f} {metrics['test_mse']:<12.2f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Validation curves
    plt.subplot(2, 3, 1)
    for name in ['Ridge', 'Lasso', 'Elastic Net']:
        if name in results:
            plt.semilogx(alphas, results[name]['train_scores'], 
                        label=f'{name} Train', linestyle='--', alpha=0.7)
            plt.semilogx(alphas, results[name]['test_scores'], 
                        label=f'{name} Validation')
    
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('R² Score')
    plt.title('Validation Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient paths for Ridge
    plt.subplot(2, 3, 2)
    ridge_alphas = np.logspace(-4, 2, 100)
    coefs_ridge = []
    
    for alpha in ridge_alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        coefs_ridge.append(ridge.coef_)
    
    coefs_ridge = np.array(coefs_ridge)
    plt.semilogx(ridge_alphas, coefs_ridge)
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title('Ridge Coefficient Paths')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Coefficient paths for Lasso
    plt.subplot(2, 3, 3)
    lasso_alphas = np.logspace(-4, 0, 100)  # Different range for Lasso
    coefs_lasso = []
    
    for alpha in lasso_alphas:
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso.fit(X, y)
        coefs_lasso.append(lasso.coef_)
    
    coefs_lasso = np.array(coefs_lasso)
    plt.semilogx(lasso_alphas, coefs_lasso)
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso Coefficient Paths')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Feature selection with Lasso
    plt.subplot(2, 3, 4)
    # Get coefficients from best Lasso model
    lasso_coef = results['Lasso']['model'].coef_
    feature_indices = np.arange(len(lasso_coef))
    non_zero_coef = lasso_coef[lasso_coef != 0]
    non_zero_indices = feature_indices[lasso_coef != 0]
    
    plt.bar(range(len(non_zero_coef)), non_zero_coef)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title(f'Lasso Selected Features (n={len(non_zero_coef)})')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Coefficient comparison
    plt.subplot(2, 3, 5)
    methods_to_compare = ['Ridge', 'Lasso', 'Elastic Net']
    x_pos = np.arange(len(methods_to_compare))
    zero_coefficients = []
    
    for method in methods_to_compare:
        if method in results:
            coef = results[method]['model'].coef_
            zero_count = np.sum(np.abs(coef) < 1e-5)
            zero_coefficients.append(zero_count)
    
    plt.bar(x_pos, zero_coefficients, alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Number of Zero Coefficients')
    plt.title('Feature Selection Comparison')
    plt.xticks(x_pos, methods_to_compare)
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Model performance comparison
    plt.subplot(2, 3, 6)
    methods_names = list(results.keys())
    test_r2_scores = [results[name]['test_r2'] for name in methods_names]
    
    bars = plt.bar(range(len(methods_names)), test_r2_scores, alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Test R² Score')
    plt.title('Model Performance Comparison')
    plt.xticks(range(len(methods_names)), methods_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, test_r2_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def feature_selection_demo():
    """
    Demonstrate feature selection with Lasso regularization
    """
    print("\n=== Feature Selection with Lasso ===")
    
    # Generate data with many irrelevant features
    np.random.seed(42)
    n_samples, n_features = 200, 50
    
    # Create relevant features
    X_relevant = np.random.normal(0, 1, (n_samples, 10))
    
    # Create irrelevant features
    X_irrelevant = np.random.normal(0, 1, (n_samples, 40))
    
    # Combine features
    X = np.column_stack([X_relevant, X_irrelevant])
    
    # Create target with only relevant features
    true_coef = np.array([2, -1.5, 0, 0.8, -0.5, 1.2, -0.8, 0, 0.6, -0.4] + [0] * 40)
    y = np.dot(X, true_coef) + np.random.normal(0, 0.5, n_samples)
    
    print("1. Feature Selection Setup:")
    print(f"   Total features: {X.shape[1]}")
    print(f"   Relevant features: 10")
    print(f"   Irrelevant features: 40")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Compare feature selection performance
    alphas = np.logspace(-4, 1, 50)
    n_features_selected = []
    test_scores = []
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso.fit(X_train, y_train)
        
        # Count non-zero coefficients
        n_selected = np.sum(np.abs(lasso.coef_) > 1e-5)
        n_features_selected.append(n_selected)
        
        # Test score
        test_score = lasso.score(X_test, y_test)
        test_scores.append(test_score)
    
    # Find optimal alpha
    best_idx = np.argmax(test_scores)
    best_alpha = alphas[best_idx]
    best_n_features = n_features_selected[best_idx]
    
    # Fit final model
    final_lasso = Lasso(alpha=best_alpha, max_iter=2000)
    final_lasso.fit(X_train, y_train)
    
    # Identify selected features
    selected_features = np.where(np.abs(final_lasso.coef_) > 1e-5)[0]
    true_relevant_features = np.arange(10)  # First 10 features are relevant
    
    # Calculate selection accuracy
    correctly_selected = len(set(selected_features) & set(true_relevant_features))
    total_selected = len(selected_features)
    precision = correctly_selected / total_selected if total_selected > 0 else 0
    recall = correctly_selected / len(true_relevant_features)
    
    print(f"\n2. Feature Selection Results:")
    print(f"   Optimal Alpha: {best_alpha:.4f}")
    print(f"   Features selected: {best_n_features}")
    print(f"   Correctly selected relevant features: {correctly_selected}/10")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   Test R²: {test_scores[best_idx]:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Number of features vs Alpha
    plt.subplot(1, 3, 1)
    plt.semilogx(alphas, n_features_selected, 'o-', linewidth=2)
    plt.axvline(best_alpha, color='r', linestyle='--', alpha=0.7, label=f'Best α = {best_alpha:.4f}')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Number of Selected Features')
    plt.title('Feature Selection Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Performance vs Number of features
    plt.subplot(1, 3, 2)
    plt.plot(n_features_selected, test_scores, 'o-', linewidth=2)
    plt.axvline(best_n_features, color='r', linestyle='--', alpha=0.7, 
                label=f'Best = {best_n_features} features')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Test R² Score')
    plt.title('Performance vs Feature Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Coefficient magnitudes
    plt.subplot(1, 3, 3)
    coef_magnitudes = np.abs(final_lasso.coef_)
    feature_indices = np.arange(len(coef_magnitudes))
    
    # Color code relevant vs irrelevant features
    colors = ['red' if i < 10 else 'blue' for i in range(len(coef_magnitudes))]
    plt.bar(feature_indices, coef_magnitudes, color=colors, alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Magnitude')
    plt.title('Feature Coefficient Magnitudes\n(Red=Relevant, Blue=Irrelevant)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def real_world_example():
    """
    Real-world example: Housing price prediction with regularization
    """
    print("\n=== Real-World Example: Housing Price Prediction ===")
    
    # Create realistic housing dataset
    np.random.seed(42)
    n_houses = 1000
    
    # Features
    features = {
        'size': np.random.normal(2000, 500, n_houses),  # Square feet
        'bedrooms': np.random.randint(1, 6, n_houses),  # Number of bedrooms
        'bathrooms': np.random.randint(1, 4, n_houses),  # Number of bathrooms
        'age': np.random.randint(0, 50, n_houses),      # House age in years
        'location_score': np.random.uniform(1, 10, n_houses),  # Location quality score
        'garage_size': np.random.randint(0, 4, n_houses),  # Garage capacity
        'lot_size': np.random.normal(8000, 2000, n_houses),  # Lot size in sq ft
        'school_rating': np.random.uniform(1, 10, n_houses),  # School district rating
        'crime_rate': np.random.exponential(2, n_houses),  # Crime rate (lower is better)
        'proximity_to_city': np.random.uniform(0, 30, n_houses),  # Distance to city center
        # Add some correlated features
        'size_sqft': np.random.normal(2000, 500, n_houses),  # Correlated with size
        'bedrooms_alt': np.random.randint(1, 6, n_houses),  # Correlated with bedrooms
        'age_alt': np.random.randint(0, 50, n_houses),  # Correlated with age
        # Add some irrelevant features
        'random_feature_1': np.random.normal(0, 1, n_houses),
        'random_feature_2': np.random.uniform(0, 1, n_houses),
        'random_feature_3': np.random.exponential(1, n_houses)
    }
    
    # Create DataFrame
    df = pd.DataFrame(features)
    
    # Create realistic price based on features
    price = (
        df['size'] * 100 +                    # $100 per sq ft
        df['bedrooms'] * 10000 +              # $10,000 per bedroom
        df['bathrooms'] * 15000 +             # $15,000 per bathroom
        -df['age'] * 1000 +                   # -$1,000 per year of age
        df['location_score'] * 5000 +         # $5,000 per location point
        df['garage_size'] * 8000 +            # $8,000 per garage space
        df['lot_size'] * 2 +                  # $2 per sq ft of lot
        df['school_rating'] * 3000 +          # $3,000 per school rating point
        -df['crime_rate'] * 2000 +            # -$2,000 per crime unit
        -df['proximity_to_city'] * 1000 +     # -$1,000 per mile from city
        np.random.normal(0, 20000, n_houses)  # Random noise
    )
    
    # Ensure positive prices
    price = np.maximum(price, 50000)
    df['price'] = price
    
    print("Housing Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"Average price: ${df['price'].mean():,.0f}")
    
    # Prepare data
    feature_columns = [col for col in df.columns if col != 'price']
    X = df[feature_columns].values
    y = df['price'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compare models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=100),
        'Lasso': Lasso(alpha=1000, max_iter=2000),
        'Elastic Net': ElasticNet(alpha=500, l1_ratio=0.5, max_iter=2000)
    }
    
    results = {}
    for name, model in models.items():
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'model': pipeline,
            'coefficients': pipeline.named_steps['regressor'].coef_
        }
    
    print(f"\nModel Comparison:")
    print(f"{'Method':<15} {'Train R²':<10} {'Test R²':<10} {'Train MAE':<12} {'Test MAE':<12}")
    print("-" * 65)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['train_r2']:<10.3f} {metrics['test_r2']:<10.3f} "
              f"${metrics['train_mae']:<11,.0f} ${metrics['test_mae']:<11,.0f}")
    
    # Feature importance analysis
    print(f"\nFeature Selection Analysis:")
    lasso_coef = results['Lasso']['coefficients']
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'linear_coef': results['Linear Regression']['coefficients'],
        'ridge_coef': results['Ridge']['coefficients'],
        'lasso_coef': lasso_coef,
        'lasso_abs_coef': np.abs(lasso_coef)
    }).sort_values('lasso_abs_coef', ascending=False)
    
    print("Top 10 features by Lasso coefficient magnitude:")
    print(feature_importance.head(10)[['feature', 'linear_coef', 'ridge_coef', 'lasso_coef']])
    
    # Count zero coefficients (feature selection)
    zero_coef_count = np.sum(np.abs(lasso_coef) < 1e-5)
    print(f"\nLasso selected {len(feature_columns) - zero_coef_count} features out of {len(feature_columns)}")

def when_to_use_regularization():
    """
    Guidelines for when to use different regularization techniques
    """
    print("\n=== When to Use Different Regularization Techniques ===")
    
    guidelines = {
        "Ridge Regression (L2)": {
            "When to use": "When you have multicollinearity or many features with small effects",
            "Strengths": "Handles multicollinearity well, keeps all features, stable coefficients",
            "Weaknesses": "Doesn't perform feature selection, can include irrelevant features",
            "Best for": "When all features are relevant but correlated, when you want stable predictions"
        },
        "Lasso Regression (L1)": {
            "When to use": "When you want automatic feature selection, when many features are irrelevant",
            "Strengths": "Performs feature selection, creates sparse models, interpretable",
            "Weaknesses": "Can arbitrarily select one from correlated features, unstable with highly correlated features",
            "Best for": "When you suspect many features are irrelevant, when you want interpretable models"
        },
        "Elastic Net": {
            "When to use": "When you have many correlated features and want some feature selection",
            "Strengths": "Combines benefits of Ridge and Lasso, handles correlated features better than Lasso",
            "Weaknesses": "Has two hyperparameters to tune, more complex than Ridge or Lasso",
            "Best for": "When you have many correlated features and want feature selection, high-dimensional data"
        },
        "No Regularization": {
            "When to use": "When you have few features, no multicollinearity, and sufficient data",
            "Strengths": "Simple, interpretable coefficients, no hyperparameters to tune",
            "Weaknesses": "Prone to overfitting with many features, unstable with multicollinearity",
            "Best for": "Simple problems with few features, exploratory analysis"
        }
    }
    
    for method, info in guidelines.items():
        print(f"\n{method}:")
        for key, value in info.items():
            print(f"   {key}: {value}")

# Example usage and testing
if __name__ == "__main__":
    # Regularization demo
    regularization_demo()
    print("\n" + "="*50 + "\n")
    
    # Feature selection demo
    feature_selection_demo()
    print("\n" + "="*50 + "\n")
    
    # Real-world example
    real_world_example()
    print("\n" + "="*50 + "\n")
    
    # When to use guidelines
    when_to_use_regularization()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Ridge, Lasso, and Elastic Net regularization techniques")
    print("2. Comparison of regularization methods on multicollinear data")
    print("3. Feature selection capabilities of Lasso regularization")
    print("4. Real-world housing price prediction example")
    print("5. When to use different regularization techniques")
    print("6. Hyperparameter tuning for regularization")
    print("\nKey takeaways:")
    print("- Regularization prevents overfitting and handles multicollinearity")
    print("- Ridge keeps all features but shrinks coefficients")
    print("- Lasso performs feature selection by setting some coefficients to zero")
    print("- Elastic Net combines Ridge and Lasso benefits")
    print("- Always scale features when using regularization")
    print("- Use cross-validation to select optimal regularization parameters")