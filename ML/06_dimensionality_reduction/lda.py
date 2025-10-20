"""
Linear Discriminant Analysis (LDA) Implementation
============================================

This module provides a comprehensive implementation of Linear Discriminant Analysis
for dimensionality reduction with supervised learning capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class LinearDiscriminantAnalysis:
    """
    Linear Discriminant Analysis implementation.
    
    Parameters:
    -----------
    n_components : int, default=None
        Number of components to keep. If None, n_components = min(n_classes-1, n_features).
    
    Attributes:
    -----------
    components : array, shape (n_components, n_features)
        Linear discriminant vectors.
    explained_variance_ratio : array, shape (n_components,)
        Percentage of variance explained by each component.
    means : array, shape (n_classes, n_features)
        Class means.
    priors : array, shape (n_classes,)
        Class priors.
    scalings : array, shape (n_features, n_components)
        Scaling of the features in the space spanned by the class centroids.
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
    
    def fit(self, X, y):
        """
        Fit the LDA model with X and y.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Get unique classes and their counts
        classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        
        # Set default n_components
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        else:
            self.n_components = min(self.n_components, n_classes - 1, n_features)
        
        # Calculate class priors
        self.priors = class_counts / n_samples
        self.classes = classes
        
        # Calculate class means
        self.means = np.array([np.mean(X[y == cls], axis=0) for cls in classes])
        
        # Calculate overall mean
        mean_overall = np.mean(X, axis=0)
        
        # Calculate within-class scatter matrix
        Sw = np.zeros((n_features, n_features))
        for i, cls in enumerate(classes):
            class_data = X[y == cls]
            class_mean = self.means[i]
            diff = class_data - class_mean
            Sw += np.dot(diff.T, diff)
        
        # Calculate between-class scatter matrix
        Sb = np.zeros((n_features, n_features))
        for i, cls in enumerate(classes):
            class_mean = self.means[i]
            diff = (class_mean - mean_overall).reshape(-1, 1)
            Sb += class_counts[i] * np.dot(diff, diff.T)
        
        # Solve the generalized eigenvalue problem: Sb * v = lambda * Sw * v
        # This is equivalent to solving Sw^(-1) * Sb * v = lambda * v
        try:
            # Regularized version to handle singular matrices
            Sw_reg = Sw + 1e-6 * np.eye(Sw.shape[0])
            eigenvalues, eigenvectors = np.linalg.eigh(np.dot(np.linalg.inv(Sw_reg), Sb))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            Sw_pinv = np.linalg.pinv(Sw)
            eigenvalues, eigenvectors = np.linalg.eigh(np.dot(Sw_pinv, Sb))
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance_ratio = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        self.scalings = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Project data to maximize class separation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.
            
        Returns:
        --------
        X_new : array, shape (n_samples, n_components)
            Transformed values.
        """
        return np.dot(X, self.scalings[:, :self.n_components])
    
    def fit_transform(self, X, y):
        """
        Fit the model with X and y and apply the dimensionality reduction on X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
            
        Returns:
        --------
        X_new : array, shape (n_samples, n_components)
            Transformed values.
        """
        return self.fit(X, y).transform(X)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict.
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels.
        """
        # Transform data
        X_transformed = self.transform(X)
        
        # Project class means
        means_transformed = np.dot(self.means, self.scalings[:, :self.n_components])
        
        # Calculate distances to class means
        predictions = []
        for sample in X_transformed:
            distances = [np.linalg.norm(sample - mean) for mean in means_transformed]
            predictions.append(self.classes[np.argmin(distances)])
        
        return np.array(predictions)


def compare_lda_implementations(X, y, n_components=2):
    """
    Compare custom LDA implementation with scikit-learn's implementation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to transform.
    y : array-like, shape (n_samples,)
        Target values.
    n_components : int, default=2
        Number of components.
        
    Returns:
    --------
    results : dict
        Dictionary containing results from both implementations.
    """
    # Custom implementation
    custom_lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_custom = custom_lda.fit_transform(X, y)
    y_pred_custom = custom_lda.predict(X)
    
    # Scikit-learn implementation
    sklearn_lda = SklearnLDA(n_components=n_components)
    X_sklearn = sklearn_lda.fit_transform(X, y)
    y_pred_sklearn = sklearn_lda.predict(X)
    
    # Calculate accuracies
    acc_custom = accuracy_score(y, y_pred_custom)
    acc_sklearn = accuracy_score(y, y_pred_sklearn)
    
    results = {
        'custom': {
            'transformed_data': X_custom,
            'predictions': y_pred_custom,
            'accuracy': acc_custom
        },
        'sklearn': {
            'transformed_data': X_sklearn,
            'predictions': y_pred_sklearn,
            'accuracy': acc_sklearn
        }
    }
    
    return results


def visualize_lda_2d(X, y, title="LDA Visualization"):
    """
    Visualize 2D LDA results with class labels.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to transform.
    y : array-like, shape (n_samples,)
        Class labels.
    title : str, default="LDA Visualization"
        Title for the plot.
    """
    # Apply LDA
    lda = SklearnLDA(n_components=2)
    X_lda = lda.fit_transform(X, y)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return lda


def plot_lda_decision_boundaries(X, y, title="LDA Decision Boundaries"):
    """
    Plot LDA decision boundaries for 2D data.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data (should be 2D for visualization).
    y : array-like, shape (n_samples,)
        Class labels.
    title : str, default="LDA Decision Boundaries"
        Title for the plot.
    """
    if X.shape[1] != 2:
        print("Decision boundary visualization only works for 2D data")
        return
    
    # Apply LDA
    lda = SklearnLDA()
    lda.fit(X, y)
    
    # Create a mesh to plot the decision boundaries
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on the mesh
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data
    X, y = load_iris(return_X_y=True)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)
    
    # Print results
    print("LDA Results:")
    print(f"Original dimensions: {X_scaled.shape}")
    print(f"Reduced dimensions: {X_lda.shape}")
    print(f"Class means shape: {lda.means.shape}")
    print(f"Explained variance ratio: {lda.explained_variance_ratio}")
    print(f"Total variance explained: {np.sum(lda.explained_variance_ratio):.1%}")
    
    # Visualize 2D projection
    visualize_lda_2d(X_scaled, y, "Iris Dataset LDA Projection")
    
    # Compare implementations
    print("\nComparing LDA implementations:")
    comparison_results = compare_lda_implementations(X_scaled, y, n_components=2)
    print(f"Custom LDA accuracy: {comparison_results['custom']['accuracy']:.4f}")
    print(f"Scikit-learn LDA accuracy: {comparison_results['sklearn']['accuracy']:.4f}")
    
    # Demonstrate with wine dataset
    print("\nWine Dataset Example:")
    X_wine, y_wine = load_wine(return_X_y=True)
    scaler_wine = StandardScaler()
    X_wine_scaled = scaler_wine.fit_transform(X_wine)
    
    lda_wine = LinearDiscriminantAnalysis()
    X_wine_lda = lda_wine.fit_transform(X_wine_scaled, y_wine)
    
    print(f"Wine dataset original dimensions: {X_wine_scaled.shape}")
    print(f"Wine dataset reduced dimensions: {X_wine_lda.shape}")
    print(f"Number of classes: {len(np.unique(y_wine))}")
    print(f"LDA components: {lda_wine.n_components}")
    
    # Predict and evaluate
    y_pred_wine = lda_wine.predict(X_wine_scaled)
    accuracy_wine = accuracy_score(y_wine, y_pred_wine)
    print(f"Wine dataset accuracy: {accuracy_wine:.4f}")
    
    # Visualize wine dataset
    visualize_lda_2d(X_wine_scaled, y_wine, "Wine Dataset LDA Projection")