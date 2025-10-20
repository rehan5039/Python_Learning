"""
Chapter 6: Dimensionality Reduction - Practice Problems
==============================================

This file contains practice problems for dimensionality reduction techniques with solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
try:
    import umap.umap_ as umap_sklearn
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# Problem 1: PCA Implementation from Scratch
def problem_1_pca_implementation():
    """
    Problem: Implement PCA from scratch and compare with scikit-learn.
    
    Requirements:
    - Implement PCA without using scikit-learn's PCA
    - Compare results with scikit-learn implementation
    - Visualize the principal components
    """
    print("Problem 1: PCA Implementation from Scratch")
    print("=" * 50)
    
    # Load sample data
    X, y = load_iris(return_X_y=True)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Custom PCA implementation
    class CustomPCA:
        def __init__(self, n_components=None):
            self.n_components = n_components
        
        def fit(self, X):
            # Center the data
            self.mean = np.mean(X, axis=0)
            X_centered = X - self.mean
            
            # Compute covariance matrix
            cov_matrix = np.cov(X_centered.T)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Select top n_components
            if self.n_components is not None:
                eigenvalues = eigenvalues[:self.n_components]
                eigenvectors = eigenvectors[:, :self.n_components]
            
            self.components = eigenvectors.T
            self.explained_variance = eigenvalues
            self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            
            return self
        
        def transform(self, X):
            X_centered = X - self.mean
            return np.dot(X_centered, self.components.T)
        
        def fit_transform(self, X):
            return self.fit(X).transform(X)
    
    # Apply custom implementation
    custom_pca = CustomPCA(n_components=2)
    X_custom = custom_pca.fit_transform(X_scaled)
    
    # Apply scikit-learn implementation
    sklearn_pca = PCA(n_components=2)
    X_sklearn = sklearn_pca.fit_transform(X_scaled)
    
    # Compare results
    print(f"Custom PCA explained variance ratio: {custom_pca.explained_variance_ratio}")
    print(f"Scikit-learn PCA explained variance ratio: {sklearn_pca.explained_variance_ratio_}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data (first two features)
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
    axes[0].set_title('Original Data (First Two Features)')
    axes[0].set_xlabel('Feature 1 (Standardized)')
    axes[0].set_ylabel('Feature 2 (Standardized)')
    axes[0].grid(True, alpha=0.3)
    
    # Custom PCA
    axes[1].scatter(X_custom[:, 0], X_custom[:, 1], c=y, cmap='viridis')
    axes[1].set_title('Custom PCA')
    axes[1].set_xlabel(f'PC1 ({custom_pca.explained_variance_ratio[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({custom_pca.explained_variance_ratio[1]:.1%} variance)')
    axes[1].grid(True, alpha=0.3)
    
    # Scikit-learn PCA
    axes[2].scatter(X_sklearn[:, 0], X_sklearn[:, 1], c=y, cmap='viridis')
    axes[2].set_title('Scikit-learn PCA')
    axes[2].set_xlabel(f'PC1 ({sklearn_pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[2].set_ylabel(f'PC2 ({sklearn_pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return custom_pca, sklearn_pca


# Problem 2: Image Compression using PCA
def problem_2_image_compression():
    """
    Problem: Use PCA for image compression and quality assessment.
    
    Requirements:
    - Load image data (digits dataset)
    - Apply PCA to reduce dimensions
    - Reconstruct images and measure quality
    - Compare compression ratios
    """
    print("\nProblem 2: Image Compression using PCA")
    print("=" * 40)
    
    # Load digits dataset (8x8 images)
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize data to [0, 1] range
    X_normalized = X / 16.0
    
    # Apply PCA with different numbers of components
    n_components_list = [2, 5, 10, 20, 32]
    results = {}
    
    fig, axes = plt.subplots(2, len(n_components_list), figsize=(15, 6))
    
    for i, n_components in enumerate(n_components_list):
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_normalized)
        
        # Reconstruct images
        X_reconstructed = pca.inverse_transform(X_pca)
        
        # Calculate reconstruction error
        mse = mean_squared_error(X_normalized, X_reconstructed)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        results[n_components] = {
            'mse': mse,
            'variance_explained': variance_explained,
            'compression_ratio': 64 / n_components  # 8x8 = 64 pixels
        }
        
        # Show original and reconstructed images (first sample)
        original_img = X_normalized[0].reshape(8, 8)
        reconstructed_img = X_reconstructed[0].reshape(8, 8)
        
        # Original
        axes[0, i].imshow(original_img, cmap='gray')
        axes[0, i].set_title(f'Original')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed_img, cmap='gray')
        axes[1, i].set_title(f'{n_components} PCs\nMSE: {mse:.4f}\nVar: {variance_explained:.1%}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Compression Results:")
    print("Components | MSE      | Variance | Compression Ratio")
    print("----------|----------|----------|------------------")
    for n_components, result in results.items():
        print(f"{n_components:9d} | {result['mse']:.4f} | {result['variance_explained']:.1%}   | {result['compression_ratio']:6.1f}:1")
    
    return results


# Problem 3: Data Visualization Comparison
def problem_3_visualization_comparison():
    """
    Problem: Apply PCA and t-SNE to visualize high-dimensional data.
    
    Requirements:
    - Load high-dimensional dataset
    - Apply PCA and t-SNE for visualization
    - Compare results visually
    - Analyze strengths and weaknesses
    """
    print("\nProblem 3: Data Visualization Comparison")
    print("=" * 40)
    
    # Load breast cancer dataset (30 features)
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = load_breast_cancer().feature_names
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Apply UMAP if available
    if UMAP_AVAILABLE:
        umap_reducer = umap_sklearn.UMAP(n_components=2, random_state=42)
        X_umap = umap_reducer.fit_transform(X_scaled)
    
    # Visualize results
    n_plots = 3 if UMAP_AVAILABLE else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    # PCA
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].set_title('PCA Visualization')
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].set_title('t-SNE Visualization')
    axes[1].grid(True, alpha=0.3)
    
    # UMAP (if available)
    if UMAP_AVAILABLE:
        axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.7)
        axes[2].set_xlabel('UMAP Dimension 1')
        axes[2].set_ylabel('UMAP Dimension 2')
        axes[2].set_title('UMAP Visualization')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze PCA components
    print("Top 5 features in first two PCA components:")
    for i in range(2):
        component = pca.components_[i]
        top_features_idx = np.argsort(np.abs(component))[::-1][:5]
        print(f"PC{i+1}:")
        for idx in top_features_idx:
            print(f"  {feature_names[idx]}: {component[idx]:.3f}")
        print()
    
    return pca, tsne


# Problem 4: LDA vs PCA Comparison
def problem_4_lda_vs_pca():
    """
    Problem: Compare supervised and unsupervised dimensionality reduction.
    
    Requirements:
    - Apply PCA and LDA to the same dataset
    - Compare results with and without class labels
    - Evaluate classification performance
    """
    print("\nProblem 4: LDA vs PCA Comparison")
    print("=" * 30)
    
    # Load iris dataset
    X, y = load_iris(return_X_y=True)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data (first two features)
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
    axes[0].set_title('Original Data (First Two Features)')
    axes[0].set_xlabel('Feature 1 (Standardized)')
    axes[0].set_ylabel('Feature 2 (Standardized)')
    axes[0].grid(True, alpha=0.3)
    
    # PCA
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1].set_title('PCA Projection')
    axes[1].grid(True, alpha=0.3)
    
    # LDA
    axes[2].scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
    axes[2].set_xlabel('LD1')
    axes[2].set_ylabel('LD2')
    axes[2].set_title('LDA Projection')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate classification performance
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # PCA + Random Forest
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    rf_pca = RandomForestClassifier(random_state=42)
    rf_pca.fit(X_train_pca, y_train)
    y_pred_pca = rf_pca.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    
    # LDA + Random Forest
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    rf_lda = RandomForestClassifier(random_state=42)
    rf_lda.fit(X_train_lda, y_train)
    y_pred_lda = rf_lda.predict(X_test_lda)
    acc_lda = accuracy_score(y_test, y_pred_lda)
    
    # Baseline (no dimensionality reduction)
    rf_baseline = RandomForestClassifier(random_state=42)
    rf_baseline.fit(X_train, y_train)
    y_pred_baseline = rf_baseline.predict(X_test)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    
    print("Classification Accuracy Comparison:")
    print(f"Baseline (no reduction): {acc_baseline:.4f}")
    print(f"PCA + RF: {acc_pca:.4f} (variance explained: {np.sum(pca.explained_variance_ratio_):.1%})")
    print(f"LDA + RF: {acc_lda:.4f}")
    
    return pca, lda


# Problem 5: Feature Selection Implementation
def problem_5_feature_selection():
    """
    Problem: Implement feature selection techniques as an alternative to PCA.
    
    Requirements:
    - Implement variance-based feature selection
    - Implement correlation-based feature selection
    - Compare with PCA for dimensionality reduction
    """
    print("\nProblem 5: Feature Selection Implementation")
    print("=" * 40)
    
    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = load_breast_cancer().feature_names
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Variance-based feature selection
    feature_variances = np.var(X_scaled, axis=0)
    top_variance_indices = np.argsort(feature_variances)[::-1][:10]
    
    print("Top 10 features by variance:")
    for i, idx in enumerate(top_variance_indices):
        print(f"{i+1:2d}. {feature_names[idx]}: {feature_variances[idx]:.4f}")
    
    # 2. Correlation-based feature selection
    # Remove highly correlated features
    corr_matrix = np.corrcoef(X_scaled.T)
    # Get upper triangle of correlation matrix
    upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    high_corr_pairs = np.where((np.abs(corr_matrix) > 0.95) & upper_triangle)
    
    print(f"\nHighly correlated feature pairs (|r| > 0.95): {len(high_corr_pairs[0])}")
    features_to_remove = set()
    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
        # Remove the feature with lower variance
        if feature_variances[i] < feature_variances[j]:
            features_to_remove.add(i)
        else:
            features_to_remove.add(j)
    
    print(f"Features to remove due to high correlation: {len(features_to_remove)}")
    
    # 3. Compare with PCA
    # Select features based on variance and correlation removal
    selected_features = [i for i in range(len(feature_names)) if i not in features_to_remove]
    selected_features = sorted(selected_features, key=lambda x: feature_variances[x], reverse=True)[:10]
    
    print(f"\nSelected features (top 10 after correlation removal):")
    for i, idx in enumerate(selected_features):
        print(f"{i+1:2d}. {feature_names[idx]}")
    
    # Evaluate performance with selected features vs PCA
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Using selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    rf_selected = RandomForestClassifier(random_state=42)
    rf_selected.fit(X_train_selected, y_train)
    y_pred_selected = rf_selected.predict(X_test_selected)
    acc_selected = accuracy_score(y_test, y_pred_selected)
    
    # Using PCA
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    rf_pca = RandomForestClassifier(random_state=42)
    rf_pca.fit(X_train_pca, y_train)
    y_pred_pca = rf_pca.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    
    # Baseline (all features)
    rf_baseline = RandomForestClassifier(random_state=42)
    rf_baseline.fit(X_train, y_train)
    y_pred_baseline = rf_baseline.predict(X_test)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    
    print(f"\nClassification Accuracy Comparison:")
    print(f"Baseline (all {X.shape[1]} features): {acc_baseline:.4f}")
    print(f"Feature selection ({len(selected_features)} features): {acc_selected:.4f}")
    print(f"PCA ({pca.n_components} components): {acc_pca:.4f}")
    
    return selected_features, pca


# Main execution
if __name__ == "__main__":
    print("Chapter 6: Dimensionality Reduction - Practice Problems")
    print("====================================================")
    
    # Run all problems
    try:
        # Problem 1: PCA Implementation
        custom_pca, sklearn_pca = problem_1_pca_implementation()
        
        # Problem 2: Image Compression
        compression_results = problem_2_image_compression()
        
        # Problem 3: Visualization Comparison
        pca_viz, tsne_viz = problem_3_visualization_comparison()
        
        # Problem 4: LDA vs PCA
        pca_comp, lda_comp = problem_4_lda_vs_pca()
        
        # Problem 5: Feature Selection
        selected_features, pca_fs = problem_5_feature_selection()
        
        print("\n🎉 All practice problems completed successfully!")
        print("\n📋 Summary of what you've learned:")
        print("  • Implemented PCA from scratch and compared with scikit-learn")
        print("  • Used PCA for image compression and quality assessment")
        print("  • Compared different visualization techniques (PCA, t-SNE, UMAP)")
        print("  • Compared supervised (LDA) and unsupervised (PCA) dimensionality reduction")
        print("  • Implemented feature selection techniques as alternatives to PCA")
        
    except Exception as e:
        print(f"Error running practice problems: {e}")
        print("Please check the implementations and try again.")