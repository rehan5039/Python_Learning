"""
Decision Trees and Random Forest Implementation

This module covers the implementation of Decision Trees and Random Forest algorithms:
- Decision tree construction with ID3/CART algorithms
- Tree pruning techniques
- Random Forest ensemble method
- Feature importance calculation
- Handling overfitting
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DecisionTreeNode:
    """
    Node class for Decision Tree.
    """
    
    def __init__(self, feature_index: Optional[int] = None, 
                 threshold: Optional[float] = None,
                 left: Optional['DecisionTreeNode'] = None,
                 right: Optional['DecisionTreeNode'] = None,
                 value: Optional[int] = None):
        """
        Initialize Decision Tree Node.
        
        Args:
            feature_index: Index of feature to split on
            threshold: Threshold value for splitting
            left: Left child node
            right: Right child node
            value: Class value for leaf nodes
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes


class DecisionTree:
    """
    Decision Tree classifier implementation from scratch.
    
    Uses CART (Classification and Regression Trees) algorithm for splitting.
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        """
        Initialize Decision Tree.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes = None
        self.n_features = None
        self.feature_importances_ = None
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity of a set of labels.
        
        Args:
            y: Array of labels
            
        Returns:
            Gini impurity value
        """
        if len(y) == 0:
            return 0
        
        # Count occurrences of each class
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calculate Gini impurity
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _information_gain(self, y: np.ndarray, left_y: np.ndarray, 
                         right_y: np.ndarray) -> float:
        """
        Calculate information gain from a split.
        
        Args:
            y: Original labels
            left_y: Labels in left split
            right_y: Labels in right split
            
        Returns:
            Information gain
        """
        # Calculate weighted Gini impurity after split
        n = len(y)
        n_left, n_right = len(left_y), len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Weighted impurity after split
        weighted_impurity = (n_left / n) * self._gini_impurity(left_y) + \
                           (n_right / n) * self._gini_impurity(right_y)
        
        # Information gain
        info_gain = self._gini_impurity(y) - weighted_impurity
        return info_gain
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Find the best split for a dataset.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            Tuple of (best_feature_index, best_threshold)
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Try all features
        for feature_idx in range(X.shape[1]):
            # Get unique values for this feature
            feature_values = np.unique(X[:, feature_idx])
            
            # Try all possible thresholds
            for threshold in feature_values:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                # Skip if split doesn't create valid partitions
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                left_y = y[left_mask]
                right_y = y[right_mask]
                gain = self._information_gain(y, left_y, right_y)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, 
                   depth: int = 0) -> DecisionTreeNode:
        """
        Recursively build decision tree.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            depth: Current depth of the tree
            
        Returns:
            DecisionTreeNode
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # Create leaf node
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # If no valid split found, create leaf node
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Create child nodes
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Create internal node
        return DecisionTreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Train the decision tree.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """
        Predict class for a single sample.
        
        Args:
            x: Single sample (n_features,)
            node: Current tree node
            
        Returns:
            Predicted class
        """
        # If leaf node, return value
        if node.value is not None:
            return node.value
        
        # Navigate to appropriate child
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for multiple samples.
        
        Args:
            X: Samples (n_samples, n_features)
            
        Returns:
            Predicted classes (n_samples,)
        """
        predictions = []
        for x in X:
            pred = self._predict_sample(x, self.root)
            predictions.append(pred)
        return np.array(predictions)


class RandomForest:
    """
    Random Forest classifier implementation.
    
    Ensemble of decision trees with bagging and random feature selection.
    """
    
    def __init__(self, n_trees: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Optional[int] = None, random_state: Optional[int] = None):
        """
        Initialize Random Forest.
        
        Args:
            n_trees: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed for reproducibility
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        
        if random_state:
            np.random.seed(random_state)
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create bootstrap sample of data.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            Tuple of (X_bootstrap, y_bootstrap)
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _random_features(self, X: np.ndarray) -> np.ndarray:
        """
        Select random subset of features.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Array of selected feature indices
        """
        n_features = X.shape[1]
        if self.max_features is None:
            max_features = int(np.sqrt(n_features))
        else:
            max_features = min(self.max_features, n_features)
        
        return np.random.choice(n_features, max_features, replace=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """
        Train the random forest.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        self.trees = []
        
        for _ in range(self.n_trees):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # Select random features
            feature_indices = self._random_features(X)
            X_selected = X_bootstrap[:, feature_indices]
            
            # Train decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_selected, y_bootstrap)
            
            # Store tree and feature indices
            self.trees.append((tree, feature_indices))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes using ensemble voting.
        
        Args:
            X: Samples (n_samples, n_features)
            
        Returns:
            Predicted classes (n_samples,)
        """
        # Get predictions from all trees
        tree_predictions = []
        for tree, feature_indices in self.trees:
            X_selected = X[:, feature_indices]
            predictions = tree.predict(X_selected)
            tree_predictions.append(predictions)
        
        # Convert to array and transpose
        tree_predictions = np.array(tree_predictions).T
        
        # Majority voting
        final_predictions = []
        for sample_predictions in tree_predictions:
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)


def generate_sample_data(n_samples: int = 1000, n_features: int = 10, 
                        n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        
    Returns:
        Tuple of (X, y) features and labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=min(n_features, 5),
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y


def plot_decision_tree_boundary(model: DecisionTree, X: np.ndarray, y: np.ndarray):
    """
    Plot decision boundary for 2D decision tree classification.
    
    Args:
        model: Trained DecisionTree model
        X: Features (n_samples, 2)
        y: Labels (n_samples,)
    """
    if X.shape[1] != 2:
        print("Decision boundary plotting only supported for 2D features")
        return
    
    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Tree Decision Boundary')
    plt.show()


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray):
    """
    Compare Decision Tree and Random Forest performance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    """
    # Decision Tree
    dt = DecisionTree(max_depth=5)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    # Random Forest
    rf = RandomForest(n_trees=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print("Model Comparison:")
    print(f"  Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"  Random Forest Accuracy: {rf_accuracy:.4f}")
    
    return dt, rf


def demo():
    """Demonstrate Decision Tree and Random Forest implementation."""
    print("=== Decision Trees and Random Forest Demo ===\n")
    
    # Generate sample data
    print("1. Generating Sample Data:")
    X, y = generate_sample_data(n_samples=1000, n_features=5, n_classes=3)
    print(f"   Data shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compare models
    print("\n2. Model Comparison:")
    dt, rf = compare_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Detailed evaluation for Random Forest
    print("\n3. Random Forest Detailed Evaluation:")
    rf_pred = rf.predict(X_test_scaled)
    print(f"   Classification Report:")
    print(classification_report(y_test, rf_pred))
    
    # Test with 2D data for visualization
    print("\n4. Decision Tree Visualization (2D Data):")
    X_2d, y_2d = generate_sample_data(n_samples=300, n_features=2, n_classes=3)
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d, y_2d, test_size=0.2, random_state=42
    )
    
    # Scale 2D data
    scaler_2d = StandardScaler()
    X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler_2d.transform(X_test_2d)
    
    # Train decision tree for visualization
    dt_2d = DecisionTree(max_depth=4)
    dt_2d.fit(X_train_2d_scaled, y_train_2d)
    dt_2d_pred = dt_2d.predict(X_test_2d_scaled)
    dt_2d_accuracy = accuracy_score(y_test_2d, dt_2d_pred)
    
    print(f"   2D Decision Tree Accuracy: {dt_2d_accuracy:.4f}")
    plot_decision_tree_boundary(dt_2d, X_test_2d_scaled, y_test_2d)
    
    # Compare with sklearn
    print("\n5. Comparison with Scikit-learn:")
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # Scikit-learn Decision Tree
    sklearn_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    sklearn_dt.fit(X_train_scaled, y_train)
    sklearn_dt_pred = sklearn_dt.predict(X_test_scaled)
    sklearn_dt_accuracy = accuracy_score(y_test, sklearn_dt_pred)
    
    # Scikit-learn Random Forest
    sklearn_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    sklearn_rf.fit(X_train_scaled, y_train)
    sklearn_rf_pred = sklearn_rf.predict(X_test_scaled)
    sklearn_rf_accuracy = accuracy_score(y_test, sklearn_rf_pred)
    
    print(f"   Our Decision Tree vs Scikit-learn: {dt_accuracy:.4f} vs {sklearn_dt_accuracy:.4f}")
    print(f"   Our Random Forest vs Scikit-learn: {rf_accuracy:.4f} vs {sklearn_rf_accuracy:.4f}")


if __name__ == "__main__":
    demo()