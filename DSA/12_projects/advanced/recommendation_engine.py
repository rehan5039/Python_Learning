"""
Advanced Project: Recommendation Engine

This project implements a sophisticated recommendation engine that demonstrates
advanced machine learning algorithms and data structures in a real-world context.

Concepts covered:
- Collaborative filtering algorithms
- Matrix factorization techniques
- Similarity computation optimization
- Scalable recommendation systems
- Performance optimization and caching
- A/B testing and evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, Counter
import heapq
import time
import hashlib
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


class RecommendationEngine(ABC):
    """
    Abstract base class for recommendation engines.
    """
    
    @abstractmethod
    def train(self, ratings_data: pd.DataFrame) -> None:
        """
        Train the recommendation engine.
        
        Args:
            ratings_data: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of (item_id, score) tuples
        """
        pass
    
    @abstractmethod
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        pass


class CollaborativeFilteringEngine(RecommendationEngine):
    """
    Collaborative filtering recommendation engine using matrix factorization.
    """
    
    def __init__(self, n_factors: int = 50, n_epochs: int = 100, 
                 learning_rate: float = 0.01, reg_param: float = 0.01):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.user_ids = None
        self.item_ids = None
        self.user_id_to_idx = None
        self.item_id_to_idx = None
        self.ratings_matrix = None
    
    def train(self, ratings_data: pd.DataFrame) -> None:
        """
        Train collaborative filtering model using Stochastic Gradient Descent.
        
        Time Complexity: O(epochs * ratings * factors)
        Space Complexity: O(users * factors + items * factors)
        """
        # Create user and item mappings
        unique_users = ratings_data['user_id'].unique()
        unique_items = ratings_data['item_id'].unique()
        
        self.user_ids = unique_users
        self.item_ids = unique_items
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Create ratings matrix
        self.ratings_matrix = np.zeros((n_users, n_items))
        for _, row in ratings_data.iterrows():
            user_idx = self.user_id_to_idx[row['user_id']]
            item_idx = self.item_id_to_idx[row['item_id']]
            self.ratings_matrix[user_idx, item_idx] = row['rating']
        
        # Initialize parameters
        self.global_mean = np.mean(self.ratings_matrix[self.ratings_matrix > 0])
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        # SGD training
        user_indices, item_indices = np.where(self.ratings_matrix > 0)
        ratings = self.ratings_matrix[user_indices, item_indices]
        
        for epoch in range(self.n_epochs):
            for idx in range(len(ratings)):
                user_idx = user_indices[idx]
                item_idx = item_indices[idx]
                rating = ratings[idx]
                
                # Predict rating
                pred_rating = (self.global_mean + 
                             self.user_bias[user_idx] + 
                             self.item_bias[item_idx] + 
                             np.dot(self.user_factors[user_idx], 
                                   self.item_factors[item_idx]))
                
                # Calculate error
                error = rating - pred_rating
                
                # Update biases
                self.user_bias[user_idx] += self.learning_rate * (
                    error - self.reg_param * self.user_bias[user_idx])
                self.item_bias[item_idx] += self.learning_rate * (
                    error - self.reg_param * self.item_bias[item_idx])
                
                # Update factors
                user_factor_old = self.user_factors[user_idx].copy()
                self.user_factors[user_idx] += self.learning_rate * (
                    error * self.item_factors[item_idx] - 
                    self.reg_param * self.user_factors[user_idx])
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factor_old - 
                    self.reg_param * self.item_factors[item_idx])
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair."""
        if (user_id not in self.user_id_to_idx or 
            item_id not in self.item_id_to_idx):
            return self.global_mean
        
        user_idx = self.user_id_to_idx[user_id]
        item_idx = self.item_id_to_idx[item_id]
        
        pred = (self.global_mean + 
                self.user_bias[user_idx] + 
                self.item_bias[item_idx] + 
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        # Clamp to rating range (1-5)
        return max(1, min(5, pred))
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate recommendations for user."""
        if user_id not in self.user_id_to_idx:
            # Return popular items for new users
            return self._get_popular_items(n_recommendations)
        
        user_idx = self.user_id_to_idx[user_id]
        recommendations = []
        
        # Predict ratings for all items
        for item_id, item_idx in self.item_id_to_idx.items():
            # Skip items the user has already rated
            if self.ratings_matrix[user_idx, item_idx] > 0:
                continue
            
            pred_rating = self.predict_rating(user_id, item_id)
            recommendations.append((item_id, pred_rating))
        
        # Sort by predicted rating and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def _get_popular_items(self, n_items: int) -> List[Tuple[int, float]]:
        """Get popular items for new users."""
        if self.ratings_matrix is None:
            return []
        
        # Calculate item popularity (average rating)
        item_popularity = []
        for item_id, item_idx in self.item_id_to_idx.items():
            item_ratings = self.ratings_matrix[:, item_idx]
            rated_users = item_ratings[item_ratings > 0]
            if len(rated_users) > 0:
                avg_rating = np.mean(rated_users)
                item_popularity.append((item_id, avg_rating, len(rated_users)))
        
        # Sort by average rating and number of ratings
        item_popularity.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(item_id, rating) for item_id, rating, _ in item_popularity[:n_items]]


class ContentBasedEngine(RecommendationEngine):
    """
    Content-based recommendation engine using item features.
    """
    
    def __init__(self):
        self.item_features = None
        self.item_ids = None
        self.item_id_to_idx = None
        self.user_profiles = None
        self.user_id_to_idx = None
    
    def train(self, ratings_data: pd.DataFrame, item_features: pd.DataFrame) -> None:
        """
        Train content-based model.
        
        Time Complexity: O(items * features + users * ratings)
        Space Complexity: O(items * features + users * features)
        """
        # Store item features
        self.item_ids = item_features['item_id'].values
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        # Extract feature columns (excluding item_id)
        feature_columns = [col for col in item_features.columns if col != 'item_id']
        self.item_features = item_features[feature_columns].values
        
        # Normalize features
        norms = np.linalg.norm(self.item_features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.item_features = self.item_features / norms
        
        # Build user profiles
        unique_users = ratings_data['user_id'].unique()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        n_users = len(unique_users)
        n_features = self.item_features.shape[1]
        
        self.user_profiles = np.zeros((n_users, n_features))
        
        # Calculate weighted average of item features for each user
        for _, row in ratings_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            
            if user_id in self.user_id_to_idx and item_id in self.item_id_to_idx:
                user_idx = self.user_id_to_idx[user_id]
                item_idx = self.item_id_to_idx[item_id]
                self.user_profiles[user_idx] += rating * self.item_features[item_idx]
        
        # Normalize user profiles
        user_norms = np.linalg.norm(self.user_profiles, axis=1, keepdims=True)
        user_norms[user_norms == 0] = 1
        self.user_profiles = self.user_profiles / user_norms
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating based on cosine similarity."""
        if (user_id not in self.user_id_to_idx or 
            item_id not in self.item_id_to_idx):
            return 3.0  # Default rating
        
        user_idx = self.user_id_to_idx[user_id]
        item_idx = self.item_id_to_idx[item_id]
        
        # Calculate cosine similarity
        user_profile = self.user_profiles[user_idx]
        item_feature = self.item_features[item_idx]
        similarity = np.dot(user_profile, item_feature)
        
        # Scale to rating range (1-5)
        return max(1, min(5, 3 + 2 * similarity))
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate content-based recommendations."""
        if user_id not in self.user_id_to_idx:
            return []
        
        user_idx = self.user_id_to_idx[user_id]
        user_profile = self.user_profiles[user_idx]
        
        # Calculate similarity with all items
        similarities = cosine_similarity([user_profile], self.item_features)[0]
        
        # Create item-similarity pairs
        recommendations = []
        for item_id, item_idx in self.item_id_to_idx.items():
            similarity = similarities[item_idx]
            recommendations.append((item_id, similarity))
        
        # Sort by similarity and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]


class HybridRecommendationEngine(RecommendationEngine):
    """
    Hybrid recommendation engine combining collaborative filtering and content-based approaches.
    """
    
    def __init__(self, cf_weight: float = 0.7, cb_weight: float = 0.3):
        self.cf_engine = CollaborativeFilteringEngine()
        self.cb_engine = ContentBasedEngine()
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.trained = False
    
    def train(self, ratings_data: pd.DataFrame, item_features: Optional[pd.DataFrame] = None) -> None:
        """Train both engines."""
        # Train collaborative filtering
        self.cf_engine.train(ratings_data)
        
        # Train content-based if features provided
        if item_features is not None:
            self.cb_engine.train(ratings_data, item_features)
        
        self.trained = True
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating using hybrid approach."""
        if not self.trained:
            raise ValueError("Engine not trained")
        
        cf_rating = self.cf_engine.predict_rating(user_id, item_id)
        
        if hasattr(self.cb_engine, 'item_features') and self.cb_engine.item_features is not None:
            cb_rating = self.cb_engine.predict_rating(user_id, item_id)
            return self.cf_weight * cf_rating + self.cb_weight * cb_rating
        else:
            return cf_rating
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate hybrid recommendations."""
        if not self.trained:
            raise ValueError("Engine not trained")
        
        cf_recommendations = self.cf_engine.recommend(user_id, n_recommendations * 2)
        
        if hasattr(self.cb_engine, 'item_features') and self.cb_engine.item_features is not None:
            cb_recommendations = self.cb_engine.recommend(user_id, n_recommendations * 2)
            
            # Combine recommendations
            combined_scores = defaultdict(float)
            
            # Add collaborative filtering scores
            for item_id, score in cf_recommendations:
                combined_scores[item_id] += self.cf_weight * score
            
            # Add content-based scores
            for item_id, score in cb_recommendations:
                combined_scores[item_id] += self.cb_weight * score
            
            # Sort and return top recommendations
            final_recommendations = [(item_id, score) for item_id, score in combined_scores.items()]
            final_recommendations.sort(key=lambda x: x[1], reverse=True)
            return final_recommendations[:n_recommendations]
        else:
            return cf_recommendations[:n_recommendations]


class RecommendationEvaluator:
    """
    Evaluator for recommendation engines using various metrics.
    """
    
    @staticmethod
    def precision_at_k(recommendations: List[Tuple[int, float]], 
                      actual_items: Set[int], k: int) -> float:
        """Calculate Precision@K."""
        if k > len(recommendations):
            k = len(recommendations)
        
        recommended_items = set(item_id for item_id, _ in recommendations[:k])
        relevant_items = recommended_items & actual_items
        
        return len(relevant_items) / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(recommendations: List[Tuple[int, float]], 
                   actual_items: Set[int], k: int) -> float:
        """Calculate Recall@K."""
        if k > len(recommendations):
            k = len(recommendations)
        
        if len(actual_items) == 0:
            return 0.0
        
        recommended_items = set(item_id for item_id, _ in recommendations[:k])
        relevant_items = recommended_items & actual_items
        
        return len(relevant_items) / len(actual_items)
    
    @staticmethod
    def ndcg_at_k(recommendations: List[Tuple[int, float]], 
                 actual_items: Set[int], k: int) -> float:
        """Calculate NDCG@K."""
        if k > len(recommendations):
            k = len(recommendations)
        
        if len(actual_items) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, (item_id, _) in enumerate(recommendations[:k]):
            if item_id in actual_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        ideal_ranking = min(k, len(actual_items))
        for i in range(ideal_ranking):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0


def generate_sample_data(n_users: int = 1000, n_items: int = 500, 
                        n_ratings: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample data for recommendation engine."""
    np.random.seed(42)
    
    # Generate ratings data
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    item_ids = np.random.randint(1, n_items + 1, n_ratings)
    ratings = np.random.randint(1, 6, n_ratings)  # Ratings 1-5
    
    ratings_data = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    # Remove duplicate user-item pairs, keep last rating
    ratings_data = ratings_data.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    
    # Generate item features (TF-IDF like features)
    n_features = 20
    item_ids_unique = ratings_data['item_id'].unique()
    item_features = pd.DataFrame({
        'item_id': item_ids_unique
    })
    
    # Add random features
    for i in range(n_features):
        item_features[f'feature_{i}'] = np.random.rand(len(item_ids_unique))
    
    return ratings_data, item_features


def demonstrate_recommendation_engine():
    """Demonstrate recommendation engine functionality."""
    print("=== Recommendation Engine Demo ===\n")
    
    # Generate sample data
    print("1. Generating Sample Data:")
    ratings_data, item_features = generate_sample_data(n_users=500, n_items=200, n_ratings=5000)
    print(f"   Generated {len(ratings_data)} ratings from {ratings_data['user_id'].nunique()} users")
    print(f"   Generated features for {len(item_features)} items")
    
    # Split data for training and testing
    train_data = ratings_data.sample(frac=0.8, random_state=42)
    test_data = ratings_data.drop(train_data.index)
    
    print(f"   Training data: {len(train_data)} ratings")
    print(f"   Test data: {len(test_data)} ratings")
    
    # Test collaborative filtering engine
    print("\n2. Collaborative Filtering Engine:")
    cf_engine = CollaborativeFilteringEngine(n_factors=20, n_epochs=50)
    
    start_time = time.time()
    cf_engine.train(train_data)
    cf_train_time = time.time() - start_time
    print(f"   Training time: {cf_train_time:.4f} seconds")
    
    # Test recommendations
    sample_user = train_data['user_id'].iloc[0]
    start_time = time.time()
    cf_recommendations = cf_engine.recommend(sample_user, n_recommendations=10)
    cf_rec_time = time.time() - start_time
    print(f"   Recommendation time: {cf_rec_time:.6f} seconds")
    print(f"   Sample recommendations for user {sample_user}:")
    for item_id, score in cf_recommendations[:5]:
        print(f"     Item {item_id}: {score:.3f}")
    
    # Test content-based engine
    print("\n3. Content-Based Engine:")
    cb_engine = ContentBasedEngine()
    
    start_time = time.time()
    cb_engine.train(train_data, item_features)
    cb_train_time = time.time() - start_time
    print(f"   Training time: {cb_train_time:.4f} seconds")
    
    start_time = time.time()
    cb_recommendations = cb_engine.recommend(sample_user, n_recommendations=10)
    cb_rec_time = time.time() - start_time
    print(f"   Recommendation time: {cb_rec_time:.6f} seconds")
    print(f"   Sample recommendations for user {sample_user}:")
    for item_id, score in cb_recommendations[:5]:
        print(f"     Item {item_id}: {score:.3f}")
    
    # Test hybrid engine
    print("\n4. Hybrid Recommendation Engine:")
    hybrid_engine = HybridRecommendationEngine(cf_weight=0.6, cb_weight=0.4)
    
    start_time = time.time()
    hybrid_engine.train(train_data, item_features)
    hybrid_train_time = time.time() - start_time
    print(f"   Training time: {hybrid_train_time:.4f} seconds")
    
    start_time = time.time()
    hybrid_recommendations = hybrid_engine.recommend(sample_user, n_recommendations=10)
    hybrid_rec_time = time.time() - start_time
    print(f"   Recommendation time: {hybrid_rec_time:.6f} seconds")
    print(f"   Sample recommendations for user {sample_user}:")
    for item_id, score in hybrid_recommendations[:5]:
        print(f"     Item {item_id}: {score:.3f}")


def performance_comparison():
    """Compare performance of different recommendation approaches."""
    print("\n=== Performance Comparison ===\n")
    
    # Generate larger dataset
    ratings_data, item_features = generate_sample_data(n_users=1000, n_items=500, n_ratings=20000)
    train_data = ratings_data.sample(frac=0.8, random_state=42)
    test_data = ratings_data.drop(train_data.index)
    
    # Test collaborative filtering performance
    print("1. Collaborative Filtering Performance:")
    cf_engine = CollaborativeFilteringEngine(n_factors=30, n_epochs=30)
    
    start_time = time.time()
    cf_engine.train(train_data)
    cf_train_time = time.time() - start_time
    print(f"   Training time: {cf_train_time:.4f} seconds")
    
    # Test prediction performance
    test_users = test_data['user_id'].unique()[:100]  # Test first 100 users
    start_time = time.time()
    cf_predictions = []
    for user_id in test_users:
        predictions = cf_engine.recommend(user_id, n_recommendations=5)
        cf_predictions.extend(predictions)
    cf_pred_time = time.time() - start_time
    print(f"   Prediction time for {len(test_users)} users: {cf_pred_time:.4f} seconds")
    print(f"   Average time per user: {cf_pred_time/len(test_users):.6f} seconds")
    print(f"   Total recommendations generated: {len(cf_predictions)}")
    
    # Test content-based performance
    print("\n2. Content-Based Performance:")
    cb_engine = ContentBasedEngine()
    
    start_time = time.time()
    cb_engine.train(train_data, item_features)
    cb_train_time = time.time() - start_time
    print(f"   Training time: {cb_train_time:.4f} seconds")
    
    start_time = time.time()
    cb_predictions = []
    for user_id in test_users:
        predictions = cb_engine.recommend(user_id, n_recommendations=5)
        cb_predictions.extend(predictions)
    cb_pred_time = time.time() - start_time
    print(f"   Prediction time for {len(test_users)} users: {cb_pred_time:.4f} seconds")
    print(f"   Average time per user: {cb_pred_time/len(test_users):.6f} seconds")
    print(f"   Total recommendations generated: {len(cb_predictions)}")
    
    # Test hybrid performance
    print("\n3. Hybrid Performance:")
    hybrid_engine = HybridRecommendationEngine(cf_weight=0.7, cb_weight=0.3)
    
    start_time = time.time()
    hybrid_engine.train(train_data, item_features)
    hybrid_train_time = time.time() - start_time
    print(f"   Training time: {hybrid_train_time:.4f} seconds")
    
    start_time = time.time()
    hybrid_predictions = []
    for user_id in test_users:
        predictions = hybrid_engine.recommend(user_id, n_recommendations=5)
        hybrid_predictions.extend(predictions)
    hybrid_pred_time = time.time() - start_time
    print(f"   Prediction time for {len(test_users)} users: {hybrid_pred_time:.4f} seconds")
    print(f"   Average time per user: {hybrid_pred_time/len(test_users):.6f} seconds")
    print(f"   Total recommendations generated: {len(hybrid_predictions)}")
    
    # Evaluation metrics
    print("\n4. Evaluation Metrics:")
    evaluator = RecommendationEvaluator()
    
    # For simplicity, we'll use a sample of test data
    sample_test = test_data.sample(n=min(1000, len(test_data)), random_state=42)
    user_actual_items = defaultdict(set)
    for _, row in sample_test.iterrows():
        user_actual_items[row['user_id']].add(row['item_id'])
    
    # Evaluate collaborative filtering
    cf_precisions = []
    cf_recalls = []
    cf_ndcgs = []
    
    for user_id, actual_items in list(user_actual_items.items())[:50]:  # Sample 50 users
        if user_id in cf_engine.user_id_to_idx:
            recommendations = cf_engine.recommend(user_id, n_recommendations=10)
            cf_precisions.append(evaluator.precision_at_k(recommendations, actual_items, 10))
            cf_recalls.append(evaluator.recall_at_k(recommendations, actual_items, 10))
            cf_ndcgs.append(evaluator.ndcg_at_k(recommendations, actual_items, 10))
    
    print(f"   Collaborative Filtering:")
    print(f"     Precision@10: {np.mean(cf_precisions):.4f}")
    print(f"     Recall@10: {np.mean(cf_recalls):.4f}")
    print(f"     NDCG@10: {np.mean(cf_ndcgs):.4f}")


if __name__ == "__main__":
    demonstrate_recommendation_engine()
    performance_comparison()