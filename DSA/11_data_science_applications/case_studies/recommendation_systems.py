"""
Recommendation Systems Case Study

This case study demonstrates the application of Data Structures and Algorithms in building recommendation systems:
- Collaborative filtering algorithms
- Content-based recommendation techniques
- Matrix factorization optimization
- Similarity computation optimization
- Scalability considerations for large datasets
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time


class CollaborativeFiltering:
    """
    Collaborative Filtering Recommendation System using Matrix Factorization.
    """
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01, 
                 reg_param: float = 0.01, n_epochs: int = 100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
    
    def fit(self, ratings_matrix: np.ndarray) -> 'CollaborativeFiltering':
        """
        Train the collaborative filtering model using Stochastic Gradient Descent.
        
        Time Complexity: O(n_epochs * n_ratings * n_factors)
        Space Complexity: O(n_users * n_factors + n_items * n_factors)
        """
        n_users, n_items = ratings_matrix.shape
        self.global_mean = np.mean(ratings_matrix[ratings_matrix > 0])
        
        # Initialize factor matrices
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        # Get indices of non-zero ratings
        user_indices, item_indices = np.where(ratings_matrix > 0)
        ratings = ratings_matrix[user_indices, item_indices]
        
        # Stochastic Gradient Descent
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
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for a user-item pair."""
        pred = (self.global_mean + 
                self.user_bias[user_idx] + 
                self.item_bias[item_idx] + 
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        return max(1, min(5, pred))  # Clamp to rating range
    
    def recommend(self, user_idx: int, n_recommendations: int = 10, 
                 items_to_exclude: List[int] = None) -> List[Tuple[int, float]]:
        """Recommend items for a user."""
        if items_to_exclude is None:
            items_to_exclude = []
        
        # Get all items
        all_items = set(range(len(self.item_factors)))
        candidate_items = list(all_items - set(items_to_exclude))
        
        # Predict ratings for candidate items
        predictions = []
        for item_idx in candidate_items:
            pred_rating = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred_rating))
        
        # Sort by predicted rating and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class ContentBasedFiltering:
    """
    Content-Based Recommendation System using TF-IDF and Cosine Similarity.
    """
    
    def __init__(self):
        self.item_profiles = None
        self.item_ids = None
    
    def fit(self, item_features: np.ndarray, item_ids: List[int]) -> 'ContentBasedFiltering':
        """
        Build item profiles from features.
        
        Time Complexity: O(n_items * n_features)
        Space Complexity: O(n_items * n_features)
        """
        self.item_profiles = item_features
        self.item_ids = item_ids
        return self
    
    def recommend(self, user_profile: np.ndarray, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend items based on user profile similarity.
        
        Time Complexity: O(n_items * n_features)
        Space Complexity: O(n_items)
        """
        # Calculate cosine similarity between user profile and all items
        similarities = cosine_similarity([user_profile], self.item_profiles)[0]
        
        # Create item-similarity pairs
        item_similarities = [(self.item_ids[i], similarities[i]) for i in range(len(self.item_ids))]
        
        # Sort by similarity and return top recommendations
        item_similarities.sort(key=lambda x: x[1], reverse=True)
        return item_similarities[:n_recommendations]


class HybridRecommender:
    """
    Hybrid Recommendation System combining collaborative and content-based filtering.
    """
    
    def __init__(self, cf_weight: float = 0.7, cb_weight: float = 0.3):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = None
        self.cb_model = None
    
    def fit(self, ratings_matrix: np.ndarray, item_features: np.ndarray, 
           item_ids: List[int]) -> 'HybridRecommender':
        """Train both collaborative and content-based models."""
        # Train collaborative filtering
        self.cf_model = CollaborativeFiltering(n_factors=20, n_epochs=50)
        self.cf_model.fit(ratings_matrix)
        
        # Train content-based filtering
        self.cb_model = ContentBasedFiltering()
        self.cb_model.fit(item_features, item_ids)
        
        return self
    
    def recommend(self, user_idx: int, user_profile: np.ndarray, 
                 n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Recommend items using hybrid approach."""
        # Get recommendations from both models
        cf_recommendations = self.cf_model.recommend(user_idx, n_recommendations * 2)
        cb_recommendations = self.cb_model.recommend(user_profile, n_recommendations * 2)
        
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


def generate_sample_data(n_users: int = 1000, n_items: int = 500) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate sample data for recommendation system demonstration.
    
    Time Complexity: O(n_users * n_items)
    Space Complexity: O(n_users * n_items)
    """
    # Generate random ratings matrix (sparse)
    np.random.seed(42)
    ratings_matrix = np.zeros((n_users, n_items))
    
    # Each user rates 10-50 random items
    for user_idx in range(n_users):
        n_ratings = np.random.randint(10, 51)
        item_indices = np.random.choice(n_items, n_ratings, replace=False)
        ratings = np.random.randint(1, 6, n_ratings)  # Ratings 1-5
        ratings_matrix[user_idx, item_indices] = ratings
    
    # Generate item features (TF-IDF like features)
    n_features = 100
    item_features = np.random.rand(n_items, n_features)
    
    # Normalize features
    item_features = item_features / np.linalg.norm(item_features, axis=1, keepdims=True)
    
    item_ids = list(range(n_items))
    
    return ratings_matrix, item_features, item_ids


def performance_evaluation():
    """Evaluate performance of recommendation algorithms."""
    print("=== Recommendation Systems Performance Evaluation ===\n")
    
    # Generate sample data
    print("1. Generating Sample Data:")
    ratings_matrix, item_features, item_ids = generate_sample_data(n_users=500, n_items=200)
    print(f"   Ratings matrix shape: {ratings_matrix.shape}")
    print(f"   Sparsity: {1 - np.count_nonzero(ratings_matrix) / ratings_matrix.size:.4f}")
    print(f"   Item features shape: {item_features.shape}")
    
    # Test Collaborative Filtering
    print("\n2. Collaborative Filtering:")
    start_time = time.time()
    cf_model = CollaborativeFiltering(n_factors=10, n_epochs=20)
    cf_model.fit(ratings_matrix)
    cf_training_time = time.time() - start_time
    print(f"   Training time: {cf_training_time:.4f} seconds")
    
    # Test recommendations
    start_time = time.time()
    recommendations = cf_model.recommend(user_idx=0, n_recommendations=10)
    cf_recommendation_time = time.time() - start_time
    print(f"   Recommendation time: {cf_recommendation_time:.6f} seconds")
    print(f"   Sample recommendations: {recommendations[:3]}")
    
    # Test Content-Based Filtering
    print("\n3. Content-Based Filtering:")
    start_time = time.time()
    cb_model = ContentBasedFiltering()
    cb_model.fit(item_features, item_ids)
    cb_training_time = time.time() - start_time
    print(f"   Training time: {cb_training_time:.4f} seconds")
    
    # Test recommendations
    user_profile = np.random.rand(100)
    user_profile = user_profile / np.linalg.norm(user_profile)
    start_time = time.time()
    cb_recommendations = cb_model.recommend(user_profile, n_recommendations=10)
    cb_recommendation_time = time.time() - start_time
    print(f"   Recommendation time: {cb_recommendation_time:.6f} seconds")
    print(f"   Sample recommendations: {cb_recommendations[:3]}")
    
    # Test Hybrid Recommender
    print("\n4. Hybrid Recommender:")
    start_time = time.time()
    hybrid_model = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
    hybrid_model.fit(ratings_matrix, item_features, item_ids)
    hybrid_training_time = time.time() - start_time
    print(f"   Training time: {hybrid_training_time:.4f} seconds")
    
    # Test hybrid recommendations
    start_time = time.time()
    hybrid_recommendations = hybrid_model.recommend(user_idx=0, user_profile=user_profile, 
                                                  n_recommendations=10)
    hybrid_recommendation_time = time.time() - start_time
    print(f"   Recommendation time: {hybrid_recommendation_time:.6f} seconds")
    print(f"   Sample recommendations: {hybrid_recommendations[:3]}")


def demo():
    """Demonstrate recommendation systems case study."""
    print("=== Recommendation Systems Case Study ===\n")
    
    # Generate sample data
    ratings_matrix, item_features, item_ids = generate_sample_data(n_users=100, n_items=50)
    print("Sample data generated:")
    print(f"  Ratings matrix: {ratings_matrix.shape}")
    print(f"  Item features: {item_features.shape}")
    print(f"  Non-zero ratings: {np.count_nonzero(ratings_matrix)}")
    
    # Demonstrate Collaborative Filtering
    print("\n1. Collaborative Filtering:")
    cf_model = CollaborativeFiltering(n_factors=5, n_epochs=10)
    cf_model.fit(ratings_matrix)
    
    # Show recommendations for first user
    user_0_recommendations = cf_model.recommend(user_idx=0, n_recommendations=5)
    print(f"  Recommendations for User 0: {user_0_recommendations}")
    
    # Demonstrate Content-Based Filtering
    print("\n2. Content-Based Filtering:")
    cb_model = ContentBasedFiltering()
    cb_model.fit(item_features, item_ids)
    
    # Create sample user profile
    user_profile = np.mean(item_features[:10], axis=0)  # Average of first 10 items
    cb_recommendations = cb_model.recommend(user_profile, n_recommendations=5)
    print(f"  Content-based recommendations: {cb_recommendations}")
    
    # Demonstrate Hybrid Recommender
    print("\n3. Hybrid Recommender:")
    hybrid_model = HybridRecommender(cf_weight=0.7, cb_weight=0.3)
    hybrid_model.fit(ratings_matrix, item_features, item_ids)
    
    hybrid_recommendations = hybrid_model.recommend(user_idx=0, user_profile=user_profile, 
                                                  n_recommendations=5)
    print(f"  Hybrid recommendations: {hybrid_recommendations}")
    
    # Performance evaluation
    print("\n" + "="*60)
    performance_evaluation()


if __name__ == "__main__":
    demo()