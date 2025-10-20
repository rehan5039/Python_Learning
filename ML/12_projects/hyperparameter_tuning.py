"""
Advanced Hyperparameter Tuning
============================

This module provides advanced hyperparameter tuning techniques for machine learning models.
It covers grid search, random search, Bayesian optimization, and other advanced methods.

Key Components:
- Grid search and random search
- Bayesian optimization
- Evolutionary algorithms
- Multi-objective optimization
- Automated hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Advanced hyperparameter tuning framework.
    
    Parameters:
    -----------
    search_method : str, default='grid'
        Search method ('grid', 'random', 'bayesian')
    cv_folds : int, default=5
        Number of cross-validation folds
    n_iter : int, default=10
        Number of iterations for random search
    """
    
    def __init__(self, search_method='grid', cv_folds=5, n_iter=10):
        self.search_method = search_method
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.best_params = None
        self.best_score = None
        self.search_results = None
        
        print(f"HyperparameterTuner initialized with {search_method} search")
        print(f"CV folds: {cv_folds}, Iterations: {n_iter}")
    
    def tune_classifier(self, model, param_grid, X, y, scoring='accuracy'):
        """
        Tune hyperparameters for classification model.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to tune
        param_grid : dict
            Parameter grid for tuning
        X : array-like
            Features
        y : array-like
            Target
        scoring : str, default='accuracy'
            Scoring metric
            
        Returns:
        --------
        best_model : sklearn estimator
            Best model with tuned parameters
        """
        print("Starting hyperparameter tuning for classification...")
        print("=" * 50)
        
        if self.search_method == 'grid':
            # Grid search
            search = GridSearchCV(
                model, param_grid, cv=self.cv_folds,
                scoring=scoring, n_jobs=-1, verbose=1
            )
            
        elif self.search_method == 'random':
            # Random search
            search = RandomizedSearchCV(
                model, param_grid, cv=self.cv_folds, n_iter=self.n_iter,
                scoring=scoring, n_jobs=-1, verbose=1, random_state=42
            )
            
        else:
            raise ValueError(f"Unsupported search method: {self.search_method}")
        
        # Fit search
        search.fit(X, y)
        
        # Store results
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.search_results = search.cv_results_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best {scoring}: {self.best_score:.4f}")
        print(f"Rank 1 parameters: {search.cv_results_['params'][0]}")
        print(f"Rank 1 score: {search.cv_results_['mean_test_score'][0]:.4f}")
        
        return search.best_estimator_
    
    def tune_regressor(self, model, param_grid, X, y, scoring='neg_mean_squared_error'):
        """
        Tune hyperparameters for regression model.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to tune
        param_grid : dict
            Parameter grid for tuning
        X : array-like
            Features
        y : array-like
            Target
        scoring : str, default='neg_mean_squared_error'
            Scoring metric
            
        Returns:
        --------
        best_model : sklearn estimator
            Best model with tuned parameters
        """
        print("Starting hyperparameter tuning for regression...")
        print("=" * 45)
        
        if self.search_method == 'grid':
            # Grid search
            search = GridSearchCV(
                model, param_grid, cv=self.cv_folds,
                scoring=scoring, n_jobs=-1, verbose=1
            )
            
        elif self.search_method == 'random':
            # Random search
            search = RandomizedSearchCV(
                model, param_grid, cv=self.cv_folds, n_iter=self.n_iter,
                scoring=scoring, n_jobs=-1, verbose=1, random_state=42
            )
            
        else:
            raise ValueError(f"Unsupported search method: {self.search_method}")
        
        # Fit search
        search.fit(X, y)
        
        # Store results
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.search_results = search.cv_results_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best {scoring}: {self.best_score:.4f}")
        print(f"Rank 1 parameters: {search.cv_results_['params'][0]}")
        print(f"Rank 1 score: {search.cv_results_['mean_test_score'][0]:.4f}")
        
        return search.best_estimator_
    
    def get_search_results(self, top_n=10):
        """
        Get top search results.
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top results to return
            
        Returns:
        --------
        results_df : pandas.DataFrame
            DataFrame with search results
        """
        if self.search_results is None:
            print("No search results available. Run tuning first.")
            return None
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.search_results)
        
        # Select relevant columns
        score_cols = [col for col in results_df.columns if 'mean_test' in col or 'std_test' in col]
        param_cols = [col for col in results_df.columns if 'param_' in col]
        relevant_cols = param_cols + score_cols + ['rank_test_score']
        
        results_summary = results_df[relevant_cols].sort_values('rank_test_score').head(top_n)
        
        print(f"\nTop {top_n} Search Results:")
        print("=" * 25)
        print(results_summary.to_string(index=False))
        
        return results_summary


class BayesianOptimizer:
    """
    Simplified Bayesian optimization for hyperparameter tuning.
    """
    
    def __init__(self, n_calls=20):
        self.n_calls = n_calls
        self.evaluated_points = []
        self.scores = []
        
        print(f"BayesianOptimizer initialized with {n_calls} calls")
    
    def optimize_rf_classifier(self, X, y, param_space=None):
        """
        Optimize Random Forest classifier using simplified Bayesian approach.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        param_space : dict, optional
            Parameter space to search
            
        Returns:
        --------
        best_params : dict
            Best parameters found
        """
        from sklearn.model_selection import cross_val_score
        
        if param_space is None:
            param_space = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        print("Starting Bayesian optimization for Random Forest...")
        print("=" * 50)
        
        best_score = -np.inf
        best_params = None
        
        # Random search as simplified Bayesian optimization
        param_combinations = []
        for n_est in param_space['n_estimators']:
            for max_d in param_space['max_depth']:
                for min_split in param_space['min_samples_split']:
                    for min_leaf in param_space['min_samples_leaf']:
                        param_combinations.append({
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'min_samples_split': min_split,
                            'min_samples_leaf': min_leaf
                        })
        
        # Evaluate random combinations
        np.random.seed(42)
        selected_combinations = np.random.choice(
            param_combinations, 
            size=min(self.n_calls, len(param_combinations)), 
            replace=False
        )
        
        for i, params in enumerate(selected_combinations):
            # Create model with parameters
            model = RandomForestClassifier(**params, random_state=42)
            
            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            mean_score = scores.mean()
            
            print(f"Iteration {i+1}/{len(selected_combinations)}: "
                  f"Score = {mean_score:.4f}, Params = {params}")
            
            # Update best
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")
        
        return best_params, best_score


class EvolutionaryTuner:
    """
    Evolutionary algorithm for hyperparameter tuning.
    """
    
    def __init__(self, population_size=20, generations=10, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        print(f"EvolutionaryTuner initialized with {population_size} population, "
              f"{generations} generations")
    
    def tune_svm_classifier(self, X, y, param_bounds=None):
        """
        Tune SVM classifier using evolutionary algorithm.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        param_bounds : dict, optional
            Parameter bounds for search
            
        Returns:
        --------
        best_params : dict
            Best parameters found
        """
        from sklearn.model_selection import cross_val_score
        
        if param_bounds is None:
            param_bounds = {
                'C': (0.1, 100),
                'gamma': (0.001, 1)
            }
        
        print("Starting evolutionary optimization for SVM...")
        print("=" * 42)
        
        # Initialize population
        np.random.seed(42)
        population = []
        for _ in range(self.population_size):
            individual = {
                'C': np.random.uniform(param_bounds['C'][0], param_bounds['C'][1]),
                'gamma': np.random.uniform(param_bounds['gamma'][0], param_bounds['gamma'][1])
            }
            population.append(individual)
        
        best_score = -np.inf
        best_params = None
        
        # Evolutionary process
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    model = SVC(C=individual['C'], gamma=individual['gamma'], 
                              random_state=42)
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                    fitness = scores.mean()
                except:
                    fitness = 0  # Invalid parameters
                
                fitness_scores.append(fitness)
            
            # Find best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_score:
                best_score = fitness_scores[best_idx]
                best_params = population[best_idx].copy()
            
            print(f"Generation {generation+1}/{self.generations}: "
                  f"Best score = {best_score:.4f}")
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(self.population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(
                    len(population), size=tournament_size, replace=False
                )
                tournament_scores = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_scores)]
                new_population.append(population[winner_idx].copy())
            
            # Crossover and mutation
            for i in range(0, len(new_population)-1, 2):
                # Simple arithmetic crossover
                if np.random.random() < 0.7:  # Crossover probability
                    alpha = np.random.random()
                    for param in ['C', 'gamma']:
                        new_population[i][param] = (
                            alpha * new_population[i][param] + 
                            (1 - alpha) * new_population[i+1][param]
                        )
                        new_population[i+1][param] = (
                            (1 - alpha) * new_population[i][param] + 
                            alpha * new_population[i+1][param]
                        )
                
                # Mutation
                for individual in [new_population[i], new_population[i+1]]:
                    if np.random.random() < self.mutation_rate:
                        param = np.random.choice(['C', 'gamma'])
                        if param == 'C':
                            individual[param] = np.random.uniform(
                                param_bounds['C'][0], param_bounds['C'][1]
                            )
                        else:
                            individual[param] = np.random.uniform(
                                param_bounds['gamma'][0], param_bounds['gamma'][1]
                            )
            
            population = new_population
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")
        
        return best_params, best_score


class AutomatedTuner:
    """
    Automated hyperparameter tuning with multiple strategies.
    """
    
    def __init__(self):
        print("AutomatedTuner initialized")
    
    def auto_tune_classifier(self, X, y, model_type='rf', time_budget=60):
        """
        Automatically tune classifier with time budget.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        model_type : str, default='rf'
            Type of model ('rf', 'svm', 'lr')
        time_budget : int, default=60
            Time budget in seconds
            
        Returns:
        --------
        best_model : sklearn estimator
            Best tuned model
        """
        import time
        
        print(f"Starting automated tuning for {model_type} classifier...")
        print("=" * 50)
        
        start_time = time.time()
        
        if model_type == 'rf':
            # Random Forest
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
        elif model_type == 'svm':
            # SVM
            model = SVC(random_state=42)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            }
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Quick grid search
        tuner = HyperparameterTuner(search_method='grid', cv_folds=3)
        best_model = tuner.tune_classifier(model, param_grid, X, y)
        
        elapsed_time = time.time() - start_time
        print(f"Auto-tuning completed in {elapsed_time:.2f} seconds")
        
        return best_model


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data for demonstration
    print("Advanced Hyperparameter Tuning Demonstration")
    print("=" * 45)
    
    # Generate sample classification data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=3, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Sample data created: {X_train.shape} (train), {X_test.shape} (test)")
    
    # Grid Search Demonstration
    print("\n1. Grid Search Tuning:")
    print("-" * 20)
    
    # Define model and parameter grid
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5]
    }
    
    # Initialize tuner
    grid_tuner = HyperparameterTuner(search_method='grid', cv_folds=3)
    
    # Tune model
    best_rf = grid_tuner.tune_classifier(rf_model, param_grid, X_train, y_train)
    
    # Get search results
    search_results = grid_tuner.get_search_results(top_n=5)
    
    # Random Search Demonstration
    print("\n2. Random Search Tuning:")
    print("-" * 22)
    
    # Larger parameter space for random search
    large_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Initialize random tuner
    random_tuner = HyperparameterTuner(search_method='random', cv_folds=3, n_iter=10)
    
    # Tune model
    best_rf_random = random_tuner.tune_classifier(rf_model, large_param_grid, X_train, y_train)
    
    # Bayesian Optimization Demonstration
    print("\n3. Bayesian Optimization:")
    print("-" * 22)
    
    bayesian_optimizer = BayesianOptimizer(n_calls=15)
    bayes_params, bayes_score = bayesian_optimizer.optimize_rf_classifier(X_train, y_train)
    
    # Evolutionary Algorithm Demonstration
    print("\n4. Evolutionary Algorithm:")
    print("-" * 22)
    
    # Generate binary classification data for SVM
    X_bin, y_bin = make_classification(
        n_samples=500, n_features=10, n_classes=2, random_state=42
    )
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42
    )
    
    evolutionary_tuner = EvolutionaryTuner(population_size=15, generations=8)
    evo_params, evo_score = evolutionary_tuner.tune_svm_classifier(X_train_bin, y_train_bin)
    
    # Automated Tuning Demonstration
    print("\n5. Automated Tuning:")
    print("-" * 18)
    
    auto_tuner = AutomatedTuner()
    auto_model = auto_tuner.auto_tune_classifier(X_train, y_train, model_type='rf')
    
    # Compare results
    print("\n" + "="*50)
    print("Hyperparameter Tuning Results Comparison")
    print("="*50)
    
    # Evaluate all models
    from sklearn.metrics import accuracy_score
    
    models = {
        'Default RF': RandomForestClassifier(random_state=42),
        'Grid Search RF': best_rf,
        'Random Search RF': best_rf_random,
        'Bayesian RF': RandomForestClassifier(**bayes_params, random_state=42),
        'Auto-tuned RF': auto_model
    }
    
    print("Model Performance Comparison:")
    print("-" * 30)
    for name, model in models.items():
        if name == 'Default RF':
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name:20}: {accuracy:.4f}")
    
    # Advanced tuning techniques summary
    print("\n" + "="*50)
    print("Advanced Hyperparameter Tuning Techniques")
    print("="*50)
    print("1. Grid Search:")
    print("   - Exhaustive search over parameter grid")
    print("   - Guaranteed to find optimal in grid")
    print("   - Computationally expensive")
    print("   - Good for small parameter spaces")
    
    print("\n2. Random Search:")
    print("   - Random sampling from parameter distributions")
    print("   - More efficient than grid search")
    print("   - Better for large parameter spaces")
    print("   - Can find good solutions faster")
    
    print("\n3. Bayesian Optimization:")
    print("   - Uses probabilistic model of objective function")
    print("   - Sequential model-based optimization")
    print("   - Learns from previous evaluations")
    print("   - Efficient for expensive evaluations")
    
    print("\n4. Evolutionary Algorithms:")
    print("   - Population-based optimization")
    print("   - Genetic algorithms, evolution strategies")
    print("   - Good for complex parameter spaces")
    print("   - Can handle constraints well")
    
    print("\n5. Multi-Armed Bandits:")
    print("   - Online learning approach")
    print("   - Balances exploration and exploitation")
    print("   - Good for continuous tuning")
    print("   - Adaptive to changing conditions")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Hyperparameter Tuning")
    print("="*50)
    print("1. Parameter Space Design:")
    print("   - Start with reasonable ranges")
    print("   - Use logarithmic scales for parameters like learning rate")
    print("   - Consider parameter dependencies")
    print("   - Validate parameter combinations")
    
    print("\n2. Search Strategy:")
    print("   - Use random search for initial exploration")
    print("   - Apply Bayesian optimization for refinement")
    print("   - Consider computational budget")
    print("   - Balance exploration and exploitation")
    
    print("\n3. Validation:")
    print("   - Use proper cross-validation")
    print("   - Avoid data leakage")
    print("   - Monitor for overfitting")
    print("   - Validate on multiple datasets")
    
    print("\n4. Implementation:")
    print("   - Use parallel processing when possible")
    print("   - Implement early stopping")
    print("   - Log all experiments")
    print("   - Make results reproducible")
    
    print("\n5. Evaluation:")
    print("   - Use appropriate metrics")
    print("   - Consider multiple objectives")
    print("   - Analyze parameter importance")
    print("   - Document findings thoroughly")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using:")
    print("- Scikit-optimize: Bayesian optimization")
    print("- Optuna: Automated hyperparameter optimization")
    print("- Hyperopt: Distributed asynchronous hyperparameter optimization")
    print("- These provide enterprise-grade tuning capabilities")