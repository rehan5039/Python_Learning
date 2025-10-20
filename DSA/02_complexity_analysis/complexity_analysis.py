"""
Complexity Analysis - Detailed Explanations
=========================================

This module provides detailed explanations of complexity analysis concepts
with comprehensive examples and practical applications.

Topics Covered:
- Formal definitions of Big O, Big Omega, and Big Theta
- Mathematical analysis techniques
- Amortized analysis
- Best, average, and worst case analysis
- Practical optimization strategies
"""

import math
import time
from typing import List, Callable

def big_o_notation():
    """
    Explanation of Big O notation and its formal definition
    """
    print("=== Big O Notation ===")
    print("Big O notation describes the upper bound of an algorithm's running time.")
    print("Formal definition: f(n) = O(g(n)) if there exist positive constants c and n₀")
    print("such that 0 ≤ f(n) ≤ c·g(n) for all n ≥ n₀")
    print()
    
    print("Key Points:")
    print("1. Big O focuses on the worst-case scenario")
    print("2. We ignore constants and lower-order terms")
    print("3. It describes the growth rate as input size increases")
    print("4. Used to compare algorithm efficiency")
    print()

def big_omega_notation():
    """
    Explanation of Big Omega notation
    """
    print("=== Big Omega Notation (Ω) ===")
    print("Big Omega describes the lower bound of an algorithm's running time.")
    print("Formal definition: f(n) = Ω(g(n)) if there exist positive constants c and n₀")
    print("such that 0 ≤ c·g(n) ≤ f(n) for all n ≥ n₀")
    print()
    
    print("Key Points:")
    print("1. Big Omega focuses on the best-case scenario")
    print("2. Describes the minimum time an algorithm will take")
    print("3. Useful for understanding algorithm guarantees")

def big_theta_notation():
    """
    Explanation of Big Theta notation
    """
    print("=== Big Theta Notation (Θ) ===")
    print("Big Theta describes the tight bound of an algorithm's running time.")
    print("Formal definition: f(n) = Θ(g(n)) if there exist positive constants c₁, c₂, and n₀")
    print("such that 0 ≤ c₁·g(n) ≤ f(n) ≤ c₂·g(n) for all n ≥ n₀")
    print()
    
    print("Key Points:")
    print("1. Big Theta provides both upper and lower bounds")
    print("2. Describes the exact growth rate when possible")
    print("3. When we say an algorithm is Θ(f(n)), it means it's both O(f(n)) and Ω(f(n))")

def complexity_analysis_techniques():
    """
    Techniques for analyzing algorithm complexity
    """
    print("=== Complexity Analysis Techniques ===")
    
    techniques = {
        "1. Counting Operations": "Count the number of basic operations as a function of input size",
        "2. Recurrence Relations": "Express complexity in terms of smaller inputs for recursive algorithms",
        "3. Amortized Analysis": "Average time per operation over a sequence of operations",
        "4. Master Theorem": "Direct formula for divide-and-conquer recurrences",
        "5. Substitution Method": "Guess and prove the solution to a recurrence"
    }
    
    for technique, description in techniques.items():
        print(f"{technique}: {description}")

def recurrence_relations():
    """
    Explanation of recurrence relations in complexity analysis
    """
    print("\n=== Recurrence Relations ===")
    print("Recurrence relations express the time complexity of recursive algorithms.")
    print()
    
    examples = {
        "Binary Search": "T(n) = T(n/2) + O(1) → O(log n)",
        "Merge Sort": "T(n) = 2T(n/2) + O(n) → O(n log n)",
        "Binary Tree Traversal": "T(n) = 2T(n/2) + O(1) → O(n)",
        "Fibonacci (naive)": "T(n) = T(n-1) + T(n-2) + O(1) → O(2^n)"
    }
    
    for algorithm, relation in examples.items():
        print(f"{algorithm}: {relation}")

def amortized_analysis():
    """
    Explanation of amortized analysis
    """
    print("\n=== Amortized Analysis ===")
    print("Amortized analysis considers the average time per operation over a sequence.")
    print()
    
    print("Example: Dynamic Array (List) Append Operation")
    print("- Most appends take O(1) time")
    print("- When capacity is exceeded, resizing takes O(n) time")
    print("- But resizing happens infrequently")
    print("- Amortized time per append: O(1)")
    print()
    
    print("Methods of Amortized Analysis:")
    print("1. Aggregate Method: Total cost / number of operations")
    print("2. Accounting Method: Assign credits to operations")
    print("3. Potential Method: Use a potential function")

def best_average_worst_case():
    """
    Explanation of best, average, and worst case analysis
    """
    print("\n=== Best, Average, and Worst Case Analysis ===")
    
    print("Best Case (Ω): Minimum time for any input of size n")
    print("Average Case (Θ): Expected time over all inputs of size n")
    print("Worst Case (O): Maximum time for any input of size n")
    print()
    
    # Example with linear search
    print("Example: Linear Search in an Array")
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target = 5
    
    print(f"Array: {arr}")
    print(f"Target: {target}")
    print()
    print("Best Case: Target is first element → O(1)")
    print("Worst Case: Target is last element or not present → O(n)")
    print("Average Case: Target is in the middle on average → O(n/2) = O(n)")
    print()

def mathematical_analysis():
    """
    Mathematical techniques for complexity analysis
    """
    print("=== Mathematical Analysis Techniques ===")
    
    print("1. Summation Formulas:")
    print("   - Σ(i=1 to n) i = n(n+1)/2 = O(n²)")
    print("   - Σ(i=1 to n) i² = n(n+1)(2n+1)/6 = O(n³)")
    print("   - Σ(i=0 to n) 2^i = 2^(n+1) - 1 = O(2^n)")
    print()
    
    print("2. Logarithmic Properties:")
    print("   - log(n^k) = k·log(n) = O(log n)")
    print("   - log(n!) = Θ(n log n) (Stirling's approximation)")
    print()
    
    print("3. Limits and Growth Rates:")
    print("   - Constant < log n < n^ε < n < n log n < n^2 < 2^n < n!")
    print("   (where ε > 0 is any positive constant)")

def optimization_strategies():
    """
    Strategies for optimizing algorithm complexity
    """
    print("\n=== Optimization Strategies ===")
    
    strategies = {
        "1. Choose Better Data Structures": "Use hash tables for O(1) lookups instead of arrays O(n)",
        "2. Preprocessing": "Sort data once O(n log n) to enable O(log n) binary searches",
        "3. Caching/Memoization": "Store results of expensive computations to avoid recomputation",
        "4. Divide and Conquer": "Break problems into smaller subproblems",
        "5. Greedy Algorithms": "Make locally optimal choices for global optimization",
        "6. Dynamic Programming": "Solve overlapping subproblems efficiently"
    }
    
    for strategy, description in strategies.items():
        print(f"{strategy}: {description}")

def complexity_in_python():
    """
    Python-specific complexity considerations
    """
    print("\n=== Python-Specific Complexity Considerations ===")
    
    print("Built-in Data Structure Operations:")
    print("1. Lists (Dynamic Arrays):")
    print("   - Access by index: O(1)")
    print("   - Append: O(1) amortized")
    print("   - Insert/Delete at end: O(1) amortized")
    print("   - Insert/Delete at beginning: O(n)")
    print("   - Search: O(n)")
    print()
    
    print("2. Dictionaries (Hash Tables):")
    print("   - Access/Add/Delete: O(1) average, O(n) worst case")
    print()
    
    print("3. Sets (Hash Tables):")
    print("   - Add/Remove/Check membership: O(1) average, O(n) worst case")
    print()
    
    print("4. Tuples:")
    print("   - Access by index: O(1)")
    print("   - Creation: O(n)")
    print()

def practical_examples():
    """
    Practical examples of complexity analysis
    """
    print("\n=== Practical Examples ===")
    
    # Example 1: Finding duplicates
    print("1. Finding Duplicates in an Array:")
    print("   Naive approach (nested loops): O(n²)")
    print("   Optimized approach (hash set): O(n)")
    print()
    
    # Example 2: Matrix multiplication
    print("2. Matrix Multiplication:")
    print("   Standard algorithm: O(n³)")
    print("   Strassen's algorithm: O(n^2.807)")
    print("   Coppersmith-Winograd: O(n^2.373)")
    print()
    
    # Example 3: Graph algorithms
    print("3. Graph Algorithms:")
    print("   - BFS/DFS: O(V + E) where V=vertices, E=edges")
    print("   - Dijkstra's algorithm: O((V + E) log V) with priority queue")
    print("   - Floyd-Warshall: O(V³) for all-pairs shortest paths")

# Example usage and testing
if __name__ == "__main__":
    # Explain Big O notation
    big_o_notation()
    print("\n" + "="*50 + "\n")
    
    # Explain Big Omega notation
    big_omega_notation()
    print("\n" + "="*50 + "\n")
    
    # Explain Big Theta notation
    big_theta_notation()
    print("\n" + "="*50 + "\n")
    
    # Complexity analysis techniques
    complexity_analysis_techniques()
    print("\n" + "="*50 + "\n")
    
    # Recurrence relations
    recurrence_relations()
    print("\n" + "="*50 + "\n")
    
    # Amortized analysis
    amortized_analysis()
    print("\n" + "="*50 + "\n")
    
    # Best, average, worst case analysis
    best_average_worst_case()
    print("\n" + "="*50 + "\n")
    
    # Mathematical analysis
    mathematical_analysis()
    print("\n" + "="*50 + "\n")
    
    # Optimization strategies
    optimization_strategies()
    print("\n" + "="*50 + "\n")
    
    # Python-specific considerations
    complexity_in_python()
    print("\n" + "="*50 + "\n")
    
    # Practical examples
    practical_examples()
    
    print("\n=== Summary ===")
    print("This module covered:")
    print("1. Formal definitions of Big O, Big Omega, and Big Theta notations")
    print("2. Mathematical techniques for complexity analysis")
    print("3. Recurrence relations and their solutions")
    print("4. Amortized analysis methods")
    print("5. Best, average, and worst case analysis")
    print("6. Optimization strategies for improving algorithm efficiency")
    print("7. Python-specific complexity considerations")
    print("\nMastering complexity analysis enables you to:")
    print("- Write more efficient algorithms")
    print("- Choose appropriate data structures")
    print("- Optimize existing code")
    print("- Understand algorithm scalability")