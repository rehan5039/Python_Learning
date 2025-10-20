"""
Complexity Analysis - Big O Notation Examples
==========================================

This module provides practical examples of complexity analysis using Big O notation,
with a focus on Python implementations and data science applications.

Topics Covered:
- Time and space complexity analysis
- Common complexity classes
- Python-specific examples
- Data science use cases
"""

import time
import sys
from typing import List, Tuple

def constant_time_example(arr: List[int]) -> int:
    """
    O(1) - Constant time complexity
    Accessing an element by index in an array/list
    """
    if len(arr) > 0:
        return arr[0]  # Always takes the same time regardless of array size
    return 0

def linear_time_example(arr: List[int], target: int) -> bool:
    """
    O(n) - Linear time complexity
    Linear search through an array
    """
    for element in arr:
        if element == target:
            return True
    return False

def quadratic_time_example(arr: List[int]) -> List[Tuple[int, int]]:
    """
    O(n²) - Quadratic time complexity
    Finding all pairs in an array
    """
    pairs = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            pairs.append((arr[i], arr[j]))
    return pairs

def logarithmic_time_example(sorted_arr: List[int], target: int) -> int:
    """
    O(log n) - Logarithmic time complexity
    Binary search in a sorted array
    """
    left, right = 0, len(sorted_arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if sorted_arr[mid] == target:
            return mid
        elif sorted_arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def linearithmic_time_example(arr: List[int]) -> List[int]:
    """
    O(n log n) - Linearithmic time complexity
    Sorting an array using merge sort
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = linearithmic_time_example(arr[:mid])
    right = linearithmic_time_example(arr[mid:])
    
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    """Helper function for merge sort"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def exponential_time_example(n: int) -> int:
    """
    O(2^n) - Exponential time complexity
    Calculating Fibonacci numbers recursively (inefficient)
    """
    if n <= 1:
        return n
    return exponential_time_example(n - 1) + exponential_time_example(n - 2)

def factorial_time_example(n: int) -> List[List[int]]:
    """
    O(n!) - Factorial time complexity
    Generating all permutations of a list
    """
    if n == 0:
        return [[]]
    
    result = []
    for i in range(n):
        rest = list(range(n))
        rest.pop(i)
        for perm in factorial_time_example_helper(rest):
            result.append([i] + perm)
    return result

def factorial_time_example_helper(arr: List[int]) -> List[List[int]]:
    """Helper function for generating permutations"""
    if len(arr) <= 1:
        return [arr]
    
    result = []
    for i in range(len(arr)):
        rest = arr[:i] + arr[i+1:]
        for perm in factorial_time_example_helper(rest):
            result.append([arr[i]] + perm)
    return result

def space_complexity_examples():
    """
    Examples of space complexity analysis
    """
    print("=== Space Complexity Examples ===")
    
    # O(1) - Constant space
    def constant_space(n: int) -> int:
        total = 0
        for i in range(n):
            total += i
        return total
    
    # O(n) - Linear space
    def linear_space(n: int) -> List[int]:
        return [i for i in range(n)]
    
    # O(n²) - Quadratic space
    def quadratic_space(n: int) -> List[List[int]]:
        return [[i * j for j in range(n)] for i in range(n)]
    
    print("Space complexity examples demonstrated in code.")

def complexity_comparison_demo():
    """
    Demonstrate the difference in performance between different complexities
    """
    print("=== Complexity Comparison Demo ===")
    
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        arr = list(range(size))
        sorted_arr = sorted(arr)
        target = size // 2
        
        print(f"\nArray size: {size}")
        
        # O(1) example
        start = time.time()
        constant_time_example(arr)
        end = time.time()
        print(f"O(1) - Constant time: {end - start:.6f} seconds")
        
        # O(n) example
        start = time.time()
        linear_time_example(arr, target)
        end = time.time()
        print(f"O(n) - Linear time: {end - start:.6f} seconds")
        
        # O(log n) example
        start = time.time()
        logarithmic_time_example(sorted_arr, target)
        end = time.time()
        print(f"O(log n) - Logarithmic time: {end - start:.6f} seconds")
        
        # O(n log n) example (only for smaller sizes due to recursion depth)
        if size <= 1000:
            start = time.time()
            linearithmic_time_example(arr[:min(size, 100)])  # Limit size for demo
            end = time.time()
            print(f"O(n log n) - Linearithmic time: {end - start:.6f} seconds")

def common_complexity_classes():
    """
    Summary of common complexity classes
    """
    print("=== Common Complexity Classes ===")
    complexity_info = {
        "O(1)": "Constant time - Direct access, simple operations",
        "O(log n)": "Logarithmic time - Binary search, balanced tree operations",
        "O(n)": "Linear time - Iterating through elements",
        "O(n log n)": "Linearithmic time - Efficient sorting algorithms",
        "O(n²)": "Quadratic time - Nested loops, bubble sort",
        "O(n³)": "Cubic time - Triple nested loops",
        "O(2^n)": "Exponential time - Recursive algorithms without memoization",
        "O(n!)": "Factorial time - Generating all permutations"
    }
    
    for complexity, description in complexity_info.items():
        print(f"{complexity}: {description}")

def data_science_applications():
    """
    Examples of complexity analysis in data science contexts
    """
    print("\n=== Data Science Applications ===")
    
    # Example 1: DataFrame operations
    print("1. Pandas DataFrame operations:")
    print("   - Selecting a column by name: O(1)")
    print("   - Filtering rows based on condition: O(n)")
    print("   - Grouping by a column: O(n log n)")
    print("   - Merging two DataFrames: O(n * m) where n, m are DataFrame sizes")
    
    # Example 2: Machine learning algorithms
    print("\n2. Machine learning algorithm complexities:")
    print("   - Linear Regression (scikit-learn): O(n * p²) where n= samples, p= features")
    print("   - K-Means Clustering: O(n * k * i * d) where k= clusters, i= iterations, d= dimensions")
    print("   - Decision Trees: O(n * log n) for training")
    print("   - Random Forest: O(t * n * log n) where t= number of trees")
    
    # Example 3: Database queries
    print("\n3. Database query optimization:")
    print("   - Indexed lookups: O(log n)")
    print("   - Full table scans: O(n)")
    print("   - Joins without indexes: O(n * m)")

# Example usage and testing
if __name__ == "__main__":
    # Demonstrate complexity analysis
    complexity_comparison_demo()
    print("\n" + "="*50 + "\n")
    
    # Show common complexity classes
    common_complexity_classes()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Space complexity examples
    space_complexity_examples()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Different time complexity classes with practical examples")
    print("2. Space complexity analysis")
    print("3. Performance comparison between complexity classes")
    print("4. Applications in data science contexts")
    print("\nUnderstanding complexity analysis helps you:")
    print("- Choose the most efficient algorithms for your problems")
    print("- Optimize existing code for better performance")
    print("- Anticipate how algorithms will scale with larger inputs")
    print("- Make informed decisions about data structure choices")