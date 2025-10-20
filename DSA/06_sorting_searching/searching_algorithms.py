"""
Sorting and Searching - Searching Algorithms
======================================

This module provides implementations and analysis of various searching algorithms,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Linear search and its variants
- Binary search and its applications
- Interpolation search
- Exponential search
- Hash-based searching
- Time and space complexity analysis
"""

from typing import List, Optional, Any, Callable
import bisect

def linear_search(arr: List[Any], target: Any) -> int:
    """
    Search for target in array using linear search
    Time Complexity: O(n)
    Space Complexity: O(1)
    Works on: Unsorted arrays
    """
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1  # Target not found

def linear_search_last_occurrence(arr: List[Any], target: Any) -> int:
    """
    Find last occurrence of target using linear search
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    last_index = -1
    for i, element in enumerate(arr):
        if element == target:
            last_index = i
    return last_index

def binary_search(arr: List[Any], target: Any) -> int:
    """
    Search for target in sorted array using binary search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    Works on: Sorted arrays
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found

def binary_search_first_occurrence(arr: List[Any], target: Any) -> int:
    """
    Find first occurrence of target in sorted array
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching in left half
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def binary_search_last_occurrence(arr: List[Any], target: Any) -> int:
    """
    Find last occurrence of target in sorted array
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching in right half
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def binary_search_range(arr: List[Any], target: Any) -> List[int]:
    """
    Find range (first and last occurrence) of target in sorted array
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    first = binary_search_first_occurrence(arr, target)
    if first == -1:
        return [-1, -1]
    
    last = binary_search_last_occurrence(arr, target)
    return [first, last]

def interpolation_search(arr: List[int], target: int) -> int:
    """
    Search for target in sorted uniformly distributed array
    Time Complexity: O(log log n) average, O(n) worst case
    Space Complexity: O(1)
    Works on: Sorted arrays with uniform distribution
    """
    left, right = 0, len(arr) - 1
    
    while left <= right and target >= arr[left] and target <= arr[right]:
        # If array has only one element
        if left == right:
            if arr[left] == target:
                return left
            return -1
        
        # Probing the position with uniform distribution formula
        pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])
        
        # Check if target is found
        if arr[pos] == target:
            return pos
        
        # If target is larger, search in right half
        if arr[pos] < target:
            left = pos + 1
        # If target is smaller, search in left half
        else:
            right = pos - 1
    
    return -1  # Target not found

def exponential_search(arr: List[Any], target: Any) -> int:
    """
    Search for target in sorted array using exponential search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    Works on: Sorted arrays, especially unbounded searches
    """
    if not arr:
        return -1
    
    # If target is first element
    if arr[0] == target:
        return 0
    
    # Find range where target might be present
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    
    # Perform binary search in found range
    left = i // 2
    right = min(i, len(arr) - 1)
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found

def ternary_search(arr: List[int], target: int) -> int:
    """
    Search for target in sorted array using ternary search
    Time Complexity: O(log₃ n)
    Space Complexity: O(1)
    Works on: Sorted arrays, unimodal functions
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # Divide array into three parts
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        # Check if target is at any mid
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        
        # Determine which segment to search
        if target < arr[mid1]:
            # Search in first third
            right = mid1 - 1
        elif target > arr[mid2]:
            # Search in last third
            left = mid2 + 1
        else:
            # Search in middle third
            left = mid1 + 1
            right = mid2 - 1
    
    return -1  # Target not found

def jump_search(arr: List[Any], target: Any) -> int:
    """
    Search for target in sorted array using jump search
    Time Complexity: O(√n)
    Space Complexity: O(1)
    Works on: Sorted arrays
    """
    import math
    
    n = len(arr)
    if n == 0:
        return -1
    
    # Finding block size to jump
    step = int(math.sqrt(n))
    
    # Finding the block where element is present
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Linear search in the block
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    
    # If element is found
    if arr[prev] == target:
        return prev
    
    return -1  # Target not found

def searching_algorithms_demo():
    """
    Demonstrate various searching algorithms
    """
    print("=== Searching Algorithms Demo ===")
    
    # Test with sorted array
    sorted_arr = [1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10]
    target = 4
    
    print(f"Sorted Array: {sorted_arr}")
    print(f"Target: {target}")
    
    # Linear search
    index = linear_search(sorted_arr, target)
    print(f"Linear Search: Index {index}")
    
    # Binary search
    index = binary_search(sorted_arr, target)
    print(f"Binary Search: Index {index}")
    
    # First and last occurrence
    first = binary_search_first_occurrence(sorted_arr, target)
    last = binary_search_last_occurrence(sorted_arr, target)
    print(f"First Occurrence: Index {first}")
    print(f"Last Occurrence: Index {last}")
    
    # Range search
    range_result = binary_search_range(sorted_arr, target)
    print(f"Range of Target: [{range_result[0]}, {range_result[1]}]")
    
    # Test with unsorted array
    unsorted_arr = [64, 34, 25, 12, 22, 11, 90]
    target = 25
    
    print(f"\nUnsorted Array: {unsorted_arr}")
    print(f"Target: {target}")
    
    # Linear search on unsorted array
    index = linear_search(unsorted_arr, target)
    print(f"Linear Search: Index {index}")
    
    # Test interpolation search
    uniform_arr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    target = 70
    
    print(f"\nUniform Array: {uniform_arr}")
    print(f"Target: {target}")
    
    index = interpolation_search(uniform_arr, target)
    print(f"Interpolation Search: Index {index}")
    
    # Test exponential search
    large_sorted_arr = list(range(1, 1001, 2))  # Odd numbers 1 to 999
    target = 501
    
    print(f"\nLarge Sorted Array (first 10): {large_sorted_arr[:10]}...{large_sorted_arr[-10:]}")
    print(f"Target: {target}")
    
    index = exponential_search(large_sorted_arr, target)
    print(f"Exponential Search: Index {index}")

def search_algorithm_properties():
    """
    Display properties of searching algorithms
    """
    print("\n=== Searching Algorithm Properties ===")
    
    properties = {
        "Linear Search": {
            "Time Complexity": "O(n)",
            "Space Complexity": "O(1)",
            "Prerequisite": "None",
            "Best Case": "O(1) - Target at first position",
            "Worst Case": "O(n) - Target at last or not present",
            "Use Case": "Unsorted data, small datasets"
        },
        "Binary Search": {
            "Time Complexity": "O(log n)",
            "Space Complexity": "O(1)",
            "Prerequisite": "Sorted array",
            "Best Case": "O(1) - Target at middle",
            "Worst Case": "O(log n) - Target at extreme or not present",
            "Use Case": "Sorted data, frequent searches"
        },
        "Interpolation Search": {
            "Time Complexity": "O(log log n) average, O(n) worst",
            "Space Complexity": "O(1)",
            "Prerequisite": "Sorted, uniformly distributed data",
            "Best Case": "O(1) - Uniform distribution, target found quickly",
            "Worst Case": "O(n) - Non-uniform distribution",
            "Use Case": "Large sorted datasets with uniform distribution"
        },
        "Exponential Search": {
            "Time Complexity": "O(log n)",
            "Space Complexity": "O(1)",
            "Prerequisite": "Sorted array",
            "Best Case": "O(1) - Target at first position",
            "Worst Case": "O(log n) - Target at end or not present",
            "Use Case": "Unbounded searches, target near beginning"
        },
        "Jump Search": {
            "Time Complexity": "O(√n)",
            "Space Complexity": "O(1)",
            "Prerequisite": "Sorted array",
            "Best Case": "O(1) - Target at first block",
            "Worst Case": "O(√n) - Target at end or not present",
            "Use Case": "Sorted data when binary search is costly"
        },
        "Ternary Search": {
            "Time Complexity": "O(log₃ n)",
            "Space Complexity": "O(1)",
            "Prerequisite": "Sorted array or unimodal function",
            "Best Case": "O(1) - Target at mid1 or mid2",
            "Worst Case": "O(log₃ n) - Target not present",
            "Use Case": "Unimodal functions, divide search space in three"
        }
    }
    
    for name, props in properties.items():
        print(f"\n{name}:")
        for prop, value in props.items():
            print(f"   {prop}: {value}")

def data_science_applications():
    """
    Examples of searching algorithms in data science
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Database indexing
    print("1. Database Indexing:")
    print("   - Binary search on B-trees and B+ trees")
    print("   - Hash-based indexing for exact matches")
    print("   - Range queries using sorted indices")
    
    # 2. Machine learning
    print("\n2. Machine Learning:")
    print("   - K-nearest neighbors search")
    print("   - Feature selection using search algorithms")
    print("   - Hyperparameter tuning with search spaces")
    
    # 3. Information retrieval
    print("\n3. Information Retrieval:")
    print("   - Search engines use various search techniques")
    print("   - Document ranking and retrieval")
    print("   - Similarity search in recommendation systems")
    
    # 4. Data analysis
    print("\n4. Data Analysis:")
    print("   - Finding percentiles and quantiles")
    print("   - Searching for outliers and anomalies")
    print("   - Time series analysis and pattern matching")

def performance_comparison():
    """
    Compare performance of different searching algorithms
    """
    print("\n=== Performance Comparison ===")
    
    import time
    import random
    
    # Create test data
    sorted_data = list(range(0, 100000, 2))  # Even numbers 0 to 99998
    unsorted_data = sorted_data.copy()
    random.shuffle(unsorted_data)
    
    target = 50000  # Middle element
    
    algorithms = [
        ("Linear Search (sorted)", lambda: linear_search(sorted_data, target)),
        ("Linear Search (unsorted)", lambda: linear_search(unsorted_data, target)),
        ("Binary Search", lambda: binary_search(sorted_data, target)),
        ("Interpolation Search", lambda: interpolation_search(sorted_data, target)),
        ("Exponential Search", lambda: exponential_search(sorted_data, target)),
        ("Jump Search", lambda: jump_search(sorted_data, target))
    ]
    
    print(f"Searching for {target} in arrays of size {len(sorted_data)}:")
    
    for name, func in algorithms:
        start = time.time()
        result = func()
        elapsed = time.time() - start
        print(f"   {name}: Index {result}, Time {elapsed:.8f}s")

# Example usage and testing
if __name__ == "__main__":
    # Searching algorithms demo
    searching_algorithms_demo()
    print("\n" + "="*50 + "\n")
    
    # Algorithm properties
    search_algorithm_properties()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Implementation of major searching algorithms")
    print("2. Properties and characteristics of each algorithm")
    print("3. Applications in data science and real-world scenarios")
    print("4. Performance comparison of different algorithms")
    print("\nKey takeaways:")
    print("- Linear search works on any data but is slow")
    print("- Binary search requires sorted data but is very efficient")
    print("- Interpolation search can be faster than binary for uniform data")
    print("- Exponential search is good for unbounded or large datasets")
    print("- Choice of algorithm depends on data characteristics and requirements")