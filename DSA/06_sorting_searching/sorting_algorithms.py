"""
Sorting and Searching - Sorting Algorithms
====================================

This module provides implementations and analysis of various sorting algorithms,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Comparison-based sorting algorithms
- Non-comparison sorting algorithms
- Stability and in-place properties
- Time and space complexity analysis
- Applications in data science
"""

from typing import List, Callable, Optional
import random
import time

def bubble_sort(arr: List[int]) -> List[int]:
    """
    Sort array using bubble sort algorithm
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: Yes
    In-place: Yes
    """
    arr = arr.copy()  # Don't modify original array
    n = len(arr)
    
    for i in range(n):
        swapped = False
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # If no swapping occurred, array is sorted
        if not swapped:
            break
    
    return arr

def selection_sort(arr: List[int]) -> List[int]:
    """
    Sort array using selection sort algorithm
    Time Complexity: O(n²)
    Space Complexity: O(1)
    Stable: No
    In-place: Yes
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(n):
        # Find minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # Swap the found minimum element with the first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

def insertion_sort(arr: List[int]) -> List[int]:
    """
    Sort array using insertion sort algorithm
    Time Complexity: O(n²) worst case, O(n) best case
    Space Complexity: O(1)
    Stable: Yes
    In-place: Yes
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        # Move elements greater than key one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    
    return arr

def merge_sort(arr: List[int]) -> List[int]:
    """
    Sort array using merge sort algorithm
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable: Yes
    In-place: No
    """
    if len(arr) <= 1:
        return arr.copy()
    
    def merge(left: List[int], right: List[int]) -> List[int]:
        """Merge two sorted arrays"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Add remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer
    return merge(left, right)

def quick_sort(arr: List[int]) -> List[int]:
    """
    Sort array using quick sort algorithm
    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(log n) average, O(n) worst case
    Stable: No
    In-place: Yes (this implementation creates new arrays)
    """
    if len(arr) <= 1:
        return arr.copy()
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def heap_sort(arr: List[int]) -> List[int]:
    """
    Sort array using heap sort algorithm
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    Stable: No
    In-place: Yes
    """
    import heapq
    
    arr = arr.copy()
    heapq.heapify(arr)  # Transform list into heap
    
    result = []
    while arr:
        result.append(heapq.heappop(arr))
    
    return result

def counting_sort(arr: List[int], max_val: Optional[int] = None) -> List[int]:
    """
    Sort array using counting sort algorithm (non-comparison based)
    Time Complexity: O(n + k) where k is range of input
    Space Complexity: O(k)
    Stable: Yes
    In-place: No
    """
    if not arr:
        return arr.copy()
    
    if max_val is None:
        max_val = max(arr)
    min_val = min(arr)
    
    # Create count array
    range_size = max_val - min_val + 1
    count = [0] * range_size
    
    # Count occurrences
    for num in arr:
        count[num - min_val] += 1
    
    # Reconstruct sorted array
    result = []
    for i in range(range_size):
        result.extend([i + min_val] * count[i])
    
    return result

def radix_sort(arr: List[int]) -> List[int]:
    """
    Sort array using radix sort algorithm (non-comparison based)
    Time Complexity: O(d * (n + k)) where d is number of digits, k is base
    Space Complexity: O(n + k)
    Stable: Yes
    In-place: No
    """
    if not arr:
        return arr.copy()
    
    # Handle negative numbers
    negative = [x for x in arr if x < 0]
    positive = [x for x in arr if x >= 0]
    
    def radix_sort_positive(arr):
        if not arr:
            return arr
        
        max_val = max(arr)
        exp = 1
        
        while max_val // exp > 0:
            # Counting sort for current digit
            output = [0] * len(arr)
            count = [0] * 10
            
            # Count occurrences of each digit
            for num in arr:
                digit = (num // exp) % 10
                count[digit] += 1
            
            # Change count[i] to actual position
            for i in range(1, 10):
                count[i] += count[i - 1]
            
            # Build output array
            for i in range(len(arr) - 1, -1, -1):
                digit = (arr[i] // exp) % 10
                output[count[digit] - 1] = arr[i]
                count[digit] -= 1
            
            arr = output.copy()
            exp *= 10
        
        return arr
    
    # Sort positive numbers
    sorted_positive = radix_sort_positive(positive) if positive else []
    
    # Sort negative numbers (convert to positive, sort, then convert back)
    sorted_negative = []
    if negative:
        # Make negative numbers positive for sorting
        neg_positive = [-x for x in negative]
        sorted_neg_positive = radix_sort_positive(neg_positive)
        # Convert back to negative and reverse order
        sorted_negative = [-x for x in reversed(sorted_neg_positive)]
    
    return sorted_negative + sorted_positive

def sorting_algorithms_demo():
    """
    Demonstrate various sorting algorithms
    """
    print("=== Sorting Algorithms Demo ===")
    
    # Test with different arrays
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 4, 6, 1, 3],
        [1, 2, 3, 4, 5],  # Already sorted
        [5, 4, 3, 2, 1],  # Reverse sorted
        [3, 3, 3, 3, 3],  # All same elements
        []  # Empty array
    ]
    
    algorithms = [
        ("Bubble Sort", bubble_sort),
        ("Selection Sort", selection_sort),
        ("Insertion Sort", insertion_sort),
        ("Merge Sort", merge_sort),
        ("Quick Sort", quick_sort),
        ("Heap Sort", heap_sort),
        ("Counting Sort", counting_sort),
        ("Radix Sort", radix_sort)
    ]
    
    for i, arr in enumerate(test_arrays):
        print(f"\nTest Array {i+1}: {arr}")
        for name, func in algorithms:
            try:
                sorted_arr = func(arr)
                print(f"   {name}: {sorted_arr}")
            except Exception as e:
                print(f"   {name}: Error - {e}")

def algorithm_properties():
    """
    Display properties of sorting algorithms
    """
    print("\n=== Sorting Algorithm Properties ===")
    
    properties = {
        "Bubble Sort": {
            "Time Complexity": "O(n²) worst/average, O(n) best",
            "Space Complexity": "O(1)",
            "Stable": "Yes",
            "In-place": "Yes",
            "Use Case": "Educational, small datasets"
        },
        "Selection Sort": {
            "Time Complexity": "O(n²) all cases",
            "Space Complexity": "O(1)",
            "Stable": "No",
            "In-place": "Yes",
            "Use Case": "Memory-constrained environments"
        },
        "Insertion Sort": {
            "Time Complexity": "O(n²) worst/average, O(n) best",
            "Space Complexity": "O(1)",
            "Stable": "Yes",
            "In-place": "Yes",
            "Use Case": "Small datasets, nearly sorted data"
        },
        "Merge Sort": {
            "Time Complexity": "O(n log n) all cases",
            "Space Complexity": "O(n)",
            "Stable": "Yes",
            "In-place": "No",
            "Use Case": "Large datasets, stable sorting needed"
        },
        "Quick Sort": {
            "Time Complexity": "O(n log n) average, O(n²) worst",
            "Space Complexity": "O(log n) average, O(n) worst",
            "Stable": "No",
            "In-place": "Yes",
            "Use Case": "General-purpose, good average performance"
        },
        "Heap Sort": {
            "Time Complexity": "O(n log n) all cases",
            "Space Complexity": "O(1)",
            "Stable": "No",
            "In-place": "Yes",
            "Use Case": "Guaranteed O(n log n), memory-constrained"
        },
        "Counting Sort": {
            "Time Complexity": "O(n + k) where k is range",
            "Space Complexity": "O(k)",
            "Stable": "Yes",
            "In-place": "No",
            "Use Case": "Small integer ranges"
        },
        "Radix Sort": {
            "Time Complexity": "O(d * (n + k)) where d is digits",
            "Space Complexity": "O(n + k)",
            "Stable": "Yes",
            "In-place": "No",
            "Use Case": "Fixed-width integers/strings"
        }
    }
    
    for name, props in properties.items():
        print(f"\n{name}:")
        for prop, value in props.items():
            print(f"   {prop}: {value}")

def data_science_applications():
    """
    Examples of sorting algorithms in data science
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Data preprocessing
    print("1. Data Preprocessing:")
    print("   - Sorting data for efficient analysis")
    print("   - Ordering features for better visualization")
    print("   - Preparing time series data")
    
    # 2. Feature engineering
    print("\n2. Feature Engineering:")
    print("   - Ranking features by importance")
    print("   - Creating ordered categorical variables")
    print("   - Generating sorted indices for group operations")
    
    # 3. Machine learning
    print("\n3. Machine Learning:")
    print("   - K-nearest neighbors require sorted distances")
    print("   - Decision trees sort features for splits")
    print("   - Clustering algorithms sort data points")
    
    # 4. Database operations
    print("\n4. Database Operations:")
    print("   - Index creation uses sorting algorithms")
    print("   - Query optimization with sorted data")
    print("   - Merge join operations")

def performance_comparison():
    """
    Compare performance of different sorting algorithms
    """
    print("\n=== Performance Comparison ===")
    
    # Test with different array sizes
    sizes = [100, 1000, 5000]
    
    algorithms = [
        ("Bubble Sort", bubble_sort),
        ("Selection Sort", selection_sort),
        ("Insertion Sort", insertion_sort),
        ("Merge Sort", merge_sort),
        ("Quick Sort", quick_sort),
        ("Heap Sort", heap_sort)
    ]
    
    for size in sizes:
        print(f"\nTesting with {size} elements:")
        
        # Generate random data
        data = [random.randint(1, 10000) for _ in range(size)]
        
        for name, func in algorithms:
            # Skip slow algorithms for large arrays
            if size > 1000 and name in ["Bubble Sort", "Selection Sort"]:
                print(f"   {name}: Skipped (too slow for large arrays)")
                continue
            
            start = time.time()
            try:
                _ = func(data)
                elapsed = time.time() - start
                print(f"   {name}: {elapsed:.6f}s")
            except Exception as e:
                print(f"   {name}: Error - {e}")

# Example usage and testing
if __name__ == "__main__":
    # Sorting algorithms demo
    sorting_algorithms_demo()
    print("\n" + "="*50 + "\n")
    
    # Algorithm properties
    algorithm_properties()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Implementation of major sorting algorithms")
    print("2. Properties and characteristics of each algorithm")
    print("3. Applications in data science and real-world scenarios")
    print("4. Performance comparison of different algorithms")
    print("\nKey takeaways:")
    print("- Comparison-based sorts have Ω(n log n) lower bound")
    print("- Non-comparison sorts can be faster for specific data types")
    print("- Stability and in-place properties matter for different use cases")
    print("- Choose algorithm based on data characteristics and constraints")
    print("- Python's built-in sort (Timsort) combines merge sort and insertion sort")