"""
Algorithm Design Techniques - Divide and Conquer
=========================================

This module provides implementations and examples of divide and conquer algorithms,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Divide and conquer paradigm
- Classic algorithms (merge sort, quick sort, binary search)
- Problem-solving strategies
- Complexity analysis
- Applications in data science
"""

from typing import List, Optional, Tuple
import random

def merge_sort(arr: List[int]) -> List[int]:
    """
    Sort array using merge sort algorithm (divide and conquer)
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    # Base case: arrays with 0 or 1 element are already sorted
    if len(arr) <= 1:
        return arr
    
    # Divide: split array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Conquer: recursively sort both halves
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)
    
    # Combine: merge sorted halves
    return merge(left_sorted, right_sorted)

def merge(left: List[int], right: List[int]) -> List[int]:
    """
    Merge two sorted arrays into one sorted array
    Time Complexity: O(n + m) where n, m are lengths of arrays
    """
    result = []
    i = j = 0
    
    # Compare elements and merge in sorted order
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

def quick_sort(arr: List[int]) -> List[int]:
    """
    Sort array using quick sort algorithm (divide and conquer)
    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(log n) average, O(n) worst case
    """
    if len(arr) <= 1:
        return arr
    
    # Divide: partition array around pivot
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Conquer: recursively sort subarrays
    # Combine: concatenate sorted subarrays
    return quick_sort(left) + middle + quick_sort(right)

def quick_sort_inplace(arr: List[int], low: int = 0, high: Optional[int] = None) -> None:
    """
    In-place quick sort implementation
    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(log n) average, O(n) worst case
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition and get pivot index
        pivot_index = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quick_sort_inplace(arr, low, pivot_index - 1)
        quick_sort_inplace(arr, pivot_index + 1, high)

def partition(arr: List[int], low: int, high: int) -> int:
    """
    Partition array for quick sort
    """
    # Choose rightmost element as pivot
    pivot = arr[high]
    
    # Index of smaller element (indicates right position of pivot)
    i = low - 1
    
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def binary_search(arr: List[int], target: int) -> int:
    """
    Search for target in sorted array using binary search
    Time Complexity: O(log n)
    Space Complexity: O(1)
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

def binary_search_recursive(arr: List[int], target: int, left: int = 0, right: Optional[int] = None) -> int:
    """
    Recursive binary search implementation
    Time Complexity: O(log n)
    Space Complexity: O(log n) due to recursion stack
    """
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1  # Target not found
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def closest_pair(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Find closest pair of points using divide and conquer
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def brute_force_closest(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Brute force approach for small arrays"""
        min_dist = float('inf')
        pair = (points[0], points[1])
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = distance(points[i], points[j])
                if dist < min_dist:
                    min_dist = dist
                    pair = (points[i], points[j])
        
        return pair[0], pair[1], min_dist
    
    def closest_pair_rec(px: List[Tuple[float, float]], py: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Recursive function to find closest pair"""
        n = len(px)
        
        # Base case: use brute force for small arrays
        if n <= 3:
            return brute_force_closest(px)
        
        # Divide
        mid = n // 2
        midpoint = px[mid]
        
        pyl = [point for point in py if point[0] <= midpoint[0]]
        pyr = [point for point in py if point[0] > midpoint[0]]
        
        # Conquer
        left_pair = closest_pair_rec(px[:mid], pyl)
        right_pair = closest_pair_rec(px[mid:], pyr)
        
        # Find minimum of the two halves
        if left_pair[2] <= right_pair[2]:
            min_pair = left_pair
        else:
            min_pair = right_pair
        
        # Combine: check points near the dividing line
        strip = [point for point in py if abs(point[0] - midpoint[0]) < min_pair[2]]
        
        # Check points in strip
        strip_pair = closest_in_strip(strip, min_pair[2])
        if strip_pair[2] < min_pair[2]:
            return strip_pair
        
        return min_pair
    
    def closest_in_strip(strip: List[Tuple[float, float]], d: float) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Find closest pair in strip"""
        min_dist = d
        pair = (strip[0], strip[1]) if len(strip) >= 2 else ((0, 0), (0, 0))
        
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) < min_dist:
                dist = distance(strip[i], strip[j])
                if dist < min_dist:
                    min_dist = dist
                    pair = (strip[i], strip[j])
                j += 1
        
        return pair[0], pair[1], min_dist
    
    # Sort points by x and y coordinates
    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    
    return closest_pair_rec(px, py)

def divide_and_conquer_demo():
    """
    Demonstrate divide and conquer algorithms
    """
    print("=== Divide and Conquer Demo ===")
    
    # Merge sort example
    print("1. Merge Sort:")
    arr1 = [64, 34, 25, 12, 22, 11, 90]
    print(f"   Original array: {arr1}")
    sorted_arr1 = merge_sort(arr1)
    print(f"   Sorted array: {sorted_arr1}")
    
    # Quick sort example
    print("\n2. Quick Sort:")
    arr2 = [64, 34, 25, 12, 22, 11, 90]
    print(f"   Original array: {arr2}")
    sorted_arr2 = quick_sort(arr2)
    print(f"   Sorted array: {sorted_arr2}")
    
    # In-place quick sort
    print("\n3. In-place Quick Sort:")
    arr3 = [64, 34, 25, 12, 22, 11, 90]
    print(f"   Original array: {arr3}")
    quick_sort_inplace(arr3)
    print(f"   Sorted array: {arr3}")
    
    # Binary search example
    print("\n4. Binary Search:")
    sorted_arr = [11, 12, 22, 25, 34, 64, 90]
    targets = [25, 55, 11]
    print(f"   Sorted array: {sorted_arr}")
    for target in targets:
        index = binary_search(sorted_arr, target)
        if index != -1:
            print(f"   Found {target} at index {index}")
        else:
            print(f"   {target} not found")

def closest_pair_demo():
    """
    Demonstrate closest pair algorithm
    """
    print("\n=== Closest Pair Demo ===")
    
    # Generate random points
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
    print("1. Random Points:")
    for i, point in enumerate(points):
        print(f"   Point {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
    
    # Find closest pair
    p1, p2, distance = closest_pair(points)
    print(f"\n2. Closest Pair:")
    print(f"   Point 1: ({p1[0]:.2f}, {p1[1]:.2f})")
    print(f"   Point 2: ({p2[0]:.2f}, {p2[1]:.2f})")
    print(f"   Distance: {distance:.2f}")

def divide_and_conquer_applications():
    """
    Demonstrate applications of divide and conquer
    """
    print("\n=== Divide and Conquer Applications ===")
    
    # 1. Matrix multiplication (Strassen's algorithm concept)
    print("1. Matrix Multiplication:")
    print("   Standard algorithm: O(n³)")
    print("   Strassen's algorithm: O(n^2.807)")
    print("   Divide large matrices into smaller submatrices")
    
    # 2. Fast Fourier Transform (FFT)
    print("\n2. Fast Fourier Transform:")
    print("   Converts signal from time domain to frequency domain")
    print("   Time Complexity: O(n log n)")
    print("   Used in signal processing, polynomial multiplication")
    
    # 3. Karatsuba multiplication
    print("\n3. Karatsuba Multiplication:")
    print("   Fast multiplication of large numbers")
    print("   Time Complexity: O(n^1.585)")
    print("   Divide numbers into parts and reduce multiplications")
    
    # 4. Convex hull (Quickhull algorithm)
    print("\n4. Convex Hull:")
    print("   Find smallest convex polygon containing all points")
    print("   Quickhull algorithm uses divide and conquer")
    print("   Time Complexity: O(n log n) average, O(n²) worst case")

def data_science_applications():
    """
    Examples of divide and conquer in data science
    """
    print("\n=== Data Science Applications ===")
    
    # 1. K-D trees for nearest neighbor search
    print("1. K-D Trees:")
    print("   Spatial data structure for nearest neighbor search")
    print("   Used in recommendation systems and computer vision")
    print("   Build using divide and conquer approach")
    
    # 2. Decision trees in machine learning
    print("\n2. Decision Trees:")
    print("   Machine learning models built using divide and conquer")
    print("   Recursively partition data based on features")
    print("   Used in classification and regression tasks")
    
    # 3. Hierarchical clustering
    print("\n3. Hierarchical Clustering:")
    print("   Build cluster hierarchy using divide and conquer")
    print("   Agglomerative: bottom-up merging")
    print("   Divisive: top-down splitting")
    
    # 4. MapReduce framework
    print("\n4. MapReduce Framework:")
    print("   Distributed computing paradigm")
    print("   Map phase: divide data into chunks")
    print("   Reduce phase: combine results")
    print("   Used in big data processing")

def performance_comparison():
    """
    Compare performance of different divide and conquer algorithms
    """
    print("\n=== Performance Comparison ===")
    
    import time
    
    # Test with different array sizes
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTesting with {size} elements:")
        
        # Generate random data
        data = [random.randint(1, 100000) for _ in range(size)]
        
        # Merge sort
        arr1 = data.copy()
        start = time.time()
        merge_sort(arr1)
        merge_time = time.time() - start
        
        # Quick sort
        arr2 = data.copy()
        start = time.time()
        quick_sort(arr2)
        quick_time = time.time() - start
        
        # In-place quick sort
        arr3 = data.copy()
        start = time.time()
        quick_sort_inplace(arr3)
        quick_inplace_time = time.time() - start
        
        # Python's built-in sort (Timsort)
        arr4 = data.copy()
        start = time.time()
        arr4.sort()
        builtin_time = time.time() - start
        
        print(f"   Merge Sort: {merge_time:.6f}s")
        print(f"   Quick Sort: {quick_time:.6f}s")
        print(f"   In-place Quick Sort: {quick_inplace_time:.6f}s")
        print(f"   Python's sort: {builtin_time:.6f}s")

# Example usage and testing
if __name__ == "__main__":
    # Divide and conquer demo
    divide_and_conquer_demo()
    print("\n" + "="*50 + "\n")
    
    # Closest pair demo
    closest_pair_demo()
    print("\n" + "="*50 + "\n")
    
    # Applications
    divide_and_conquer_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Merge sort and quick sort implementations")
    print("2. Binary search algorithms")
    print("3. Closest pair of points algorithm")
    print("4. Applications in computer science and data science")
    print("5. Performance characteristics of divide and conquer algorithms")
    print("\nKey takeaways:")
    print("- Divide and conquer breaks problems into smaller subproblems")
    print("- Often provides optimal time complexity")
    print("- Recursion is a key component")
    print("- Many efficient algorithms use this paradigm")
    print("- Widely applicable in sorting, searching, and geometric problems")