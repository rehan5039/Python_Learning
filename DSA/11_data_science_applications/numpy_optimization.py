"""
NumPy Optimization with DSA Principles

This module demonstrates how to optimize NumPy operations using Data Structures and Algorithms:
- Efficient array operations and broadcasting
- Memory management techniques
- Vectorization for performance improvement
- Advanced indexing and slicing strategies
"""

import numpy as np
from typing import List, Tuple
import time


def efficient_array_creation(shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
    """
    Create arrays efficiently using appropriate methods.
    
    Time Complexity: O(n) where n is number of elements
    Space Complexity: O(n)
    
    Args:
        shape: Shape of the array
        dtype: Data type of the array
        
    Returns:
        Efficiently created NumPy array
    """
    # Use appropriate creation functions
    if len(shape) == 1 and shape[0] == 0:
        return np.array([], dtype=dtype)
    elif len(shape) == 1:
        # For 1D arrays, consider pre-allocation if size is known
        return np.empty(shape, dtype=dtype)
    else:
        # For multi-dimensional arrays
        return np.empty(shape, dtype=dtype)


def vectorized_operations(arr: np.ndarray) -> np.ndarray:
    """
    Perform vectorized operations instead of loops for better performance.
    
    Time Complexity: O(n) where n is number of elements
    Space Complexity: O(n)
    
    Args:
        arr: Input array
        
    Returns:
        Array with operations applied
    """
    # Vectorized mathematical operations
    result = np.sqrt(np.abs(arr))  # Instead of loop with math.sqrt
    result = np.where(result > 1, result, 0)  # Vectorized conditional operations
    
    return result


def broadcasting_optimization(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Optimize operations using NumPy broadcasting rules.
    
    Time Complexity: O(n * m) where n, m are array sizes
    Space Complexity: O(n * m)
    
    Args:
        arr1: First array
        arr2: Second array
        
    Returns:
        Result of broadcasting operation
    """
    # NumPy automatically handles broadcasting
    try:
        result = arr1 + arr2  # Broadcasting will occur automatically
        return result
    except ValueError as e:
        print(f"Broadcasting error: {e}")
        return np.array([])


def memory_efficient_operations(arr: np.ndarray) -> np.ndarray:
    """
    Perform memory-efficient operations using in-place modifications.
    
    Time Complexity: O(n)
    Space Complexity: O(1) additional space
    
    Args:
        arr: Input array
        
    Returns:
        Modified array (in-place)
    """
    # In-place operations to save memory
    arr += 1  # Instead of arr = arr + 1
    arr *= 2  # Instead of arr = arr * 2
    np.sqrt(arr, out=arr)  # In-place square root
    
    return arr


def advanced_indexing(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Use advanced indexing techniques for efficient array access.
    
    Time Complexity: O(k) where k is number of indices
    Space Complexity: O(k)
    
    Args:
        arr: Input array
        indices: Array of indices
        
    Returns:
        Array elements at specified indices
    """
    # Fancy indexing
    return arr[indices]


def sorting_optimization(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize sorting operations using appropriate algorithms.
    
    Time Complexity: O(n log n) for efficient sorting
    Space Complexity: O(n)
    
    Args:
        arr: Input array
        
    Returns:
        Tuple of (sorted_array, sort_indices)
    """
    # Use np.argsort for indices, np.sort for sorted array
    sorted_indices = np.argsort(arr)
    sorted_array = arr[sorted_indices]
    
    return sorted_array, sorted_indices


def aggregation_optimization(arr: np.ndarray, axis: int = None) -> dict:
    """
    Optimize aggregation operations using vectorized functions.
    
    Time Complexity: O(n) where n is number of elements
    Space Complexity: O(1)
    
    Args:
        arr: Input array
        axis: Axis along which to perform operations
        
    Returns:
        Dictionary of aggregation results
    """
    results = {
        'sum': np.sum(arr, axis=axis),
        'mean': np.mean(arr, axis=axis),
        'std': np.std(arr, axis=axis),
        'min': np.min(arr, axis=axis),
        'max': np.max(arr, axis=axis)
    }
    
    return results


def matrix_operations_optimization(matrix_a: np.ndarray, matrix_b: np.ndarray) -> dict:
    """
    Optimize matrix operations using appropriate NumPy functions.
    
    Time Complexity: O(n^3) for matrix multiplication
    Space Complexity: O(n^2)
    
    Args:
        matrix_a: First matrix
        matrix_b: Second matrix
        
    Returns:
        Dictionary of matrix operation results
    """
    results = {}
    
    # Matrix multiplication
    if matrix_a.shape[1] == matrix_b.shape[0]:
        results['dot_product'] = np.dot(matrix_a, matrix_b)
    
    # Element-wise operations
    if matrix_a.shape == matrix_b.shape:
        results['element_wise_add'] = np.add(matrix_a, matrix_b)
        results['element_wise_multiply'] = np.multiply(matrix_a, matrix_b)
    
    # Matrix properties
    results['transpose_a'] = matrix_a.T
    if matrix_a.shape[0] == matrix_a.shape[1]:
        results['determinant_a'] = np.linalg.det(matrix_a)
        results['inverse_a'] = np.linalg.inv(matrix_a) if np.linalg.det(matrix_a) != 0 else None
    
    return results


def performance_comparison():
    """Compare performance of different NumPy optimization techniques."""
    print("=== NumPy Optimization Performance Comparison ===\n")
    
    # Create sample arrays
    size = 1000000
    arr1 = np.random.randn(size)
    arr2 = np.random.randn(size)
    
    # Test vectorized operations
    print("1. Vectorized Operations:")
    start_time = time.time()
    result1 = vectorized_operations(arr1)
    vectorized_time = time.time() - start_time
    print(f"   Vectorized time: {vectorized_time:.6f} seconds")
    
    # Compare with loop (inefficient)
    start_time = time.time()
    result2 = np.array([np.sqrt(abs(x)) for x in arr1])
    loop_time = time.time() - start_time
    print(f"   Loop time: {loop_time:.6f} seconds")
    print(f"   Speedup: {loop_time / vectorized_time:.1f}x")
    
    # Test broadcasting
    print("\n2. Broadcasting:")
    matrix_2d = np.random.randn(1000, 1000)
    vector_1d = np.random.randn(1000)
    
    start_time = time.time()
    broadcast_result = broadcasting_optimization(matrix_2d, vector_1d)
    broadcast_time = time.time() - start_time
    print(f"   Broadcasting time: {broadcast_time:.6f} seconds")
    print(f"   Result shape: {broadcast_result.shape}")
    
    # Test memory efficiency
    print("\n3. Memory Efficiency:")
    test_array = np.random.randn(100000)
    original_memory = test_array.nbytes
    
    start_time = time.time()
    memory_efficient_operations(test_array)
    memory_time = time.time() - start_time
    print(f"   Memory efficient time: {memory_time:.6f} seconds")
    print(f"   Memory usage: {original_memory / 1024 / 1024:.2f} MB")
    
    # Test sorting optimization
    print("\n4. Sorting Optimization:")
    unsorted_array = np.random.randn(100000)
    
    start_time = time.time()
    sorted_arr, sort_indices = sorting_optimization(unsorted_array)
    sort_time = time.time() - start_time
    print(f"   Sorting time: {sort_time:.6f} seconds")
    
    # Test aggregation optimization
    print("\n5. Aggregation Optimization:")
    start_time = time.time()
    agg_results = aggregation_optimization(arr1)
    agg_time = time.time() - start_time
    print(f"   Aggregation time: {agg_time:.6f} seconds")
    print(f"   Mean: {agg_results['mean']:.4f}")


def demo():
    """Demonstrate NumPy optimization techniques."""
    print("=== NumPy Optimization with DSA ===\n")
    
    # Create sample arrays
    arr_1d = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    arr_2d = np.random.randn(5, 5)
    indices = np.array([0, 2, 4, 6, 8])
    
    print("Original 1D array:")
    print(arr_1d)
    
    # Vectorized operations
    vec_result = vectorized_operations(arr_1d)
    print(f"\nVectorized operations result:")
    print(vec_result)
    
    # Advanced indexing
    indexed_result = advanced_indexing(arr_1d, indices)
    print(f"\nAdvanced indexing result:")
    print(indexed_result)
    
    # Sorting optimization
    sorted_arr, sort_indices = sorting_optimization(arr_1d)
    print(f"\nSorting results:")
    print(f"  Sorted array: {sorted_arr}")
    print(f"  Sort indices: {sort_indices}")
    
    # Aggregation optimization
    agg_results = aggregation_optimization(arr_1d)
    print(f"\nAggregation results:")
    for key, value in agg_results.items():
        print(f"  {key}: {value}")
    
    # Matrix operations
    matrix_a = np.random.randn(3, 3)
    matrix_b = np.random.randn(3, 3)
    matrix_results = matrix_operations_optimization(matrix_a, matrix_b)
    print(f"\nMatrix operations results:")
    for key, value in matrix_results.items():
        if value is not None:
            print(f"  {key}: shape {getattr(value, 'shape', 'scalar')}")
    
    # Performance comparison
    print("\n" + "="*60)
    performance_comparison()


if __name__ == "__main__":
    demo()