"""
Beginner Project: Array Operations Library

This project implements a comprehensive array operations library that demonstrates
fundamental data structures and algorithms concepts.

Concepts covered:
- Array manipulation and traversal
- Time complexity analysis
- Memory management
- Error handling and edge cases
"""

import numpy as np
from typing import List, Union, Tuple


class ArrayOperations:
    """
    A comprehensive array operations library demonstrating fundamental DSA concepts.
    """
    
    @staticmethod
    def bubble_sort(arr: List[Union[int, float]]) -> List[Union[int, float]]:
        """
        Sort array using bubble sort algorithm.
        
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        arr = arr.copy()  # Don't modify original array
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            if not swapped:  # Early termination optimization
                break
        
        return arr
    
    @staticmethod
    def binary_search(arr: List[Union[int, float]], target: Union[int, float]) -> int:
        """
        Search for target in sorted array using binary search.
        
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
        
        return -1  # Not found
    
    @staticmethod
    def merge_sorted_arrays(arr1: List[Union[int, float]], 
                          arr2: List[Union[int, float]]) -> List[Union[int, float]]:
        """
        Merge two sorted arrays into one sorted array.
        
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        merged = []
        i = j = 0
        
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                merged.append(arr1[i])
                i += 1
            else:
                merged.append(arr2[j])
                j += 1
        
        # Add remaining elements
        while i < len(arr1):
            merged.append(arr1[i])
            i += 1
        
        while j < len(arr2):
            merged.append(arr2[j])
            j += 1
        
        return merged
    
    @staticmethod
    def rotate_array(arr: List[Union[int, float]], k: int) -> List[Union[int, float]]:
        """
        Rotate array to the right by k positions.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not arr or k == 0:
            return arr.copy()
        
        n = len(arr)
        k = k % n  # Handle cases where k > n
        
        # Reverse entire array
        arr = arr.copy()
        arr.reverse()
        
        # Reverse first k elements
        arr[:k] = reversed(arr[:k])
        
        # Reverse remaining elements
        arr[k:] = reversed(arr[k:])
        
        return arr
    
    @staticmethod
    def find_duplicates(arr: List[Union[int, float]]) -> List[Union[int, float]]:
        """
        Find all duplicate elements in array.
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        seen = set()
        duplicates = set()
        
        for element in arr:
            if element in seen:
                duplicates.add(element)
            else:
                seen.add(element)
        
        return list(duplicates)
    
    @staticmethod
    def maximum_subarray_sum(arr: List[Union[int, float]]) -> Union[int, float]:
        """
        Find maximum sum of contiguous subarray using Kadane's algorithm.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not arr:
            return 0
        
        max_sum = current_sum = arr[0]
        
        for i in range(1, len(arr)):
            current_sum = max(arr[i], current_sum + arr[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum


def demonstrate_array_operations():
    """Demonstrate all array operations."""
    print("=== Array Operations Library Demo ===\n")
    
    # Create sample arrays
    arr1 = [64, 34, 25, 12, 22, 11, 90]
    arr2 = [5, 15, 25, 35, 45]
    arr3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print(f"Original arrays:")
    print(f"  arr1: {arr1}")
    print(f"  arr2: {arr2}")
    print(f"  arr3: {arr3}")
    
    # Test bubble sort
    print(f"\n1. Bubble Sort:")
    sorted_arr = ArrayOperations.bubble_sort(arr1)
    print(f"  Sorted arr1: {sorted_arr}")
    
    # Test binary search
    print(f"\n2. Binary Search:")
    sorted_arr1 = sorted(arr1)
    target = 25
    index = ArrayOperations.binary_search(sorted_arr1, target)
    print(f"  Searching for {target} in {sorted_arr1}: index {index}")
    
    # Test merge sorted arrays
    print(f"\n3. Merge Sorted Arrays:")
    merged = ArrayOperations.merge_sorted_arrays(arr1, arr2)
    print(f"  Merged {sorted(arr1)} and {sorted(arr2)}: {sorted(merged)}")
    
    # Test rotate array
    print(f"\n4. Rotate Array:")
    rotated = ArrayOperations.rotate_array(arr3, 3)
    print(f"  Rotate {arr3} by 3 positions: {rotated}")
    
    # Test find duplicates
    print(f"\n5. Find Duplicates:")
    arr_with_duplicates = [1, 2, 3, 2, 4, 5, 3, 6]
    duplicates = ArrayOperations.find_duplicates(arr_with_duplicates)
    print(f"  Duplicates in {arr_with_duplicates}: {duplicates}")
    
    # Test maximum subarray sum
    print(f"\n6. Maximum Subarray Sum:")
    subarray_arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = ArrayOperations.maximum_subarray_sum(subarray_arr)
    print(f"  Maximum subarray sum in {subarray_arr}: {max_sum}")


def performance_comparison():
    """Compare performance of different array operations."""
    import time
    
    print("\n=== Performance Comparison ===\n")
    
    # Create large array for testing
    large_arr = list(range(10000, 0, -1))  # Reverse sorted array
    
    # Test bubble sort performance
    print("1. Bubble Sort Performance:")
    start_time = time.time()
    sorted_arr = ArrayOperations.bubble_sort(large_arr)
    bubble_time = time.time() - start_time
    print(f"   Time taken: {bubble_time:.6f} seconds")
    
    # Test binary search performance
    print("\n2. Binary Search Performance:")
    sorted_large_arr = sorted(large_arr)
    target = 5000
    
    start_time = time.time()
    for _ in range(1000):  # Perform 1000 searches
        index = ArrayOperations.binary_search(sorted_large_arr, target)
    binary_time = time.time() - start_time
    print(f"   Time for 1000 searches: {binary_time:.6f} seconds")
    print(f"   Average time per search: {binary_time/1000:.8f} seconds")
    
    # Test merge sorted arrays performance
    print("\n3. Merge Sorted Arrays Performance:")
    arr1_large = list(range(0, 5000, 2))  # Even numbers
    arr2_large = list(range(1, 5000, 2))  # Odd numbers
    
    start_time = time.time()
    merged_large = ArrayOperations.merge_sorted_arrays(arr1_large, arr2_large)
    merge_time = time.time() - start_time
    print(f"   Time taken: {merge_time:.6f} seconds")
    print(f"   Merged array size: {len(merged_large)}")
    
    # Test rotate array performance
    print("\n4. Rotate Array Performance:")
    start_time = time.time()
    rotated_large = ArrayOperations.rotate_array(large_arr, 1000)
    rotate_time = time.time() - start_time
    print(f"   Time taken: {rotate_time:.6f} seconds")


if __name__ == "__main__":
    demonstrate_array_operations()
    performance_comparison()