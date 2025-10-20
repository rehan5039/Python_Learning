"""
Linear Data Structures - Arrays and Lists
=====================================

This module provides implementations and examples of array and list data structures,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Static arrays vs dynamic arrays
- Python lists as dynamic arrays
- Array operations and their complexities
- Multidimensional arrays
- Applications in data science
"""

import array
from typing import List, Any, Union

class StaticArray:
    """
    Implementation of a static array with fixed size.
    In Python, we use the array module for true static arrays.
    """
    
    def __init__(self, size: int, typecode: str = 'i'):
        """
        Initialize a static array.
        
        Args:
            size: Fixed size of the array
            typecode: Type of elements ('i' for integers, 'f' for floats, etc.)
        """
        self._array = array.array(typecode, [0] * size)
        self._size = size
        self._typecode = typecode
    
    def __getitem__(self, index: int) -> Any:
        """Access element at index - O(1)"""
        if 0 <= index < self._size:
            return self._array[index]
        raise IndexError("Array index out of range")
    
    def __setitem__(self, index: int, value: Any) -> None:
        """Set element at index - O(1)"""
        if 0 <= index < self._size:
            self._array[index] = value
        else:
            raise IndexError("Array index out of range")
    
    def __len__(self) -> int:
        """Return the size of the array - O(1)"""
        return self._size
    
    def __str__(self) -> str:
        """String representation of the array"""
        return str(self._array.tolist())

class DynamicArray:
    """
    Implementation of a dynamic array that grows as needed.
    Similar to Python's built-in list.
    """
    
    def __init__(self):
        """Initialize an empty dynamic array"""
        self._data = []
        self._size = 0
    
    def append(self, item: Any) -> None:
        """
        Add an item to the end of the array - O(1) amortized
        """
        self._data.append(item)
        self._size += 1
    
    def insert(self, index: int, item: Any) -> None:
        """
        Insert an item at a specific index - O(n)
        """
        if 0 <= index <= self._size:
            self._data.insert(index, item)
            self._size += 1
        else:
            raise IndexError("Index out of range")
    
    def remove(self, item: Any) -> None:
        """
        Remove the first occurrence of an item - O(n)
        """
        self._data.remove(item)
        self._size -= 1
    
    def pop(self, index: int = -1) -> Any:
        """
        Remove and return item at index (default last) - O(n) or O(1) for last
        """
        if 0 <= index < self._size or (index == -1 and self._size > 0):
            item = self._data.pop(index)
            self._size -= 1
            return item
        raise IndexError("Index out of range")
    
    def __getitem__(self, index: int) -> Any:
        """Access element at index - O(1)"""
        return self._data[index]
    
    def __setitem__(self, index: int, value: Any) -> None:
        """Set element at index - O(1)"""
        self._data[index] = value
    
    def __len__(self) -> int:
        """Return the size of the array - O(1)"""
        return self._size
    
    def __str__(self) -> str:
        """String representation of the array"""
        return str(self._data)

def array_operations_demo():
    """
    Demonstrate common array operations and their complexities
    """
    print("=== Array Operations Demo ===")
    
    # Static array example
    print("1. Static Array (size 5):")
    static_arr = StaticArray(5)
    for i in range(5):
        static_arr[i] = i * 2
    print(f"   Array: {static_arr}")
    print(f"   Access element at index 2: {static_arr[2]} (O(1))")
    
    # Dynamic array example
    print("\n2. Dynamic Array:")
    dynamic_arr = DynamicArray()
    for i in range(5):
        dynamic_arr.append(i * 3)
    print(f"   Array: {[dynamic_arr[i] for i in range(len(dynamic_arr))]}")
    print(f"   Length: {len(dynamic_arr)} (O(1))")
    
    # Insert operation
    dynamic_arr.insert(2, 99)
    print(f"   After inserting 99 at index 2: {[dynamic_arr[i] for i in range(len(dynamic_arr))]} (O(n))")
    
    # Remove operation
    dynamic_arr.remove(99)
    print(f"   After removing 99: {[dynamic_arr[i] for i in range(len(dynamic_arr))]} (O(n))")

def multidimensional_arrays():
    """
    Examples of multidimensional arrays
    """
    print("\n=== Multidimensional Arrays ===")
    
    # 2D array using lists of lists
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    print("2D Matrix:")
    for row in matrix:
        print(f"   {row}")
    
    # Accessing elements
    print(f"\nElement at [1][2]: {matrix[1][2]} (O(1))")
    
    # Using NumPy for efficient multidimensional arrays (data science)
    try:
        import numpy as np
        print("\nUsing NumPy for multidimensional arrays:")
        np_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(f"NumPy Matrix:\n{np_matrix}")
        print(f"Element at [1][2]: {np_matrix[1, 2]} (O(1))")
        print(f"Matrix shape: {np_matrix.shape}")
    except ImportError:
        print("\nNumPy not available. Install with: pip install numpy")

def data_science_applications():
    """
    Examples of array applications in data science
    """
    print("\n=== Data Science Applications ===")
    
    # Example 1: Feature vectors in machine learning
    print("1. Feature Vectors in Machine Learning:")
    print("   Each row in a dataset can be represented as an array of features")
    print("   Example: [age, income, education_level] = [25, 50000, 16]")
    
    # Example 2: Time series data
    print("\n2. Time Series Data:")
    print("   Stock prices over time: [100.5, 101.2, 99.8, 102.1, 103.4]")
    time_series = [100.5, 101.2, 99.8, 102.1, 103.4]
    print(f"   Average price: {sum(time_series) / len(time_series):.2f}")
    
    # Example 3: Image representation
    print("\n3. Image Representation:")
    print("   Grayscale image as 2D array of pixel intensities (0-255)")
    grayscale_image = [
        [0, 50, 100, 50, 0],
        [50, 150, 200, 150, 50],
        [100, 200, 255, 200, 100],
        [50, 150, 200, 150, 50],
        [0, 50, 100, 50, 0]
    ]
    print("   Simple 5x5 grayscale image:")
    for row in grayscale_image:
        print(f"   {row}")

def performance_comparison():
    """
    Compare performance of different array implementations
    """
    print("\n=== Performance Comparison ===")
    
    import time
    
    # Test with different sizes
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nArray size: {size}")
        
        # Python list
        start = time.time()
        py_list = []
        for i in range(size):
            py_list.append(i)
        end = time.time()
        print(f"   Python list append: {end - start:.6f} seconds")
        
        # DynamicArray implementation
        start = time.time()
        dyn_arr = DynamicArray()
        for i in range(size):
            dyn_arr.append(i)
        end = time.time()
        print(f"   DynamicArray append: {end - start:.6f} seconds")
        
        # Access operations
        start = time.time()
        for i in range(0, size, 100):  # Access every 100th element
            _ = py_list[i]
        end = time.time()
        print(f"   Python list access: {end - start:.6f} seconds")

def common_algorithms():
    """
    Common algorithms using arrays
    """
    print("\n=== Common Array Algorithms ===")
    
    # Linear search
    def linear_search(arr: List[int], target: int) -> int:
        """O(n) time complexity"""
        for i, value in enumerate(arr):
            if value == target:
                return i
        return -1
    
    # Binary search (requires sorted array)
    def binary_search(arr: List[int], target: int) -> int:
        """O(log n) time complexity"""
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    # Array reversal
    def reverse_array(arr: List[int]) -> List[int]:
        """O(n) time complexity"""
        return arr[::-1]
    
    # Example usage
    test_array = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"Test array: {test_array}")
    print(f"Linear search for 7: Index {linear_search(test_array, 7)}")
    print(f"Binary search for 7: Index {binary_search(test_array, 7)}")
    print(f"Reversed array: {reverse_array(test_array)}")

# Example usage and testing
if __name__ == "__main__":
    # Demonstrate array operations
    array_operations_demo()
    print("\n" + "="*50 + "\n")
    
    # Multidimensional arrays
    multidimensional_arrays()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    print("\n" + "="*50 + "\n")
    
    # Common algorithms
    common_algorithms()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Static and dynamic array implementations")
    print("2. Common array operations and their complexities")
    print("3. Multidimensional arrays")
    print("4. Data science applications of arrays")
    print("5. Performance comparisons")
    print("6. Common algorithms using arrays")
    print("\nKey takeaways:")
    print("- Arrays provide O(1) access time")
    print("- Dynamic arrays (like Python lists) provide flexibility")
    print("- Static arrays are more memory efficient")
    print("- Arrays are fundamental in data science for storing numerical data")