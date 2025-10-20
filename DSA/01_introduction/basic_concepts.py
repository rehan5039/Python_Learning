"""
Data Structures and Algorithms - Introduction
===========================================

This module provides a basic introduction to data structures and algorithms,
focusing on Python implementations and data science applications.

Topics Covered:
- What are Data Structures?
- What are Algorithms?
- Why are they important in data science?
- Python-specific implementations
"""

# Basic Data Structures in Python

# 1. Lists - Dynamic arrays
def list_example():
    """Demonstrate basic list operations"""
    # Creating a list
    numbers = [1, 2, 3, 4, 5]
    
    # Accessing elements
    first = numbers[0]  # O(1)
    last = numbers[-1]  # O(1)
    
    # Adding elements
    numbers.append(6)   # O(1) amortized
    numbers.insert(0, 0)  # O(n)
    
    # Removing elements
    numbers.pop()       # O(1)
    numbers.pop(0)      # O(n)
    
    return numbers

# 2. Dictionaries - Hash maps
def dict_example():
    """Demonstrate basic dictionary operations"""
    # Creating a dictionary
    student_grades = {
        "Alice": 85,
        "Bob": 92,
        "Charlie": 78
    }
    
    # Accessing elements
    alice_grade = student_grades["Alice"]  # O(1) average
    
    # Adding/updating elements
    student_grades["David"] = 88  # O(1) average
    
    # Removing elements
    del student_grades["Charlie"]  # O(1) average
    
    return student_grades

# 3. Sets - Unordered collections of unique elements
def set_example():
    """Demonstrate basic set operations"""
    # Creating a set
    unique_numbers = {1, 2, 3, 4, 5}
    
    # Adding elements
    unique_numbers.add(6)  # O(1) average
    
    # Removing elements
    unique_numbers.remove(1)  # O(1) average
    
    # Set operations
    set_a = {1, 2, 3, 4}
    set_b = {3, 4, 5, 6}
    
    union = set_a | set_b  # {1, 2, 3, 4, 5, 6}
    intersection = set_a & set_b  # {3, 4}
    difference = set_a - set_b  # {1, 2}
    
    return union, intersection, difference

# 4. Tuples - Immutable sequences
def tuple_example():
    """Demonstrate basic tuple operations"""
    # Creating a tuple
    coordinates = (10, 20)
    
    # Accessing elements
    x = coordinates[0]  # O(1)
    y = coordinates[1]  # O(1)
    
    # Tuples are immutable - this would raise an error:
    # coordinates[0] = 15  # TypeError
    
    # Tuples can be used as dictionary keys
    location_data = {
        (0, 0): "origin",
        (10, 20): "point A",
        (30, 40): "point B"
    }
    
    return coordinates, location_data

# Algorithm Example: Linear Search
def linear_search(arr, target):
    """
    Linear search algorithm - O(n) time complexity
    
    Args:
        arr: List to search in
        target: Element to find
    
    Returns:
        Index of target if found, -1 otherwise
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Algorithm Example: Binary Search (for sorted arrays)
def binary_search(arr, target):
    """
    Binary search algorithm - O(log n) time complexity
    
    Args:
        arr: Sorted list to search in
        target: Element to find
    
    Returns:
        Index of target if found, -1 otherwise
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
    
    return -1

# Data Science Application: Efficient Data Processing
def process_large_dataset(data):
    """
    Example of using appropriate data structures for data science
    
    Args:
        data: List of dictionaries representing dataset
    
    Returns:
        Processed results using efficient data structures
    """
    # Using dictionary for O(1) lookups
    processed_data = {}
    
    # Using set for unique values
    unique_categories = set()
    
    for record in data:
        # Efficient lookup and update
        category = record.get('category')
        value = record.get('value', 0)
        
        if category in processed_data:
            processed_data[category] += value
        else:
            processed_data[category] = value
        
        # Add to set for unique categories
        unique_categories.add(category)
    
    return {
        'aggregated_data': processed_data,
        'unique_categories': list(unique_categories),
        'total_records': len(data)
    }

# Example usage and testing
if __name__ == "__main__":
    # Test basic data structures
    print("=== Basic Data Structures ===")
    print("List example:", list_example())
    print("Dict example:", dict_example())
    print("Set example:", set_example())
    print("Tuple example:", tuple_example())
    
    # Test algorithms
    print("\n=== Algorithm Examples ===")
    test_array = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"Linear search for 7 in {test_array}: Index {linear_search(test_array, 7)}")
    print(f"Binary search for 7 in {test_array}: Index {binary_search(test_array, 7)}")
    
    # Test data science application
    print("\n=== Data Science Application ===")
    sample_data = [
        {'category': 'A', 'value': 100},
        {'category': 'B', 'value': 200},
        {'category': 'A', 'value': 150},
        {'category': 'C', 'value': 300},
        {'category': 'B', 'value': 250}
    ]
    
    result = process_large_dataset(sample_data)
    print("Processed data:", result)