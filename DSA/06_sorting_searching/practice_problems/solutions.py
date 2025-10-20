"""
Sorting and Searching - Practice Problem Solutions
==========================================

This file contains detailed solutions and explanations for the practice problems.
"""

# Solution 1: Basic Sorting Implementation
def solution_1():
    """
    Detailed solutions for basic sorting implementation problems:
    """
    
    print("Solution 1: Basic Sorting Implementation")
    print("=" * 40)
    
    # 1. Sort array of strings by length
    print("1. Sort Strings by Length:")
    print("Approach: Bubble sort with custom comparison")
    print("Time Complexity: O(n²), Space Complexity: O(1)")
    print("Key insight: Compare lengths instead of lexicographic order")
    print("Alternative approaches:")
    print("  - Using built-in sort with key parameter: O(n log n) time")
    print("  - Counting sort by length: O(n + k) time where k is max length")
    print()
    
    # 2. Sort array with custom comparator
    print("2. Custom Sorting Rule:")
    print("Approach: Using custom key function with stable sort")
    print("Time Complexity: O(n log n), Space Complexity: O(n)")
    print("Key insight: Create tuple key (priority, value) for complex sorting")
    print("Alternative approaches:")
    print("  - Partition approach: O(n) time, O(1) space")
    print()
    
    # 3. Stable sort implementation
    print("3. Stable Sort for Students:")
    print("Approach: Two-step sorting with stable algorithms")
    print("Time Complexity: O(n log n), Space Complexity: O(n)")
    print("Key insight: Sort by secondary key first, then primary key")
    print("Alternative approaches:")
    print("  - Single sort with tuple key: O(n log n) time")

# Solution 2: Searching Algorithm Practice
def solution_2():
    """
    Detailed solutions for searching algorithm practice problems:
    """
    
    print("Solution 2: Searching Algorithm Practice")
    print("=" * 40)
    
    # 1. Search in rotated sorted array
    print("1. Search in Rotated Sorted Array:")
    print("Approach: Modified binary search")
    print("Time Complexity: O(log n), Space Complexity: O(1)")
    print("Key insight: One half of rotated array is always sorted")
    print("Alternative approaches:")
    print("  - Find pivot then binary search: O(log n) time")
    print("  - Linear search: O(n) time but simpler")
    print()
    
    # 2. Find peak element
    print("2. Find Peak Element:")
    print("Approach: Binary search based on gradient")
    print("Time Complexity: O(log n), Space Complexity: O(1)")
    print("Key insight: Move toward increasing direction")
    print("Alternative approaches:")
    print("  - Linear scan: O(n) time")
    print()
    
    # 3. Search in matrix
    print("3. Search in Sorted Matrix:")
    print("Approach: Start from top-right corner")
    print("Time Complexity: O(m + n), Space Complexity: O(1)")
    print("Key insight: Eliminate row or column at each step")
    print("Alternative approaches:")
    print("  - Binary search on rows: O(m log n) time")
    print("  - Treat as 1D array with binary search: O(log(mn)) time")

# Solution 3: Advanced Sorting Problems
def solution_3():
    """
    Detailed solutions for advanced sorting problems:
    """
    
    print("Solution 3: Advanced Sorting Problems")
    print("=" * 35)
    
    # 1. Sort colors (Dutch National Flag problem)
    print("1. Sort Colors:")
    print("Approach: Three-way partitioning")
    print("Time Complexity: O(n), Space Complexity: O(1)")
    print("Key insight: Maintain three regions (0s, 1s, 2s) with three pointers")
    print("Alternative approaches:")
    print("  - Counting sort: O(n) time, O(1) space")
    print("  - Two-pass partitioning: O(n) time, O(1) space")
    print()
    
    # 2. Merge intervals
    print("2. Merge Intervals:")
    print("Approach: Sort then merge overlapping intervals")
    print("Time Complexity: O(n log n), Space Complexity: O(n)")
    print("Key insight: After sorting, only adjacent intervals can overlap")
    print("Alternative approaches:")
    print("  - Using interval tree: O(n log n) time, O(n) space")
    print()
    
    # 3. Top K frequent elements
    print("3. Top K Frequent Elements:")
    print("Approach: Hash map + min heap")
    print("Time Complexity: O(n log k), Space Complexity: O(n + k)")
    print("Key insight: Use min heap of size k to keep track of top k elements")
    print("Alternative approaches:")
    print("  - Sorting frequencies: O(n log n) time")
    print("  - Quickselect: O(n) average time")
    print("  - Bucket sort: O(n) time when elements are limited")

# Solution 4: Searching Variations
def solution_4():
    """
    Detailed solutions for searching variation problems:
    """
    
    print("Solution 4: Searching Variations")
    print("=" * 30)
    
    # 1. Find minimum in rotated sorted array
    print("1. Find Minimum in Rotated Sorted Array:")
    print("Approach: Binary search based on comparison with rightmost element")
    print("Time Complexity: O(log n), Space Complexity: O(1)")
    print("Key insight: Minimum element is in the unsorted half")
    print("Alternative approaches:")
    print("  - Linear scan: O(n) time")
    print()
    
    # 2. Search for range (first and last position)
    print("2. Search for Range:")
    print("Approach: Two binary searches for first and last positions")
    print("Time Complexity: O(log n), Space Complexity: O(1)")
    print("Key insight: Modify binary search to find boundaries")
    print("Alternative approaches:")
    print("  - Linear scan: O(n) time")
    print("  - Find one position then expand: O(n) worst case")
    print()
    
    # 3. Find kth largest element
    print("3. Find Kth Largest Element:")
    print("Approach: Quickselect (partition-based selection)")
    print("Time Complexity: O(n) average, O(n²) worst case")
    print("Space Complexity: O(1)")
    print("Key insight: Use partitioning to eliminate half of elements")
    print("Alternative approaches:")
    print("  - Sort then index: O(n log n) time")
    print("  - Min heap of size k: O(n log k) time")
    print("  - Max heap: O(n log n) time")

# Solution 5: Hybrid Algorithm Design
def solution_5():
    """
    Detailed solutions for hybrid algorithm design problems:
    """
    
    print("Solution 5: Hybrid Algorithm Design")
    print("=" * 32)
    
    # 1. Count smaller elements to the right
    print("1. Count Smaller Elements to Right:")
    print("Approach: Process right to left with binary insertion")
    print("Time Complexity: O(n log n), Space Complexity: O(n)")
    print("Key insight: Maintain sorted list of processed elements")
    print("Alternative approaches:")
    print("  - Binary Indexed Tree: O(n log m) time where m is value range")
    print("  - Segment Tree: O(n log m) time")
    print()
    
    # 2. Merge k sorted lists
    print("2. Merge K Sorted Lists:")
    print("Approach: Min heap to track smallest elements")
    print("Time Complexity: O(N log k) where N is total elements")
    print("Space Complexity: O(k)")
    print("Key insight: Always select minimum among current elements")
    print("Alternative approaches:")
    print("  - Sequential merging: O(kN) time")
    print("  - Divide and conquer: O(N log k) time")
    print()
    
    # 3. Find median of two sorted arrays
    print("3. Median of Two Sorted Arrays:")
    print("Approach: Binary search on smaller array")
    print("Time Complexity: O(log(min(m, n))), Space Complexity: O(1)")
    print("Key insight: Partition arrays such that left parts ≤ right parts")
    print("Alternative approaches:")
    print("  - Merge arrays then find median: O(m + n) time")
    print("  - Binary search on both arrays: O(log(m + n)) time")

# Additional Solutions and Explanations
def additional_solutions():
    """
    Additional solutions and advanced techniques:
    """
    
    print("Additional Solutions and Techniques")
    print("=" * 35)
    
    print("1. Algorithm Selection Guidelines:")
    print("   - Small datasets (< 50): Insertion sort")
    print("   - Nearly sorted: Insertion sort or bubble sort")
    print("   - Large datasets: Merge sort or quick sort")
    print("   - Limited range integers: Counting sort or radix sort")
    print("   - Memory constrained: Heap sort")
    print()
    
    print("2. Searching Strategy Selection:")
    print("   - Unsorted data: Linear search")
    print("   - Sorted data: Binary search")
    print("   - Uniform distribution: Interpolation search")
    print("   - Large datasets: Exponential search")
    print("   - Approximate results: Hash-based search")
    print()
    
    print("3. Optimization Techniques:")
    print("   - Early termination conditions")
    print("   - Hybrid algorithms for different input sizes")
    print("   - Preprocessing for repeated operations")
    print("   - Cache-friendly implementations")
    print("   - Parallel processing for large datasets")
    print()
    
    print("4. Common Pitfalls to Avoid:")
    print("   - Off-by-one errors in binary search")
    print("   - Not handling edge cases (empty arrays, single elements)")
    print("   - Incorrect partitioning in quick sort/select")
    print("   - Memory leaks in recursive implementations")
    print("   - Integer overflow in calculations")

# Run all solutions
if __name__ == "__main__":
    print("=== Sorting and Searching Practice Problem Solutions ===\n")
    
    solution_1()
    print("\n" + "="*50 + "\n")
    
    solution_2()
    print("\n" + "="*50 + "\n")
    
    solution_3()
    print("\n" + "="*50 + "\n")
    
    solution_4()
    print("\n" + "="*50 + "\n")
    
    solution_5()
    print("\n" + "="*50 + "\n")
    
    additional_solutions()
    print("\n" + "="*50 + "\n")
    
    print("=== Summary ===")
    print("These solutions demonstrate:")
    print("1. Multiple approaches to the same problem")
    print("2. Time and space complexity analysis")
    print("3. Trade-offs between different implementations")
    print("4. Advanced techniques and optimizations")
    print("5. Mathematical principles behind algorithms")
    print("\nKey takeaways:")
    print("- Understand problem requirements before choosing approach")
    print("- Consider both time and space complexity")
    print("- Think about edge cases and error handling")
    print("- Know multiple solutions for common problems")
    print("- Practice implementing algorithms from scratch")