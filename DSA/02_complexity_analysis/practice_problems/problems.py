"""
Complexity Analysis - Practice Problems
====================================

This file contains practice problems for complexity analysis with solutions.
"""

# Problem 1: Basic Complexity Identification
def problem_1():
    """
    Identify the time complexity of the following functions:
    """
    
    # Function A
    def func_a(n):
        for i in range(n):
            print(i)
    
    # Function B
    def func_b(n):
        for i in range(n):
            for j in range(n):
                print(i, j)
    
    # Function C
    def func_c(n):
        i = 1
        while i < n:
            print(i)
            i *= 2
    
    # Function D
    def func_d(n):
        if n <= 1:
            return 1
        return func_d(n-1) + func_d(n-2)
    
    print("Problem 1: Identify the time complexity of each function")
    print("Function A: O(n) - Single loop from 0 to n-1")
    print("Function B: O(n²) - Nested loops, both from 0 to n-1")
    print("Function C: O(log n) - i doubles each iteration, so log₂(n) iterations")
    print("Function D: O(2^n) - Recursive Fibonacci without memoization")

# Problem 2: Common Algorithm Analysis
def problem_2():
    """
    Analyze the complexity of these common operations:
    """
    
    # Find maximum element
    def find_max(arr):
        max_val = arr[0]
        for i in range(1, len(arr)):
            if arr[i] > max_val:
                max_val = arr[i]
        return max_val
    
    # Binary search
    def binary_search(arr, target):
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
    
    print("Problem 2: Analyze complexity of common operations")
    print("Find Maximum: O(n) - Single pass through array")
    print("Binary Search: O(log n) - Halves search space each iteration")

# Problem 3: Recurrence Relations
def problem_3():
    """
    Solve these recurrence relations:
    """
    
    print("Problem 3: Solve recurrence relations")
    print("1. T(n) = 2T(n/2) + n → O(n log n) - Merge sort type")
    print("2. T(n) = T(n-1) + 1 → O(n) - Linear recursion")
    print("3. T(n) = 4T(n/2) + n² → O(n² log n) - Using Master Theorem")
    print("4. T(n) = T(n/3) + T(2n/3) + n → O(n log n) - Uneven divide")

# Problem 4: Amortized Analysis
def problem_4():
    """
    Analyze the amortized complexity:
    """
    
    print("Problem 4: Amortized analysis")
    print("Dynamic Array Append:")
    print("- Individual append: O(1) best case, O(n) worst case")
    print("- Amortized: O(1) - Cost is distributed over many operations")
    print()
    print("Stack with Multi-Pop:")
    print("- Pop operation: O(1)")
    print("- Multi-Pop(k): O(k) worst case")
    print("- Amortized for sequence of n operations: O(1) per operation")

# Problem 5: Real-World Scenarios
def problem_5():
    """
    Analyze real-world algorithm complexities:
    """
    
    print("Problem 5: Real-world complexity analysis")
    print("1. Database Index Lookup: O(log n) - B-tree index")
    print("2. Linear Search in Database: O(n) - Full table scan")
    print("3. Hash Table Lookup: O(1) average, O(n) worst case")
    print("4. Sorting in Database: O(n log n) - Typically merge sort or similar")
    print()
    print("5. K-means Clustering: O(n * k * i * d)")
    print("   where n=samples, k=clusters, i=iterations, d=dimensions")
    print("6. Decision Tree Training: O(n * log n) for sorting-based algorithms")

# Problem 6: Optimization Challenges
def problem_6():
    """
    Optimize these inefficient algorithms:
    """
    
    # Inefficient approach to find duplicates
    def find_duplicates_slow(arr):
        """O(n²) approach"""
        duplicates = []
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if arr[i] == arr[j] and arr[i] not in duplicates:
                    duplicates.append(arr[i])
        return duplicates
    
    # Efficient approach to find duplicates
    def find_duplicates_fast(arr):
        """O(n) approach using hash set"""
        seen = set()
        duplicates = set()
        for item in arr:
            if item in seen:
                duplicates.add(item)
            else:
                seen.add(item)
        return list(duplicates)
    
    print("Problem 6: Algorithm optimization")
    print("Finding Duplicates:")
    print("- Slow approach: O(n²) time, O(1) space")
    print("- Fast approach: O(n) time, O(n) space")
    print("- Trade-off: Time vs Space complexity")

# Run all problems
if __name__ == "__main__":
    print("=== Complexity Analysis Practice Problems ===\n")
    
    problem_1()
    print("\n" + "="*50 + "\n")
    
    problem_2()
    print("\n" + "="*50 + "\n")
    
    problem_3()
    print("\n" + "="*50 + "\n")
    
    problem_4()
    print("\n" + "="*50 + "\n")
    
    problem_5()
    print("\n" + "="*50 + "\n")
    
    problem_6()
    print("\n" + "="*50 + "\n")
    
    print("=== Summary ===")
    print("These practice problems covered:")
    print("1. Basic complexity identification")
    print("2. Analysis of common algorithms")
    print("3. Solving recurrence relations")
    print("4. Amortized analysis concepts")
    print("5. Real-world complexity scenarios")
    print("6. Algorithm optimization techniques")