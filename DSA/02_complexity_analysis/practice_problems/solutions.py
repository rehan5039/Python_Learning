"""
Complexity Analysis - Practice Problem Solutions
============================================

This file contains detailed solutions and explanations for the practice problems.
"""

# Solution 1: Basic Complexity Identification
def solution_1():
    """
    Detailed solutions for basic complexity identification:
    """
    
    print("Solution 1: Basic Complexity Identification")
    print("=" * 40)
    
    # Function A - O(n)
    print("Function A:")
    print("Time Complexity: O(n)")
    print("Explanation: Single loop that iterates n times")
    print("Space Complexity: O(1)")
    print("Explanation: No additional space that grows with input")
    print()
    
    # Function B - O(n²)
    print("Function B:")
    print("Time Complexity: O(n²)")
    print("Explanation: Nested loops, both iterate n times, resulting in n*n operations")
    print("Space Complexity: O(1)")
    print("Explanation: No additional space that grows with input")
    print()
    
    # Function C - O(log n)
    print("Function C:")
    print("Time Complexity: O(log n)")
    print("Explanation: Loop where i doubles each iteration, so it runs log₂(n) times")
    print("Space Complexity: O(1)")
    print("Explanation: No additional space that grows with input")
    print()
    
    # Function D - O(2^n)
    print("Function D:")
    print("Time Complexity: O(2^n)")
    print("Explanation: Recursive Fibonacci without memoization, creates binary tree of calls")
    print("Space Complexity: O(n)")
    print("Explanation: Maximum recursion depth is n, so O(n) stack space")

# Solution 2: Common Algorithm Analysis
def solution_2():
    """
    Detailed solutions for common algorithm analysis:
    """
    
    print("Solution 2: Common Algorithm Analysis")
    print("=" * 40)
    
    # Find maximum element - O(n)
    print("Find Maximum Element:")
    print("Time Complexity: O(n)")
    print("Explanation: Single pass through array, comparing each element once")
    print("Space Complexity: O(1)")
    print("Explanation: Only storing the current maximum value")
    print()
    
    # Binary search - O(log n)
    print("Binary Search:")
    print("Time Complexity: O(log n)")
    print("Explanation: Search space is halved in each iteration, so log₂(n) iterations")
    print("Space Complexity: O(1)")
    print("Explanation: Only storing left, right, and mid indices")

# Solution 3: Recurrence Relations
def solution_3():
    """
    Detailed solutions for recurrence relations:
    """
    
    print("Solution 3: Recurrence Relations")
    print("=" * 40)
    
    print("1. T(n) = 2T(n/2) + n")
    print("   Using Master Theorem: a=2, b=2, f(n)=n")
    print("   log_b(a) = log₂(2) = 1, f(n) = n¹")
    print("   Case 2: f(n) = Θ(n^(log_b(a))) → T(n) = Θ(n log n)")
    print()
    
    print("2. T(n) = T(n-1) + 1")
    print("   Expanding: T(n) = T(n-1) + 1 = T(n-2) + 2 = ... = T(1) + (n-1)")
    print("   Therefore: T(n) = O(n)")
    print()
    
    print("3. T(n) = 4T(n/2) + n²")
    print("   Using Master Theorem: a=4, b=2, f(n)=n²")
    print("   log_b(a) = log₂(4) = 2, f(n) = n²")
    print("   Case 2: f(n) = Θ(n^(log_b(a))) → T(n) = Θ(n² log n)")
    print()
    
    print("4. T(n) = T(n/3) + T(2n/3) + n")
    print("   This doesn't fit the Master Theorem directly")
    print("   Using recursion tree method: Each level does O(n) work")
    print("   Depth of tree is log_{3/2}(n) = O(log n)")
    print("   Therefore: T(n) = O(n log n)")

# Solution 4: Amortized Analysis
def solution_4():
    """
    Detailed solutions for amortized analysis:
    """
    
    print("Solution 4: Amortized Analysis")
    print("=" * 40)
    
    print("Dynamic Array Append:")
    print("Worst case: O(n) when resizing is needed")
    print("However, resizing happens infrequently")
    print("Amortized analysis:")
    print("- First resize at 1 element (cost 1)")
    print("- Second resize at 2 elements (cost 2)")
    print("- Third resize at 4 elements (cost 4)")
    print("- k-th resize at 2^k elements (cost 2^k)")
    print("- Total cost for n operations: 1 + 2 + 4 + ... + n = 2n - 1")
    print("- Amortized cost per operation: O(n)/n = O(1)")
    print()
    
    print("Stack with Multi-Pop:")
    print("Each element can be pushed at most once")
    print("Each element can be popped at most once")
    print("Total pops across all operations ≤ Total pushes")
    print("Therefore, amortized cost per operation is O(1)")

# Solution 5: Real-World Scenarios
def solution_5():
    """
    Detailed solutions for real-world complexity analysis:
    """
    
    print("Solution 5: Real-World Scenarios")
    print("=" * 40)
    
    print("1. Database Index Lookup - O(log n)")
    print("   B-trees keep data sorted and balanced")
    print("   Height of B-tree is O(log n)")
    print()
    
    print("2. Linear Search in Database - O(n)")
    print("   Must check each row until finding target or end")
    print()
    
    print("3. Hash Table Lookup - O(1) average")
    print("   Direct access via hash function")
    print("   Worst case O(n) when all keys hash to same bucket")
    print()
    
    print("4. K-means Clustering - O(n * k * i * d)")
    print("   For each of i iterations:")
    print("   - Assign n points to k clusters: O(n * k * d)")
    print("   - Update k cluster centers: O(n * k * d)")
    print("   Total: O(n * k * i * d)")

# Solution 6: Optimization Challenges
def solution_6():
    """
    Detailed solutions for optimization challenges:
    """
    
    print("Solution 6: Optimization Challenges")
    print("=" * 40)
    
    print("Finding Duplicates:")
    print("Slow approach - O(n²) time, O(1) space:")
    print("- Nested loops checking each pair")
    print("- No extra space needed")
    print()
    
    print("Fast approach - O(n) time, O(n) space:")
    print("- Single pass with hash set to track seen elements")
    print("- Extra space to store seen elements")
    print()
    
    print("Trade-off Analysis:")
    print("- Time: O(n²) vs O(n) - Significant improvement for large inputs")
    print("- Space: O(1) vs O(n) - Uses more memory")
    print("- Choice depends on constraints: memory vs speed requirements")

# Run all solutions
if __name__ == "__main__":
    print("=== Complexity Analysis Practice Problem Solutions ===\n")
    
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
    
    solution_6()
    print("\n" + "="*50 + "\n")
    
    print("=== Summary ===")
    print("These solutions demonstrate:")
    print("1. Systematic approach to complexity analysis")
    print("2. Application of Master Theorem and other techniques")
    print("3. Amortized analysis methods")
    print("4. Real-world complexity considerations")
    print("5. Trade-off analysis between time and space complexity")