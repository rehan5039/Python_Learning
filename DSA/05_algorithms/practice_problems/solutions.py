"""
Algorithm Design Techniques - Practice Problem Solutions
================================================

This file contains detailed solutions and explanations for the practice problems.
"""

# Solution 1: Divide and Conquer
def solution_1():
    """
    Detailed solutions for divide and conquer problems:
    """
    
    print("Solution 1: Divide and Conquer")
    print("=" * 30)
    
    # 1. Peak element in array
    print("1. Peak Element:")
    print("Approach: Binary search to find peak element")
    print("Time Complexity: O(log n), Space Complexity: O(log n)")
    print("Key insight: Compare middle element with neighbors to determine search direction")
    print("Alternative approaches:")
    print("  - Linear scan: O(n) time, O(1) space")
    print()
    
    # 2. Count inversions in array
    print("2. Count Inversions:")
    print("Approach: Modified merge sort to count inversions during merge")
    print("Time Complexity: O(n log n), Space Complexity: O(n)")
    print("Key insight: During merge, if element from right array is smaller,")
    print("             it forms inversions with all remaining elements in left array")
    print("Alternative approaches:")
    print("  - Brute force: O(n²) time, O(1) space")
    print("  - Using balanced BST: O(n log n) time, O(n) space")
    print()

# Solution 2: Greedy Algorithms
def solution_2():
    """
    Detailed solutions for greedy algorithm problems:
    """
    
    print("Solution 2: Greedy Algorithms")
    print("=" * 30)
    
    # 1. Job scheduling
    print("1. Job Scheduling:")
    print("Approach: Greedy by profit (sort by descending profit)")
    print("Time Complexity: O(n²), Space Complexity: O(1)")
    print("Key insight: Always select job with highest profit that can be completed")
    print("Alternative approaches:")
    print("  - Greedy by deadline: O(n log n) time")
    print()
    
    # 2. Minimum number of platforms
    print("2. Minimum Platforms:")
    print("Approach: Sort and use two pointers to track overlaps")
    print("Time Complexity: O(n log n), Space Complexity: O(1)")
    print("Key insight: Maximum platforms needed equals maximum concurrent trains")
    print("Alternative approaches:")
    - Using auxiliary array: O(max_time) time, O(max_time) space")
    print()

# Solution 3: Dynamic Programming
def solution_3():
    """
    Detailed solutions for dynamic programming problems:
    """
    
    print("Solution 3: Dynamic Programming")
    print("=" * 30)
    
    # 1. Rod cutting problem
    print("1. Rod Cutting:")
    print("Approach: Bottom-up DP with optimal substructure")
    print("Time Complexity: O(n²), Space Complexity: O(n)")
    print("Key insight: For rod of length i, try all possible cuts and take maximum")
    print("Alternative approaches:")
    print("  - Top-down with memoization: O(n²) time, O(n) space")
    print("  - Brute force recursion: O(2^n) time")
    print()
    
    # 2. Matrix chain multiplication
    print("2. Matrix Chain Multiplication:")
    print("Approach: DP with 2D table for chain lengths")
    print("Time Complexity: O(n³), Space Complexity: O(n²)")
    print("Key insight: Try all possible parenthesizations and choose minimum")
    print("Alternative approaches:")
    print("  - Recursive with memoization: O(n³) time, O(n²) space")
    print()

# Solution 4: Backtracking
def solution_4():
    """
    Detailed solutions for backtracking problems:
    """
    
    print("Solution 4: Backtracking")
    print("=" * 25)
    
    # 1. Permutations of string
    print("1. String Permutations:")
    print("Approach: Backtracking with used array to avoid duplicates")
    print("Time Complexity: O(n! * n), Space Complexity: O(n)")
    print("Key insight: Generate permutations by choosing each character at each position")
    print("Alternative approaches:")
    print("  - Iterative with queue: O(n! * n) time, O(n! * n) space")
    print()
    
    # 2. Word break problem
    print("2. Word Break:")
    print("Approach: Backtracking with memoization")
    print("Time Complexity: O(n² * m) where m is dictionary size")
    print("Space Complexity: O(n)")
    print("Key insight: Try all possible word breaks and memoize results")
    print("Alternative approaches:")
    print("  - DP approach: O(n² * m) time, O(n) space")
    print("  - BFS: O(n² * m) time, O(n) space")
    print()

# Solution 5: Mixed Techniques
def solution_5():
    """
    Detailed solutions for mixed technique problems:
    """
    
    print("Solution 5: Mixed Techniques")
    print("=" * 28)
    
    # 1. Longest palindromic subsequence
    print("1. Longest Palindromic Subsequence:")
    print("Approach: DP with 2D table for substring lengths")
    print("Time Complexity: O(n²), Space Complexity: O(n²)")
    print("Key insight: If characters match, add 2 to inner substring LPS")
    print("Alternative approaches:")
    print("  - Recursion with memoization: O(n²) time, O(n²) space")
    print("  - Space optimized DP: O(n²) time, O(n) space")
    print()
    
    # 2. Egg drop problem
    print("2. Egg Drop Problem:")
    print("Approach: DP with eggs and floors dimensions")
    print("Time Complexity: O(eggs * floors²), Space Complexity: O(eggs * floors)")
    print("Key insight: For each floor, consider both cases (egg breaks or not)")
    print("Alternative approaches:")
    print("  - Binary search optimization: O(eggs * floors * log floors)")
    print("  - Mathematical approach: O(eggs * sqrt(floors))")

# Additional Solutions and Explanations
def additional_solutions():
    """
    Additional solutions and advanced techniques:
    """
    
    print("Additional Solutions and Techniques")
    print("=" * 35)
    
    print("1. When to Use Each Technique:")
    print("   Divide and Conquer: Problems with optimal substructure and independent subproblems")
    print("   Greedy: Problems with greedy choice property and optimal substructure")
    print("   Dynamic Programming: Problems with overlapping subproblems and optimal substructure")
    print("   Backtracking: Constraint satisfaction problems and exhaustive search")
    print()
    
    print("2. Optimization Strategies:")
    print("   - Memoization for recursive solutions")
    print("   - Tabulation for iterative solutions")
    print("   - Space optimization in DP")
    print("   - Pruning in backtracking")
    print("   - Heuristics for greedy algorithms")
    print()
    
    print("3. Common Problem Patterns:")
    print("   - Interval scheduling (greedy)")
    print("   - Sequence alignment (DP)")
    print("   - Subset generation (backtracking)")
    print("   - Tree traversal (divide and conquer)")
    print()
    
    print("4. Complexity Analysis Tips:")
    print("   - Identify recurrence relations")
    print("   - Use Master Theorem when applicable")
    print("   - Consider best, average, and worst cases")
    print("   - Account for space used by recursion stack")

# Run all solutions
if __name__ == "__main__":
    print("=== Algorithm Design Techniques Practice Problem Solutions ===\n")
    
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
    print("- Understand problem requirements before choosing technique")
    print("- Consider both time and space complexity")
    print("- Think about edge cases and error handling")
    print("- Know multiple solutions for common problems")
    print("- Practice implementing algorithms from scratch")