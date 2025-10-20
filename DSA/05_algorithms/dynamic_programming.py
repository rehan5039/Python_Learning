"""
Algorithm Design Techniques - Dynamic Programming
==========================================

This module provides implementations and examples of dynamic programming algorithms,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Dynamic programming paradigm
- Memoization vs tabulation
- Classic DP problems (Fibonacci, LCS, Knapsack)
- Optimization techniques
- Applications in data science
"""

from typing import List, Tuple, Dict, Optional
import sys

def fibonacci_recursive(n: int) -> int:
    """
    Calculate nth Fibonacci number using naive recursion
    Time Complexity: O(2^n)
    Space Complexity: O(n)
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_memo(n: int, memo: Dict[int, int] = None) -> int:
    """
    Calculate nth Fibonacci number using memoization
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

def fibonacci_dp(n: int) -> int:
    """
    Calculate nth Fibonacci number using tabulation
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

def longest_common_subsequence(str1: str, str2: str) -> Tuple[int, str]:
    """
    Find length and actual LCS of two strings using dynamic programming
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Args:
        str1: First string
        str2: Second string
    
    Returns:
        Tuple of (length of LCS, actual LCS string)
    """
    m, n = len(str1), len(str2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return dp[m][n], ''.join(reversed(lcs))

def longest_increasing_subsequence(arr: List[int]) -> int:
    """
    Find length of longest increasing subsequence using DP
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    if not arr:
        return 0
    
    n = len(arr)
    # dp[i] represents length of LIS ending at index i
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def longest_increasing_subsequence_optimized(arr: List[int]) -> int:
    """
    Find length of LIS using binary search optimization
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if not arr:
        return 0
    
    # tails[i] stores the smallest tail of all increasing subsequences of length i+1
    tails = []
    
    for num in arr:
        # Binary search for position to insert/replace
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        # If num is larger than all elements in tails, append it
        if left == len(tails):
            tails.append(num)
        else:
            # Replace element at left position
            tails[left] = num
    
    return len(tails)

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Solve 0/1 knapsack problem using dynamic programming
    Time Complexity: O(n * W)
    Space Complexity: O(n * W)
    
    Args:
        weights: List of item weights
        values: List of item values
        capacity: Knapsack capacity
    
    Returns:
        Maximum value that can be obtained
    """
    n = len(weights)
    
    # Create DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            # If current item's weight is more than capacity, skip it
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                # Max of including or excluding current item
                dp[i][w] = max(
                    dp[i - 1][w],  # Exclude current item
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]  # Include current item
                )
    
    return dp[n][capacity]

def edit_distance(str1: str, str2: str) -> int:
    """
    Calculate minimum edit distance (Levenshtein distance) between two strings
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Args:
        str1: First string
        str2: Second string
    
    Returns:
        Minimum number of operations (insert, delete, replace) to transform str1 to str2
    """
    m, n = len(str1), len(str2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters from str1
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters to empty string
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )
    
    return dp[m][n]

def coin_change(coins: List[int], amount: int) -> int:
    """
    Find minimum number of coins needed to make given amount
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    Args:
        coins: List of coin denominations
        amount: Target amount
    
    Returns:
        Minimum number of coins needed, -1 if impossible
    """
    # dp[i] represents minimum coins needed for amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 0 coins needed for amount 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def dynamic_programming_demo():
    """
    Demonstrate dynamic programming algorithms
    """
    print("=== Dynamic Programming Demo ===")
    
    # Fibonacci numbers
    print("1. Fibonacci Numbers:")
    n = 10
    print(f"   Calculating {n}th Fibonacci number:")
    print(f"   Recursive (inefficient): {fibonacci_recursive(n)}")
    print(f"   Memoization: {fibonacci_memo(n)}")
    print(f"   Tabulation: {fibonacci_dp(n)}")
    
    # Longest Common Subsequence
    print("\n2. Longest Common Subsequence:")
    str1, str2 = "ABCDGH", "AEDFHR"
    length, lcs = longest_common_subsequence(str1, str2)
    print(f"   String 1: '{str1}'")
    print(f"   String 2: '{str2}'")
    print(f"   LCS length: {length}")
    print(f"   LCS: '{lcs}'")
    
    # Longest Increasing Subsequence
    print("\n3. Longest Increasing Subsequence:")
    arr = [10, 9, 2, 5, 3, 7, 101, 18]
    lis_length = longest_increasing_subsequence(arr)
    lis_length_opt = longest_increasing_subsequence_optimized(arr)
    print(f"   Array: {arr}")
    print(f"   LIS length (DP): {lis_length}")
    print(f"   LIS length (Optimized): {lis_length_opt}")
    
    # 0/1 Knapsack
    print("\n4. 0/1 Knapsack:")
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    capacity = 7
    max_value = knapsack_01(weights, values, capacity)
    print(f"   Weights: {weights}")
    print(f"   Values: {values}")
    print(f"   Capacity: {capacity}")
    print(f"   Maximum value: {max_value}")
    
    # Edit Distance
    print("\n5. Edit Distance:")
    str1, str2 = "saturday", "sunday"
    distance = edit_distance(str1, str2)
    print(f"   String 1: '{str1}'")
    print(f"   String 2: '{str2}'")
    print(f"   Edit distance: {distance}")
    
    # Coin Change
    print("\n6. Coin Change:")
    coins = [1, 3, 4]
    amount = 6
    min_coins = coin_change(coins, amount)
    print(f"   Coins: {coins}")
    print(f"   Amount: {amount}")
    print(f"   Minimum coins needed: {min_coins}")

def dp_optimization_techniques():
    """
    Demonstrate DP optimization techniques
    """
    print("\n=== DP Optimization Techniques ===")
    
    # 1. Space optimization
    print("1. Space Optimization:")
    print("   Convert 2D DP to 1D when only previous row is needed")
    print("   Example: Fibonacci - O(n) space → O(1) space")
    
    # 2. State reduction
    print("\n2. State Reduction:")
    print("   Reduce state space by identifying redundant states")
    print("   Example: In LIS, only track tails array instead of full DP table")
    
    # 3. Memoization vs Tabulation
    print("\n3. Memoization vs Tabulation:")
    print("   Memoization:")
    print("   - Top-down approach")
    print("   - Recursive with caching")
    print("   - Space for recursion stack")
    print("   Tabulation:")
    print("   - Bottom-up approach")
    print("   - Iterative")
    print("   - Usually more space efficient")

def dp_applications():
    """
    Demonstrate applications of dynamic programming
    """
    print("\n=== Dynamic Programming Applications ===")
    
    # 1. String algorithms
    print("1. String Algorithms:")
    print("   - Longest common subsequence")
    print("   - Edit distance")
    print("   - Palindrome partitioning")
    print("   - Regular expression matching")
    
    # 2. Graph algorithms
    print("\n2. Graph Algorithms:")
    print("   - Shortest paths (Floyd-Warshall)")
    print("   - All-pairs shortest paths")
    print("   - Transitive closure")
    
    # 3. Game theory
    print("\n3. Game Theory:")
    print("   - Optimal game strategies")
    print("   - Minimax algorithms")
    print("   - Winning/losing positions")
    
    # 4. Probability and statistics
    print("\n4. Probability and Statistics:")
    print("   - Probability calculations")
    print("   - Expected values")
    print("   - Markov chains")

def data_science_applications():
    """
    Examples of dynamic programming in data science
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Sequence alignment in bioinformatics
    print("1. Sequence Alignment:")
    print("   DNA/RNA sequence alignment using edit distance")
    print("   Protein structure prediction")
    print("   Phylogenetic tree construction")
    
    # 2. Time series analysis
    print("\n2. Time Series Analysis:")
    print("   Dynamic time warping for sequence comparison")
    print("   Optimal subsequence matching")
    print("   Change point detection")
    
    # 3. Natural language processing
    print("\n3. Natural Language Processing:")
    print("   Word segmentation")
    print("   Part-of-speech tagging")
    print("   Named entity recognition")
    print("   Machine translation alignment")
    
    # 4. Machine learning
    print("\n4. Machine Learning:")
    print("   Viterbi algorithm for Hidden Markov Models")
    print("   Forward-backward algorithm")
    print("   Optimal decision tree pruning")
    print("   Feature selection with dynamic criteria")

def performance_comparison():
    """
    Compare performance of different DP approaches
    """
    print("\n=== Performance Comparison ===")
    
    import time
    
    # Test Fibonacci implementations
    n = 30
    print(f"Testing Fibonacci({n}):")
    
    # Recursive (only for small n due to exponential time)
    if n <= 35:
        start = time.time()
        result = fibonacci_recursive(n)
        recursive_time = time.time() - start
        print(f"   Recursive: {result} in {recursive_time:.6f}s")
    
    # Memoization
    start = time.time()
    result = fibonacci_memo(n)
    memo_time = time.time() - start
    print(f"   Memoization: {result} in {memo_time:.6f}s")
    
    # Tabulation
    start = time.time()
    result = fibonacci_dp(n)
    tab_time = time.time() - start
    print(f"   Tabulation: {result} in {tab_time:.6f}s")
    
    # Test LIS with different array sizes
    sizes = [100, 500, 1000]
    for size in sizes:
        print(f"\nTesting LIS with {size} elements:")
        arr = list(range(size, 0, -1))  # Worst case: decreasing sequence
        
        # Standard DP
        start = time.time()
        length1 = longest_increasing_subsequence(arr)
        dp_time = time.time() - start
        
        # Optimized DP
        start = time.time()
        length2 = longest_increasing_subsequence_optimized(arr)
        opt_time = time.time() - start
        
        print(f"   Standard DP: Length {length1} in {dp_time:.6f}s")
        print(f"   Optimized DP: Length {length2} in {opt_time:.6f}s")

# Example usage and testing
if __name__ == "__main__":
    # Dynamic programming demo
    dynamic_programming_demo()
    print("\n" + "="*50 + "\n")
    
    # DP optimization techniques
    dp_optimization_techniques()
    print("\n" + "="*50 + "\n")
    
    # Applications
    dp_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Classic DP problems (Fibonacci, LCS, LIS, Knapsack)")
    print("2. Different DP approaches (memoization, tabulation)")
    print("3. Optimization techniques for DP algorithms")
    print("4. Applications in computer science and data science")
    print("5. Performance characteristics of DP algorithms")
    print("\nKey takeaways:")
    print("- DP solves problems by breaking them into overlapping subproblems")
    print("- Memoization (top-down) vs tabulation (bottom-up)")
    print("- Trade-offs between time and space complexity")
    print("- Many optimization problems can be solved with DP")
    print("- Essential for sequence analysis in data science")