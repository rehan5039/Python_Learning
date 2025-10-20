"""
Dynamic Programming Fundamentals

This module covers the core concepts of dynamic programming, including:
- Memoization vs. Tabulation
- Fibonacci sequence optimization
- Basic DP patterns and techniques
"""

import functools
import time
from typing import Dict, List


def fibonacci_naive(n: int) -> int:
    """
    Naive recursive implementation of Fibonacci sequence.
    
    Time Complexity: O(2^n)
    Space Complexity: O(n) - due to recursion stack
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_memo(n: int, memo: Dict[int, int] = None) -> int:
    """
    Memoized implementation of Fibonacci sequence.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Args:
        n: The position in the Fibonacci sequence
        memo: Dictionary to store computed values
        
    Returns:
        The nth Fibonacci number
    """
    if memo is None:
        memo = {}
        
    if n in memo:
        return memo[n]
        
    if n <= 1:
        return n
        
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


def fibonacci_tab(n: int) -> int:
    """
    Tabulated implementation of Fibonacci sequence.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
        
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
        
    return dp[n]


def fibonacci_optimized(n: int) -> int:
    """
    Space-optimized implementation of Fibonacci sequence.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
        
    prev2, prev1 = 0, 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
        
    return prev1


@functools.lru_cache(maxsize=None)
def fibonacci_cached(n: int) -> int:
    """
    Using Python's built-in LRU cache decorator.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)


def climbing_stairs(n: int) -> int:
    """
    Calculate the number of distinct ways to climb to the top of a staircase.
    
    You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Args:
        n: Number of steps in the staircase
        
    Returns:
        Number of distinct ways to climb to the top
    """
    if n <= 2:
        return n
        
    prev2, prev1 = 1, 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
        
    return prev1


def min_cost_climbing_stairs(cost: List[int]) -> int:
    """
    Find the minimum cost to reach the top of the floor.
    
    You are given an integer array cost where cost[i] is the cost of ith step.
    Once you pay the cost, you can either climb one or two steps.
    You can either start from the step with index 0, or the step with index 1.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Args:
        cost: Array of costs for each step
        
    Returns:
        Minimum cost to reach the top
    """
    n = len(cost)
    if n <= 2:
        return min(cost)
        
    # dp[i] represents the minimum cost to reach step i
    prev2, prev1 = cost[0], cost[1]
    
    for i in range(2, n):
        current = cost[i] + min(prev1, prev2)
        prev2, prev1 = prev1, current
        
    return min(prev1, prev2)


def house_robber(nums: List[int]) -> int:
    """
    Determine the maximum amount of money you can rob tonight without alerting the police.
    
    You are a professional robber planning to rob houses along a street.
    Each house has a certain amount of money stashed.
    All houses at this place are arranged in a circle.
    Adjacent houses have a security system connected, and it will automatically contact the police
    if two adjacent houses were broken into on the same night.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Args:
        nums: Array representing the amount of money in each house
        
    Returns:
        Maximum amount of money that can be robbed
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
        
    # Helper function for linear arrangement
    def rob_linear(houses: List[int]) -> int:
        prev2, prev1 = 0, 0
        for money in houses:
            current = max(prev1, prev2 + money)
            prev2, prev1 = prev1, current
        return prev1
    
    # Since houses are in a circle, we have two scenarios:
    # 1. Rob the first house but skip the last
    # 2. Skip the first house and possibly rob the last
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))


def demo():
    """Demonstrate the dynamic programming concepts."""
    print("=== Dynamic Programming Fundamentals ===\n")
    
    # Test Fibonacci implementations
    n = 10
    print(f"Fibonacci({n}):")
    
    start = time.time()
    result = fibonacci_naive(n)
    naive_time = time.time() - start
    print(f"  Naive: {result} (Time: {naive_time:.6f}s)")
    
    start = time.time()
    result = fibonacci_memo(n)
    memo_time = time.time() - start
    print(f"  Memoized: {result} (Time: {memo_time:.6f}s)")
    
    start = time.time()
    result = fibonacci_tab(n)
    tab_time = time.time() - start
    print(f"  Tabulated: {result} (Time: {tab_time:.6f}s)")
    
    start = time.time()
    result = fibonacci_optimized(n)
    opt_time = time.time() - start
    print(f"  Optimized: {result} (Time: {opt_time:.6f}s)")
    
    start = time.time()
    result = fibonacci_cached(n)
    cached_time = time.time() - start
    print(f"  Cached: {result} (Time: {cached_time:.6f}s)")
    
    print("\n" + "="*50)
    
    # Test climbing stairs
    steps = 5
    ways = climbing_stairs(steps)
    print(f"\nClimbing Stairs ({steps} steps): {ways} ways")
    
    # Test min cost climbing stairs
    cost = [10, 15, 20]
    min_cost = min_cost_climbing_stairs(cost)
    print(f"\nMin Cost Climbing Stairs {cost}: ${min_cost}")
    
    # Test house robber
    houses = [2, 7, 9, 3, 1]
    max_money = house_robber(houses)
    print(f"\nHouse Robber {houses}: ${max_money}")


if __name__ == "__main__":
    demo()