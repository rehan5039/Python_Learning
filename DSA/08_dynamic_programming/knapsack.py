"""
Knapsack Problem Variants

This module covers various variants of the knapsack problem:
- 0/1 Knapsack Problem
- Unbounded Knapsack Problem
- Fractional Knapsack Problem (Greedy approach)
- Subset Sum Problem
- Coin Change Problem
"""

from typing import List, Tuple


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Solve the 0/1 Knapsack problem using dynamic programming.
    
    In the 0/1 Knapsack problem, we can either take an item or leave it (0 or 1).
    
    Time Complexity: O(n * W) where n is number of items and W is capacity
    Space Complexity: O(n * W)
    
    Args:
        weights: List of item weights
        values: List of item values
        capacity: Maximum weight capacity of knapsack
        
    Returns:
        Maximum value that can be obtained
    """
    n = len(weights)
    
    # Create a 2D DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            # If current item's weight is more than capacity, skip it
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                # Max of including or excluding the current item
                dp[i][w] = max(
                    dp[i - 1][w],  # Exclude current item
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]  # Include current item
                )
    
    return dp[n][capacity]


def knapsack_01_optimized(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Space-optimized version of 0/1 Knapsack problem.
    
    Time Complexity: O(n * W)
    Space Complexity: O(W)
    
    Args:
        weights: List of item weights
        values: List of item values
        capacity: Maximum weight capacity of knapsack
        
    Returns:
        Maximum value that can be obtained
    """
    # Only need previous row
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Traverse from right to left to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def knapsack_01_items(weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int]]:
    """
    Solve 0/1 Knapsack and return both max value and selected items.
    
    Time Complexity: O(n * W)
    Space Complexity: O(n * W)
    
    Args:
        weights: List of item weights
        values: List of item values
        capacity: Maximum weight capacity of knapsack
        
    Returns:
        Tuple of (max_value, list of selected item indices)
    """
    n = len(weights)
    
    # Create a 2D DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(
                    dp[i - 1][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )
    
    # Backtrack to find selected items
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        # If value comes from including the item
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)  # Item index
            w -= weights[i - 1]
    
    selected_items.reverse()
    return dp[n][capacity], selected_items


def unbounded_knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Solve the Unbounded Knapsack problem where items can be taken multiple times.
    
    Time Complexity: O(n * W)
    Space Complexity: O(W)
    
    Args:
        weights: List of item weights
        values: List of item values
        capacity: Maximum weight capacity of knapsack
        
    Returns:
        Maximum value that can be obtained
    """
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def subset_sum(nums: List[int], target: int) -> bool:
    """
    Determine if there is a subset of the given set with sum equal to given target.
    
    Time Complexity: O(n * target)
    Space Complexity: O(target)
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        True if subset with target sum exists, False otherwise
    """
    # dp[i] will be True if there is a subset with sum i
    dp = [False] * (target + 1)
    dp[0] = True  # Sum of 0 is always possible (empty subset)
    
    # Process all elements one by one
    for num in nums:
        # Traverse from right to left to avoid using updated values
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    
    return dp[target]


def coin_change_min_coins(coins: List[int], amount: int) -> int:
    """
    Find the minimum number of coins needed to make up the given amount.
    
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


def coin_change_combinations(coins: List[int], amount: int) -> int:
    """
    Find the number of combinations that make up the given amount.
    
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    
    Args:
        coins: List of coin denominations
        amount: Target amount
        
    Returns:
        Number of combinations that make up the amount
    """
    # dp[i] represents number of ways to make amount i
    dp = [0] * (amount + 1)
    dp[0] = 1  # One way to make amount 0 (use no coins)
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]


def rod_cutting(prices: List[int], n: int) -> int:
    """
    Find the maximum value obtainable by cutting up a rod and selling the pieces.
    
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Args:
        prices: List where prices[i] is the price of a rod of length i+1
        n: Length of the rod
        
    Returns:
        Maximum value obtainable
    """
    # dp[i] represents maximum value for rod of length i
    dp = [0] * (n + 1)
    
    for i in range(1, n + 1):
        max_val = float('-inf')
        for j in range(i):
            max_val = max(max_val, prices[j] + dp[i - j - 1])
        dp[i] = max_val
    
    return dp[n]


def demo():
    """Demonstrate the knapsack problem variants."""
    print("=== Knapsack Problem Variants ===\n")
    
    # Test 0/1 Knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    
    max_value = knapsack_01(weights, values, capacity)
    max_value_opt = knapsack_01_optimized(weights, values, capacity)
    max_value_items, selected = knapsack_01_items(weights, values, capacity)
    
    print("0/1 Knapsack Problem:")
    print(f"  Weights: {weights}")
    print(f"  Values: {values}")
    print(f"  Capacity: {capacity}")
    print(f"  Maximum Value: {max_value}")
    print(f"  Selected Items (indices): {selected}")
    
    print("\n" + "="*50)
    
    # Test Unbounded Knapsack
    weights = [1, 3, 4]
    values = [1, 4, 5]
    capacity = 7
    
    max_value = unbounded_knapsack(weights, values, capacity)
    print("\nUnbounded Knapsack Problem:")
    print(f"  Weights: {weights}")
    print(f"  Values: {values}")
    print(f"  Capacity: {capacity}")
    print(f"  Maximum Value: {max_value}")
    
    print("\n" + "="*50)
    
    # Test Subset Sum
    nums = [3, 34, 4, 12, 5, 2]
    target = 9
    
    exists = subset_sum(nums, target)
    print(f"\nSubset Sum Problem:")
    print(f"  Array: {nums}")
    print(f"  Target: {target}")
    print(f"  Subset exists: {exists}")
    
    print("\n" + "="*50)
    
    # Test Coin Change
    coins = [1, 2, 5]
    amount = 11
    
    min_coins = coin_change_min_coins(coins, amount)
    combinations = coin_change_combinations(coins, amount)
    
    print(f"\nCoin Change Problem:")
    print(f"  Coins: {coins}")
    print(f"  Amount: {amount}")
    print(f"  Minimum coins needed: {min_coins}")
    print(f"  Number of combinations: {combinations}")
    
    print("\n" + "="*50)
    
    # Test Rod Cutting
    prices = [1, 5, 8, 9, 10, 17, 17, 20]
    n = 8
    
    max_value = rod_cutting(prices, n)
    print(f"\nRod Cutting Problem:")
    print(f"  Prices: {prices}")
    print(f"  Rod length: {n}")
    print(f"  Maximum value: {max_value}")


if __name__ == "__main__":
    demo()