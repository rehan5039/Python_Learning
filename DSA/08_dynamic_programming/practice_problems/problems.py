"""
Practice Problems for Dynamic Programming

This file contains solutions to the practice problems with detailed explanations.
"""

from typing import List, Tuple
import math


def fibonacci(n: int) -> int:
    """Problem 1: Fibonacci Series"""
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1


def climb_stairs(n: int) -> int:
    """Problem 2: Climbing Stairs"""
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1


def min_cost_climbing_stairs(cost: List[int]) -> int:
    """Problem 3: Min Cost Climbing Stairs"""
    n = len(cost)
    if n <= 2:
        return min(cost)
    
    prev2, prev1 = cost[0], cost[1]
    for i in range(2, n):
        current = cost[i] + min(prev1, prev2)
        prev2, prev1 = prev1, current
    
    return min(prev1, prev2)


def house_robber(nums: List[int]) -> int:
    """Problem 4: House Robber"""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2, prev1 = 0, nums[0]
    for i in range(1, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, current
    
    return prev1


def longest_palindrome(s: str) -> str:
    """Problem 5: Longest Palindromic Substring"""
    if not s:
        return ""
    
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start = 0
    max_len = 1
    
    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = True
    
    # Check for substrings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    # Check for substrings of length 3 and more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]


def regex_matching(s: str, p: str) -> bool:
    """Problem 6: Regular Expression Matching"""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, a*b*c*
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2]  # Zero occurrence
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]  # One or more
    
    return dp[m][n]


def wildcard_matching(s: str, p: str) -> bool:
    """Problem 7: Wildcard Matching"""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns starting with '*'
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
    
    return dp[m][n]


def distinct_subsequences(s: str, t: str) -> int:
    """Problem 8: Distinct Subsequences"""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    
    return dp[m][n]


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """Problem 9: 0/1 Knapsack"""
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def unbounded_knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """Problem 10: Unbounded Knapsack"""
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def coin_change_min(coins: List[int], amount: int) -> int:
    """Problem 11: Coin Change (Minimum Coins)"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def rod_cutting(prices: List[int], n: int) -> int:
    """Problem 12: Rod Cutting"""
    dp = [0] * (n + 1)
    
    for i in range(1, n + 1):
        max_val = float('-inf')
        for j in range(i):
            max_val = max(max_val, prices[j] + dp[i - j - 1])
        dp[i] = max_val
    
    return dp[n]


def longest_common_subsequence(text1: str, text2: str) -> int:
    """Problem 13: Longest Common Subsequence"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def longest_increasing_subsequence(nums: List[int]) -> int:
    """Problem 14: Longest Increasing Subsequence"""
    if not nums:
        return 0
    
    tails = []
    for num in nums:
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)


def edit_distance(word1: str, word2: str) -> int:
    """Problem 15: Edit Distance"""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n]


def max_subarray(nums: List[int]) -> int:
    """Problem 17: Maximum Subarray (Kadane's Algorithm)"""
    max_sum = float('-inf')
    current_sum = 0
    
    for num in nums:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum


def palindrome_partitioning(s: str) -> int:
    """Problem 18: Palindrome Partitioning"""
    n = len(s)
    # Precompute palindromes
    is_palindrome = [[False] * n for _ in range(n)]
    for i in range(n):
        is_palindrome[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length == 2 or is_palindrome[i + 1][j - 1]):
                is_palindrome[i][j] = True
    
    # DP for minimum cuts
    dp = [0] * n
    for i in range(n):
        if is_palindrome[0][i]:
            dp[i] = 0
        else:
            dp[i] = float('inf')
            for j in range(i):
                if is_palindrome[j + 1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n - 1]


def word_break(s: str, word_dict: List[str]) -> bool:
    """Problem 19: Word Break"""
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[len(s)]


def num_decodings(s: str) -> int:
    """Problem 20: Decode Ways"""
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, n + 1):
        # Single digit decoding
        if s[i - 1] != '0':
            dp[i] += dp[i - 1]
        
        # Two digit decoding
        two_digit = int(s[i - 2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i - 2]
    
    return dp[n]


def run_all_problems():
    """Run all practice problems to verify solutions."""
    print("=== Running All Practice Problems ===\n")
    
    # Problem 1: Fibonacci
    print(f"1. Fibonacci(10) = {fibonacci(10)}")
    
    # Problem 2: Climbing Stairs
    print(f"2. Climb Stairs(5) = {climb_stairs(5)}")
    
    # Problem 3: Min Cost Climbing Stairs
    cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
    print(f"3. Min Cost Climbing Stairs = {min_cost_climbing_stairs(cost)}")
    
    # Problem 4: House Robber
    nums = [2, 7, 9, 3, 1]
    print(f"4. House Robber = {house_robber(nums)}")
    
    # Problem 5: Longest Palindrome
    s = "babad"
    print(f"5. Longest Palindrome in '{s}' = '{longest_palindrome(s)}'")
    
    # Problem 6: Regex Matching
    print(f"6. Regex Matching 'aa' with 'a*' = {regex_matching('aa', 'a*')}")
    
    # Problem 7: Wildcard Matching
    print(f"7. Wildcard Matching 'adceb' with '*a*b' = {wildcard_matching('adceb', '*a*b')}")
    
    # Problem 8: Distinct Subsequences
    print(f"8. Distinct Subsequences 'rabbbit' -> 'rabbit' = {distinct_subsequences('rabbbit', 'rabbit')}")
    
    # Problem 9: 0/1 Knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    print(f"9. 0/1 Knapsack = {knapsack_01(weights, values, capacity)}")
    
    # Problem 10: Unbounded Knapsack
    weights = [1, 3, 4]
    values = [1, 4, 5]
    capacity = 7
    print(f"10. Unbounded Knapsack = {unbounded_knapsack(weights, values, capacity)}")
    
    # Problem 11: Coin Change
    coins = [1, 2, 5]
    amount = 11
    print(f"11. Coin Change (min coins) = {coin_change_min(coins, amount)}")
    
    # Problem 12: Rod Cutting
    prices = [1, 5, 8, 9, 10, 17, 17, 20]
    n = 8
    print(f"12. Rod Cutting = {rod_cutting(prices, n)}")
    
    # Problem 13: Longest Common Subsequence
    text1 = "abcde"
    text2 = "ace"
    print(f"13. LCS of '{text1}' and '{text2}' = {longest_common_subsequence(text1, text2)}")
    
    # Problem 14: Longest Increasing Subsequence
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print(f"14. LIS of {nums} = {longest_increasing_subsequence(nums)}")
    
    # Problem 15: Edit Distance
    word1 = "horse"
    word2 = "ros"
    print(f"15. Edit Distance '{word1}' -> '{word2}' = {edit_distance(word1, word2)}")
    
    # Problem 17: Maximum Subarray
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"17. Maximum Subarray = {max_subarray(nums)}")
    
    # Problem 18: Palindrome Partitioning
    s = "aab"
    print(f"18. Palindrome Partitioning of '{s}' = {palindrome_partitioning(s)}")
    
    # Problem 19: Word Break
    s = "leetcode"
    word_dict = ["leet", "code"]
    print(f"19. Word Break '{s}' = {word_break(s, word_dict)}")
    
    # Problem 20: Decode Ways
    s = "12"
    print(f"20. Decode Ways '{s}' = {num_decodings(s)}")


if __name__ == "__main__":
    run_all_problems()