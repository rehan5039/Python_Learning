"""
Longest Common Subsequence (LCS) and Longest Increasing Subsequence (LIS)

This module covers two important dynamic programming problems:
- Longest Common Subsequence: Finding the longest subsequence common to two sequences
- Longest Increasing Subsequence: Finding the longest subsequence of a sequence that is strictly increasing
"""

from typing import List, Tuple


def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Find the length of the longest common subsequence between two strings.
    
    A subsequence is a sequence that can be derived from another sequence by deleting
    some or no elements without changing the order of the remaining elements.
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Args:
        text1: First string
        text2: Second string
        
    Returns:
        Length of the longest common subsequence
    """
    m, n = len(text1), len(text2)
    
    # Create a 2D DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def longest_common_subsequence_string(text1: str, text2: str) -> str:
    """
    Find the actual longest common subsequence string between two strings.
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Args:
        text1: First string
        text2: Second string
        
    Returns:
        The longest common subsequence string
    """
    m, n = len(text1), len(text2)
    
    # Create a 2D DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct the LCS string
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))


def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    Find the length of the longest strictly increasing subsequence.
    
    Time Complexity: O(n^2) - Can be optimized to O(n log n) with binary search
    Space Complexity: O(n)
    
    Args:
        nums: Array of integers
        
    Returns:
        Length of the longest increasing subsequence
    """
    if not nums:
        return 0
    
    n = len(nums)
    # dp[i] represents the length of LIS ending at index i
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def longest_increasing_subsequence_optimized(nums: List[int]) -> int:
    """
    Find the length of the longest strictly increasing subsequence using binary search.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Args:
        nums: Array of integers
        
    Returns:
        Length of the longest increasing subsequence
    """
    if not nums:
        return 0
    
    # tails[i] stores the smallest tail of all increasing subsequences of length i+1
    tails = []
    
    for num in nums:
        # Binary search for the position to insert/replace
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
            # Replace the element at position left
            tails[left] = num
    
    return len(tails)


def longest_increasing_subsequence_path(nums: List[int]) -> List[int]:
    """
    Find the actual longest increasing subsequence.
    
    Time Complexity: O(n^2)
    Space Complexity: O(n^2) in worst case
    
    Args:
        nums: Array of integers
        
    Returns:
        The longest increasing subsequence
    """
    if not nums:
        return []
    
    n = len(nums)
    # dp[i] represents the length of LIS ending at index i
    dp = [1] * n
    # parent[i] stores the previous index in the LIS ending at index i
    parent = [-1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # Find the index with maximum LIS length
    max_length = max(dp)
    max_index = dp.index(max_length)
    
    # Reconstruct the LIS
    lis = []
    current = max_index
    while current != -1:
        lis.append(nums[current])
        current = parent[current]
    
    return list(reversed(lis))


def edit_distance(word1: str, word2: str) -> int:
    """
    Calculate the minimum number of operations required to convert word1 to word2.
    
    Operations allowed:
    1. Insert a character
    2. Delete a character
    3. Replace a character
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Args:
        word1: Source string
        word2: Target string
        
    Returns:
        Minimum number of operations (edit distance)
    """
    m, n = len(word1), len(word2)
    
    # Create a 2D DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters from word1
    
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters to get word2
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # Delete
                    dp[i][j - 1],    # Insert
                    dp[i - 1][j - 1] # Replace
                )
    
    return dp[m][n]


def demo():
    """Demonstrate the LCS and LIS algorithms."""
    print("=== Longest Common Subsequence and Longest Increasing Subsequence ===\n")
    
    # Test LCS
    text1 = "ABCDGH"
    text2 = "AEDFHR"
    lcs_length = longest_common_subsequence(text1, text2)
    lcs_string = longest_common_subsequence_string(text1, text2)
    print(f"LCS of '{text1}' and '{text2}':")
    print(f"  Length: {lcs_length}")
    print(f"  String: '{lcs_string}'")
    
    text1 = "AGGTAB"
    text2 = "GXTXAYB"
    lcs_length = longest_common_subsequence(text1, text2)
    lcs_string = longest_common_subsequence_string(text1, text2)
    print(f"\nLCS of '{text1}' and '{text2}':")
    print(f"  Length: {lcs_length}")
    print(f"  String: '{lcs_string}'")
    
    print("\n" + "="*60)
    
    # Test LIS
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    lis_length = longest_increasing_subsequence(nums)
    lis_path = longest_increasing_subsequence_path(nums)
    lis_length_opt = longest_increasing_subsequence_optimized(nums)
    print(f"\nLIS of {nums}:")
    print(f"  Length (O(n^2)): {lis_length}")
    print(f"  Length (O(n log n)): {lis_length_opt}")
    print(f"  Actual LIS: {lis_path}")
    
    nums = [0, 1, 0, 3, 2, 3]
    lis_length = longest_increasing_subsequence(nums)
    lis_path = longest_increasing_subsequence_path(nums)
    print(f"\nLIS of {nums}:")
    print(f"  Length: {lis_length}")
    print(f"  Actual LIS: {lis_path}")
    
    print("\n" + "="*60)
    
    # Test Edit Distance
    word1 = "horse"
    word2 = "ros"
    distance = edit_distance(word1, word2)
    print(f"\nEdit distance between '{word1}' and '{word2}': {distance}")
    
    word1 = "intention"
    word2 = "execution"
    distance = edit_distance(word1, word2)
    print(f"Edit distance between '{word1}' and '{word2}': {distance}")


if __name__ == "__main__":
    demo()