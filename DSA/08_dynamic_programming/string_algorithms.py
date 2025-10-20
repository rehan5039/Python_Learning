"""
String Algorithms using Dynamic Programming

This module covers various string algorithms that can be solved efficiently using dynamic programming:
- Palindrome-related problems
- Regular expression matching
- Wildcard pattern matching
- Distinct subsequences
- Interleaving strings
"""

from typing import List


def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Args:
        s: Input string
        
    Returns:
        True if string is palindrome, False otherwise
    """
    return s == s[::-1]


def longest_palindromic_substring(s: str) -> str:
    """
    Find the longest palindromic substring in a string.
    
    Time Complexity: O(n^2)
    Space Complexity: O(n^2)
    
    Args:
        s: Input string
        
    Returns:
        Longest palindromic substring
    """
    if not s:
        return ""
    
    n = len(s)
    # dp[i][j] will be True if substring s[i:j+1] is palindrome
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
            
            # Check if s[i:j+1] is palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]


def count_palindromic_substrings(s: str) -> int:
    """
    Count the number of palindromic substrings in a string.
    
    Time Complexity: O(n^2)
    Space Complexity: O(n^2)
    
    Args:
        s: Input string
        
    Returns:
        Number of palindromic substrings
    """
    if not s:
        return 0
    
    n = len(s)
    # dp[i][j] will be True if substring s[i:j+1] is palindrome
    dp = [[False] * n for _ in range(n)]
    count = 0
    
    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = True
        count += 1
    
    # Check for substrings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            count += 1
    
    # Check for substrings of length 3 and more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Check if s[i:j+1] is palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                count += 1
    
    return count


def regular_expression_matching(s: str, p: str) -> bool:
    """
    Implement regular expression matching with support for '.' and '*'.
    
    '.' Matches any single character.
    '*' Matches zero or more of the preceding element.
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Args:
        s: Input string
        p: Pattern string
        
    Returns:
        True if pattern matches entire input string, False otherwise
    """
    m, n = len(s), len(p)
    
    # dp[i][j] represents if s[0:i] matches p[0:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True  # Empty string matches empty pattern
    
    # Handle patterns like a*, a*b*, a*b*c* that can match empty string
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                # Zero occurrence of the preceding character
                dp[i][j] = dp[i][j - 2]
                
                # One or more occurrences of the preceding character
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
    
    return dp[m][n]


def wildcard_matching(s: str, p: str) -> bool:
    """
    Implement wildcard pattern matching with support for '?' and '*'.
    
    '?' Matches any single character.
    '*' Matches any sequence of characters (including the empty sequence).
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Args:
        s: Input string
        p: Pattern string
        
    Returns:
        True if pattern matches entire input string, False otherwise
    """
    m, n = len(s), len(p)
    
    # dp[i][j] represents if s[0:i] matches p[0:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True  # Empty string matches empty pattern
    
    # Handle patterns starting with '*' that can match empty string
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                # '*' can match empty sequence or any sequence
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
    
    return dp[m][n]


def distinct_subsequences(s: str, t: str) -> int:
    """
    Count the number of distinct subsequences of s which equals t.
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Args:
        s: Input string
        t: Target subsequence
        
    Returns:
        Number of distinct subsequences of s equal to t
    """
    m, n = len(s), len(t)
    
    # dp[i][j] represents number of ways to form t[0:j] from s[0:i]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Empty string t can be formed in 1 way from any s
    for i in range(m + 1):
        dp[i][0] = 1
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # We can always exclude current character of s
            dp[i][j] = dp[i - 1][j]
            
            # If characters match, we can also include current character
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    
    return dp[m][n]


def is_interleave(s1: str, s2: str, s3: str) -> bool:
    """
    Check if s3 is formed by an interleaving of s1 and s2.
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Args:
        s1: First string
        s2: Second string
        s3: String to check for interleaving
        
    Returns:
        True if s3 is an interleaving of s1 and s2, False otherwise
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    # Length check
    if m + n != k:
        return False
    
    # dp[i][j] represents if s3[0:i+j] is interleaving of s1[0:i] and s2[0:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Fill first row (s1 is empty)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]
    
    # Fill first column (s2 is empty)
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
    
    # Fill the rest of the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s3[i + j - 1]:
                dp[i][j] = dp[i][j] or dp[i - 1][j]
            if s2[j - 1] == s3[i + j - 1]:
                dp[i][j] = dp[i][j] or dp[i][j - 1]
    
    return dp[m][n]


def demo():
    """Demonstrate the string algorithms using dynamic programming."""
    print("=== String Algorithms using Dynamic Programming ===\n")
    
    # Test Palindrome algorithms
    s = "babad"
    longest_palindrome = longest_palindromic_substring(s)
    palindrome_count = count_palindromic_substrings(s)
    
    print(f"String: '{s}'")
    print(f"  Longest palindromic substring: '{longest_palindrome}'")
    print(f"  Number of palindromic substrings: {palindrome_count}")
    
    s = "abc"
    palindrome_count = count_palindromic_substrings(s)
    print(f"\nString: '{s}'")
    print(f"  Number of palindromic substrings: {palindrome_count}")
    
    print("\n" + "="*60)
    
    # Test Regular Expression Matching
    s = "aa"
    p = "a*"
    match = regular_expression_matching(s, p)
    print(f"\nRegular Expression Matching:")
    print(f"  String: '{s}', Pattern: '{p}' -> {match}")
    
    s = "ab"
    p = ".*"
    match = regular_expression_matching(s, p)
    print(f"  String: '{s}', Pattern: '{p}' -> {match}")
    
    s = "aab"
    p = "c*a*b"
    match = regular_expression_matching(s, p)
    print(f"  String: '{s}', Pattern: '{p}' -> {match}")
    
    print("\n" + "="*60)
    
    # Test Wildcard Matching
    s = "adceb"
    p = "*a*b"
    match = wildcard_matching(s, p)
    print(f"\nWildcard Matching:")
    print(f"  String: '{s}', Pattern: '{p}' -> {match}")
    
    s = "acdcb"
    p = "a*c?b"
    match = wildcard_matching(s, p)
    print(f"  String: '{s}', Pattern: '{p}' -> {match}")
    
    print("\n" + "="*60)
    
    # Test Distinct Subsequences
    s = "rabbbit"
    t = "rabbit"
    count = distinct_subsequences(s, t)
    print(f"\nDistinct Subsequences:")
    print(f"  String: '{s}', Target: '{t}' -> {count}")
    
    s = "babgbag"
    t = "bag"
    count = distinct_subsequences(s, t)
    print(f"  String: '{s}', Target: '{t}' -> {count}")
    
    print("\n" + "="*60)
    
    # Test Interleaving Strings
    s1 = "aab"
    s2 = "axy"
    s3 = "aaxaby"
    interleave = is_interleave(s1, s2, s3)
    print(f"\nInterleaving Strings:")
    print(f"  s1: '{s1}', s2: '{s2}', s3: '{s3}' -> {interleave}")
    
    s1 = "aab"
    s2 = "axy"
    s3 = "abaaxy"
    interleave = is_interleave(s1, s2, s3)
    print(f"  s1: '{s1}', s2: '{s2}', s3: '{s3}' -> {interleave}")


if __name__ == "__main__":
    demo()