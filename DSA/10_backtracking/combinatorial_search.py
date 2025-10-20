"""
Combinatorial Search with Backtracking

This module covers various combinatorial search problems using backtracking:
- Permutations and combinations
- Subset generation
- Partition problems
- Palindrome partitioning
- Expression evaluation
"""

from typing import List, Set


def combine(n: int, k: int) -> List[List[int]]:
    """
    Generate all possible combinations of k numbers out of range [1, n].
    
    Time Complexity: O(C(n,k) * k)
    Space Complexity: O(C(n,k) * k) + O(k)
    
    Args:
        n: Upper bound of range [1, n]
        k: Size of each combination
        
    Returns:
        List of all possible combinations
    """
    result = []
    
    def backtrack(start: int, current_combination: List[int]):
        # Base case: combination is complete
        if len(current_combination) == k:
            result.append(current_combination[:])
            return
        
        # Pruning: if not enough numbers left to complete combination
        if len(current_combination) + (n - start + 1) < k:
            return
        
        # Try each number from start to n
        for i in range(start, n + 1):
            current_combination.append(i)
            backtrack(i + 1, current_combination)
            current_combination.pop()
    
    backtrack(1, [])
    return result


def combination_sum3(k: int, n: int) -> List[List[int]]:
    """
    Find all valid combinations of k numbers that sum up to n.
    Only numbers 1 through 9 are used, each at most once.
    
    Time Complexity: O(C(9,k))
    Space Complexity: O(C(9,k) * k) + O(k)
    
    Args:
        k: Number of elements in each combination
        n: Target sum
        
    Returns:
        List of all valid combinations
    """
    result = []
    
    def backtrack(start: int, current_combination: List[int], remaining_sum: int):
        # Base case: valid combination found
        if len(current_combination) == k and remaining_sum == 0:
            result.append(current_combination[:])
            return
        
        # Pruning
        if len(current_combination) >= k or remaining_sum < 0:
            return
        
        # Try each number from start to 9
        for i in range(start, 10):
            # Pruning: if current number is too large
            if i > remaining_sum:
                break
            
            current_combination.append(i)
            backtrack(i + 1, current_combination, remaining_sum - i)
            current_combination.pop()
    
    backtrack(1, [], n)
    return result


def partition_equal_subset_sum(nums: List[int]) -> bool:
    """
    Determine if array can be partitioned into two subsets with equal sum.
    
    Time Complexity: O(n * sum)
    Space Complexity: O(sum)
    
    Args:
        nums: List of positive integers
        
    Returns:
        True if partition is possible, False otherwise
    """
    total_sum = sum(nums)
    
    # If total sum is odd, partition is impossible
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    
    # Use backtracking with memoization
    memo = {}
    
    def backtrack(index: int, current_sum: int) -> bool:
        # Base cases
        if current_sum == target:
            return True
        if index >= len(nums) or current_sum > target:
            return False
        
        # Memoization
        if (index, current_sum) in memo:
            return memo[(index, current_sum)]
        
        # Try including or excluding current element
        result = (backtrack(index + 1, current_sum + nums[index]) or 
                 backtrack(index + 1, current_sum))
        
        memo[(index, current_sum)] = result
        return result
    
    return backtrack(0, 0)


def palindrome_partitioning(s: str) -> List[List[str]]:
    """
    Partition string such that every substring is a palindrome.
    
    Time Complexity: O(N * 2^N) where N is length of string
    Space Complexity: O(N * 2^N) + O(N)
    
    Args:
        s: Input string
        
    Returns:
        List of all possible palindrome partitions
    """
    def is_palindrome(string: str, left: int, right: int) -> bool:
        """Check if substring is palindrome."""
        while left < right:
            if string[left] != string[right]:
                return False
            left += 1
            right -= 1
        return True
    
    result = []
    
    def backtrack(start: int, current_partition: List[str]):
        # Base case: entire string processed
        if start >= len(s):
            result.append(current_partition[:])
            return
        
        # Try all possible partitions starting from 'start'
        for end in range(start, len(s)):
            # If substring is palindrome, continue backtracking
            if is_palindrome(s, start, end):
                current_partition.append(s[start:end+1])
                backtrack(end + 1, current_partition)
                current_partition.pop()
    
    backtrack(0, [])
    return result


def word_break(s: str, word_dict: List[str]) -> bool:
    """
    Determine if string can be segmented into a space-separated sequence of dictionary words.
    
    Time Complexity: O(n^3) where n is length of string
    Space Complexity: O(n)
    
    Args:
        s: Input string
        word_dict: List of dictionary words
        
    Returns:
        True if segmentation is possible, False otherwise
    """
    word_set = set(word_dict)
    memo = {}
    
    def backtrack(start: int) -> bool:
        # Base case: reached end of string
        if start == len(s):
            return True
        
        # Memoization
        if start in memo:
            return memo[start]
        
        # Try all possible substrings starting from 'start'
        for end in range(start + 1, len(s) + 1):
            # If substring is in dictionary and rest can be segmented
            if s[start:end] in word_set and backtrack(end):
                memo[start] = True
                return True
        
        memo[start] = False
        return False
    
    return backtrack(0)


def restore_ip_addresses(s: str) -> List[str]:
    """
    Generate all possible valid IP addresses from string.
    
    Time Complexity: O(3^4) = O(1) since IP has 4 parts
    Space Complexity: O(1) for output + O(1) for recursion
    
    Args:
        s: String of digits
        
    Returns:
        List of all valid IP addresses
    """
    result = []
    
    def is_valid(segment: str) -> bool:
        """Check if segment is valid IP part."""
        # Empty segment or leading zero (except "0")
        if not segment or (len(segment) > 1 and segment[0] == '0'):
            return False
        
        # Value out of range
        if int(segment) > 255:
            return False
        
        return True
    
    def backtrack(start: int, current_ip: List[str]):
        # Base case: 4 parts and entire string used
        if len(current_ip) == 4:
            if start == len(s):
                result.append('.'.join(current_ip))
            return
        
        # Pruning: too many parts or not enough characters left
        if len(current_ip) > 4 or start >= len(s):
            return
        
        # Try segments of length 1, 2, 3
        for length in range(1, 4):
            if start + length <= len(s):
                segment = s[start:start + length]
                if is_valid(segment):
                    current_ip.append(segment)
                    backtrack(start + length, current_ip)
                    current_ip.pop()
    
    backtrack(0, [])
    return result


def expression_add_operators(num: str, target: int) -> List[str]:
    """
    Generate all possible expressions that evaluate to target by adding +, -, * operators.
    
    Time Complexity: O(4^n) where n is length of num
    Space Complexity: O(n) for recursion stack
    
    Args:
        num: String of digits
        target: Target value
        
    Returns:
        List of expressions that evaluate to target
    """
    result = []
    
    def backtrack(index: int, path: str, value: int, last: int):
        # Base case: processed entire string
        if index == len(num):
            if value == target:
                result.append(path)
            return
        
        # Try all possible numbers starting from current index
        for i in range(index, len(num)):
            # Skip numbers with leading zeros
            if i != index and num[index] == '0':
                break
            
            current_str = num[index:i+1]
            current_num = int(current_str)
            
            # First number in expression
            if index == 0:
                backtrack(i + 1, current_str, current_num, current_num)
            else:
                # Addition
                backtrack(i + 1, path + '+' + current_str, value + current_num, current_num)
                # Subtraction
                backtrack(i + 1, path + '-' + current_str, value - current_num, -current_num)
                # Multiplication
                backtrack(i + 1, path + '*' + current_str, value - last + last * current_num, last * current_num)
    
    backtrack(0, '', 0, 0)
    return result


def demo():
    """Demonstrate the combinatorial search algorithms."""
    print("=== Combinatorial Search with Backtracking ===\n")
    
    # Test Combinations
    n, k = 4, 2
    result = combine(n, k)
    print(f"Combinations of {k} numbers from 1 to {n}:")
    for combo in result:
        print(f"  {combo}")
    
    print("\n" + "="*50)
    
    # Test Combination Sum 3
    k, n = 3, 7
    result = combination_sum3(k, n)
    print(f"\nCombination sum 3 (k={k}, n={n}):")
    for combo in result:
        print(f"  {combo}")
    
    print("\n" + "="*50)
    
    # Test Partition Equal Subset Sum
    nums = [1, 5, 11, 5]
    can_partition = partition_equal_subset_sum(nums)
    print(f"\nCan partition {nums} into equal subsets? {can_partition}")
    
    nums = [1, 2, 3, 5]
    can_partition = partition_equal_subset_sum(nums)
    print(f"Can partition {nums} into equal subsets? {can_partition}")
    
    print("\n" + "="*50)
    
    # Test Palindrome Partitioning
    s = "aab"
    result = palindrome_partitioning(s)
    print(f"\nPalindrome partitions of '{s}':")
    for partition in result:
        print(f"  {partition}")
    
    print("\n" + "="*50)
    
    # Test Word Break
    s = "leetcode"
    word_dict = ["leet", "code"]
    can_break = word_break(s, word_dict)
    print(f"\nCan break '{s}' with {word_dict}? {can_break}")
    
    s = "applepenapple"
    word_dict = ["apple", "pen"]
    can_break = word_break(s, word_dict)
    print(f"Can break '{s}' with {word_dict}? {can_break}")
    
    print("\n" + "="*50)
    
    # Test Restore IP Addresses
    s = "25525511135"
    result = restore_ip_addresses(s)
    print(f"\nValid IP addresses from '{s}':")
    for ip in result:
        print(f"  {ip}")
    
    print("\n" + "="*50)
    
    # Test Expression Add Operators
    num = "123"
    target = 6
    result = expression_add_operators(num, target)
    print(f"\nExpressions from '{num}' that equal {target}:")
    for expr in result:
        print(f"  {expr}")


if __name__ == "__main__":
    demo()