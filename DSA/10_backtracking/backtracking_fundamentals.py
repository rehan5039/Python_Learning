"""
Backtracking Fundamentals

This module covers the core concepts of backtracking algorithms, including:
- Backtracking framework and principles
- Basic backtracking patterns
- State space tree traversal
- Pruning techniques
"""

from typing import List, Set


def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all possible subsets (power set) of a given array.
    
    Time Complexity: O(2^n * n) where n is length of nums
    Space Complexity: O(2^n * n) for output + O(n) for recursion stack
    
    Args:
        nums: List of integers
        
    Returns:
        List of all possible subsets
    """
    result = []
    
    def backtrack(start: int, current_subset: List[int]):
        # Add current subset to result (make a copy)
        result.append(current_subset[:])
        
        # Generate subsets by including each remaining element
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()  # Backtrack
    
    backtrack(0, [])
    return result


def subsets_with_duplicates(nums: List[int]) -> List[List[int]]:
    """
    Generate all possible subsets of array that may contain duplicates.
    
    Time Complexity: O(2^n * n)
    Space Complexity: O(2^n * n) + O(n)
    
    Args:
        nums: List of integers (may contain duplicates)
        
    Returns:
        List of all possible subsets without duplicates
    """
    nums.sort()  # Sort to handle duplicates
    result = []
    
    def backtrack(start: int, current_subset: List[int]):
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[i - 1]:
                continue
            
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    backtrack(0, [])
    return result


def permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all possible permutations of a given array.
    
    Time Complexity: O(n! * n)
    Space Complexity: O(n! * n) + O(n)
    
    Args:
        nums: List of integers
        
    Returns:
        List of all possible permutations
    """
    result = []
    used = [False] * len(nums)
    
    def backtrack(current_permutation: List[int]):
        # Base case: if permutation is complete
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])
            return
        
        # Try each unused number
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                current_permutation.append(nums[i])
                backtrack(current_permutation)
                current_permutation.pop()
                used[i] = False
    
    backtrack([])
    return result


def permutations_unique(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of array that may contain duplicates.
    
    Time Complexity: O(n! * n)
    Space Complexity: O(n! * n) + O(n)
    
    Args:
        nums: List of integers (may contain duplicates)
        
    Returns:
        List of all unique permutations
    """
    nums.sort()  # Sort to handle duplicates
    result = []
    used = [False] * len(nums)
    
    def backtrack(current_permutation: List[int]):
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])
            return
        
        for i in range(len(nums)):
            # Skip used elements
            if used[i]:
                continue
            
            # Skip duplicates: if current element is same as previous
            # and previous element is not used, skip current element
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            used[i] = True
            current_permutation.append(nums[i])
            backtrack(current_permutation)
            current_permutation.pop()
            used[i] = False
    
    backtrack([])
    return result


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find all unique combinations where chosen numbers sum to target.
    
    Time Complexity: O(N^(T/M)) where N is number of candidates, T is target, M is minimal value
    Space Complexity: O(T/M) for recursion stack
    
    Args:
        candidates: List of distinct integers
        target: Target sum
        
    Returns:
        List of all unique combinations that sum to target
    """
    result = []
    
    def backtrack(start: int, current_combination: List[int], remaining: int):
        # Base case: found valid combination
        if remaining == 0:
            result.append(current_combination[:])
            return
        
        # If remaining is negative, no valid combination
        if remaining < 0:
            return
        
        # Try each candidate from start index
        for i in range(start, len(candidates)):
            current_combination.append(candidates[i])
            # Use same index i since we can reuse elements
            backtrack(i, current_combination, remaining - candidates[i])
            current_combination.pop()
    
    backtrack(0, [], target)
    return result


def combination_sum_unique(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find all unique combinations where chosen numbers sum to target (each number used once).
    
    Time Complexity: O(2^n)
    Space Complexity: O(target) for recursion stack
    
    Args:
        candidates: List of integers (may contain duplicates)
        target: Target sum
        
    Returns:
        List of all unique combinations that sum to target
    """
    candidates.sort()  # Sort to handle duplicates
    result = []
    
    def backtrack(start: int, current_combination: List[int], remaining: int):
        if remaining == 0:
            result.append(current_combination[:])
            return
        
        if remaining < 0:
            return
        
        for i in range(start, len(candidates)):
            # Skip duplicates
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            
            current_combination.append(candidates[i])
            # Use i + 1 since each element can be used only once
            backtrack(i + 1, current_combination, remaining - candidates[i])
            current_combination.pop()
    
    backtrack(0, [], target)
    return result


def generate_parentheses(n: int) -> List[str]:
    """
    Generate all combinations of well-formed parentheses.
    
    Time Complexity: O(4^n / sqrt(n)) - Catalan number
    Space Complexity: O(4^n / sqrt(n)) + O(n)
    
    Args:
        n: Number of pairs of parentheses
        
    Returns:
        List of all valid combinations of parentheses
    """
    result = []
    
    def backtrack(current: str, open_count: int, close_count: int):
        # Base case: valid combination found
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Add opening parenthesis if count < n
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)
        
        # Add closing parenthesis if count < open_count
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)
    
    backtrack("", 0, 0)
    return result


def letter_combinations(digits: str) -> List[str]:
    """
    Generate all possible letter combinations for phone number digits.
    
    Time Complexity: O(3^N * 4^M) where N is digits with 3 letters, M with 4 letters
    Space Complexity: O(3^N * 4^M) + O(N + M)
    
    Args:
        digits: String of digits (2-9)
        
    Returns:
        List of all possible letter combinations
    """
    if not digits:
        return []
    
    # Mapping of digits to letters
    phone_map = {
        "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
        "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
    }
    
    result = []
    
    def backtrack(index: int, current_combination: str):
        # Base case: combination is complete
        if index == len(digits):
            result.append(current_combination)
            return
        
        # Get letters for current digit
        letters = phone_map[digits[index]]
        
        # Try each letter
        for letter in letters:
            backtrack(index + 1, current_combination + letter)
    
    backtrack(0, "")
    return result


def demo():
    """Demonstrate the backtracking fundamentals."""
    print("=== Backtracking Fundamentals ===\n")
    
    # Test Subsets
    nums = [1, 2, 3]
    result = subsets(nums)
    print(f"Subsets of {nums}:")
    for subset in result:
        print(f"  {subset}")
    
    print("\n" + "="*40)
    
    # Test Subsets with Duplicates
    nums = [1, 2, 2]
    result = subsets_with_duplicates(nums)
    print(f"\nSubsets of {nums} (with duplicates):")
    for subset in result:
        print(f"  {subset}")
    
    print("\n" + "="*40)
    
    # Test Permutations
    nums = [1, 2, 3]
    result = permutations(nums)
    print(f"\nPermutations of {nums}:")
    for perm in result:
        print(f"  {perm}")
    
    print("\n" + "="*40)
    
    # Test Unique Permutations
    nums = [1, 1, 2]
    result = permutations_unique(nums)
    print(f"\nUnique permutations of {nums}:")
    for perm in result:
        print(f"  {perm}")
    
    print("\n" + "="*40)
    
    # Test Combination Sum
    candidates = [2, 3, 6, 7]
    target = 7
    result = combination_sum(candidates, target)
    print(f"\nCombination sum for {candidates}, target {target}:")
    for combo in result:
        print(f"  {combo}")
    
    print("\n" + "="*40)
    
    # Test Generate Parentheses
    n = 3
    result = generate_parentheses(n)
    print(f"\nValid parentheses combinations for n={n}:")
    for combo in result:
        print(f"  {combo}")
    
    print("\n" + "="*40)
    
    # Test Letter Combinations
    digits = "23"
    result = letter_combinations(digits)
    print(f"\nLetter combinations for digits '{digits}':")
    for combo in result:
        print(f"  {combo}")


if __name__ == "__main__":
    demo()