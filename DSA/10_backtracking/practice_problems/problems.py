"""
Practice Problems for Backtracking

This file contains solutions to the practice problems with detailed explanations.
"""

from typing import List


def subsets(nums: List[int]) -> List[List[int]]:
    """Problem 1: Subsets"""
    result = []
    def backtrack(start: int, current: List[int]):
        result.append(current[:])
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    backtrack(0, [])
    return result


def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    """Problem 2: Subsets with Duplicates"""
    nums.sort()
    result = []
    def backtrack(start: int, current: List[int]):
        result.append(current[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    backtrack(0, [])
    return result


def permute(nums: List[int]) -> List[List[int]]:
    """Problem 3: Permutations"""
    result = []
    used = [False] * len(nums)
    def backtrack(current: List[int]):
        if len(current) == len(nums):
            result.append(current[:])
            return
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                current.append(nums[i])
                backtrack(current)
                current.pop()
                used[i] = False
    backtrack([])
    return result


def permute_unique(nums: List[int]) -> List[List[int]]:
    """Problem 4: Permutations with Duplicates"""
    nums.sort()
    result = []
    used = [False] * len(nums)
    def backtrack(current: List[int]):
        if len(current) == len(nums):
            result.append(current[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            used[i] = False
    backtrack([])
    return result


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """Problem 5: Combination Sum"""
    result = []
    def backtrack(start: int, current: List[int], remaining: int):
        if remaining == 0:
            result.append(current[:])
            return
        if remaining < 0:
            return
        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])
            current.pop()
    backtrack(0, [], target)
    return result


def combination_sum2(candidates: List[int], target: int) -> List[List[int]]:
    """Problem 6: Combination Sum II"""
    candidates.sort()
    result = []
    def backtrack(start: int, current: List[int], remaining: int):
        if remaining == 0:
            result.append(current[:])
            return
        if remaining < 0:
            return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i-1]:
                continue
            current.append(candidates[i])
            backtrack(i + 1, current, remaining - candidates[i])
            current.pop()
    backtrack(0, [], target)
    return result


def generate_parenthesis(n: int) -> List[str]:
    """Problem 7: Generate Parentheses"""
    result = []
    def backtrack(current: str, open_count: int, close_count: int):
        if len(current) == 2 * n:
            result.append(current)
            return
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)
    backtrack("", 0, 0)
    return result


def letter_combinations(digits: str) -> List[str]:
    """Problem 8: Letter Combinations"""
    if not digits:
        return []
    phone_map = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
                 "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    result = []
    def backtrack(index: int, current: str):
        if index == len(digits):
            result.append(current)
            return
        for letter in phone_map[digits[index]]:
            backtrack(index + 1, current + letter)
    backtrack(0, "")
    return result


def combine(n: int, k: int) -> List[List[int]]:
    """Problem 9: Combinations"""
    result = []
    def backtrack(start: int, current: List[int]):
        if len(current) == k:
            result.append(current[:])
            return
        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()
    backtrack(1, [])
    return result


def combination_sum3(k: int, n: int) -> List[List[int]]:
    """Problem 10: Combination Sum III"""
    result = []
    def backtrack(start: int, current: List[int], remaining: int):
        if len(current) == k and remaining == 0:
            result.append(current[:])
            return
        if len(current) >= k or remaining < 0:
            return
        for i in range(start, 10):
            if i > remaining:
                break
            current.append(i)
            backtrack(i + 1, current, remaining - i)
            current.pop()
    backtrack(1, [], n)
    return result


def can_partition(nums: List[int]) -> bool:
    """Problem 11: Partition Equal Subset Sum"""
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    memo = {}
    def backtrack(index: int, current_sum: int) -> bool:
        if current_sum == target:
            return True
        if index >= len(nums) or current_sum > target:
            return False
        if (index, current_sum) in memo:
            return memo[(index, current_sum)]
        result = (backtrack(index + 1, current_sum + nums[index]) or 
                 backtrack(index + 1, current_sum))
        memo[(index, current_sum)] = result
        return result
    return backtrack(0, 0)


def partition(s: str) -> List[List[str]]:
    """Problem 12: Palindrome Partitioning"""
    def is_palindrome(string: str, left: int, right: int) -> bool:
        while left < right:
            if string[left] != string[right]:
                return False
            left += 1
            right -= 1
        return True
    result = []
    def backtrack(start: int, current: List[str]):
        if start >= len(s):
            result.append(current[:])
            return
        for end in range(start, len(s)):
            if is_palindrome(s, start, end):
                current.append(s[start:end+1])
                backtrack(end + 1, current)
                current.pop()
    backtrack(0, [])
    return result


def solve_n_queens(n: int) -> List[List[str]]:
    """Problem 13: N-Queens"""
    def is_safe(board: List[int], row: int, col: int) -> bool:
        for i in range(row):
            if board[i] == col or abs(row - i) == abs(col - board[i]):
                return False
        return True
    def backtrack(board: List[int], row: int, solutions: List[List[str]]):
        if row == n:
            solution = []
            for i in range(n):
                row_str = "." * board[i] + "Q" + "." * (n - board[i] - 1)
                solution.append(row_str)
            solutions.append(solution)
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(board, row + 1, solutions)
                board[row] = -1
    board = [-1] * n
    solutions = []
    backtrack(board, 0, solutions)
    return solutions


def solve_sudoku(board: List[List[str]]) -> None:
    """Problem 14: Sudoku Solver"""
    def is_valid(b: List[List[str]], r: int, c: int, num: str) -> bool:
        for j in range(9):
            if b[r][j] == num:
                return False
        for i in range(9):
            if b[i][c] == num:
                return False
        box_row, box_col = (r // 3) * 3, (c // 3) * 3
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if b[i][j] == num:
                    return False
        return True
    def backtrack() -> bool:
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'
                    return False
        return True
    backtrack()


def exist(board: List[List[str]], word: str) -> bool:
    """Problem 15: Word Search"""
    if not board or not board[0] or not word:
        return False
    rows, cols = len(board), len(board[0])
    def backtrack(row: int, col: int, index: int) -> bool:
        if index == len(word):
            return True
        if (row < 0 or row >= rows or col < 0 or col >= cols or 
            board[row][col] != word[index]):
            return False
        temp = board[row][col]
        board[row][col] = '#'
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            if backtrack(row + dr, col + dc, index + 1):
                board[row][col] = temp
                return True
        board[row][col] = temp
        return False
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    return False


def find_words(board: List[List[str]], words: List[str]) -> List[str]:
    """Problem 16: Word Search II"""
    if not board or not board[0] or not words:
        return []
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None
    def build_trie(words_list: List[str]) -> TrieNode:
        root = TrieNode()
        for word in words_list:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
        return root
    rows, cols = len(board), len(board[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    result = set()
    def backtrack(row: int, col: int, node: TrieNode):
        char = board[row][col]
        if char not in node.children:
            return
        next_node = node.children[char]
        if next_node.word:
            result.add(next_node.word)
            next_node.word = None
        board[row][col] = '#'
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                board[new_row][new_col] != '#'):
                backtrack(new_row, new_col, next_node)
        board[row][col] = char
    trie = build_trie(words)
    for i in range(rows):
        for j in range(cols):
            backtrack(i, j, trie)
    return list(result)


def run_all_problems():
    """Run all practice problems to verify solutions."""
    print("=== Running All Practice Problems ===\n")
    
    # Problem 1: Subsets
    nums = [1, 2, 3]
    result = subsets(nums)
    print(f"1. Subsets of {nums}: {len(result)} subsets")
    
    # Problem 3: Permutations
    nums = [1, 2, 3]
    result = permute(nums)
    print(f"3. Permutations of {nums}: {len(result)} permutations")
    
    # Problem 5: Combination Sum
    candidates = [2, 3, 6, 7]
    target = 7
    result = combination_sum(candidates, target)
    print(f"5. Combination sum for {candidates}, target {target}: {len(result)} combinations")
    
    # Problem 7: Generate Parentheses
    n = 3
    result = generate_parenthesis(n)
    print(f"7. Generate parentheses for n={n}: {len(result)} combinations")
    
    # Problem 9: Combinations
    n, k = 4, 2
    result = combine(n, k)
    print(f"9. Combinations of {k} from 1 to {n}: {len(result)} combinations")
    
    # Problem 13: N-Queens
    n = 4
    result = solve_n_queens(n)
    print(f"13. N-Queens for n={n}: {len(result)} solutions")
    
    # Problem 15: Word Search
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    word = "ABCCED"
    found = exist(board, word)
    print(f"15. Word search for '{word}': {found}")
    
    # Problem 11: Partition Equal Subset Sum
    nums = [1, 5, 11, 5]
    can_part = can_partition(nums)
    print(f"11. Can partition {nums}: {can_part}")


if __name__ == "__main__":
    run_all_problems()