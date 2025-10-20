"""
Algorithm Design Techniques - Practice Problems
=======================================

This file contains practice problems for algorithm design techniques with solutions.
"""

# Problem 1: Divide and Conquer
def problem_1():
    """
    Divide and conquer problems:
    """
    
    print("Problem 1: Divide and Conquer")
    print("=" * 30)
    
    # 1. Peak element in array
    def find_peak_element(arr):
        """
        Find a peak element in array (element greater than neighbors)
        Time Complexity: O(log n)
        """
        def binary_search_peak(arr, left, right):
            mid = (left + right) // 2
            
            # Check if mid is peak
            if ((mid == 0 or arr[mid-1] <= arr[mid]) and 
                (mid == len(arr)-1 or arr[mid+1] <= arr[mid])):
                return mid
            
            # If left neighbor is greater, peak is on left side
            if mid > 0 and arr[mid-1] > arr[mid]:
                return binary_search_peak(arr, left, mid-1)
            else:
                # Peak is on right side
                return binary_search_peak(arr, mid+1, right)
        
        if not arr:
            return -1
        return binary_search_peak(arr, 0, len(arr)-1)
    
    # 2. Count inversions in array
    def count_inversions(arr):
        """
        Count number of inversions in array
        Time Complexity: O(n log n)
        """
        def merge_and_count(arr, temp, left, mid, right):
            i, j, k = left, mid + 1, left
            inv_count = 0
            
            while i <= mid and j <= right:
                if arr[i] <= arr[j]:
                    temp[k] = arr[i]
                    i += 1
                else:
                    temp[k] = arr[j]
                    inv_count += (mid - i + 1)
                    j += 1
                k += 1
            
            # Copy remaining elements
            while i <= mid:
                temp[k] = arr[i]
                i += 1
                k += 1
            
            while j <= right:
                temp[k] = arr[j]
                j += 1
                k += 1
            
            # Copy back to original array
            for i in range(left, right + 1):
                arr[i] = temp[i]
            
            return inv_count
        
        def merge_sort_and_count(arr, temp, left, right):
            inv_count = 0
            if left < right:
                mid = (left + right) // 2
                inv_count += merge_sort_and_count(arr, temp, left, mid)
                inv_count += merge_sort_and_count(arr, temp, mid + 1, right)
                inv_count += merge_and_count(arr, temp, left, mid, right)
            return inv_count
        
        if len(arr) <= 1:
            return 0
        temp = [0] * len(arr)
        arr_copy = arr.copy()
        return merge_sort_and_count(arr_copy, temp, 0, len(arr) - 1)
    
    # Test cases
    print("1. Peak Element:")
    arr1 = [1, 3, 20, 4, 1, 0]
    peak_index = find_peak_element(arr1)
    print(f"   Array: {arr1}")
    print(f"   Peak element at index: {peak_index}, value: {arr1[peak_index]}")
    
    print("\n2. Count Inversions:")
    arr2 = [8, 4, 2, 1]
    inversions = count_inversions(arr2)
    print(f"   Array: {arr2}")
    print(f"   Number of inversions: {inversions}")

# Problem 2: Greedy Algorithms
def problem_2():
    """
    Greedy algorithm problems:
    """
    
    print("\nProblem 2: Greedy Algorithms")
    print("=" * 30)
    
    # 1. Job scheduling
    def job_scheduling(jobs):
        """
        Schedule jobs to maximize profit
        Time Complexity: O(n log n)
        """
        # Sort jobs by profit in descending order
        jobs.sort(key=lambda x: x[2], reverse=True)
        
        # Find maximum deadline to determine slots
        max_deadline = max(job[1] for job in jobs)
        slots = [False] * (max_deadline + 1)
        result = []
        
        for job in jobs:
            # Find a free slot for this job (starting from its deadline)
            for j in range(min(max_deadline, job[1]), 0, -1):
                if not slots[j]:
                    slots[j] = True
                    result.append(job)
                    break
        
        return result
    
    # 2. Minimum number of platforms
    def min_platforms(arrival, departure):
        """
        Find minimum number of platforms needed
        Time Complexity: O(n log n)
        """
        # Sort arrival and departure times
        arrival.sort()
        departure.sort()
        
        platforms = 1
        result = 1
        i = 1  # Index for arrival
        j = 0  # Index for departure
        
        while i < len(arrival) and j < len(departure):
            if arrival[i] <= departure[j]:
                platforms += 1
                i += 1
            else:
                platforms -= 1
                j += 1
            result = max(result, platforms)
        
        return result
    
    # Test cases
    print("1. Job Scheduling:")
    jobs = [('A', 2, 100), ('B', 1, 19), ('C', 2, 27), ('D', 1, 25), ('E', 3, 15)]
    print(f"   Jobs (id, deadline, profit): {jobs}")
    scheduled = job_scheduling(jobs)
    print(f"   Scheduled jobs: {scheduled}")
    total_profit = sum(job[2] for job in scheduled)
    print(f"   Total profit: {total_profit}")
    
    print("\n2. Minimum Platforms:")
    arrival = [900, 940, 950, 1100, 1500, 1800]
    departure = [910, 1200, 1120, 1130, 1900, 2000]
    print(f"   Arrival times: {arrival}")
    print(f"   Departure times: {departure}")
    platforms = min_platforms(arrival, departure)
    print(f"   Minimum platforms needed: {platforms}")

# Problem 3: Dynamic Programming
def problem_3():
    """
    Dynamic programming problems:
    """
    
    print("\nProblem 3: Dynamic Programming")
    print("=" * 30)
    
    # 1. Rod cutting problem
    def rod_cutting(prices, n):
        """
        Find maximum value obtainable by cutting rod
        Time Complexity: O(n²)
        """
        # dp[i] represents maximum value for rod of length i
        dp = [0] * (n + 1)
        
        # Build dp table in bottom-up manner
        for i in range(1, n + 1):
            max_val = float('-inf')
            for j in range(i):
                max_val = max(max_val, prices[j] + dp[i - j - 1])
            dp[i] = max_val
        
        return dp[n]
    
    # 2. Matrix chain multiplication
    def matrix_chain_multiplication(dimensions):
        """
        Find minimum number of multiplications for matrix chain
        Time Complexity: O(n³)
        """
        n = len(dimensions) - 1
        # dp[i][j] represents minimum multiplications for matrices i to j
        dp = [[0] * n for _ in range(n)]
        
        # L is chain length
        for L in range(2, n + 1):
            for i in range(n - L + 1):
                j = i + L - 1
                dp[i][j] = float('inf')
                for k in range(i, j):
                    cost = (dp[i][k] + dp[k + 1][j] + 
                           dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n - 1]
    
    # Test cases
    print("1. Rod Cutting:")
    prices = [1, 5, 8, 9, 10, 17, 17, 20]
    length = 8
    print(f"   Prices for lengths 1-{len(prices)}: {prices}")
    print(f"   Rod length: {length}")
    max_value = rod_cutting(prices, length)
    print(f"   Maximum value: {max_value}")
    
    print("\n2. Matrix Chain Multiplication:")
    dimensions = [1, 2, 3, 4, 3]  # Matrices of dimensions 1x2, 2x3, 3x4, 4x3
    print(f"   Matrix dimensions: {dimensions}")
    min_multiplications = matrix_chain_multiplication(dimensions)
    print(f"   Minimum multiplications: {min_multiplications}")

# Problem 4: Backtracking
def problem_4():
    """
    Backtracking problems:
    """
    
    print("\nProblem 4: Backtracking")
    print("=" * 25)
    
    # 1. Permutations of string
    def string_permutations(s):
        """
        Generate all permutations of string
        Time Complexity: O(n! * n)
        """
        def backtrack(chars, path, result, used):
            if len(path) == len(chars):
                result.append(''.join(path))
                return
            
            for i in range(len(chars)):
                if used[i]:
                    continue
                # Skip duplicates
                if i > 0 and chars[i] == chars[i-1] and not used[i-1]:
                    continue
                
                used[i] = True
                path.append(chars[i])
                backtrack(chars, path, result, used)
                path.pop()
                used[i] = False
        
        chars = sorted(list(s))
        result = []
        used = [False] * len(chars)
        backtrack(chars, [], result, used)
        return result
    
    # 2. Word break problem
    def word_break(s, word_dict):
        """
        Check if string can be segmented into dictionary words
        Time Complexity: O(n² * m) where m is dictionary size
        """
        def backtrack(s, word_dict, start, memo):
            if start == len(s):
                return True
            if start in memo:
                return memo[start]
            
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_dict and backtrack(s, word_dict, end, memo):
                    memo[start] = True
                    return True
            
            memo[start] = False
            return False
        
        return backtrack(s, set(word_dict), 0, {})
    
    # Test cases
    print("1. String Permutations:")
    s = "abc"
    print(f"   String: '{s}'")
    permutations = string_permutations(s)
    print(f"   Permutations: {permutations}")
    
    print("\n2. Word Break:")
    s = "leetcode"
    word_dict = ["leet", "code"]
    print(f"   String: '{s}'")
    print(f"   Dictionary: {word_dict}")
    can_break = word_break(s, word_dict)
    print(f"   Can be segmented: {can_break}")

# Problem 5: Mixed Techniques
def problem_5():
    """
    Problems combining multiple techniques:
    """
    
    print("\nProblem 5: Mixed Techniques")
    print("=" * 28)
    
    # 1. Longest palindromic subsequence
    def longest_palindromic_subsequence(s):
        """
        Find length of longest palindromic subsequence
        Uses DP technique
        Time Complexity: O(n²)
        """
        n = len(s)
        # dp[i][j] represents length of LPS in substring s[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Every single character is palindrome of length 1
        for i in range(n):
            dp[i][i] = 1
        
        # Build table for substrings of length 2 to n
        for cl in range(2, n + 1):  # cl is length of substring
            for i in range(n - cl + 1):
                j = i + cl - 1
                if s[i] == s[j] and cl == 2:
                    dp[i][j] = 2
                elif s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])
        
        return dp[0][n - 1]
    
    # 2. Egg dropping problem
    def egg_drop(eggs, floors):
        """
        Find minimum attempts to find critical floor
        Uses DP technique
        Time Complexity: O(eggs * floors²)
        """
        # dp[i][j] represents minimum attempts for i eggs and j floors
        dp = [[0] * (floors + 1) for _ in range(eggs + 1)]
        
        # Base cases
        # 0 floors - 0 attempts needed
        # 1 floor - 1 attempt needed
        for i in range(1, eggs + 1):
            dp[i][1] = 1
            dp[i][0] = 0
        
        # Only 1 egg - try all floors from 1 to j
        for j in range(1, floors + 1):
            dp[1][j] = j
        
        # Fill rest of table
        for i in range(2, eggs + 1):
            for j in range(2, floors + 1):
                dp[i][j] = float('inf')
                for k in range(1, j + 1):
                    # Egg breaks: dp[i-1][k-1], Egg doesn't break: dp[i][j-k]
                    result = 1 + max(dp[i - 1][k - 1], dp[i][j - k])
                    dp[i][j] = min(dp[i][j], result)
        
        return dp[eggs][floors]
    
    # Test cases
    print("1. Longest Palindromic Subsequence:")
    s = "bbbab"
    print(f"   String: '{s}'")
    lps_length = longest_palindromic_subsequence(s)
    print(f"   Length of LPS: {lps_length}")
    
    print("\n2. Egg Drop Problem:")
    eggs = 2
    floors = 10
    print(f"   Number of eggs: {eggs}")
    print(f"   Number of floors: {floors}")
    min_attempts = egg_drop(eggs, floors)
    print(f"   Minimum attempts needed: {min_attempts}")

# Run all problems
if __name__ == "__main__":
    print("=== Algorithm Design Techniques Practice Problems ===\n")
    
    problem_1()
    print("\n" + "="*50 + "\n")
    
    problem_2()
    print("\n" + "="*50 + "\n")
    
    problem_3()
    print("\n" + "="*50 + "\n")
    
    problem_4()
    print("\n" + "="*50 + "\n")
    
    problem_5()
    print("\n" + "="*50 + "\n")
    
    print("=== Summary ===")
    print("These practice problems covered:")
    print("1. Divide and conquer algorithms")
    print("2. Greedy algorithm implementations")
    print("3. Dynamic programming solutions")
    print("4. Backtracking techniques")
    print("5. Mixed algorithm design problems")
    print("\nEach problem demonstrates:")
    print("- Implementation of algorithmic techniques")
    print("- Common problem patterns and solutions")
    print("- Real-world applications")
    print("- Time and space complexity considerations")