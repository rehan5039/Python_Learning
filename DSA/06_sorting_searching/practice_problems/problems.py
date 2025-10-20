"""
Sorting and Searching - Practice Problems
=================================

This file contains practice problems for sorting and searching algorithms with solutions.
"""

# Problem 1: Basic Sorting Implementation
def problem_1():
    """
    Basic sorting algorithm implementation problems:
    """
    
    print("Problem 1: Basic Sorting Implementation")
    print("=" * 40)
    
    # 1. Sort array of strings by length
    def sort_by_length(arr):
        """
        Sort array of strings by their lengths
        Time Complexity: O(n²) with bubble sort, O(n log n) with merge sort
        """
        # Using bubble sort for demonstration
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if len(arr[j]) > len(arr[j + 1]):
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        
        return arr
    
    # 2. Sort array with custom comparator
    def sort_by_custom_rule(arr):
        """
        Sort array where even numbers come before odd numbers
        Time Complexity: O(n log n)
        """
        def custom_key(x):
            # Even numbers get priority (0), odd numbers get secondary (1)
            return (x % 2, x)
        
        return sorted(arr, key=custom_key)
    
    # 3. Stable sort implementation
    def stable_sort_students(students):
        """
        Sort students by grade (descending) and then by name (ascending)
        Time Complexity: O(n log n)
        """
        # Sort by name first (stable sort maintains relative order)
        students.sort(key=lambda x: x[1])  # Sort by name
        # Then sort by grade (stable sort maintains name order for same grades)
        students.sort(key=lambda x: x[0], reverse=True)  # Sort by grade descending
        return students
    
    # Test cases
    print("1. Sort Strings by Length:")
    strings = ["python", "java", "c", "javascript", "go"]
    sorted_strings = sort_by_length(strings)
    print(f"   Original: {strings}")
    print(f"   Sorted: {sorted_strings}")
    
    print("\n2. Custom Sorting Rule:")
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sorted_numbers = sort_by_custom_rule(numbers)
    print(f"   Original: {numbers}")
    print(f"   Even first: {sorted_numbers}")
    
    print("\n3. Stable Sort for Students:")
    students = [(85, "Alice"), (92, "Bob"), (85, "Charlie"), (78, "David"), (92, "Eve")]
    print(f"   Original: {students}")
    sorted_students = stable_sort_students(students)
    print(f"   Sorted by grade (desc) then name (asc): {sorted_students}")

# Problem 2: Searching Algorithm Practice
def problem_2():
    """
    Searching algorithm practice problems:
    """
    
    print("\nProblem 2: Searching Algorithm Practice")
    print("=" * 40)
    
    # 1. Search in rotated sorted array
    def search_rotated_array(arr, target):
        """
        Search for target in rotated sorted array
        Time Complexity: O(log n)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            
            # Left half is sorted
            if arr[left] <= arr[mid]:
                if arr[left] <= target < arr[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Right half is sorted
            else:
                if arr[mid] < target <= arr[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    # 2. Find peak element
    def find_peak_element(arr):
        """
        Find a peak element in array
        Time Complexity: O(log n)
        """
        left, right = 0, len(arr) - 1
        
        while left < right:
            mid = (left + right) // 2
            
            if arr[mid] > arr[mid + 1]:
                # Peak is in left half (including mid)
                right = mid
            else:
                # Peak is in right half
                left = mid + 1
        
        return left
    
    # 3. Search in matrix
    def search_matrix(matrix, target):
        """
        Search for target in matrix where rows and columns are sorted
        Time Complexity: O(m + n)
        """
        if not matrix or not matrix[0]:
            return False
        
        row, col = 0, len(matrix[0]) - 1
        
        while row < len(matrix) and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False
    
    # Test cases
    print("1. Search in Rotated Sorted Array:")
    rotated_arr = [4, 5, 6, 7, 0, 1, 2]
    target = 0
    print(f"   Array: {rotated_arr}")
    print(f"   Target: {target}")
    index = search_rotated_array(rotated_arr, target)
    print(f"   Index: {index}")
    
    print("\n2. Find Peak Element:")
    peak_arr = [1, 2, 3, 1]
    print(f"   Array: {peak_arr}")
    peak_index = find_peak_element(peak_arr)
    print(f"   Peak index: {peak_index}, value: {peak_arr[peak_index]}")
    
    print("\n3. Search in Sorted Matrix:")
    matrix = [
        [1,  4,  7,  11],
        [2,  5,  8,  12],
        [3,  6,  9,  16],
        [10, 13, 14, 17]
    ]
    target = 5
    print(f"   Matrix:")
    for row in matrix:
        print(f"     {row}")
    print(f"   Target: {target}")
    found = search_matrix(matrix, target)
    print(f"   Found: {found}")

# Problem 3: Advanced Sorting Problems
def problem_3():
    """
    Advanced sorting problems:
    """
    
    print("\nProblem 3: Advanced Sorting Problems")
    print("=" * 35)
    
    # 1. Sort colors (Dutch National Flag problem)
    def sort_colors(arr):
        """
        Sort array of 0s, 1s, and 2s
        Time Complexity: O(n), Space Complexity: O(1)
        """
        low = 0      # Next position for 0
        mid = 0      # Current element being processed
        high = len(arr) - 1  # Next position for 2
        
        while mid <= high:
            if arr[mid] == 0:
                arr[low], arr[mid] = arr[mid], arr[low]
                low += 1
                mid += 1
            elif arr[mid] == 1:
                mid += 1
            else:  # arr[mid] == 2
                arr[mid], arr[high] = arr[high], arr[mid]
                high -= 1
        
        return arr
    
    # 2. Merge intervals
    def merge_intervals(intervals):
        """
        Merge overlapping intervals
        Time Complexity: O(n log n)
        """
        if not intervals:
            return []
        
        # Sort by start time
        intervals.sort(key=lambda x: x[0])
        
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            # If current interval overlaps with last merged interval
            if current[0] <= last[1]:
                # Merge them by updating the end time
                last[1] = max(last[1], current[1])
            else:
                # No overlap, add current interval
                merged.append(current)
        
        return merged
    
    # 3. Top K frequent elements
    def top_k_frequent(arr, k):
        """
        Find top k frequent elements
        Time Complexity: O(n log k)
        """
        from collections import Counter
        import heapq
        
        # Count frequencies
        freq_map = Counter(arr)
        
        # Use min heap of size k
        heap = []
        for num, freq in freq_map.items():
            if len(heap) < k:
                heapq.heappush(heap, (freq, num))
            elif freq > heap[0][0]:
                heapq.heapreplace(heap, (freq, num))
        
        # Extract elements
        return [num for freq, num in heap]
    
    # Test cases
    print("1. Sort Colors (0, 1, 2):")
    colors = [2, 0, 2, 1, 1, 0]
    print(f"   Original: {colors}")
    sorted_colors = sort_colors(colors.copy())
    print(f"   Sorted: {sorted_colors}")
    
    print("\n2. Merge Intervals:")
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    print(f"   Original: {intervals}")
    merged = merge_intervals(intervals)
    print(f"   Merged: {merged}")
    
    print("\n3. Top K Frequent Elements:")
    elements = [1, 1, 1, 2, 2, 3]
    k = 2
    print(f"   Array: {elements}")
    print(f"   K: {k}")
    top_k = top_k_frequent(elements, k)
    print(f"   Top {k} frequent: {top_k}")

# Problem 4: Searching Variations
def problem_4():
    """
    Searching variation problems:
    """
    
    print("\nProblem 4: Searching Variations")
    print("=" * 30)
    
    # 1. Find minimum in rotated sorted array
    def find_min_rotated(arr):
        """
        Find minimum element in rotated sorted array
        Time Complexity: O(log n)
        """
        left, right = 0, len(arr) - 1
        
        # If array is not rotated
        if arr[left] <= arr[right]:
            return arr[left]
        
        while left < right:
            mid = (left + right) // 2
            
            # If mid element is greater than rightmost element,
            # minimum is in right half
            if arr[mid] > arr[right]:
                left = mid + 1
            else:
                # Minimum is in left half (including mid)
                right = mid
        
        return arr[left]
    
    # 2. Search for range (first and last position)
    def search_range(arr, target):
        """
        Find first and last position of target in sorted array
        Time Complexity: O(log n)
        """
        def find_first(arr, target):
            left, right = 0, len(arr) - 1
            first_pos = -1
            
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    first_pos = mid
                    right = mid - 1  # Continue searching left
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return first_pos
        
        def find_last(arr, target):
            left, right = 0, len(arr) - 1
            last_pos = -1
            
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    last_pos = mid
                    left = mid + 1  # Continue searching right
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return last_pos
        
        first = find_first(arr, target)
        if first == -1:
            return [-1, -1]
        last = find_last(arr, target)
        return [first, last]
    
    # 3. Find kth largest element
    def find_kth_largest(arr, k):
        """
        Find kth largest element in array
        Time Complexity: O(n) average, O(n²) worst case
        """
        import random
        
        def quickselect(arr, left, right, k):
            if left == right:
                return arr[left]
            
            # Randomly choose pivot and partition
            pivot_index = random.randint(left, right)
            pivot_index = partition(arr, left, right, pivot_index)
            
            if k == pivot_index:
                return arr[k]
            elif k < pivot_index:
                return quickselect(arr, left, pivot_index - 1, k)
            else:
                return quickselect(arr, pivot_index + 1, right, k)
        
        def partition(arr, left, right, pivot_index):
            pivot_value = arr[pivot_index]
            # Move pivot to end
            arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
            
            store_index = left
            for i in range(left, right):
                if arr[i] > pivot_value:  # For kth largest, we want descending order
                    arr[store_index], arr[i] = arr[i], arr[store_index]
                    store_index += 1
            
            # Move pivot to its final place
            arr[right], arr[store_index] = arr[store_index], arr[right]
            return store_index
        
        # Make a copy to avoid modifying original array
        arr_copy = arr.copy()
        return quickselect(arr_copy, 0, len(arr_copy) - 1, k - 1)
    
    # Test cases
    print("1. Find Minimum in Rotated Sorted Array:")
    rotated_min = [4, 5, 6, 7, 0, 1, 2]
    print(f"   Array: {rotated_min}")
    minimum = find_min_rotated(rotated_min)
    print(f"   Minimum element: {minimum}")
    
    print("\n2. Search for Range:")
    range_arr = [5, 7, 7, 8, 8, 10]
    target = 8
    print(f"   Array: {range_arr}")
    print(f"   Target: {target}")
    range_result = search_range(range_arr, target)
    print(f"   First and last position: {range_result}")
    
    print("\n3. Find Kth Largest Element:")
    kth_arr = [3, 2, 1, 5, 6, 4]
    k = 2
    print(f"   Array: {kth_arr}")
    print(f"   K: {k}")
    kth_largest = find_kth_largest(kth_arr, k)
    print(f"   {k}th largest element: {kth_largest}")

# Problem 5: Hybrid Algorithm Design
def problem_5():
    """
    Hybrid algorithm design problems:
    """
    
    print("\nProblem 5: Hybrid Algorithm Design")
    print("=" * 32)
    
    # 1. Count smaller elements to the right
    def count_smaller_elements(arr):
        """
        Count smaller elements to the right of each element
        Time Complexity: O(n log n)
        """
        import bisect
        
        result = []
        sorted_list = []
        
        # Process from right to left
        for i in range(len(arr) - 1, -1, -1):
            # Find position where arr[i] would be inserted in sorted_list
            pos = bisect.bisect_left(sorted_list, arr[i])
            result.append(pos)
            # Insert arr[i] at correct position to maintain sorted order
            bisect.insort(sorted_list, arr[i])
        
        # Reverse result since we processed from right to left
        return result[::-1]
    
    # 2. Merge k sorted lists
    def merge_k_sorted_lists(lists):
        """
        Merge k sorted linked lists
        Time Complexity: O(N log k) where N is total elements
        """
        import heapq
        
        # Initialize heap with first element of each list
        heap = []
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))
        
        result = []
        while heap:
            val, list_idx, elem_idx = heapq.heappop(heap)
            result.append(val)
            
            # Add next element from same list
            if elem_idx + 1 < len(lists[list_idx]):
                next_val = lists[list_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
        
        return result
    
    # 3. Find median of two sorted arrays
    def find_median_sorted_arrays(nums1, nums2):
        """
        Find median of two sorted arrays
        Time Complexity: O(log(min(m, n)))
        """
        # Ensure nums1 is the smaller array
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        x, y = len(nums1), len(nums2)
        low, high = 0, x
        
        while low <= high:
            partition_x = (low + high) // 2
            partition_y = (x + y + 1) // 2 - partition_x
            
            # If partition_x is 0, there's nothing on left side of nums1
            max_left_x = float('-inf') if partition_x == 0 else nums1[partition_x - 1]
            # If partition_x is length of nums1, there's nothing on right side
            min_right_x = float('inf') if partition_x == x else nums1[partition_x]
            
            max_left_y = float('-inf') if partition_y == 0 else nums2[partition_y - 1]
            min_right_y = float('inf') if partition_y == y else nums2[partition_y]
            
            if max_left_x <= min_right_y and max_left_y <= min_right_x:
                # Found the correct partition
                if (x + y) % 2 == 0:
                    # Even total length
                    return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2
                else:
                    # Odd total length
                    return max(max_left_x, max_left_y)
            elif max_left_x > min_right_y:
                # Too far on right, move left
                high = partition_x - 1
            else:
                # Too far on left, move right
                low = partition_x + 1
        
        raise ValueError("Input arrays are not sorted")
    
    # Test cases
    print("1. Count Smaller Elements to Right:")
    count_arr = [5, 2, 6, 1]
    print(f"   Array: {count_arr}")
    counts = count_smaller_elements(count_arr)
    print(f"   Counts: {counts}")
    print(f"   Explanation: [2 has 1 smaller to right, 6 has 1 smaller, 1 has 0, 5 has 0]")
    
    print("\n2. Merge K Sorted Lists:")
    k_lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    print(f"   Lists: {k_lists}")
    merged = merge_k_sorted_lists(k_lists)
    print(f"   Merged: {merged}")
    
    print("\n3. Median of Two Sorted Arrays:")
    nums1 = [1, 3]
    nums2 = [2]
    print(f"   Array 1: {nums1}")
    print(f"   Array 2: {nums2}")
    median = find_median_sorted_arrays(nums1, nums2)
    print(f"   Median: {median}")

# Run all problems
if __name__ == "__main__":
    print("=== Sorting and Searching Practice Problems ===\n")
    
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
    print("1. Basic sorting algorithm implementations")
    print("2. Searching algorithm variations")
    print("3. Advanced sorting problems")
    print("4. Complex searching scenarios")
    print("5. Hybrid algorithm design")
    print("\nEach problem demonstrates:")
    print("- Implementation of specific algorithms")
    print("- Common problem patterns and solutions")
    print("- Real-world applications")
    print("- Time and space complexity considerations")