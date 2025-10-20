"""
Greedy Algorithms Fundamentals

This module covers the core concepts of greedy algorithms, including:
- Greedy algorithm principles
- Activity selection problem
- Fractional knapsack problem
- Huffman coding
- Job sequencing problem
"""

import heapq
from collections import defaultdict
from typing import List, Tuple


def activity_selection(activities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Select maximum number of activities that don't overlap.
    
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(1) excluding output
    
    Args:
        activities: List of tuples (start_time, end_time, activity_name)
        
    Returns:
        List of selected activities
    """
    if not activities:
        return []
    
    # Sort activities by end time (greedy choice)
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_end_time = activities[0][1]
    
    for i in range(1, len(activities)):
        # If current activity starts after last selected activity ends
        if activities[i][0] >= last_end_time:
            selected.append(activities[i])
            last_end_time = activities[i][1]
    
    return selected


def fractional_knapsack(weights: List[int], values: List[int], capacity: int) -> float:
    """
    Solve the fractional knapsack problem using greedy approach.
    
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for item ratios
    
    Args:
        weights: List of item weights
        values: List of item values
        capacity: Maximum weight capacity of knapsack
        
    Returns:
        Maximum value that can be obtained
    """
    if not weights or not values or len(weights) != len(values):
        return 0.0
    
    # Calculate value-to-weight ratios
    items = []
    for i in range(len(weights)):
        ratio = values[i] / weights[i] if weights[i] > 0 else 0
        items.append((ratio, weights[i], values[i]))
    
    # Sort by value-to-weight ratio in descending order (greedy choice)
    items.sort(reverse=True)
    
    total_value = 0.0
    remaining_capacity = capacity
    
    for ratio, weight, value in items:
        if remaining_capacity >= weight:
            # Take the whole item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction of the item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            break
    
    return total_value


def huffman_coding(frequencies: dict) -> dict:
    """
    Generate Huffman codes for characters based on their frequencies.
    
    Time Complexity: O(n log n) where n is number of unique characters
    Space Complexity: O(n)
    
    Args:
        frequencies: Dictionary mapping characters to their frequencies
        
    Returns:
        Dictionary mapping characters to their Huffman codes
    """
    if not frequencies:
        return {}
    
    # Create a priority queue (min heap) of nodes
    heap = [[freq, [char, ""]] for char, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        # Extract two nodes with minimum frequency
        low = heapq.heappop(heap)
        high = heapq.heappop(heap)
        
        # Assign '0' to left branch and '1' to right branch
        for pair in low[1:]:
            pair[1] = '0' + pair[1]
        for pair in high[1:]:
            pair[1] = '1' + pair[1]
        
        # Create new internal node with combined frequency
        heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])
    
    # Extract Huffman codes
    huffman_codes = {}
    if heap:
        for char, code in heap[0][1:]:
            huffman_codes[char] = code
    
    return huffman_codes


def job_sequencing(jobs: List[Tuple[int, int, int]]) -> Tuple[int, List[int]]:
    """
    Solve job sequencing problem to maximize profit.
    
    Time Complexity: O(n^2) where n is number of jobs
    Space Complexity: O(n)
    
    Args:
        jobs: List of tuples (job_id, deadline, profit)
        
    Returns:
        Tuple of (maximum_profit, list_of_selected_job_ids)
    """
    if not jobs:
        return 0, []
    
    # Sort jobs by profit in descending order (greedy choice)
    jobs.sort(key=lambda x: x[2], reverse=True)
    
    n = len(jobs)
    # Find maximum deadline to determine slot array size
    max_deadline = max(job[1] for job in jobs)
    
    # Create array to keep track of free time slots
    slots = [-1] * (max_deadline + 1)
    selected_jobs = []
    total_profit = 0
    
    # Iterate through all jobs
    for job_id, deadline, profit in jobs:
        # Find a free slot for this job (starting from deadline and moving backwards)
        for j in range(min(max_deadline, deadline), 0, -1):
            if slots[j] == -1:
                slots[j] = job_id
                selected_jobs.append(job_id)
                total_profit += profit
                break
    
    return total_profit, selected_jobs


def minimum_number_of_coins(coins: List[int], amount: int) -> List[int]:
    """
    Find minimum number of coins needed to make change (greedy approach).
    
    Note: This works optimally only for certain coin systems (like US coins).
    For arbitrary coin systems, dynamic programming is needed.
    
    Time Complexity: O(amount) in worst case
    Space Complexity: O(amount) for result
    
    Args:
        coins: List of coin denominations (sorted in descending order for optimal results)
        amount: Target amount
        
    Returns:
        List of coins used to make the amount
    """
    # Sort coins in descending order (greedy choice)
    coins.sort(reverse=True)
    
    result = []
    remaining_amount = amount
    
    for coin in coins:
        while remaining_amount >= coin:
            result.append(coin)
            remaining_amount -= coin
    
    # Check if exact change is possible
    if remaining_amount == 0:
        return result
    else:
        return []  # Cannot make exact change


def optimal_merge_pattern(sizes: List[int]) -> int:
    """
    Find minimum cost to merge files of given sizes.
    
    Time Complexity: O(n log n) due to heap operations
    Space Complexity: O(n) for heap
    
    Args:
        sizes: List of file sizes
        
    Returns:
        Minimum cost to merge all files
    """
    if not sizes:
        return 0
    
    if len(sizes) == 1:
        return 0
    
    # Create a min heap
    heap = sizes[:]
    heapq.heapify(heap)
    
    total_cost = 0
    
    # Keep merging two smallest files until only one file remains
    while len(heap) > 1:
        # Extract two smallest files
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        
        # Merge them and add cost
        merged_size = first + second
        total_cost += merged_size
        
        # Insert merged file back into heap
        heapq.heappush(heap, merged_size)
    
    return total_cost


def demo():
    """Demonstrate the greedy algorithms."""
    print("=== Greedy Algorithms Fundamentals ===\n")
    
    # Test Activity Selection
    activities = [
        (1, 4, "Activity A"),
        (3, 5, "Activity B"),
        (0, 6, "Activity C"),
        (5, 7, "Activity D"),
        (3, 9, "Activity E"),
        (5, 9, "Activity F"),
        (6, 10, "Activity G"),
        (8, 11, "Activity H"),
        (8, 12, "Activity I"),
        (2, 14, "Activity J"),
        (12, 16, "Activity K")
    ]
    
    selected = activity_selection(activities)
    print("Activity Selection Problem:")
    print(f"  Activities: {len(activities)} total")
    print(f"  Selected: {len(selected)} activities")
    for start, end, name in selected:
        print(f"    {name}: [{start}, {end}]")
    
    print("\n" + "="*50)
    
    # Test Fractional Knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    
    max_value = fractional_knapsack(weights, values, capacity)
    print(f"\nFractional Knapsack Problem:")
    print(f"  Weights: {weights}")
    print(f"  Values: {values}")
    print(f"  Capacity: {capacity}")
    print(f"  Maximum Value: {max_value:.2f}")
    
    print("\n" + "="*50)
    
    # Test Huffman Coding
    frequencies = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
    codes = huffman_coding(frequencies)
    print(f"\nHuffman Coding:")
    print(f"  Frequencies: {frequencies}")
    print(f"  Huffman Codes:")
    for char, code in sorted(codes.items()):
        print(f"    '{char}': {code}")
    
    print("\n" + "="*50)
    
    # Test Job Sequencing
    jobs = [
        (1, 2, 100),  # (job_id, deadline, profit)
        (2, 1, 19),
        (3, 2, 27),
        (4, 1, 25),
        (5, 3, 15)
    ]
    
    max_profit, selected_jobs = job_sequencing(jobs)
    print(f"\nJob Sequencing Problem:")
    print(f"  Jobs: {jobs}")
    print(f"  Maximum Profit: {max_profit}")
    print(f"  Selected Jobs: {selected_jobs}")
    
    print("\n" + "="*50)
    
    # Test Minimum Number of Coins
    coins = [1, 2, 5, 10, 20, 50, 100, 500, 1000]
    amount = 93
    result = minimum_number_of_coins(coins, amount)
    print(f"\nMinimum Number of Coins:")
    print(f"  Coins: {coins}")
    print(f"  Amount: {amount}")
    print(f"  Coins used: {result}")
    print(f"  Total coins: {len(result)}")
    
    print("\n" + "="*50)
    
    # Test Optimal Merge Pattern
    sizes = [5, 10, 20, 30]
    min_cost = optimal_merge_pattern(sizes)
    print(f"\nOptimal Merge Pattern:")
    print(f"  File sizes: {sizes}")
    print(f"  Minimum merge cost: {min_cost}")


if __name__ == "__main__":
    demo()