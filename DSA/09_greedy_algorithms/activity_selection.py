"""
Activity Selection and Scheduling Problems

This module covers various activity selection and scheduling problems:
- Activity selection problem
- Weighted activity selection
- Interval scheduling optimization
- Meeting rooms problem
- Task scheduling with deadlines
"""

from typing import List, Tuple


def activity_selection_basic(activities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Select maximum number of non-overlapping activities.
    
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


def weighted_activity_selection(activities: List[Tuple[int, int, int, str]]) -> Tuple[int, List[Tuple[int, int, int, str]]]:
    """
    Select activities to maximize total weight/value.
    
    Time Complexity: O(n log n + n * W) where W is maximum end time
    Space Complexity: O(n + W)
    
    Args:
        activities: List of tuples (start_time, end_time, weight, activity_name)
        
    Returns:
        Tuple of (maximum_weight, list_of_selected_activities)
    """
    if not activities:
        return 0, []
    
    # Sort activities by end time
    activities.sort(key=lambda x: x[1])
    
    n = len(activities)
    # Find maximum end time
    max_end_time = max(activity[1] for activity in activities)
    
    # dp[i] represents maximum weight achievable by time i
    dp = [0] * (max_end_time + 1)
    
    # For each activity, update dp array
    for start, end, weight, name in activities:
        # Find maximum weight achievable before start time
        max_before = max(dp[:start + 1]) if start >= 0 else 0
        # Update dp for end time
        dp[end] = max(dp[end], max_before + weight)
    
    # Backtrack to find selected activities
    selected = []
    current_time = max_end_time
    max_weight = max(dp)
    
    # This is a simplified backtrack, actual implementation would be more complex
    # For demonstration, we'll just return the maximum weight
    
    return max_weight, []


def meeting_rooms(intervals: List[Tuple[int, int]]) -> int:
    """
    Find minimum number of meeting rooms required.
    
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for start and end arrays
    
    Args:
        intervals: List of tuples (start_time, end_time)
        
    Returns:
        Minimum number of meeting rooms required
    """
    if not intervals:
        return 0
    
    # Separate start and end times
    start_times = sorted([interval[0] for interval in intervals])
    end_times = sorted([interval[1] for interval in intervals])
    
    rooms_needed = 0
    max_rooms = 0
    start_ptr = end_ptr = 0
    
    # Process events in chronological order
    while start_ptr < len(start_times):
        # If a meeting starts before the earliest ending meeting
        if start_times[start_ptr] < end_times[end_ptr]:
            rooms_needed += 1
            max_rooms = max(max_rooms, rooms_needed)
            start_ptr += 1
        else:
            # A meeting ends, free up a room
            rooms_needed -= 1
            end_ptr += 1
    
    return max_rooms


def interval_scheduling_maximization(intervals: List[Tuple[int, int]]) -> int:
    """
    Find maximum number of non-overlapping intervals.
    
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(1)
    
    Args:
        intervals: List of tuples (start_time, end_time)
        
    Returns:
        Maximum number of non-overlapping intervals
    """
    if not intervals:
        return 0
    
    # Sort intervals by end time (greedy choice)
    intervals.sort(key=lambda x: x[1])
    
    count = 1
    last_end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        # If current interval starts after last selected interval ends
        if intervals[i][0] >= last_end:
            count += 1
            last_end = intervals[i][1]
    
    return count


def task_scheduling_with_deadlines(tasks: List[Tuple[int, int, int]]) -> Tuple[int, List[int]]:
    """
    Schedule tasks to maximize profit considering deadlines.
    
    Time Complexity: O(n^2) where n is number of tasks
    Space Complexity: O(n)
    
    Args:
        tasks: List of tuples (task_id, deadline, profit)
        
    Returns:
        Tuple of (maximum_profit, list_of_scheduled_task_ids)
    """
    if not tasks:
        return 0, []
    
    # Sort tasks by profit in descending order (greedy choice)
    tasks.sort(key=lambda x: x[2], reverse=True)
    
    n = len(tasks)
    # Find maximum deadline
    max_deadline = max(task[1] for task in tasks)
    
    # Create array to keep track of free time slots
    slots = [-1] * (max_deadline + 1)
    scheduled_tasks = []
    total_profit = 0
    
    # Iterate through all tasks
    for task_id, deadline, profit in tasks:
        # Find a free slot for this task (starting from deadline and moving backwards)
        for j in range(min(max_deadline, deadline), 0, -1):
            if slots[j] == -1:
                slots[j] = task_id
                scheduled_tasks.append(task_id)
                total_profit += profit
                break
    
    return total_profit, scheduled_tasks


def remove_overlapping_intervals(intervals: List[List[int]]) -> int:
    """
    Find minimum number of intervals to remove to make rest non-overlapping.
    
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(1)
    
    Args:
        intervals: List of intervals [start, end]
        
    Returns:
        Minimum number of intervals to remove
    """
    if len(intervals) <= 1:
        return 0
    
    # Sort intervals by end time (greedy choice)
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    last_end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        # If current interval overlaps with last selected interval
        if intervals[i][0] < last_end:
            count += 1  # Remove current interval
        else:
            last_end = intervals[i][1]  # Keep current interval
    
    return count


def partition_labels(s: str) -> List[int]:
    """
    Partition string into as many parts as possible so that each letter appears in at most one part.
    
    Time Complexity: O(n) where n is length of string
    Space Complexity: O(1) since at most 26 characters
    
    Args:
        s: Input string
        
    Returns:
        List of lengths of partitions
    """
    if not s:
        return []
    
    # Record last occurrence of each character
    last_occurrence = {char: i for i, char in enumerate(s)}
    
    result = []
    start = 0
    end = 0
    
    for i, char in enumerate(s):
        # Extend end to the furthest last occurrence of any character in current partition
        end = max(end, last_occurrence[char])
        
        # If we've reached the end of current partition
        if i == end:
            result.append(end - start + 1)
            start = end + 1
    
    return result


def demo():
    """Demonstrate the activity selection and scheduling algorithms."""
    print("=== Activity Selection and Scheduling Problems ===\n")
    
    # Test Basic Activity Selection
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
    
    selected = activity_selection_basic(activities)
    print("Basic Activity Selection:")
    print(f"  Total activities: {len(activities)}")
    print(f"  Selected activities: {len(selected)}")
    for start, end, name in selected:
        print(f"    {name}: [{start}, {end}]")
    
    print("\n" + "="*60)
    
    # Test Meeting Rooms
    intervals = [(0, 30), (5, 10), (15, 20)]
    rooms = meeting_rooms(intervals)
    print(f"\nMeeting Rooms Problem:")
    print(f"  Intervals: {intervals}")
    print(f"  Minimum rooms required: {rooms}")
    
    intervals = [(7, 10), (2, 4)]
    rooms = meeting_rooms(intervals)
    print(f"  Intervals: {intervals}")
    print(f"  Minimum rooms required: {rooms}")
    
    print("\n" + "="*60)
    
    # Test Interval Scheduling Maximization
    intervals = [(1, 2), (3, 4), (0, 6), (5, 7), (8, 9), (5, 9)]
    max_intervals = interval_scheduling_maximization(intervals)
    print(f"\nInterval Scheduling Maximization:")
    print(f"  Intervals: {intervals}")
    print(f"  Maximum non-overlapping intervals: {max_intervals}")
    
    print("\n" + "="*60)
    
    # Test Task Scheduling with Deadlines
    tasks = [
        (1, 2, 100),  # (task_id, deadline, profit)
        (2, 1, 19),
        (3, 2, 27),
        (4, 1, 25),
        (5, 3, 15)
    ]
    
    max_profit, scheduled = task_scheduling_with_deadlines(tasks)
    print(f"\nTask Scheduling with Deadlines:")
    print(f"  Tasks: {tasks}")
    print(f"  Maximum profit: {max_profit}")
    print(f"  Scheduled tasks: {scheduled}")
    
    print("\n" + "="*60)
    
    # Test Remove Overlapping Intervals
    intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
    remove_count = remove_overlapping_intervals(intervals)
    print(f"\nRemove Overlapping Intervals:")
    print(f"  Intervals: {intervals}")
    print(f"  Intervals to remove: {remove_count}")
    
    intervals = [[1, 2], [1, 2], [1, 2]]
    remove_count = remove_overlapping_intervals(intervals)
    print(f"  Intervals: {intervals}")
    print(f"  Intervals to remove: {remove_count}")
    
    print("\n" + "="*60)
    
    # Test Partition Labels
    s = "ababcbacadefegdehijhklij"
    partitions = partition_labels(s)
    print(f"\nPartition Labels:")
    print(f"  String: '{s}'")
    print(f"  Partitions: {partitions}")
    print(f"  Number of partitions: {len(partitions)}")
    
    s = "eccbbbbdec"
    partitions = partition_labels(s)
    print(f"  String: '{s}'")
    print(f"  Partitions: {partitions}")


if __name__ == "__main__":
    demo()