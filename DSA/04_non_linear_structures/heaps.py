"""
Non-Linear Data Structures - Heaps and Priority Queues
================================================

This module provides implementations and examples of heap data structures and priority queues,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Heap properties and types (Min-Heap, Max-Heap)
- Heap operations and their complexities
- Priority queue implementations
- Heap algorithms (heapify, heap sort)
- Applications in algorithms and data science
"""

import heapq
from typing import Any, List, Optional, Tuple

class MinHeap:
    """
    Implementation of a Min-Heap (smallest element at root)
    """
    
    def __init__(self):
        """Initialize an empty min heap"""
        self.heap: List[Any] = []
    
    def push(self, item: Any) -> None:
        """
        Insert an item into the heap
        Time Complexity: O(log n)
        """
        heapq.heappush(self.heap, item)
    
    def pop(self) -> Any:
        """
        Remove and return the smallest item from the heap
        Time Complexity: O(log n)
        """
        if self.is_empty():
            raise IndexError("pop from empty heap")
        return heapq.heappop(self.heap)
    
    def peek(self) -> Any:
        """
        Return the smallest item without removing it
        Time Complexity: O(1)
        """
        if self.is_empty():
            raise IndexError("peek from empty heap")
        return self.heap[0]
    
    def heapify(self, iterable: List[Any]) -> None:
        """
        Transform list into a heap in-place
        Time Complexity: O(n)
        """
        heapq.heapify(iterable)
        self.heap = iterable
    
    def is_empty(self) -> bool:
        """
        Check if the heap is empty
        Time Complexity: O(1)
        """
        return len(self.heap) == 0
    
    def size(self) -> int:
        """
        Return the number of items in the heap
        Time Complexity: O(1)
        """
        return len(self.heap)
    
    def __str__(self) -> str:
        """String representation of the heap"""
        return f"MinHeap({self.heap})"

class MaxHeap:
    """
    Implementation of a Max-Heap (largest element at root)
    Using negative values with heapq to simulate max-heap behavior
    """
    
    def __init__(self):
        """Initialize an empty max heap"""
        self.heap: List[Any] = []
    
    def push(self, item: Any) -> None:
        """
        Insert an item into the heap
        Time Complexity: O(log n)
        """
        heapq.heappush(self.heap, -item)
    
    def pop(self) -> Any:
        """
        Remove and return the largest item from the heap
        Time Complexity: O(log n)
        """
        if self.is_empty():
            raise IndexError("pop from empty heap")
        return -heapq.heappop(self.heap)
    
    def peek(self) -> Any:
        """
        Return the largest item without removing it
        Time Complexity: O(1)
        """
        if self.is_empty():
            raise IndexError("peek from empty heap")
        return -self.heap[0]
    
    def heapify(self, iterable: List[Any]) -> None:
        """
        Transform list into a max heap in-place
        Time Complexity: O(n)
        """
        self.heap = [-x for x in iterable]
        heapq.heapify(self.heap)
    
    def is_empty(self) -> bool:
        """
        Check if the heap is empty
        Time Complexity: O(1)
        """
        return len(self.heap) == 0
    
    def size(self) -> int:
        """
        Return the number of items in the heap
        Time Complexity: O(1)
        """
        return len(self.heap)
    
    def __str__(self) -> str:
        """String representation of the heap"""
        return f"MaxHeap({[-x for x in self.heap]})"

class PriorityQueue:
    """
    Implementation of a Priority Queue using heap
    Items with lower priority numbers are dequeued first
    """
    
    def __init__(self):
        """Initialize an empty priority queue"""
        self.heap: List[Tuple[int, Any]] = []
        self.entry_count = 0  # To handle items with same priority
    
    def enqueue(self, item: Any, priority: int) -> None:
        """
        Add an item with a priority to the queue
        Time Complexity: O(log n)
        """
        heapq.heappush(self.heap, (priority, self.entry_count, item))
        self.entry_count += 1
    
    def dequeue(self) -> Any:
        """
        Remove and return the highest priority item
        Time Complexity: O(log n)
        """
        if self.is_empty():
            raise IndexError("dequeue from empty priority queue")
        priority, count, item = heapq.heappop(self.heap)
        return item
    
    def peek(self) -> Any:
        """
        Return the highest priority item without removing it
        Time Complexity: O(1)
        """
        if self.is_empty():
            raise IndexError("peek from empty priority queue")
        return self.heap[0][2]  # Return just the item, not priority tuple
    
    def is_empty(self) -> bool:
        """
        Check if the priority queue is empty
        Time Complexity: O(1)
        """
        return len(self.heap) == 0
    
    def size(self) -> int:
        """
        Return the number of items in the priority queue
        Time Complexity: O(1)
        """
        return len(self.heap)
    
    def __str__(self) -> str:
        """String representation of the priority queue"""
        items = [item for priority, count, item in self.heap]
        return f"PriorityQueue({items})"

class HeapSort:
    """
    Implementation of Heap Sort algorithm
    """
    
    @staticmethod
    def sort(arr: List[Any]) -> List[Any]:
        """
        Sort array using heap sort algorithm
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        # Create a copy to avoid modifying original array
        heap = arr.copy()
        
        # Transform list into a heap
        heapq.heapify(heap)
        
        # Extract elements one by one
        sorted_arr = []
        while heap:
            sorted_arr.append(heapq.heappop(heap))
        
        return sorted_arr

def heap_operations_demo():
    """
    Demonstrate heap operations and their complexities
    """
    print("=== Heap Operations Demo ===")
    
    # Min-Heap example
    print("1. Min-Heap Operations:")
    min_heap = MinHeap()
    values = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    
    print(f"   Inserting values: {values}")
    for value in values:
        min_heap.push(value)
        print(f"   After inserting {value}: {min_heap}")
    
    print(f"   Peek (smallest): {min_heap.peek()}")
    
    print("   Extracting elements in sorted order:")
    extracted = []
    while not min_heap.is_empty():
        element = min_heap.pop()
        extracted.append(element)
        print(f"   Extracted: {element}, Remaining: {min_heap}")
    
    # Max-Heap example
    print("\n2. Max-Heap Operations:")
    max_heap = MaxHeap()
    values = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    
    print(f"   Inserting values: {values}")
    for value in values:
        max_heap.push(value)
        print(f"   After inserting {value}: {max_heap}")
    
    print(f"   Peek (largest): {max_heap.peek()}")
    
    print("   Extracting elements in sorted order:")
    extracted = []
    while not max_heap.is_empty():
        element = max_heap.pop()
        extracted.append(element)
        print(f"   Extracted: {element}, Remaining: {max_heap}")

def priority_queue_demo():
    """
    Demonstrate priority queue operations
    """
    print("\n=== Priority Queue Demo ===")
    
    pq = PriorityQueue()
    
    # Enqueue items with priorities
    items = [("Task A", 3), ("Task B", 1), ("Task C", 2), ("Task D", 1)]
    print("1. Enqueuing tasks with priorities:")
    for item, priority in items:
        pq.enqueue(item, priority)
        print(f"   Enqueued '{item}' with priority {priority}")
    
    print(f"   Priority Queue: {pq}")
    
    print("\n2. Dequeuing tasks (lowest priority number first):")
    while not pq.is_empty():
        item = pq.dequeue()
        print(f"   Dequeued: '{item}', Remaining: {pq}")

def heap_sort_demo():
    """
    Demonstrate heap sort algorithm
    """
    print("\n=== Heap Sort Demo ===")
    
    # Test with different arrays
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 8, 1, 9],
        [1],
        [],
        [3, 3, 3, 3],
        [9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]
    
    for i, arr in enumerate(test_arrays):
        sorted_arr = HeapSort.sort(arr)
        print(f"   Array {i+1}: {arr} → {sorted_arr}")

def heap_applications():
    """
    Demonstrate common applications of heaps
    """
    print("\n=== Heap Applications ===")
    
    # 1. Finding kth smallest/largest element
    print("1. Finding kth Smallest/Largest Element:")
    def find_kth_largest(arr: List[int], k: int) -> int:
        """
        Find kth largest element using max heap
        Time Complexity: O(n log k)
        """
        if k > len(arr):
            raise ValueError("k is larger than array size")
        
        # Use min heap of size k
        heap = []
        for num in arr:
            if len(heap) < k:
                heapq.heappush(heap, num)
            elif num > heap[0]:
                heapq.heapreplace(heap, num)
        
        return heap[0]
    
    arr = [3, 2, 1, 5, 6, 4]
    k = 2
    kth_largest = find_kth_largest(arr, k)
    print(f"   Array: {arr}")
    print(f"   {k}th largest element: {kth_largest}")
    
    # 2. Merge k sorted arrays
    print("\n2. Merge k Sorted Arrays:")
    def merge_k_sorted_arrays(arrays: List[List[int]]) -> List[int]:
        """
        Merge k sorted arrays using min heap
        Time Complexity: O(N log k) where N is total elements
        """
        heap = []  # (value, array_index, element_index)
        result = []
        
        # Initialize heap with first element of each array
        for i, arr in enumerate(arrays):
            if arr:
                heapq.heappush(heap, (arr[0], i, 0))
        
        # Extract minimum and add next element from same array
        while heap:
            value, array_idx, element_idx = heapq.heappop(heap)
            result.append(value)
            
            # Add next element from the same array
            if element_idx + 1 < len(arrays[array_idx]):
                next_value = arrays[array_idx][element_idx + 1]
                heapq.heappush(heap, (next_value, array_idx, element_idx + 1))
        
        return result
    
    arrays = [[1, 4, 5], [1, 3, 4], [2, 6]]
    merged = merge_k_sorted_arrays(arrays)
    print(f"   Arrays: {arrays}")
    print(f"   Merged: {merged}")
    
    # 3. Top K frequent elements
    print("\n3. Top K Frequent Elements:")
    def top_k_frequent(nums: List[int], k: int) -> List[int]:
        """
        Find k most frequent elements
        Time Complexity: O(n log k)
        """
        # Count frequencies
        freq_map = {}
        for num in nums:
            freq_map[num] = freq_map.get(num, 0) + 1
        
        # Use min heap to keep track of top k elements
        heap = []
        for num, freq in freq_map.items():
            if len(heap) < k:
                heapq.heappush(heap, (freq, num))
            elif freq > heap[0][0]:
                heapq.heapreplace(heap, (freq, num))
        
        # Extract elements
        return [num for freq, num in heap]
    
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    top_k = top_k_frequent(nums, k)
    print(f"   Numbers: {nums}")
    print(f"   Top {k} frequent elements: {top_k}")

def data_science_applications():
    """
    Examples of heaps in data science applications
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Recommendation systems
    print("1. Recommendation Systems:")
    print("   Heaps can efficiently maintain top-k recommendations")
    print("   Priority queues help rank items by relevance scores")
    print("   Enable real-time personalized recommendations")
    
    # 2. Streaming data processing
    print("\n2. Streaming Data Processing:")
    print("   Heaps maintain top-k elements in streaming data")
    print("   Useful for trending topics, popular items")
    print("   Memory efficient for large-scale data streams")
    
    # 3. Machine learning algorithms
    print("\n3. Machine Learning Algorithms:")
    print("   K-nearest neighbors use heaps for efficient neighbor search")
    print("   Decision trees use priority queues for best split selection")
    print("   Clustering algorithms use heaps for centroid management")
    
    # 4. Resource allocation
    print("\n4. Resource Allocation:")
    print("   Priority queues manage task scheduling in distributed systems")
    print("   Cloud computing uses heaps for load balancing")
    print("   Network routing algorithms use priority queues")

def performance_comparison():
    """
    Compare performance of different heap operations
    """
    print("\n=== Performance Comparison ===")
    
    import time
    import random
    
    # Test with different sizes
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTesting with {size} elements:")
        
        # Generate random data
        data = [random.randint(1, 100000) for _ in range(size)]
        
        # Min-Heap operations
        min_heap = MinHeap()
        start = time.time()
        for item in data:
            min_heap.push(item)
        insert_time = time.time() - start
        
        start = time.time()
        for _ in range(100):  # Extract 100 elements
            if not min_heap.is_empty():
                min_heap.pop()
        extract_time = time.time() - start
        
        print(f"   Min-Heap - Insert {size} items: {insert_time:.6f}s")
        print(f"   Min-Heap - Extract 100 items: {extract_time:.6f}s")
        
        # Heap Sort
        start = time.time()
        sorted_data = HeapSort.sort(data)
        sort_time = time.time() - start
        
        print(f"   Heap Sort {size} items: {sort_time:.6f}s")

# Example usage and testing
if __name__ == "__main__":
    # Heap operations demo
    heap_operations_demo()
    print("\n" + "="*50 + "\n")
    
    # Priority queue demo
    priority_queue_demo()
    print("\n" + "="*50 + "\n")
    
    # Heap sort demo
    heap_sort_demo()
    print("\n" + "="*50 + "\n")
    
    # Heap applications
    heap_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Min-Heap and Max-Heap implementations")
    print("2. Priority Queue implementation")
    print("3. Heap Sort algorithm")
    print("4. Common heap operations and their complexities")
    print("5. Practical applications in algorithms")
    print("6. Data science applications of heaps")
    print("7. Performance characteristics of heap operations")
    print("\nKey takeaways:")
    print("- Heaps provide O(log n) insertion and extraction")
    print("- Heap Sort offers guaranteed O(n log n) performance")
    print("- Priority Queues are essential for scheduling algorithms")
    print("- Heaps are fundamental in graph algorithms (Dijkstra, Prim)")
    print("- Heaps enable efficient top-k queries in data science")