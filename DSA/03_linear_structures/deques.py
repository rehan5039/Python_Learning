"""
Linear Data Structures - Deques (Double-Ended Queues)
================================================

This module provides implementations and examples of deques (double-ended queues),
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Deque implementations
- Operations and their complexities
- Applications in algorithms
- Data science use cases
"""

from collections import deque
from typing import Any, Optional, List

class DequeFromScratch:
    """
    Implementation of a deque from scratch using a circular buffer approach
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize a deque with optional maximum size.
        
        Args:
            max_size: Maximum number of elements (None for unlimited)
        """
        self._max_size = max_size
        self._data = []
        self._front = 0
        self._size = 0
    
    def appendleft(self, item: Any) -> None:
        """
        Add an item to the left end of the deque - O(1) amortized
        """
        if self._max_size and self._size >= self._max_size:
            raise OverflowError("Deque is full")
        
        self._front = (self._front - 1) % max(len(self._data), 1) if self._data else 0
        if self._size < len(self._data):
            self._data[self._front] = item
        else:
            self._data.insert(self._front, item)
        self._size += 1
    
    def append(self, item: Any) -> None:
        """
        Add an item to the right end of the deque - O(1) amortized
        """
        if self._max_size and self._size >= self._max_size:
            raise OverflowError("Deque is full")
        
        if self._size < len(self._data):
            index = (self._front + self._size) % len(self._data)
            self._data[index] = item
        else:
            if self._size == 0:
                self._data.append(item)
            else:
                index = (self._front + self._size) % len(self._data)
                if index < len(self._data):
                    self._data.insert(index, item)
                else:
                    self._data.append(item)
        self._size += 1
    
    def popleft(self) -> Any:
        """
        Remove and return an item from the left end of the deque - O(1)
        """
        if self._size == 0:
            raise IndexError("pop from empty deque")
        
        item = self._data[self._front]
        self._front = (self._front + 1) % len(self._data)
        self._size -= 1
        return item
    
    def pop(self) -> Any:
        """
        Remove and return an item from the right end of the deque - O(1)
        """
        if self._size == 0:
            raise IndexError("pop from empty deque")
        
        index = (self._front + self._size - 1) % len(self._data)
        item = self._data[index]
        self._size -= 1
        return item
    
    def peekleft(self) -> Any:
        """
        Return the leftmost item without removing it - O(1)
        """
        if self._size == 0:
            raise IndexError("peek from empty deque")
        return self._data[self._front]
    
    def peek(self) -> Any:
        """
        Return the rightmost item without removing it - O(1)
        """
        if self._size == 0:
            raise IndexError("peek from empty deque")
        index = (self._front + self._size - 1) % len(self._data)
        return self._data[index]
    
    def is_empty(self) -> bool:
        """
        Check if the deque is empty - O(1)
        """
        return self._size == 0
    
    def is_full(self) -> bool:
        """
        Check if the deque is full - O(1)
        """
        return self._max_size is not None and self._size >= self._max_size
    
    def size(self) -> int:
        """
        Return the number of items in the deque - O(1)
        """
        return self._size
    
    def to_list(self) -> List[Any]:
        """
        Convert deque to Python list - O(n)
        """
        result = []
        for i in range(self._size):
            index = (self._front + i) % len(self._data)
            result.append(self._data[index])
        return result
    
    def __str__(self) -> str:
        """String representation of the deque"""
        return f"DequeFromScratch({self.to_list()})"

class BoundedDeque:
    """
    Implementation of a bounded deque using collections.deque with maxlen
    """
    
    def __init__(self, max_size: int):
        """
        Initialize a bounded deque with maximum size.
        
        Args:
            max_size: Maximum number of elements
        """
        self._deque = deque(maxlen=max_size)
    
    def appendleft(self, item: Any) -> None:
        """
        Add an item to the left end of the deque - O(1)
        If deque is full, rightmost item is automatically removed
        """
        self._deque.appendleft(item)
    
    def append(self, item: Any) -> None:
        """
        Add an item to the right end of the deque - O(1)
        If deque is full, leftmost item is automatically removed
        """
        self._deque.append(item)
    
    def popleft(self) -> Any:
        """
        Remove and return an item from the left end of the deque - O(1)
        """
        if not self._deque:
            raise IndexError("pop from empty deque")
        return self._deque.popleft()
    
    def pop(self) -> Any:
        """
        Remove and return an item from the right end of the deque - O(1)
        """
        if not self._deque:
            raise IndexError("pop from empty deque")
        return self._deque.pop()
    
    def peekleft(self) -> Any:
        """
        Return the leftmost item without removing it - O(1)
        """
        if not self._deque:
            raise IndexError("peek from empty deque")
        return self._deque[0]
    
    def peek(self) -> Any:
        """
        Return the rightmost item without removing it - O(1)
        """
        if not self._deque:
            raise IndexError("peek from empty deque")
        return self._deque[-1]
    
    def is_empty(self) -> bool:
        """
        Check if the deque is empty - O(1)
        """
        return len(self._deque) == 0
    
    def is_full(self) -> bool:
        """
        Check if the deque is full - O(1)
        """
        return len(self._deque) == self._deque.maxlen
    
    def size(self) -> int:
        """
        Return the number of items in the deque - O(1)
        """
        return len(self._deque)
    
    def to_list(self) -> List[Any]:
        """
        Convert deque to Python list - O(n)
        """
        return list(self._deque)
    
    def __str__(self) -> str:
        """String representation of the deque"""
        return f"BoundedDeque({list(self._deque)}, max_size={self._deque.maxlen})"

def deque_operations_demo():
    """
    Demonstrate common deque operations and their complexities
    """
    print("=== Deque Operations Demo ===")
    
    # Using collections.deque
    print("1. Using collections.deque:")
    d = deque([1, 2, 3, 4, 5])
    print(f"   Initial deque: {list(d)}")
    d.appendleft(0)
    print(f"   After appendleft(0): {list(d)}")
    d.append(6)
    print(f"   After append(6): {list(d)}")
    left_item = d.popleft()
    print(f"   Popped from left: {left_item}, deque: {list(d)}")
    right_item = d.pop()
    print(f"   Popped from right: {right_item}, deque: {list(d)}")
    
    # Using custom DequeFromScratch
    print("\n2. Using custom DequeFromScratch:")
    custom_d = DequeFromScratch()
    for i in range(5):
        custom_d.append(i)
    print(f"   Initial deque: {custom_d}")
    custom_d.appendleft(-1)
    print(f"   After appendleft(-1): {custom_d}")
    custom_d.append(5)
    print(f"   After append(5): {custom_d}")
    left_item = custom_d.popleft()
    print(f"   Popped from left: {left_item}, deque: {custom_d}")
    right_item = custom_d.pop()
    print(f"   Popped from right: {right_item}, deque: {custom_d}")

def deque_applications():
    """
    Demonstrate common applications of deques
    """
    print("\n=== Deque Applications ===")
    
    # 1. Sliding window maximum
    def sliding_window_maximum(arr: List[int], k: int) -> List[int]:
        """
        Find maximum in all sliding windows of size k using deque
        Time Complexity: O(n)
        """
        if not arr or k == 0:
            return []
        
        # Deque to store indices of array elements
        dq = deque()
        result = []
        
        # Process first k elements
        for i in range(k):
            # Remove indices of smaller elements
            while dq and arr[i] >= arr[dq[-1]]:
                dq.pop()
            dq.append(i)
        
        # Process remaining elements
        for i in range(k, len(arr)):
            # Add maximum of previous window
            result.append(arr[dq[0]])
            
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices of smaller elements
            while dq and arr[i] >= arr[dq[-1]]:
                dq.pop()
            
            dq.append(i)
        
        # Add maximum of last window
        result.append(arr[dq[0]])
        return result
    
    # Example usage
    arr = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    max_values = sliding_window_maximum(arr, k)
    print(f"1. Sliding Window Maximum:")
    print(f"   Array: {arr}")
    print(f"   Window size: {k}")
    print(f"   Maximums: {max_values}")
    
    # 2. Browser history
    print(f"\n2. Browser History:")
    print(f"   Deques can efficiently implement browser history")
    print(f"   Backward history: appendleft() for new pages")
    print(f"   Forward history: append() for forward navigation")
    print(f"   Navigation: popleft() and pop() for back/forward")
    
    # 3. Undo/Redo functionality
    print(f"\n3. Undo/Redo Functionality:")
    print(f"   Undo stack: append() for new actions")
    print(f"   Redo stack: appendleft() for redo actions")
    print(f"   Undo operation: pop() from undo stack")
    print(f"   Redo operation: popleft() from redo stack")

def data_science_applications():
    """
    Examples of deques in data science applications
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Moving averages
    def moving_average(data: List[float], window_size: int) -> List[float]:
        """
        Calculate moving average using deque for efficient window management
        """
        if len(data) < window_size:
            return []
        
        window = deque(maxlen=window_size)
        averages = []
        
        for value in data:
            window.append(value)
            if len(window) == window_size:
                averages.append(sum(window) / window_size)
        
        return averages
    
    # Example usage
    stock_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    ma_3 = moving_average(stock_prices, 3)
    print(f"1. Moving Averages:")
    print(f"   Stock prices: {stock_prices}")
    print(f"   3-day moving averages: {[round(x, 2) for x in ma_3]}")
    
    # 2. FIFO cache implementation
    print(f"\n2. FIFO Cache:")
    print(f"   Deques can implement First-In-First-Out cache eviction")
    print(f"   New items appended to right end")
    print(f"   When cache is full, items popped from left end")
    
    # 3. Time series windowing
    print(f"\n3. Time Series Windowing:")
    print(f"   Deques efficiently manage time-based windows in streaming data")
    print(f"   Old data automatically removed when window size exceeded")
    print(f"   Real-time analytics on recent data points")

def performance_comparison():
    """
    Compare performance of different deque implementations
    """
    print("\n=== Performance Comparison ===")
    
    import time
    
    # Test with 10000 operations
    n = 10000
    
    # collections.deque
    start = time.time()
    d1 = deque()
    for i in range(n):
        d1.append(i)
    for i in range(n // 2):
        d1.popleft()
        d1.appendleft(i)
    end = time.time()
    print(f"collections.deque: {end - start:.6f} seconds")
    
    # Custom DequeFromScratch
    start = time.time()
    d2 = DequeFromScratch()
    for i in range(n):
        d2.append(i)
    for i in range(n // 2):
        d2.popleft()
        d2.appendleft(i)
    end = time.time()
    print(f"DequeFromScratch: {end - start:.6f} seconds")
    
    # BoundedDeque
    start = time.time()
    d3 = BoundedDeque(n * 2)
    for i in range(n):
        d3.append(i)
    for i in range(n // 2):
        d3.popleft()
        d3.appendleft(i)
    end = time.time()
    print(f"BoundedDeque: {end - start:.6f} seconds")

def advanced_deque_techniques():
    """
    Advanced techniques and patterns using deques
    """
    print("\n=== Advanced Deque Techniques ===")
    
    # 1. Rotating deque
    print("1. Rotating Deque:")
    d = deque([1, 2, 3, 4, 5])
    print(f"   Original: {list(d)}")
    d.rotate(2)  # Rotate right by 2
    print(f"   Rotated right by 2: {list(d)}")
    d.rotate(-3)  # Rotate left by 3
    print(f"   Rotated left by 3: {list(d)}")
    
    # 2. Extending deques
    print("\n2. Extending Deques:")
    d1 = deque([1, 2, 3])
    d2 = deque([4, 5, 6])
    print(f"   Deque 1: {list(d1)}")
    print(f"   Deque 2: {list(d2)}")
    d1.extend(d2)  # Add elements to right end
    print(f"   After extend: {list(d1)}")
    d1 = deque([1, 2, 3])
    d1.extendleft([0, -1])  # Add elements to left end (in reverse order)
    print(f"   After extendleft([-1, 0]): {list(d1)}")
    
    # 3. Deque as stack or queue
    print("\n3. Deque as Stack or Queue:")
    d = deque()
    
    # As stack (LIFO)
    print("   Using as Stack (LIFO):")
    for i in range(3):
        d.append(i)
        print(f"     Pushed {i}: {list(d)}")
    while d:
        item = d.pop()
        print(f"     Popped {item}: {list(d)}")
    
    # As queue (FIFO)
    print("   Using as Queue (FIFO):")
    for i in range(3):
        d.append(i)
        print(f"     Enqueued {i}: {list(d)}")
    while d:
        item = d.popleft()
        print(f"     Dequeued {item}: {list(d)}")

# Example usage and testing
if __name__ == "__main__":
    # Deque operations demo
    deque_operations_demo()
    print("\n" + "="*50 + "\n")
    
    # Deque applications
    deque_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    print("\n" + "="*50 + "\n")
    
    # Advanced techniques
    advanced_deque_techniques()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Deque implementations from scratch and using collections")
    print("2. Common operations and their time complexities")
    print("3. Practical applications in algorithms and systems")
    print("4. Data science applications of deques")
    print("5. Performance comparisons between implementations")
    print("6. Advanced techniques and patterns")
    print("\nKey takeaways:")
    print("- Deques provide O(1) operations at both ends")
    print("- collections.deque is highly optimized and recommended for most use cases")
    print("- Custom implementations give more control but require careful design")
    print("- Bounded deques automatically manage memory with fixed capacity")
    print("- Deques are versatile for sliding windows, queues, stacks, and more")