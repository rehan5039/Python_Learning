"""
Linear Data Structures - Stacks and Queues
======================================

This module provides implementations and examples of stack and queue data structures,
with a focus on Python-specific implementations and practical applications.

Topics Covered:
- Stack implementations and operations
- Queue implementations and operations
- Specialized queue variations (deque, priority queue)
- Applications in algorithms and data science
"""

from collections import deque
import heapq
from typing import Any, List, Optional

class Stack:
    """
    Implementation of a stack data structure (LIFO - Last In, First Out)
    """
    
    def __init__(self):
        """Initialize an empty stack"""
        self._items = []
    
    def push(self, item: Any) -> None:
        """
        Add an item to the top of the stack - O(1)
        """
        self._items.append(item)
    
    def pop(self) -> Any:
        """
        Remove and return the top item from the stack - O(1)
        """
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()
    
    def peek(self) -> Any:
        """
        Return the top item without removing it - O(1)
        """
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        """
        Check if the stack is empty - O(1)
        """
        return len(self._items) == 0
    
    def size(self) -> int:
        """
        Return the number of items in the stack - O(1)
        """
        return len(self._items)
    
    def __str__(self) -> str:
        """String representation of the stack"""
        return f"Stack({self._items})"

class Queue:
    """
    Implementation of a queue data structure (FIFO - First In, First Out)
    """
    
    def __init__(self):
        """Initialize an empty queue"""
        self._items = []
    
    def enqueue(self, item: Any) -> None:
        """
        Add an item to the rear of the queue - O(1)
        """
        self._items.append(item)
    
    def dequeue(self) -> Any:
        """
        Remove and return the front item from the queue - O(n)
        Note: This is inefficient with lists; use collections.deque for O(1)
        """
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._items.pop(0)
    
    def front(self) -> Any:
        """
        Return the front item without removing it - O(1)
        """
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._items[0]
    
    def is_empty(self) -> bool:
        """
        Check if the queue is empty - O(1)
        """
        return len(self._items) == 0
    
    def size(self) -> int:
        """
        Return the number of items in the queue - O(1)
        """
        return len(self._items)
    
    def __str__(self) -> str:
        """String representation of the queue"""
        return f"Queue({self._items})"

class Deque:
    """
    Implementation of a double-ended queue using collections.deque
    """
    
    def __init__(self):
        """Initialize an empty deque"""
        self._items = deque()
    
    def add_front(self, item: Any) -> None:
        """
        Add an item to the front of the deque - O(1)
        """
        self._items.appendleft(item)
    
    def add_rear(self, item: Any) -> None:
        """
        Add an item to the rear of the deque - O(1)
        """
        self._items.append(item)
    
    def remove_front(self) -> Any:
        """
        Remove and return the front item from the deque - O(1)
        """
        if self.is_empty():
            raise IndexError("remove from empty deque")
        return self._items.popleft()
    
    def remove_rear(self) -> Any:
        """
        Remove and return the rear item from the deque - O(1)
        """
        if self.is_empty():
            raise IndexError("remove from empty deque")
        return self._items.pop()
    
    def is_empty(self) -> bool:
        """
        Check if the deque is empty - O(1)
        """
        return len(self._items) == 0
    
    def size(self) -> int:
        """
        Return the number of items in the deque - O(1)
        """
        return len(self._items)
    
    def __str__(self) -> str:
        """String representation of the deque"""
        return f"Deque({list(self._items)})"

class PriorityQueue:
    """
    Implementation of a priority queue using heapq
    """
    
    def __init__(self):
        """Initialize an empty priority queue"""
        self._items = []
    
    def enqueue(self, item: Any, priority: int) -> None:
        """
        Add an item with a priority to the queue - O(log n)
        Lower priority numbers indicate higher priority
        """
        heapq.heappush(self._items, (priority, item))
    
    def dequeue(self) -> Any:
        """
        Remove and return the highest priority item - O(log n)
        """
        if self.is_empty():
            raise IndexError("dequeue from empty priority queue")
        return heapq.heappop(self._items)[1]
    
    def is_empty(self) -> bool:
        """
        Check if the priority queue is empty - O(1)
        """
        return len(self._items) == 0
    
    def size(self) -> int:
        """
        Return the number of items in the priority queue - O(1)
        """
        return len(self._items)
    
    def __str__(self) -> str:
        """String representation of the priority queue"""
        return f"PriorityQueue({self._items})"

def stack_applications():
    """
    Demonstrate common applications of stacks
    """
    print("=== Stack Applications ===")
    
    # 1. Expression evaluation
    def is_balanced_parentheses(expression: str) -> bool:
        """
        Check if parentheses in an expression are balanced
        """
        stack = Stack()
        opening = "([{"
        closing = ")]}"
        pairs = {"(": ")", "[": "]", "{": "}"}
        
        for char in expression:
            if char in opening:
                stack.push(char)
            elif char in closing:
                if stack.is_empty():
                    return False
                if pairs[stack.pop()] != char:
                    return False
        
        return stack.is_empty()
    
    expressions = ["(a + b) * c", "((a + b)", "(a + b))", "{[()]}"]
    print("1. Balanced Parentheses:")
    for expr in expressions:
        result = is_balanced_parentheses(expr)
        print(f"   '{expr}' -> {'Balanced' if result else 'Not Balanced'}")
    
    # 2. Function call simulation
    print("\n2. Function Call Simulation:")
    print("   Stacks are used by programming languages to manage function calls")
    print("   Each function call is pushed onto the call stack")
    print("   When a function returns, it's popped from the stack")
    
    # 3. Undo functionality
    print("\n3. Undo Functionality:")
    print("   Text editors use stacks to implement undo operations")
    print("   Each action is pushed onto an undo stack")
    print("   Undo pops the most recent action and reverses it")

def queue_applications():
    """
    Demonstrate common applications of queues
    """
    print("\n=== Queue Applications ===")
    
    # 1. Breadth-First Search (BFS)
    print("1. Breadth-First Search (BFS):")
    print("   Queues are used to explore nodes level by level in graphs")
    print("   Each node is enqueued when discovered and dequeued for processing")
    
    # 2. Print job scheduling
    print("\n2. Print Job Scheduling:")
    print("   Print jobs are queued in the order they arrive (FIFO)")
    print("   The printer dequeues and processes jobs one by one")
    
    # 3. Buffer for data streams
    print("\n3. Buffer for Data Streams:")
    print("   Queues buffer data in streaming applications")
    print("   Data is enqueued as it arrives and dequeued for processing")

def deque_applications():
    """
    Demonstrate common applications of deques
    """
    print("\n=== Deque Applications ===")
    
    # 1. Palindrome checker
    def is_palindrome(text: str) -> bool:
        """
        Check if a string is a palindrome using a deque
        """
        # Remove spaces and convert to lowercase
        cleaned = ''.join(text.lower().split())
        d = Deque()
        
        # Add all characters to deque
        for char in cleaned:
            d.add_rear(char)
        
        # Compare characters from both ends
        while d.size() > 1:
            if d.remove_front() != d.remove_rear():
                return False
        return True
    
    words = ["racecar", "hello", "A man a plan a canal Panama", "python"]
    print("1. Palindrome Checker:")
    for word in words:
        result = is_palindrome(word)
        print(f"   '{word}' -> {'Palindrome' if result else 'Not Palindrome'}")
    
    # 2. Browser history
    print("\n2. Browser History:")
    print("   Deques can store both backward and forward history")
    print("   Visiting a new page adds to rear, going back removes from rear")
    print("   Going forward removes from front")

def priority_queue_applications():
    """
    Demonstrate common applications of priority queues
    """
    print("\n=== Priority Queue Applications ===")
    
    # 1. Task scheduling
    print("1. Task Scheduling:")
    print("   Operating systems use priority queues to schedule processes")
    print("   High-priority tasks are executed before low-priority ones")
    
    # 2. Dijkstra's algorithm
    print("\n2. Dijkstra's Shortest Path Algorithm:")
    print("   Priority queues efficiently select the next vertex to process")
    print("   Vertex with minimum distance is given highest priority")
    
    # 3. Event-driven simulation
    print("\n3. Event-Driven Simulation:")
    print("   Events are queued with timestamps as priorities")
    print("   Events are processed in chronological order")

def performance_comparison():
    """
    Compare performance of different implementations
    """
    print("\n=== Performance Comparison ===")
    
    import time
    
    # Test with 10000 operations
    n = 10000
    
    # List-based stack
    start = time.time()
    stack = Stack()
    for i in range(n):
        stack.push(i)
    for i in range(n):
        stack.pop()
    end = time.time()
    print(f"List-based Stack: {end - start:.6f} seconds")
    
    # Deque-based queue
    start = time.time()
    queue = Deque()
    for i in range(n):
        queue.add_rear(i)
    for i in range(n):
        queue.remove_front()
    end = time.time()
    print(f"Deque-based Queue: {end - start:.6f} seconds")
    
    # Priority queue
    start = time.time()
    pq = PriorityQueue()
    for i in range(n):
        pq.enqueue(i, n - i)  # Reverse priority
    for i in range(n):
        pq.dequeue()
    end = time.time()
    print(f"Priority Queue: {end - start:.6f} seconds")

# Example usage and testing
if __name__ == "__main__":
    # Stack examples
    print("=== Stack Examples ===")
    stack = Stack()
    for i in range(5):
        stack.push(i)
    print(f"Stack after pushing 0-4: {stack}")
    print(f"Top element: {stack.peek()}")
    print(f"Popped element: {stack.pop()}")
    print(f"Stack after popping: {stack}")
    
    print("\n" + "="*50 + "\n")
    
    # Queue examples
    print("=== Queue Examples ===")
    queue = Queue()
    for i in range(5):
        queue.enqueue(i)
    print(f"Queue after enqueuing 0-4: {queue}")
    print(f"Front element: {queue.front()}")
    print(f"Dequeued element: {queue.dequeue()}")
    print(f"Queue after dequeuing: {queue}")
    
    print("\n" + "="*50 + "\n")
    
    # Deque examples
    print("=== Deque Examples ===")
    deque_obj = Deque()
    deque_obj.add_front(1)
    deque_obj.add_rear(2)
    deque_obj.add_front(0)
    print(f"Deque after operations: {deque_obj}")
    print(f"Removed from front: {deque_obj.remove_front()}")
    print(f"Removed from rear: {deque_obj.remove_rear()}")
    print(f"Deque after removals: {deque_obj}")
    
    print("\n" + "="*50 + "\n")
    
    # Priority Queue examples
    print("=== Priority Queue Examples ===")
    pq = PriorityQueue()
    pq.enqueue("Low priority", 3)
    pq.enqueue("High priority", 1)
    pq.enqueue("Medium priority", 2)
    print(f"Priority Queue: {pq}")
    print(f"Dequeued (highest priority): {pq.dequeue()}")
    print(f"Dequeued (next priority): {pq.dequeue()}")
    print(f"Priority Queue after dequeuing: {pq}")
    
    print("\n" + "="*50 + "\n")
    
    # Applications
    stack_applications()
    print("\n" + "="*50 + "\n")
    
    queue_applications()
    print("\n" + "="*50 + "\n")
    
    deque_applications()
    print("\n" + "="*50 + "\n")
    
    priority_queue_applications()
    print("\n" + "="*50 + "\n")
    
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Stack, Queue, Deque, and Priority Queue implementations")
    print("2. Common operations and their time complexities")
    print("3. Practical applications in algorithms and systems")
    print("4. Performance comparisons between implementations")
    print("\nKey takeaways:")
    print("- Stacks (LIFO) are ideal for recursive algorithms and undo operations")
    print("- Queues (FIFO) are perfect for scheduling and breadth-first traversal")
    print("- Deques provide flexibility for operations at both ends")
    print("- Priority queues are essential for optimization algorithms")