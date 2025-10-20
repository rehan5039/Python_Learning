"""
Linear Data Structures - Linked Lists
================================

This module provides implementations and examples of linked list data structures,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Singly linked lists
- Doubly linked lists
- Circular linked lists
- Operations and their complexities
- Applications in algorithms and data science
"""

from typing import Any, Optional

class ListNode:
    """
    Node class for linked list implementations
    """
    
    def __init__(self, data: Any):
        """Initialize a list node"""
        self.data = data
        self.next: Optional['ListNode'] = None

class DoublyListNode:
    """
    Node class for doubly linked list implementations
    """
    
    def __init__(self, data: Any):
        """Initialize a doubly linked list node"""
        self.data = data
        self.next: Optional['DoublyListNode'] = None
        self.prev: Optional['DoublyListNode'] = None

class SinglyLinkedList:
    """
    Implementation of a singly linked list
    """
    
    def __init__(self):
        """Initialize an empty singly linked list"""
        self.head: Optional[ListNode] = None
        self.tail: Optional[ListNode] = None
        self._size = 0
    
    def append(self, data: Any) -> None:
        """
        Add an item to the end of the list - O(1)
        """
        new_node = ListNode(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1
    
    def prepend(self, data: Any) -> None:
        """
        Add an item to the beginning of the list - O(1)
        """
        new_node = ListNode(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self._size += 1
    
    def insert(self, index: int, data: Any) -> None:
        """
        Insert an item at a specific index - O(n)
        """
        if index < 0 or index > self._size:
            raise IndexError("Index out of range")
        
        if index == 0:
            self.prepend(data)
            return
        
        if index == self._size:
            self.append(data)
            return
        
        new_node = ListNode(data)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self._size += 1
    
    def delete(self, data: Any) -> bool:
        """
        Delete the first occurrence of an item - O(n)
        """
        if not self.head:
            return False
        
        # Special case: delete head
        if self.head.data == data:
            self.head = self.head.next
            if not self.head:
                self.tail = None
            self._size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.data == data:
                # Special case: delete tail
                if current.next == self.tail:
                    self.tail = current
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next
        
        return False
    
    def search(self, data: Any) -> int:
        """
        Search for an item and return its index - O(n)
        """
        current = self.head
        index = 0
        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1
        return -1
    
    def get(self, index: int) -> Any:
        """
        Get item at a specific index - O(n)
        """
        if index < 0 or index >= self._size:
            raise IndexError("Index out of range")
        
        current = self.head
        for _ in range(index):
            current = current.next
        return current.data
    
    def reverse(self) -> None:
        """
        Reverse the linked list - O(n)
        """
        prev = None
        current = self.head
        self.tail = self.head  # Old head becomes new tail
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev
    
    def size(self) -> int:
        """
        Return the number of items in the list - O(1)
        """
        return self._size
    
    def is_empty(self) -> bool:
        """
        Check if the list is empty - O(1)
        """
        return self._size == 0
    
    def to_list(self) -> list:
        """
        Convert linked list to Python list - O(n)
        """
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result
    
    def __str__(self) -> str:
        """String representation of the linked list"""
        if not self.head:
            return "SinglyLinkedList([])"
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return f"SinglyLinkedList([{', '.join(elements)}])"

class DoublyLinkedList:
    """
    Implementation of a doubly linked list
    """
    
    def __init__(self):
        """Initialize an empty doubly linked list"""
        self.head: Optional[DoublyListNode] = None
        self.tail: Optional[DoublyListNode] = None
        self._size = 0
    
    def append(self, data: Any) -> None:
        """
        Add an item to the end of the list - O(1)
        """
        new_node = DoublyListNode(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1
    
    def prepend(self, data: Any) -> None:
        """
        Add an item to the beginning of the list - O(1)
        """
        new_node = DoublyListNode(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self._size += 1
    
    def insert(self, index: int, data: Any) -> None:
        """
        Insert an item at a specific index - O(n)
        """
        if index < 0 or index > self._size:
            raise IndexError("Index out of range")
        
        if index == 0:
            self.prepend(data)
            return
        
        if index == self._size:
            self.append(data)
            return
        
        new_node = DoublyListNode(data)
        current = self.head
        for _ in range(index):
            current = current.next
        
        new_node.next = current
        new_node.prev = current.prev
        current.prev.next = new_node
        current.prev = new_node
        self._size += 1
    
    def delete(self, data: Any) -> bool:
        """
        Delete the first occurrence of an item - O(n)
        """
        current = self.head
        while current:
            if current.data == data:
                # Update previous node's next pointer
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                
                # Update next node's previous pointer
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                
                self._size -= 1
                return True
            current = current.next
        return False
    
    def search(self, data: Any) -> int:
        """
        Search for an item and return its index - O(n)
        """
        current = self.head
        index = 0
        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1
        return -1
    
    def get(self, index: int) -> Any:
        """
        Get item at a specific index - O(n)
        Optimization: Start from head or tail based on index
        """
        if index < 0 or index >= self._size:
            raise IndexError("Index out of range")
        
        # Optimization: choose direction based on index
        if index < self._size // 2:
            # Start from head
            current = self.head
            for _ in range(index):
                current = current.next
        else:
            # Start from tail
            current = self.tail
            for _ in range(self._size - 1 - index):
                current = current.prev
        return current.data
    
    def reverse(self) -> None:
        """
        Reverse the doubly linked list - O(n)
        """
        current = self.head
        # Swap head and tail
        self.head, self.tail = self.tail, self.head
        
        # Reverse all pointers
        while current:
            # Swap next and prev pointers
            current.next, current.prev = current.prev, current.next
            # Move to the next node (which is now prev due to swap)
            current = current.prev
    
    def size(self) -> int:
        """
        Return the number of items in the list - O(1)
        """
        return self._size
    
    def is_empty(self) -> bool:
        """
        Check if the list is empty - O(1)
        """
        return self._size == 0
    
    def to_list(self) -> list:
        """
        Convert doubly linked list to Python list - O(n)
        """
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result
    
    def __str__(self) -> str:
        """String representation of the doubly linked list"""
        if not self.head:
            return "DoublyLinkedList([])"
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return f"DoublyLinkedList([{', '.join(elements)}])"

class CircularLinkedList:
    """
    Implementation of a circular linked list
    """
    
    def __init__(self):
        """Initialize an empty circular linked list"""
        self.head: Optional[ListNode] = None
        self._size = 0
    
    def append(self, data: Any) -> None:
        """
        Add an item to the end of the list - O(1)
        """
        new_node = ListNode(data)
        if not self.head:
            self.head = new_node
            new_node.next = new_node  # Point to itself
        else:
            # Find the last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            # Insert new node at the end
            current.next = new_node
            new_node.next = self.head
        self._size += 1
    
    def prepend(self, data: Any) -> None:
        """
        Add an item to the beginning of the list - O(n)
        """
        new_node = ListNode(data)
        if not self.head:
            self.head = new_node
            new_node.next = new_node
        else:
            # Find the last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            # Insert new node at the beginning
            new_node.next = self.head
            current.next = new_node
            self.head = new_node
        self._size += 1
    
    def delete(self, data: Any) -> bool:
        """
        Delete the first occurrence of an item - O(n)
        """
        if not self.head:
            return False
        
        # Special case: only one node
        if self.head.next == self.head:
            if self.head.data == data:
                self.head = None
                self._size -= 1
                return True
            return False
        
        # Find the node to delete and its previous node
        current = self.head
        prev = None
        while True:
            if current.data == data:
                if current == self.head:
                    # Find the last node to update its next pointer
                    last = self.head
                    while last.next != self.head:
                        last = last.next
                    self.head = current.next
                    last.next = self.head
                else:
                    prev.next = current.next
                self._size -= 1
                return True
            
            prev = current
            current = current.next
            
            # Break if we've traversed the entire list
            if current == self.head:
                break
        
        return False
    
    def search(self, data: Any) -> int:
        """
        Search for an item and return its index - O(n)
        """
        if not self.head:
            return -1
        
        current = self.head
        index = 0
        while True:
            if current.data == data:
                return index
            current = current.next
            index += 1
            
            # Break if we've traversed the entire list
            if current == self.head:
                break
        
        return -1
    
    def size(self) -> int:
        """
        Return the number of items in the list - O(1)
        """
        return self._size
    
    def is_empty(self) -> bool:
        """
        Check if the list is empty - O(1)
        """
        return self._size == 0
    
    def to_list(self) -> list:
        """
        Convert circular linked list to Python list - O(n)
        """
        if not self.head:
            return []
        
        result = []
        current = self.head
        while True:
            result.append(current.data)
            current = current.next
            if current == self.head:
                break
        return result
    
    def __str__(self) -> str:
        """String representation of the circular linked list"""
        if not self.head:
            return "CircularLinkedList([])"
        elements = self.to_list()
        return f"CircularLinkedList([{', '.join(map(str, elements))}])"

def linked_list_applications():
    """
    Demonstrate common applications of linked lists
    """
    print("=== Linked List Applications ===")
    
    # 1. Polynomial representation
    print("1. Polynomial Representation:")
    print("   Linked lists can represent polynomials efficiently")
    print("   Each node stores coefficient and exponent")
    print("   Example: 3x² + 2x + 1 can be stored as nodes (3,2) -> (2,1) -> (1,0)")
    
    # 2. Memory management
    print("\n2. Memory Management:")
    print("   Operating systems use linked lists for memory allocation")
    print("   Free memory blocks are linked together")
    print("   Allocation involves removing a block from the list")
    print("   Deallocation involves adding a block back to the list")
    
    # 3. Hash table collision resolution
    print("\n3. Hash Table Collision Resolution:")
    print("   Chaining method uses linked lists to handle collisions")
    print("   Each hash bucket contains a linked list of entries")
    print("   Multiple keys hashing to the same index are stored in the list")
    
    # 4. Browser history
    print("\n4. Browser History:")
    print("   Doubly linked lists can efficiently implement browser history")
    print("   Forward/backward navigation is O(1) with proper pointers")
    print("   New pages are added to the end, history can be traversed both ways")

def performance_comparison():
    """
    Compare performance of different linked list implementations
    """
    print("\n=== Performance Comparison ===")
    
    import time
    
    # Test with 10000 operations
    n = 10000
    
    # Singly linked list
    start = time.time()
    sll = SinglyLinkedList()
    for i in range(n):
        sll.append(i)
    for i in range(0, n, 100):  # Delete every 100th element
        sll.delete(i)
    end = time.time()
    print(f"Singly Linked List: {end - start:.6f} seconds")
    
    # Doubly linked list
    start = time.time()
    dll = DoublyLinkedList()
    for i in range(n):
        dll.append(i)
    for i in range(0, n, 100):  # Delete every 100th element
        dll.delete(i)
    end = time.time()
    print(f"Doubly Linked List: {end - start:.6f} seconds")
    
    # Circular linked list
    start = time.time()
    cll = CircularLinkedList()
    for i in range(n // 10):  # Smaller test for circular (more complex delete)
        cll.append(i)
    for i in range(0, n // 10, 10):  # Delete every 10th element
        cll.delete(i)
    end = time.time()
    print(f"Circular Linked List: {end - start:.6f} seconds")

def data_science_applications():
    """
    Examples of linked lists in data science applications
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Streaming data processing
    print("1. Streaming Data Processing:")
    print("   Linked lists can efficiently handle continuous data streams")
    print("   New data points are added to the end in O(1) time")
    print("   Old data can be removed from the front to maintain window size")
    
    # 2. Graph representation
    print("\n2. Adjacency List Representation of Graphs:")
    print("   Graphs are often represented using linked lists")
    print("   Each vertex has a linked list of adjacent vertices")
    print("   More memory efficient than adjacency matrices for sparse graphs")
    
    # 3. Skip lists for indexing
    print("\n3. Skip Lists for Indexing:")
    print("   Probabilistic data structure built on linked lists")
    print("   Provides O(log n) search time with O(n) space")
    print("   Used in databases and concurrent data structures")

# Example usage and testing
if __name__ == "__main__":
    # Singly Linked List examples
    print("=== Singly Linked List Examples ===")
    sll = SinglyLinkedList()
    for i in range(5):
        sll.append(i)
    print(f"List after appending 0-4: {sll}")
    sll.prepend(-1)
    print(f"List after prepending -1: {sll}")
    sll.insert(3, 99)
    print(f"List after inserting 99 at index 3: {sll}")
    print(f"Search for 99: Index {sll.search(99)}")
    print(f"Element at index 2: {sll.get(2)}")
    sll.reverse()
    print(f"List after reversing: {sll}")
    sll.delete(99)
    print(f"List after deleting 99: {sll}")
    
    print("\n" + "="*50 + "\n")
    
    # Doubly Linked List examples
    print("=== Doubly Linked List Examples ===")
    dll = DoublyLinkedList()
    for i in range(5):
        dll.append(i)
    print(f"List after appending 0-4: {dll}")
    dll.prepend(-1)
    print(f"List after prepending -1: {dll}")
    dll.insert(3, 99)
    print(f"List after inserting 99 at index 3: {dll}")
    print(f"Search for 99: Index {dll.search(99)}")
    print(f"Element at index 2: {dll.get(2)}")
    dll.reverse()
    print(f"List after reversing: {dll}")
    dll.delete(99)
    print(f"List after deleting 99: {dll}")
    
    print("\n" + "="*50 + "\n")
    
    # Circular Linked List examples
    print("=== Circular Linked List Examples ===")
    cll = CircularLinkedList()
    for i in range(5):
        cll.append(i)
    print(f"List after appending 0-4: {cll}")
    cll.prepend(-1)
    print(f"List after prepending -1: {cll}")
    print(f"Search for 2: Index {cll.search(2)}")
    cll.delete(2)
    print(f"List after deleting 2: {cll}")
    
    print("\n" + "="*50 + "\n")
    
    # Applications
    linked_list_applications()
    print("\n" + "="*50 + "\n")
    
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Singly, doubly, and circular linked list implementations")
    print("2. Common operations and their time complexities")
    print("3. Practical applications in systems and algorithms")
    print("4. Data science applications of linked lists")
    print("5. Performance comparisons between implementations")
    print("\nKey takeaways:")
    print("- Linked lists provide dynamic memory allocation")
    print("- Singly linked lists use less memory but only allow forward traversal")
    print("- Doubly linked lists allow bidirectional traversal but use more memory")
    print("- Circular linked lists are useful for round-robin scheduling")
    print("- Linked lists are fundamental in many advanced data structures")