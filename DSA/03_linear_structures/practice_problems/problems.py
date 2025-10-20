"""
Linear Data Structures - Practice Problems
=====================================

This file contains practice problems for linear data structures with solutions.
"""

# Problem 1: Array Operations
def problem_1():
    """
    Implement array operations:
    """
    
    print("Problem 1: Array Operations")
    print("=" * 30)
    
    # 1. Rotate array to the right by k steps
    def rotate_array(arr, k):
        """
        Rotate array to the right by k steps
        Example: [1,2,3,4,5,6,7], k=3 → [5,6,7,1,2,3,4]
        """
        if not arr or k == 0:
            return arr
        
        n = len(arr)
        k = k % n  # Handle cases where k > n
        return arr[-k:] + arr[:-k]
    
    # 2. Find missing number in array
    def find_missing_number(arr):
        """
        Find the missing number in array of n distinct numbers in range [0, n]
        Example: [3,0,1] → 2
        """
        n = len(arr)
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(arr)
        return expected_sum - actual_sum
    
    # 3. Merge sorted arrays
    def merge_sorted_arrays(arr1, arr2):
        """
        Merge two sorted arrays into one sorted array
        """
        result = []
        i = j = 0
        
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        
        # Add remaining elements
        result.extend(arr1[i:])
        result.extend(arr2[j:])
        return result
    
    # Test cases
    print("1. Array Rotation:")
    arr1 = [1, 2, 3, 4, 5, 6, 7]
    k = 3
    rotated = rotate_array(arr1, k)
    print(f"   Original: {arr1}")
    print(f"   Rotated by {k}: {rotated}")
    
    print("\n2. Missing Number:")
    arr2 = [3, 0, 1]
    missing = find_missing_number(arr2)
    print(f"   Array: {arr2}")
    print(f"   Missing number: {missing}")
    
    print("\n3. Merge Sorted Arrays:")
    arr3 = [1, 3, 5, 7]
    arr4 = [2, 4, 6, 8, 9, 10]
    merged = merge_sorted_arrays(arr3, arr4)
    print(f"   Array 1: {arr3}")
    print(f"   Array 2: {arr4}")
    print(f"   Merged: {merged}")

# Problem 2: Stack Implementation
def problem_2():
    """
    Stack problems and implementations:
    """
    
    print("\nProblem 2: Stack Implementation")
    print("=" * 30)
    
    # 1. Valid parentheses
    def is_valid_parentheses(s):
        """
        Check if parentheses in string are valid
        """
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping.values():
                stack.append(char)
            elif char in mapping.keys():
                if not stack or stack.pop() != mapping[char]:
                    return False
            # Ignore other characters
        
        return len(stack) == 0
    
    # 2. Evaluate reverse polish notation
    def eval_rpn(tokens):
        """
        Evaluate Reverse Polish Notation expression
        Example: ["2", "1", "+", "3", "*"] → 9
        """
        stack = []
        operators = {'+', '-', '*', '/'}
        
        for token in tokens:
            if token in operators:
                b = stack.pop()
                a = stack.pop()
                if token == '+':
                    stack.append(a + b)
                elif token == '-':
                    stack.append(a - b)
                elif token == '*':
                    stack.append(a * b)
                elif token == '/':
                    stack.append(int(a / b))  # Truncate towards zero
            else:
                stack.append(int(token))
        
        return stack[0]
    
    # 3. Implement Min Stack
    class MinStack:
        def __init__(self):
            self.stack = []
            self.min_stack = []  # Track minimums
        
        def push(self, val):
            self.stack.append(val)
            if not self.min_stack or val <= self.min_stack[-1]:
                self.min_stack.append(val)
        
        def pop(self):
            if self.stack:
                val = self.stack.pop()
                if self.min_stack and val == self.min_stack[-1]:
                    self.min_stack.pop()
        
        def top(self):
            return self.stack[-1] if self.stack else None
        
        def get_min(self):
            return self.min_stack[-1] if self.min_stack else None
    
    # Test cases
    print("1. Valid Parentheses:")
    test_strings = ["()", "()[]{}", "(]", "([)]", "{[]}"]
    for s in test_strings:
        result = is_valid_parentheses(s)
        print(f"   '{s}' → {'Valid' if result else 'Invalid'}")
    
    print("\n2. Evaluate RPN:")
    tokens = ["2", "1", "+", "3", "*"]
    result = eval_rpn(tokens)
    print(f"   {tokens} → {result}")
    
    tokens2 = ["4", "13", "5", "/", "+"]
    result2 = eval_rpn(tokens2)
    print(f"   {tokens2} → {result2}")
    
    print("\n3. Min Stack:")
    min_stack = MinStack()
    operations = [(2, "push"), (0, "push"), (3, "push"), (0, "push"), 
                  (None, "pop"), (None, "get_min"), (None, "pop"), (None, "get_min")]
    
    for val, op in operations:
        if op == "push":
            min_stack.push(val)
            print(f"   Pushed {val}")
        elif op == "pop":
            min_stack.pop()
            print(f"   Popped")
        elif op == "get_min":
            min_val = min_stack.get_min()
            print(f"   Current minimum: {min_val}")

# Problem 3: Queue Operations
def problem_3():
    """
    Queue problems and implementations:
    """
    
    print("\nProblem 3: Queue Operations")
    print("=" * 30)
    
    # 1. Implement circular queue
    class CircularQueue:
        def __init__(self, k):
            self.queue = [None] * k
            self.max_size = k
            self.head = self.tail = -1
        
        def enqueue(self, value):
            if (self.tail + 1) % self.max_size == self.head:
                return False  # Queue is full
            
            if self.head == -1:  # First element
                self.head = self.tail = 0
            else:
                self.tail = (self.tail + 1) % self.max_size
            
            self.queue[self.tail] = value
            return True
        
        def dequeue(self):
            if self.head == -1:
                return -1  # Queue is empty
            
            value = self.queue[self.head]
            if self.head == self.tail:  # Last element
                self.head = self.tail = -1
            else:
                self.head = (self.head + 1) % self.max_size
            
            return value
        
        def is_empty(self):
            return self.head == -1
        
        def is_full(self):
            return (self.tail + 1) % self.max_size == self.head
    
    # 2. Implement stack using queues
    class StackUsingQueues:
        def __init__(self):
            self.queue1 = []
            self.queue2 = []
        
        def push(self, x):
            # Add to empty queue
            self.queue2.append(x)
            # Move all elements from queue1 to queue2
            while self.queue1:
                self.queue2.append(self.queue1.pop(0))
            # Swap queues
            self.queue1, self.queue2 = self.queue2, self.queue1
        
        def pop(self):
            if not self.queue1:
                return None
            return self.queue1.pop(0)
        
        def top(self):
            if not self.queue1:
                return None
            return self.queue1[0]
        
        def empty(self):
            return len(self.queue1) == 0
    
    # Test cases
    print("1. Circular Queue:")
    cq = CircularQueue(3)
    print(f"   Enqueue 1: {cq.enqueue(1)}")
    print(f"   Enqueue 2: {cq.enqueue(2)}")
    print(f"   Enqueue 3: {cq.enqueue(3)}")
    print(f"   Enqueue 4: {cq.enqueue(4)} (should be False - full)")
    print(f"   Dequeue: {cq.dequeue()}")
    print(f"   Enqueue 4: {cq.enqueue(4)}")
    print(f"   Dequeue: {cq.dequeue()}")
    print(f"   Dequeue: {cq.dequeue()}")
    print(f"   Dequeue: {cq.dequeue()}")
    print(f"   Dequeue: {cq.dequeue()} (should be -1 - empty)")
    
    print("\n2. Stack Using Queues:")
    stack = StackUsingQueues()
    stack.push(1)
    stack.push(2)
    print(f"   Push 1, 2")
    print(f"   Top: {stack.top()}")
    print(f"   Pop: {stack.pop()}")
    print(f"   Empty: {stack.empty()}")
    print(f"   Pop: {stack.pop()}")
    print(f"   Empty: {stack.empty()}")

# Problem 4: Linked List Manipulation
def problem_4():
    """
    Linked list problems:
    """
    
    print("\nProblem 4: Linked List Manipulation")
    print("=" * 30)
    
    # Node class for linked list
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    
    # 1. Reverse linked list
    def reverse_list(head):
        prev = None
        current = head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev
    
    # 2. Detect cycle in linked list
    def has_cycle(head):
        if not head or not head.next:
            return False
        
        slow = head
        fast = head.next
        
        while fast and fast.next:
            if slow == fast:
                return True
            slow = slow.next
            fast = fast.next.next
        
        return False
    
    # 3. Merge two sorted linked lists
    def merge_two_lists(l1, l2):
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        # Attach remaining nodes
        current.next = l1 or l2
        return dummy.next
    
    # Helper function to create linked list from array
    def create_linked_list(arr):
        if not arr:
            return None
        head = ListNode(arr[0])
        current = head
        for i in range(1, len(arr)):
            current.next = ListNode(arr[i])
            current = current.next
        return head
    
    # Helper function to convert linked list to array
    def linked_list_to_array(head):
        result = []
        current = head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    # Test cases
    print("1. Reverse Linked List:")
    arr = [1, 2, 3, 4, 5]
    head = create_linked_list(arr)
    print(f"   Original: {arr}")
    reversed_head = reverse_list(head)
    reversed_arr = linked_list_to_array(reversed_head)
    print(f"   Reversed: {reversed_arr}")
    
    print("\n2. Merge Two Sorted Lists:")
    arr1 = [1, 2, 4]
    arr2 = [1, 3, 4]
    l1 = create_linked_list(arr1)
    l2 = create_linked_list(arr2)
    print(f"   List 1: {arr1}")
    print(f"   List 2: {arr2}")
    merged_head = merge_two_lists(l1, l2)
    merged_arr = linked_list_to_array(merged_head)
    print(f"   Merged: {merged_arr}")

# Problem 5: Deque Applications
def problem_5():
    """
    Deque problems and applications:
    """
    
    print("\nProblem 5: Deque Applications")
    print("=" * 30)
    
    from collections import deque
    
    # 1. Sliding window maximum
    def max_sliding_window(nums, k):
        """
        Find maximum in all sliding windows of size k
        """
        if not nums or k == 0:
            return []
        
        dq = deque()  # Store indices
        result = []
        
        for i in range(len(nums)):
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices of smaller elements
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            dq.append(i)
            
            # Add maximum for current window
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    # 2. Palindrome checker using deque
    def is_palindrome(s):
        """
        Check if string is palindrome using deque
        """
        # Clean string
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        d = deque(cleaned)
        
        while len(d) > 1:
            if d.popleft() != d.pop():
                return False
        return True
    
    # Test cases
    print("1. Sliding Window Maximum:")
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    max_values = max_sliding_window(nums, k)
    print(f"   Array: {nums}")
    print(f"   Window size: {k}")
    print(f"   Maximums: {max_values}")
    
    print("\n2. Palindrome Checker:")
    test_strings = ["A man, a plan, a canal: Panama", "race a car", "racecar"]
    for s in test_strings:
        result = is_palindrome(s)
        print(f"   '{s}' → {'Palindrome' if result else 'Not Palindrome'}")

# Run all problems
if __name__ == "__main__":
    print("=== Linear Data Structures Practice Problems ===\n")
    
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
    print("1. Array operations and manipulations")
    print("2. Stack implementations and applications")
    print("3. Queue operations and variations")
    print("4. Linked list algorithms")
    print("5. Deque applications")
    print("\nEach problem demonstrates:")
    print("- Implementation of data structures")
    print("- Common algorithms and patterns")
    print("- Real-world applications")
    print("- Time and space complexity considerations")