"""
Linear Data Structures - Practice Problem Solutions
==============================================

This file contains detailed solutions and explanations for the practice problems.
"""

# Solution 1: Array Operations
def solution_1():
    """
    Detailed solutions for array operations:
    """
    
    print("Solution 1: Array Operations")
    print("=" * 30)
    
    # 1. Rotate array to the right by k steps
    print("1. Array Rotation:")
    print("Approach: Use array slicing to create rotated array")
    print("Time Complexity: O(n), Space Complexity: O(n)")
    print("Alternative approaches:")
    print("  - In-place rotation using reversal algorithm: O(n) time, O(1) space")
    print("  - Using cyclic replacements: O(n) time, O(1) space")
    print()
    
    # 2. Find missing number in array
    print("2. Missing Number:")
    print("Approach: Use mathematical sum formula")
    print("Time Complexity: O(n), Space Complexity: O(1)")
    print("Alternative approaches:")
    print("  - Using XOR: O(n) time, O(1) space")
    print("  - Using sorting: O(n log n) time, O(1) space")
    print("  - Using hash set: O(n) time, O(n) space")
    print()
    
    # 3. Merge sorted arrays
    print("3. Merge Sorted Arrays:")
    print("Approach: Two-pointer technique")
    print("Time Complexity: O(m + n), Space Complexity: O(m + n)")
    print("For in-place merging (when arr1 has extra space): O(m + n) time, O(1) space")
    print()

# Solution 2: Stack Implementation
def solution_2():
    """
    Detailed solutions for stack problems:
    """
    
    print("Solution 2: Stack Implementation")
    print("=" * 30)
    
    # 1. Valid parentheses
    print("1. Valid Parentheses:")
    print("Approach: Use stack to match opening and closing brackets")
    print("Time Complexity: O(n), Space Complexity: O(n)")
    print("Key insight: For every closing bracket, there must be a matching opening bracket")
    print()
    
    # 2. Evaluate reverse polish notation
    print("2. Evaluate RPN:")
    print("Approach: Use stack to store operands")
    print("Time Complexity: O(n), Space Complexity: O(n)")
    print("Key insight: When encountering an operator, pop two operands and push result")
    print()
    
    # 3. Min Stack
    print("3. Min Stack:")
    print("Approach: Use auxiliary stack to track minimums")
    print("Time Complexity: O(1) for all operations, Space Complexity: O(n)")
    print("Alternative approaches:")
    print("  - Store (value, min) tuples in single stack")
    print("  - Store difference between value and min (space optimization)")
    print()

# Solution 3: Queue Operations
def solution_3():
    """
    Detailed solutions for queue problems:
    """
    
    print("Solution 3: Queue Operations")
    print("=" * 30)
    
    # 1. Circular Queue
    print("1. Circular Queue:")
    print("Approach: Use array with head/tail pointers and modulo arithmetic")
    print("Key conditions:")
    print("  - Empty: head == -1")
    print("  - Full: (tail + 1) % size == head")
    print("Time Complexity: O(1) for all operations, Space Complexity: O(k)")
    print()
    
    # 2. Stack Using Queues
    print("2. Stack Using Queues:")
    print("Approach: Make push operation expensive (O(n)) or pop operation expensive (O(n))")
    print("Our implementation makes push expensive:")
    print("  - Push: O(n) time, O(n) space")
    print("  - Pop: O(1) time, O(1) space")
    print("Alternative approach makes pop expensive:")
    print("  - Push: O(1) time, O(1) space")
    print("  - Pop: O(n) time, O(n) space")
    print()

# Solution 4: Linked List Manipulation
def solution_4():
    """
    Detailed solutions for linked list problems:
    """
    
    print("Solution 4: Linked List Manipulation")
    print("=" * 30)
    
    # 1. Reverse linked list
    print("1. Reverse Linked List:")
    print("Approach: Iterative with three pointers (prev, current, next)")
    print("Time Complexity: O(n), Space Complexity: O(1)")
    print("Recursive approach: O(n) time, O(n) space (due to call stack)")
    print()
    
    # 2. Detect cycle in linked list
    print("2. Detect Cycle:")
    print("Approach: Floyd's Cycle Detection Algorithm (Tortoise and Hare)")
    print("Time Complexity: O(n), Space Complexity: O(1)")
    print("Alternative approaches:")
    print("  - Using hash set: O(n) time, O(n) space")
    print("  - Modifying node values: O(n) time, O(1) space (destructive)")
    print()
    
    # 3. Merge two sorted linked lists
    print("3. Merge Two Sorted Lists:")
    print("Approach: Two-pointer technique with dummy node")
    print("Time Complexity: O(m + n), Space Complexity: O(1)")
    print("Recursive approach: O(m + n) time, O(m + n) space (due to call stack)")
    print()

# Solution 5: Deque Applications
def solution_5():
    """
    Detailed solutions for deque problems:
    """
    
    print("Solution 5: Deque Applications")
    print("=" * 30)
    
    # 1. Sliding window maximum
    print("1. Sliding Window Maximum:")
    print("Approach: Use deque to maintain indices of potential maximums")
    print("Key insights:")
    print("  - Deque stores indices in decreasing order of values")
    print("  - Remove indices outside current window")
    print("  - Remove indices of smaller elements")
    print("Time Complexity: O(n), Space Complexity: O(k)")
    print()
    
    # 2. Palindrome checker using deque
    print("2. Palindrome Checker:")
    print("Approach: Use deque to compare characters from both ends")
    print("Time Complexity: O(n), Space Complexity: O(n)")
    print("Alternative approaches:")
    print("  - Two-pointer on string: O(n) time, O(1) space")
    print("  - Reverse and compare: O(n) time, O(n) space")
    print()

# Additional Solutions and Explanations
def additional_solutions():
    """
    Additional solutions and advanced techniques:
    """
    
    print("Additional Solutions and Techniques")
    print("=" * 35)
    
    print("1. Array Rotation - In-place Reversal Algorithm:")
    print("   Steps:")
    print("   1. Reverse entire array")
    print("   2. Reverse first k elements")
    print("   3. Reverse remaining n-k elements")
    print("   Time: O(n), Space: O(1)")
    print()
    
    print("2. Missing Number - XOR Approach:")
    print("   Principle: XOR of number with itself is 0, XOR with 0 is number")
    print("   Steps:")
    print("   1. XOR all numbers from 0 to n")
    print("   2. XOR all elements in array")
    print("   3. XOR results to get missing number")
    print("   Time: O(n), Space: O(1)")
    print()
    
    print("3. Linked List Cycle - Find Cycle Start:")
    print("   After detecting cycle with Floyd's algorithm:")
    print("   1. Move one pointer to head")
    print("   2. Move both pointers one step at a time")
    print("   3. Meeting point is cycle start")
    print("   Mathematical proof: Distance from head to cycle start")
    print("   equals distance from meeting point to cycle start")
    print()
    
    print("4. Stack with O(1) GetMin - Space Optimized:")
    print("   Store difference between value and min instead of actual values")
    print("   When value >= 0: actual value = min + difference")
    print("   When value < 0: actual value = min, new min = min - difference")
    print("   Space: O(1) additional space in best case")
    print()

# Run all solutions
if __name__ == "__main__":
    print("=== Linear Data Structures Practice Problem Solutions ===\n")
    
    solution_1()
    print("\n" + "="*50 + "\n")
    
    solution_2()
    print("\n" + "="*50 + "\n")
    
    solution_3()
    print("\n" + "="*50 + "\n")
    
    solution_4()
    print("\n" + "="*50 + "\n")
    
    solution_5()
    print("\n" + "="*50 + "\n")
    
    additional_solutions()
    print("\n" + "="*50 + "\n")
    
    print("=== Summary ===")
    print("These solutions demonstrate:")
    print("1. Multiple approaches to the same problem")
    print("2. Time and space complexity analysis")
    print("3. Trade-offs between different implementations")
    print("4. Advanced techniques and optimizations")
    print("5. Mathematical principles behind algorithms")
    print("\nKey takeaways:")
    print("- Understand the problem requirements before choosing an approach")
    print("- Consider both time and space complexity")
    print("- Think about edge cases and error handling")
    print("- Know multiple solutions for common problems")
    print("- Practice implementing algorithms from scratch")