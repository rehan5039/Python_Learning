"""
Non-Linear Data Structures - Practice Problem Solutions
================================================

This file contains detailed solutions and explanations for the practice problems.
"""

# Solution 1: Tree Operations
def solution_1():
    """
    Detailed solutions for tree operations:
    """
    
    print("Solution 1: Tree Operations")
    print("=" * 25)
    
    # 1. Maximum depth of binary tree
    print("1. Maximum Depth:")
    print("Approach: Recursive DFS")
    print("Time Complexity: O(n), Space Complexity: O(h) where h is height")
    print("Alternative approaches:")
    print("  - Iterative DFS using stack: O(n) time, O(h) space")
    print("  - BFS using queue: O(n) time, O(w) space where w is max width")
    print()
    
    # 2. Invert binary tree
    print("2. Invert Binary Tree:")
    print("Approach: Recursive traversal with node swapping")
    print("Time Complexity: O(n), Space Complexity: O(h)")
    print("Key insight: Swap left and right subtrees recursively")
    print("Alternative approaches:")
    print("  - Iterative using stack or queue: O(n) time, O(w) space")
    print()
    
    # 3. Same tree check
    print("3. Same Tree Check:")
    print("Approach: Simultaneous traversal with value comparison")
    print("Time Complexity: O(min(m,n)), Space Complexity: O(min(h1,h2))")
    print("Key insight: Trees are same if roots match and subtrees are same")
    print()

# Solution 2: Binary Search Trees
def solution_2():
    """
    Detailed solutions for BST problems:
    """
    
    print("Solution 2: Binary Search Trees")
    print("=" * 25)
    
    # 1. Validate BST
    print("1. Validate BST:")
    print("Approach: Recursive validation with bounds")
    print("Time Complexity: O(n), Space Complexity: O(h)")
    print("Key insight: Each node must be within valid range based on ancestors")
    print("Alternative approaches:")
    print("  - Inorder traversal and check if sorted: O(n) time, O(n) space")
    print()
    
    # 2. Lowest Common Ancestor in BST
    print("2. Lowest Common Ancestor:")
    print("Approach: Exploit BST property for efficient search")
    print("Time Complexity: O(h), Space Complexity: O(1)")
    print("Key insight: LCA is first node where paths diverge")
    print("For general binary tree: O(n) time using recursive approach")
    print()

# Solution 3: Heap Operations
def solution_3():
    """
    Detailed solutions for heap problems:
    """
    
    print("Solution 3: Heap Operations")
    print("=" * 25)
    
    # 1. Kth largest element
    print("1. Kth Largest Element:")
    print("Approach: Min heap of size k")
    print("Time Complexity: O(n log k), Space Complexity: O(k)")
    print("Key insight: Maintain k largest elements seen so far")
    print("Alternative approaches:")
    print("  - Sort array: O(n log n) time, O(1) space")
    print("  - Quickselect: O(n) average time, O(1) space")
    print()
    
    # 2. Merge k sorted lists
    print("2. Merge K Sorted Lists:")
    print("Approach: Min heap with pointers to current elements")
    print("Time Complexity: O(N log k) where N is total elements")
    print("Space Complexity: O(k)")
    print("Key insight: Always extract minimum among current elements")
    print("Alternative approaches:")
    print("  - Sequential merging: O(kN) time")
    print("  - Divide and conquer: O(N log k) time, O(1) space")
    print()

# Solution 4: Graph Algorithms
def solution_4():
    """
    Detailed solutions for graph problems:
    """
    
    print("Solution 4: Graph Algorithms")
    print("=" * 25)
    
    # 1. Number of islands
    print("1. Number of Islands:")
    print("Approach: BFS/DFS to mark connected components")
    print("Time Complexity: O(m * n), Space Complexity: O(m * n)")
    print("Key insight: Each unvisited '1' starts a new island")
    print("Alternative approaches:")
    print("  - Union-Find: O(m * n) time, O(m * n) space")
    print()
    
    # 2. Course schedule
    print("2. Course Schedule:")
    print("Approach: Topological sort with cycle detection")
    print("Time Complexity: O(V + E), Space Complexity: O(V + E)")
    print("Key insight: Courses can be finished iff no cyclic dependencies")
    print("Alternative approaches:")
    print("  - Kahn's algorithm (BFS-based topological sort)")
    print()

# Solution 5: Hash Table Implementations
def solution_5():
    """
    Detailed solutions for hash table problems:
    """
    
    print("Solution 5: Hash Table Implementations")
    print("=" * 35)
    
    # 1. Two sum problem
    print("1. Two Sum:")
    print("Approach: Hash map to store seen values and indices")
    print("Time Complexity: O(n), Space Complexity: O(n)")
    print("Key insight: For each number, check if complement exists")
    print("Alternative approaches:")
    print("  - Brute force: O(n²) time, O(1) space")
    print("  - Sort + two pointers: O(n log n) time, O(1) space")
    print()
    
    # 2. Group anagrams
    print("2. Group Anagrams:")
    print("Approach: Hash map with sorted string as key")
    print("Time Complexity: O(n * k log k) where k is max string length")
    print("Space Complexity: O(n * k)")
    print("Key insight: Anagrams have same sorted characters")
    print("Alternative approaches:")
    print("  - Use character count as key: O(n * k) time, O(n * k) space")
    print()

# Additional Solutions and Explanations
def additional_solutions():
    """
    Additional solutions and advanced techniques:
    """
    
    print("Additional Solutions and Techniques")
    print("=" * 35)
    
    print("1. Tree Diameter:")
    print("   Problem: Find longest path between any two nodes")
    print("   Approach: DFS twice or single DFS with two max depths")
    print("   Time: O(n), Space: O(h)")
    print()
    
    print("2. Trie Implementation:")
    print("   Problem: Efficient prefix-based string operations")
    print("   Approach: Tree where each node represents a character")
    print("   Time: O(m) for operations where m is string length")
    print("   Space: O(ALPHABET_SIZE * N * M)")
    print()
    
    print("3. Union-Find (Disjoint Set):")
    print("   Problem: Efficiently manage disjoint sets")
    print("   Approach: Path compression and union by rank")
    print("   Time: Nearly O(1) amortized per operation")
    print("   Space: O(n)")
    print()
    
    print("4. Topological Sorting:")
    print("   Problem: Linear ordering of vertices in DAG")
    print("   Approach: DFS post-order or Kahn's algorithm")
    print("   Time: O(V + E), Space: O(V)")
    print()
    
    print("5. Bellman-Ford Algorithm:")
    print("   Problem: Shortest path with negative weights")
    print("   Approach: Relax edges V-1 times")
    print("   Time: O(V * E), Space: O(V)")
    print("   Can detect negative cycles")

# Run all solutions
if __name__ == "__main__":
    print("=== Non-Linear Data Structures Practice Problem Solutions ===\n")
    
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