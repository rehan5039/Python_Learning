"""
Non-Linear Data Structures - Trees
=============================

This module provides implementations and examples of tree data structures,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Tree terminology and concepts
- Binary trees and their traversals
- Tree implementations
- Applications in algorithms and data science
"""

from collections import deque
from typing import Any, List, Optional

class TreeNode:
    """
    Node class for tree implementations
    """
    
    def __init__(self, data: Any):
        """Initialize a tree node"""
        self.data = data
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None
        self.children: List['TreeNode'] = []

class BinaryTree:
    """
    Implementation of a binary tree
    """
    
    def __init__(self, data: Any = None):
        """Initialize a binary tree"""
        if data is not None:
            self.root = TreeNode(data)
        else:
            self.root = None
    
    def insert(self, data: Any) -> None:
        """
        Insert data into the binary tree (level-order insertion)
        """
        if not self.root:
            self.root = TreeNode(data)
            return
        
        # Level-order insertion using queue
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            
            if not node.left:
                node.left = TreeNode(data)
                return
            elif not node.right:
                node.right = TreeNode(data)
                return
            else:
                queue.append(node.left)
                queue.append(node.right)
    
    def preorder_traversal(self, node: Optional[TreeNode] = None) -> List[Any]:
        """
        Preorder traversal: Root -> Left -> Right
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return []
        
        result = [node.data]
        if node.left:
            result.extend(self.preorder_traversal(node.left))
        if node.right:
            result.extend(self.preorder_traversal(node.right))
        
        return result
    
    def inorder_traversal(self, node: Optional[TreeNode] = None) -> List[Any]:
        """
        Inorder traversal: Left -> Root -> Right
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return []
        
        result = []
        if node.left:
            result.extend(self.inorder_traversal(node.left))
        result.append(node.data)
        if node.right:
            result.extend(self.inorder_traversal(node.right))
        
        return result
    
    def postorder_traversal(self, node: Optional[TreeNode] = None) -> List[Any]:
        """
        Postorder traversal: Left -> Right -> Root
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return []
        
        result = []
        if node.left:
            result.extend(self.postorder_traversal(node.left))
        if node.right:
            result.extend(self.postorder_traversal(node.right))
        result.append(node.data)
        
        return result
    
    def level_order_traversal(self) -> List[Any]:
        """
        Level-order traversal (BFS): Left to right, level by level
        Time Complexity: O(n)
        """
        if not self.root:
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            result.append(node.data)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return result
    
    def height(self, node: Optional[TreeNode] = None) -> int:
        """
        Calculate the height of the tree
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return -1
        
        left_height = self.height(node.left) if node.left else -1
        right_height = self.height(node.right) if node.right else -1
        
        return 1 + max(left_height, right_height)
    
    def size(self, node: Optional[TreeNode] = None) -> int:
        """
        Calculate the number of nodes in the tree
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        left_size = self.size(node.left) if node.left else 0
        right_size = self.size(node.right) if node.right else 0
        
        return 1 + left_size + right_size
    
    def is_balanced(self, node: Optional[TreeNode] = None) -> bool:
        """
        Check if the tree is balanced (height difference <= 1 for all nodes)
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return True
        
        def check_balance(node):
            if node is None:
                return 0, True
            
            left_height, left_balanced = check_balance(node.left)
            right_height, right_balanced = check_balance(node.right)
            
            is_balanced = (left_balanced and right_balanced and 
                          abs(left_height - right_height) <= 1)
            
            return 1 + max(left_height, right_height), is_balanced
        
        _, balanced = check_balance(node)
        return balanced
    
    def __str__(self) -> str:
        """String representation of the binary tree"""
        if not self.root:
            return "BinaryTree([])"
        return f"BinaryTree(Height: {self.height()}, Size: {self.size()})"

class GeneralTree:
    """
    Implementation of a general tree (n-ary tree)
    """
    
    def __init__(self, data: Any = None):
        """Initialize a general tree"""
        if data is not None:
            self.root = TreeNode(data)
        else:
            self.root = None
    
    def add_child(self, parent_data: Any, child_data: Any) -> bool:
        """
        Add a child to a node with specified data
        """
        if not self.root:
            self.root = TreeNode(parent_data)
            self.root.children.append(TreeNode(child_data))
            return True
        
        # Find parent node using BFS
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node.data == parent_data:
                node.children.append(TreeNode(child_data))
                return True
            queue.extend(node.children)
        
        return False
    
    def preorder_traversal(self, node: Optional[TreeNode] = None) -> List[Any]:
        """
        Preorder traversal for general tree
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return []
        
        result = [node.data]
        for child in node.children:
            result.extend(self.preorder_traversal(child))
        
        return result
    
    def postorder_traversal(self, node: Optional[TreeNode] = None) -> List[Any]:
        """
        Postorder traversal for general tree
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return []
        
        result = []
        for child in node.children:
            result.extend(self.postorder_traversal(child))
        result.append(node.data)
        
        return result
    
    def level_order_traversal(self) -> List[Any]:
        """
        Level-order traversal for general tree
        Time Complexity: O(n)
        """
        if not self.root:
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            result.append(node.data)
            queue.extend(node.children)
        
        return result
    
    def height(self, node: Optional[TreeNode] = None) -> int:
        """
        Calculate the height of the general tree
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return -1
        
        if not node.children:
            return 0
        
        max_child_height = max(self.height(child) for child in node.children)
        return 1 + max_child_height
    
    def size(self, node: Optional[TreeNode] = None) -> int:
        """
        Calculate the number of nodes in the general tree
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        result = 1
        for child in node.children:
            result += self.size(child)
        
        return result

def tree_traversal_examples():
    """
    Demonstrate different tree traversal methods
    """
    print("=== Tree Traversal Examples ===")
    
    # Binary tree example
    print("1. Binary Tree Traversals:")
    bt = BinaryTree(1)
    for i in range(2, 8):
        bt.insert(i)
    
    print(f"   Tree structure:")
    print(f"         1")
    print(f"       /   \\")
    print(f"      2     3")
    print(f"     / \\   / \\")
    print(f"    4   5 6   7")
    
    print(f"   Preorder: {bt.preorder_traversal()}")
    print(f"   Inorder: {bt.inorder_traversal()}")
    print(f"   Postorder: {bt.postorder_traversal()}")
    print(f"   Level-order: {bt.level_order_traversal()}")
    
    # General tree example
    print("\n2. General Tree Traversals:")
    gt = GeneralTree('A')
    gt.add_child('A', 'B')
    gt.add_child('A', 'C')
    gt.add_child('A', 'D')
    gt.add_child('B', 'E')
    gt.add_child('B', 'F')
    gt.add_child('C', 'G')
    
    print(f"   Tree structure:")
    print(f"         A")
    print(f"       / | \\")
    print(f"      B  C  D")
    print(f"     /|  |")
    print(f"    E F  G")
    
    print(f"   Preorder: {gt.preorder_traversal()}")
    print(f"   Postorder: {gt.postorder_traversal()}")
    print(f"   Level-order: {gt.level_order_traversal()}")

def tree_properties():
    """
    Demonstrate tree properties and calculations
    """
    print("\n=== Tree Properties ===")
    
    # Binary tree properties
    bt = BinaryTree(1)
    for i in range(2, 16):
        bt.insert(i)
    
    print("1. Binary Tree Properties:")
    print(f"   Height: {bt.height()}")
    print(f"   Size: {bt.size()}")
    print(f"   Balanced: {bt.is_balanced()}")
    
    # Perfect binary tree properties
    print("\n2. Perfect Binary Tree Properties:")
    print("   For a perfect binary tree of height h:")
    print("   - Number of nodes: 2^(h+1) - 1")
    print("   - Number of leaf nodes: 2^h")
    print("   - Number of internal nodes: 2^h - 1")
    
    # General tree properties
    print("\n3. General Tree Properties:")
    gt = GeneralTree(1)
    for i in range(2, 11):
        gt.add_child((i-1)//3 + 1, i)  # Simple parent-child relationship
    
    print(f"   Height: {gt.height()}")
    print(f"   Size: {gt.size()}")

def tree_applications():
    """
    Demonstrate common applications of trees
    """
    print("\n=== Tree Applications ===")
    
    # 1. File system representation
    print("1. File System Representation:")
    print("   Trees naturally represent hierarchical file systems")
    print("   Root directory -> subdirectories -> files")
    print("   Each node represents a directory or file")
    
    # 2. Expression trees
    print("\n2. Expression Trees:")
    print("   Arithmetic expressions can be represented as binary trees")
    print("   Internal nodes: operators (+, -, *, /)")
    print("   Leaf nodes: operands (numbers, variables)")
    print("   Example: (3 + 4) * 5")
    print("            *")
    print("           / \\")
    print("          +   5")
    print("         / \\")
    print("        3   4")
    
    # 3. Decision trees
    print("\n3. Decision Trees:")
    print("   Used in machine learning for classification")
    print("   Internal nodes: decision criteria")
    print("   Leaf nodes: class labels or predictions")
    print("   Path from root to leaf: decision path")
    
    # 4. Syntax trees
    print("\n4. Syntax Trees:")
    print("   Compilers use parse trees to represent program structure")
    print("   Programs are parsed into tree structures for analysis")
    print("   Each node represents a syntactic construct")

def data_science_applications():
    """
    Examples of trees in data science applications
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Decision trees in machine learning
    print("1. Decision Trees in Machine Learning:")
    print("   Scikit-learn uses tree-based algorithms for:")
    print("   - Classification (DecisionTreeClassifier)")
    print("   - Regression (DecisionTreeRegressor)")
    print("   - Ensemble methods (Random Forest, Gradient Boosting)")
    
    # 2. Hierarchical clustering
    print("\n2. Hierarchical Clustering:")
    print("   Dendrograms are tree structures showing cluster relationships")
    print("   Agglomerative clustering builds tree from bottom up")
    print("   Divisive clustering builds tree from top down")
    
    # 3. JSON/XML data processing
    print("\n3. JSON/XML Data Processing:")
    print("   These formats have natural tree structures")
    print("   Libraries parse them into tree representations")
    print("   Tree traversal used for data extraction and transformation")
    
    # 4. Game trees
    print("\n4. Game Trees:")
    print("   Minimax algorithm uses trees for game playing AI")
    print("   Each node represents a game state")
    print("   Children represent possible moves")
    print("   Used in chess, tic-tac-toe, and other strategy games")

def performance_comparison():
    """
    Compare performance of different tree operations
    """
    print("\n=== Performance Comparison ===")
    
    import time
    
    # Test with different tree sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nTree size: {size}")
        
        # Binary tree operations
        bt = BinaryTree()
        start = time.time()
        for i in range(size):
            bt.insert(i)
        insert_time = time.time() - start
        
        start = time.time()
        _ = bt.preorder_traversal()
        traversal_time = time.time() - start
        
        start = time.time()
        _ = bt.height()
        height_time = time.time() - start
        
        print(f"   Insert {size} nodes: {insert_time:.6f} seconds")
        print(f"   Preorder traversal: {traversal_time:.6f} seconds")
        print(f"   Height calculation: {height_time:.6f} seconds")

# Example usage and testing
if __name__ == "__main__":
    # Tree traversal examples
    tree_traversal_examples()
    print("\n" + "="*50 + "\n")
    
    # Tree properties
    tree_properties()
    print("\n" + "="*50 + "\n")
    
    # Tree applications
    tree_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Binary tree and general tree implementations")
    print("2. Different tree traversal methods")
    print("3. Tree properties and calculations")
    print("4. Practical applications in systems and algorithms")
    print("5. Data science applications of trees")
    print("6. Performance characteristics of tree operations")
    print("\nKey takeaways:")
    print("- Trees provide hierarchical data organization")
    print("- Different traversal methods serve different purposes")
    print("- Binary trees are fundamental in many algorithms")
    print("- General trees model complex hierarchical relationships")
    print("- Trees are essential in file systems, databases, and ML")