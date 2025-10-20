"""
Non-Linear Data Structures - Binary Search Trees
==========================================

This module provides implementations and examples of binary search trees and balanced trees,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Binary Search Tree (BST) properties and operations
- Balanced tree implementations (AVL, Red-Black)
- BST algorithms and their complexities
- Applications in data science and algorithms
"""

from typing import Any, Optional, List, Tuple

class BSTNode:
    """
    Node class for Binary Search Tree
    """
    
    def __init__(self, data: Any):
        """Initialize a BST node"""
        self.data = data
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None
        self.height = 1  # For AVL tree balancing

class BinarySearchTree:
    """
    Implementation of a Binary Search Tree
    """
    
    def __init__(self):
        """Initialize an empty BST"""
        self.root: Optional[BSTNode] = None
    
    def insert(self, data: Any) -> None:
        """
        Insert data into the BST
        Time Complexity: O(h) where h is height of tree
        """
        self.root = self._insert_recursive(self.root, data)
    
    def _insert_recursive(self, node: Optional[BSTNode], data: Any) -> BSTNode:
        """Helper method for recursive insertion"""
        # Base case: empty node
        if not node:
            return BSTNode(data)
        
        # Recursive case: insert in appropriate subtree
        if data < node.data:
            node.left = self._insert_recursive(node.left, data)
        elif data > node.data:
            node.right = self._insert_recursive(node.right, data)
        # If data equals node.data, we don't insert duplicates
        
        return node
    
    def search(self, data: Any) -> bool:
        """
        Search for data in the BST
        Time Complexity: O(h) where h is height of tree
        """
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node: Optional[BSTNode], data: Any) -> bool:
        """Helper method for recursive search"""
        # Base case: empty node or found
        if not node:
            return False
        if node.data == data:
            return True
        
        # Recursive case: search in appropriate subtree
        if data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)
    
    def delete(self, data: Any) -> None:
        """
        Delete data from the BST
        Time Complexity: O(h) where h is height of tree
        """
        self.root = self._delete_recursive(self.root, data)
    
    def _delete_recursive(self, node: Optional[BSTNode], data: Any) -> Optional[BSTNode]:
        """Helper method for recursive deletion"""
        # Base case: empty node
        if not node:
            return node
        
        # Recursive case: find node to delete
        if data < node.data:
            node.left = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right = self._delete_recursive(node.right, data)
        else:
            # Node to delete found
            # Case 1: Node with no children (leaf node)
            if not node.left and not node.right:
                return None
            
            # Case 2: Node with one child
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            
            # Case 3: Node with two children
            # Find inorder successor (smallest in right subtree)
            successor = self._find_min(node.right)
            node.data = successor.data
            node.right = self._delete_recursive(node.right, successor.data)
        
        return node
    
    def find_min(self) -> Optional[Any]:
        """
        Find minimum value in the BST
        Time Complexity: O(h) where h is height of tree
        """
        if not self.root:
            return None
        return self._find_min(self.root).data
    
    def _find_min(self, node: BSTNode) -> BSTNode:
        """Helper method to find minimum node"""
        while node.left:
            node = node.left
        return node
    
    def find_max(self) -> Optional[Any]:
        """
        Find maximum value in the BST
        Time Complexity: O(h) where h is height of tree
        """
        if not self.root:
            return None
        node = self.root
        while node.right:
            node = node.right
        return node.data
    
    def inorder_traversal(self) -> List[Any]:
        """
        Inorder traversal (returns sorted order for BST)
        Time Complexity: O(n)
        """
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: Optional[BSTNode], result: List[Any]) -> None:
        """Helper method for inorder traversal"""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)
    
    def preorder_traversal(self) -> List[Any]:
        """
        Preorder traversal
        Time Complexity: O(n)
        """
        result = []
        self._preorder_recursive(self.root, result)
        return result
    
    def _preorder_recursive(self, node: Optional[BSTNode], result: List[Any]) -> None:
        """Helper method for preorder traversal"""
        if node:
            result.append(node.data)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)
    
    def postorder_traversal(self) -> List[Any]:
        """
        Postorder traversal
        Time Complexity: O(n)
        """
        result = []
        self._postorder_recursive(self.root, result)
        return result
    
    def _postorder_recursive(self, node: Optional[BSTNode], result: List[Any]) -> None:
        """Helper method for postorder traversal"""
        if node:
            self._postorder_recursive(node.left, result)
            self._postorder_recursive(node.right, result)
            result.append(node.data)
    
    def height(self) -> int:
        """
        Calculate the height of the BST
        Time Complexity: O(n)
        """
        return self._height_recursive(self.root)
    
    def _height_recursive(self, node: Optional[BSTNode]) -> int:
        """Helper method to calculate height"""
        if not node:
            return -1
        return 1 + max(self._height_recursive(node.left), 
                      self._height_recursive(node.right))
    
    def is_valid_bst(self) -> bool:
        """
        Check if the tree is a valid BST
        Time Complexity: O(n)
        """
        return self._is_valid_bst_recursive(self.root, float('-inf'), float('inf'))
    
    def _is_valid_bst_recursive(self, node: Optional[BSTNode], 
                               min_val: float, max_val: float) -> bool:
        """Helper method to validate BST property"""
        if not node:
            return True
        
        if node.data <= min_val or node.data >= max_val:
            return False
        
        return (self._is_valid_bst_recursive(node.left, min_val, node.data) and
                self._is_valid_bst_recursive(node.right, node.data, max_val))
    
    def __str__(self) -> str:
        """String representation of the BST"""
        if not self.root:
            return "BinarySearchTree([])"
        return f"BinarySearchTree(Height: {self.height()}, Size: {len(self.inorder_traversal())})"

class AVLTree:
    """
    Implementation of an AVL Tree (self-balancing BST)
    """
    
    def __init__(self):
        """Initialize an empty AVL tree"""
        self.root: Optional[BSTNode] = None
    
    def insert(self, data: Any) -> None:
        """
        Insert data into the AVL tree
        Time Complexity: O(log n)
        """
        self.root = self._insert_recursive(self.root, data)
    
    def _insert_recursive(self, node: Optional[BSTNode], data: Any) -> BSTNode:
        """Helper method for recursive insertion with balancing"""
        # Standard BST insertion
        if not node:
            return BSTNode(data)
        
        if data < node.data:
            node.left = self._insert_recursive(node.left, data)
        elif data > node.data:
            node.right = self._insert_recursive(node.right, data)
        else:
            # Duplicate data not allowed
            return node
        
        # Update height
        node.height = 1 + max(self._get_height(node.left), 
                             self._get_height(node.right))
        
        # Get balance factor
        balance = self._get_balance(node)
        
        # Left Left Case
        if balance > 1 and data < node.left.data:
            return self._rotate_right(node)
        
        # Right Right Case
        if balance < -1 and data > node.right.data:
            return self._rotate_left(node)
        
        # Left Right Case
        if balance > 1 and data > node.left.data:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right Left Case
        if balance < -1 and data < node.right.data:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def _get_height(self, node: Optional[BSTNode]) -> int:
        """Get height of node"""
        if not node:
            return 0
        return node.height
    
    def _get_balance(self, node: Optional[BSTNode]) -> int:
        """Get balance factor of node"""
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_left(self, z: BSTNode) -> BSTNode:
        """Left rotation"""
        y = z.right
        T2 = y.left
        
        # Perform rotation
        y.left = z
        z.right = T2
        
        # Update heights
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        
        # Return new root
        return y
    
    def _rotate_right(self, z: BSTNode) -> BSTNode:
        """Right rotation"""
        y = z.left
        T3 = y.right
        
        # Perform rotation
        y.right = z
        z.left = T3
        
        # Update heights
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        
        # Return new root
        return y
    
    def search(self, data: Any) -> bool:
        """
        Search for data in the AVL tree
        Time Complexity: O(log n)
        """
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node: Optional[BSTNode], data: Any) -> bool:
        """Helper method for recursive search"""
        if not node:
            return False
        if node.data == data:
            return True
        if data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)
    
    def inorder_traversal(self) -> List[Any]:
        """
        Inorder traversal
        Time Complexity: O(n)
        """
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: Optional[BSTNode], result: List[Any]) -> None:
        """Helper method for inorder traversal"""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)
    
    def height(self) -> int:
        """
        Calculate the height of the AVL tree
        Time Complexity: O(1) - stored in nodes
        """
        return self._get_height(self.root)
    
    def __str__(self) -> str:
        """String representation of the AVL tree"""
        if not self.root:
            return "AVLTree([])"
        return f"AVLTree(Height: {self.height()}, Size: {len(self.inorder_traversal())})"

def bst_operations_demo():
    """
    Demonstrate BST operations and their complexities
    """
    print("=== BST Operations Demo ===")
    
    # Create and populate BST
    bst = BinarySearchTree()
    values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45]
    
    print("1. Inserting values:", values)
    for value in values:
        bst.insert(value)
    
    print(f"   BST: {bst}")
    print(f"   Inorder traversal (sorted): {bst.inorder_traversal()}")
    print(f"   Height: {bst.height()}")
    print(f"   Valid BST: {bst.is_valid_bst()}")
    
    # Search operations
    print("\n2. Search operations:")
    search_values = [25, 55, 80]
    for value in search_values:
        found = bst.search(value)
        print(f"   Search {value}: {'Found' if found else 'Not Found'}")
    
    # Delete operations
    print("\n3. Delete operations:")
    delete_values = [10, 30, 50]
    for value in delete_values:
        print(f"   Before deleting {value}: {bst.inorder_traversal()}")
        bst.delete(value)
        print(f"   After deleting {value}: {bst.inorder_traversal()}")
        print(f"   Valid BST: {bst.is_valid_bst()}")

def avl_tree_demo():
    """
    Demonstrate AVL tree operations and balancing
    """
    print("\n=== AVL Tree Demo ===")
    
    # Create and populate AVL tree
    avl = AVLTree()
    values = [10, 20, 30, 40, 50, 25]  # This will cause rotations
    
    print("1. Inserting values:", values)
    print("   Note: Insertions may cause rotations to maintain balance")
    for value in values:
        avl.insert(value)
        print(f"   After inserting {value}: {avl.inorder_traversal()}")
        print(f"   Height: {avl.height()}")
    
    print(f"\n2. Final AVL tree:")
    print(f"   Inorder traversal: {avl.inorder_traversal()}")
    print(f"   Height: {avl.height()}")
    
    # Search operations
    print("\n3. Search operations:")
    search_values = [25, 35, 50]
    for value in search_values:
        found = avl.search(value)
        print(f"   Search {value}: {'Found' if found else 'Not Found'}")

def tree_comparison():
    """
    Compare BST and AVL tree performance
    """
    print("\n=== BST vs AVL Tree Comparison ===")
    
    import time
    import random
    
    # Generate random data
    data = list(range(1000))
    random.shuffle(data)
    
    # BST performance
    bst = BinarySearchTree()
    start = time.time()
    for value in data:
        bst.insert(value)
    bst_insert_time = time.time() - start
    
    start = time.time()
    for value in data[:100]:  # Search first 100 values
        bst.search(value)
    bst_search_time = time.time() - start
    
    # AVL tree performance
    avl = AVLTree()
    start = time.time()
    for value in data:
        avl.insert(value)
    avl_insert_time = time.time() - start
    
    start = time.time()
    for value in data[:100]:  # Search first 100 values
        avl.search(value)
    avl_search_time = time.time() - start
    
    print("Performance with 1000 random insertions and 100 searches:")
    print(f"BST - Insert time: {bst_insert_time:.6f}s, Height: {bst.height()}")
    print(f"BST - Search time: {bst_search_time:.6f}s")
    print(f"AVL - Insert time: {avl_insert_time:.6f}s, Height: {avl.height()}")
    print(f"AVL - Search time: {avl_search_time:.6f}s")
    print("\nNote: AVL tree maintains O(log n) height, BST can become O(n) in worst case")

def bst_applications():
    """
    Demonstrate common applications of BSTs
    """
    print("\n=== BST Applications ===")
    
    # 1. Sorted data storage
    print("1. Sorted Data Storage:")
    print("   BSTs maintain sorted order automatically")
    print("   Inorder traversal gives sorted sequence")
    print("   Efficient for range queries")
    
    # 2. Database indexing
    print("\n2. Database Indexing:")
    print("   B-trees and B+ trees are generalized BSTs")
    print("   Used for indexing in databases")
    print("   Enable fast search, insertion, and deletion")
    
    # 3. File systems
    print("\n3. File Systems:")
    print("   Directory structures can be implemented with BSTs")
    print("   Fast file lookup and organization")
    print("   Efficient for alphabetical ordering")
    
    # 4. Priority queues
    print("\n4. Priority Queues:")
    print("   Heaps are specialized BSTs for priority queues")
    print("   Enable efficient min/max extraction")
    print("   Used in scheduling algorithms")

def data_science_applications():
    """
    Examples of BSTs in data science applications
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Decision trees
    print("1. Decision Trees:")
    print("   Machine learning decision trees are tree structures")
    print("   Scikit-learn uses optimized tree implementations")
    print("   Enable fast classification and regression")
    
    # 2. Range queries in data analysis
    print("\n2. Range Queries:")
    print("   BSTs enable efficient range queries")
    print("   Find all data points within a range")
    print("   Useful for filtering and data exploration")
    
    # 3. Nearest neighbor search
    print("\n3. Nearest Neighbor Search:")
    print("   KD-trees are specialized BSTs for nearest neighbor")
    print("   Used in recommendation systems")
    print("   Enable fast similarity searches")
    
    # 4. Quantile computation
    print("\n4. Quantile Computation:")
    print("   Order statistics trees help compute quantiles")
    print("   Used in descriptive statistics")
    print("   Enable streaming quantile calculations")

# Example usage and testing
if __name__ == "__main__":
    # BST operations demo
    bst_operations_demo()
    print("\n" + "="*50 + "\n")
    
    # AVL tree demo
    avl_tree_demo()
    print("\n" + "="*50 + "\n")
    
    # Tree comparison
    tree_comparison()
    print("\n" + "="*50 + "\n")
    
    # BST applications
    bst_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Binary Search Tree implementation and operations")
    print("2. AVL Tree implementation with self-balancing")
    print("3. Comparison of BST and AVL tree performance")
    print("4. Practical applications in systems and algorithms")
    print("5. Data science applications of tree structures")
    print("\nKey takeaways:")
    print("- BSTs provide O(log n) operations for balanced trees")
    print("- Worst case for BSTs is O(n) when tree becomes linear")
    print("- AVL trees guarantee O(log n) operations through balancing")
    print("- BSTs are fundamental in databases, file systems, and ML")
    print("- Self-balancing trees are crucial for consistent performance")