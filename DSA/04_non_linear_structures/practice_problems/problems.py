"""
Non-Linear Data Structures - Practice Problems
=======================================

This file contains practice problems for non-linear data structures with solutions.
"""

# Problem 1: Tree Operations
def problem_1():
    """
    Tree operations and traversals:
    """
    
    print("Problem 1: Tree Operations")
    print("=" * 25)
    
    # TreeNode class for binary tree
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    # 1. Maximum depth of binary tree
    def max_depth(root):
        """
        Calculate maximum depth of binary tree
        Time Complexity: O(n)
        """
        if not root:
            return 0
        return 1 + max(max_depth(root.left), max_depth(root.right))
    
    # 2. Invert binary tree
    def invert_tree(root):
        """
        Invert binary tree (mirror)
        Time Complexity: O(n)
        """
        if not root:
            return None
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
        return root
    
    # 3. Same tree check
    def is_same_tree(p, q):
        """
        Check if two binary trees are identical
        Time Complexity: O(min(m,n))
        """
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
    
    # Test cases
    print("1. Maximum Depth:")
    # Create tree:     3
    #                 / \\
    #                9   20
    #                   /  \\
    #                  15   7
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)
    
    depth = max_depth(root)
    print(f"   Tree depth: {depth}")
    
    print("\n2. Invert Tree:")
    print("   Original tree: 3 -> [9, 20] -> [15, 7]")
    inverted = invert_tree(root)
    # After inversion: 3 -> [20, 9] -> [7, 15]
    print("   Inverted tree: 3 -> [20, 9] -> [7, 15]")
    
    print("\n3. Same Tree Check:")
    # Create identical tree
    root2 = TreeNode(3)
    root2.left = TreeNode(9)
    root2.right = TreeNode(20)
    root2.right.left = TreeNode(15)
    root2.right.right = TreeNode(7)
    
    same = is_same_tree(root, root2)
    print(f"   Trees are identical: {same}")

# Problem 2: Binary Search Trees
def problem_2():
    """
    Binary Search Tree problems:
    """
    
    print("\nProblem 2: Binary Search Trees")
    print("=" * 25)
    
    class BSTNode:
        def __init__(self, val=0):
            self.val = val
            self.left = None
            self.right = None
    
    # 1. Validate BST
    def is_valid_bst(root):
        """
        Check if binary tree is valid BST
        Time Complexity: O(n)
        """
        def validate(node, low=float('-inf'), high=float('inf')):
            if not node:
                return True
            if node.val <= low or node.val >= high:
                return False
            return (validate(node.left, low, node.val) and 
                   validate(node.right, node.val, high))
        return validate(root)
    
    # 2. Lowest Common Ancestor in BST
    def lowest_common_ancestor(root, p, q):
        """
        Find lowest common ancestor of two nodes in BST
        Time Complexity: O(h) where h is height
        """
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root
        return None
    
    # Test cases
    print("1. Validate BST:")
    # Valid BST:     5
    #               / \\
    #              3   8
    #             / \\ / \\
    #            2  4 7  9
    root = BSTNode(5)
    root.left = BSTNode(3)
    root.right = BSTNode(8)
    root.left.left = BSTNode(2)
    root.left.right = BSTNode(4)
    root.right.left = BSTNode(7)
    root.right.right = BSTNode(9)
    
    valid = is_valid_bst(root)
    print(f"   Valid BST: {valid}")
    
    # Invalid BST (4 > 3 but in left subtree of 3)
    root_invalid = BSTNode(5)
    root_invalid.left = BSTNode(3)
    root_invalid.right = BSTNode(8)
    root_invalid.left.left = BSTNode(2)
    root_invalid.left.right = BSTNode(6)  # Invalid: 6 > 5
    root_invalid.right.left = BSTNode(7)
    root_invalid.right.right = BSTNode(9)
    
    valid_invalid = is_valid_bst(root_invalid)
    print(f"   Invalid BST: {valid_invalid}")
    
    print("\n2. Lowest Common Ancestor:")
    # Find LCA of nodes 2 and 4
    lca = lowest_common_ancestor(root, root.left.left, root.left.right)
    print(f"   LCA of 2 and 4: {lca.val if lca else None}")

# Problem 3: Heap Operations
def problem_3():
    """
    Heap problems and operations:
    """
    
    print("\nProblem 3: Heap Operations")
    print("=" * 25)
    
    import heapq
    
    # 1. Kth largest element
    def find_kth_largest(nums, k):
        """
        Find kth largest element in array
        Time Complexity: O(n log k)
        """
        heap = []
        for num in nums:
            if len(heap) < k:
                heapq.heappush(heap, num)
            elif num > heap[0]:
                heapq.heapreplace(heap, num)
        return heap[0]
    
    # 2. Merge k sorted lists
    def merge_k_lists(lists):
        """
        Merge k sorted linked lists
        Time Complexity: O(N log k) where N is total elements
        """
        heap = []
        # Initialize heap with first element of each list
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))
        
        result = []
        while heap:
            val, list_idx, elem_idx = heapq.heappop(heap)
            result.append(val)
            
            # Add next element from same list
            if elem_idx + 1 < len(lists[list_idx]):
                next_val = lists[list_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
        
        return result
    
    # Test cases
    print("1. Kth Largest Element:")
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    kth_largest = find_kth_largest(nums, k)
    print(f"   Array: {nums}")
    print(f"   {k}th largest element: {kth_largest}")
    
    print("\n2. Merge K Sorted Lists:")
    lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    merged = merge_k_lists(lists)
    print(f"   Lists: {lists}")
    print(f"   Merged: {merged}")

# Problem 4: Graph Algorithms
def problem_4():
    """
    Graph algorithm problems:
    """
    
    print("\nProblem 4: Graph Algorithms")
    print("=" * 25)
    
    from collections import deque, defaultdict
    
    # 1. Number of islands
    def num_islands(grid):
        """
        Count number of islands in 2D binary grid
        Time Complexity: O(m * n)
        """
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        visited = set()
        islands = 0
        
        def bfs(r, c):
            q = deque()
            visited.add((r, c))
            q.append((r, c))
            
            while q:
                row, col = q.popleft()
                directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                
                for dr, dc in directions:
                    r_new, c_new = row + dr, col + dc
                    if (0 <= r_new < rows and 0 <= c_new < cols and
                        grid[r_new][c_new] == '1' and (r_new, c_new) not in visited):
                        visited.add((r_new, c_new))
                        q.append((r_new, c_new))
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1' and (r, c) not in visited:
                    bfs(r, c)
                    islands += 1
        
        return islands
    
    # 2. Course schedule (topological sort)
    def can_finish(num_courses, prerequisites):
        """
        Check if all courses can be finished
        Time Complexity: O(V + E)
        """
        # Build adjacency list
        adj = defaultdict(list)
        for course, prereq in prerequisites:
            adj[prereq].append(course)
        
        # Track visited states: 0=unvisited, 1=visiting, 2=visited
        visited = [0] * num_courses
        
        def has_cycle(course):
            if visited[course] == 1:  # Currently visiting (cycle detected)
                return True
            if visited[course] == 2:  # Already visited
                return False
            
            visited[course] = 1  # Mark as visiting
            for neighbor in adj[course]:
                if has_cycle(neighbor):
                    return True
            visited[course] = 2  # Mark as visited
            return False
        
        # Check for cycles starting from each course
        for course in range(num_courses):
            if has_cycle(course):
                return False
        
        return True
    
    # Test cases
    print("1. Number of Islands:")
    grid = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]
    ]
    islands = num_islands(grid)
    print(f"   Grid:")
    for row in grid:
        print(f"   {row}")
    print(f"   Number of islands: {islands}")
    
    print("\n2. Course Schedule:")
    num_courses = 4
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    can_finish_courses = can_finish(num_courses, prerequisites)
    print(f"   Number of courses: {num_courses}")
    print(f"   Prerequisites: {prerequisites}")
    print(f"   Can finish all courses: {can_finish_courses}")

# Problem 5: Hash Table Implementations
def problem_5():
    """
    Hash table problems:
    """
    
    print("\nProblem 5: Hash Table Implementations")
    print("=" * 35)
    
    # 1. Two sum problem
    def two_sum(nums, target):
        """
        Find indices of two numbers that sum to target
        Time Complexity: O(n)
        """
        hash_map = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in hash_map:
                return [hash_map[complement], i]
            hash_map[num] = i
        return []
    
    # 2. Group anagrams
    def group_anagrams(strs):
        """
        Group anagrams together
        Time Complexity: O(n * k log k) where k is max string length
        """
        anagram_groups = defaultdict(list)
        for s in strs:
            # Sort characters to create key
            key = ''.join(sorted(s))
            anagram_groups[key].append(s)
        return list(anagram_groups.values())
    
    # Test cases
    print("1. Two Sum:")
    nums = [2, 7, 11, 15]
    target = 9
    indices = two_sum(nums, target)
    print(f"   Array: {nums}")
    print(f"   Target: {target}")
    print(f"   Indices: {indices} (values: {nums[indices[0]]}, {nums[indices[1]]})")
    
    print("\n2. Group Anagrams:")
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    groups = group_anagrams(strs)
    print(f"   Strings: {strs}")
    print(f"   Groups: {groups}")

# Run all problems
if __name__ == "__main__":
    print("=== Non-Linear Data Structures Practice Problems ===\n")
    
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
    print("1. Tree operations and traversals")
    print("2. Binary Search Tree validation and algorithms")
    print("3. Heap operations and applications")
    print("4. Graph algorithms and traversals")
    print("5. Hash table implementations and applications")
    print("\nEach problem demonstrates:")
    print("- Implementation of data structures")
    print("- Common algorithms and patterns")
    print("- Real-world applications")
    print("- Time and space complexity considerations")