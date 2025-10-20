"""
Graph Problems with Backtracking

This module covers various graph problems that can be solved using backtracking:
- Hamiltonian path and cycle
- Graph coloring
- Knight's tour
- Word search in grid
- Maze solving
"""

from typing import List, Set


def hamiltonian_path(graph: List[List[int]]) -> List[int]:
    """
    Find a Hamiltonian path in a graph (visits each vertex exactly once).
    
    Time Complexity: O(N! * N) where N is number of vertices
    Space Complexity: O(N) for recursion stack
    
    Args:
        graph: Adjacency list representation of graph
        
    Returns:
        Hamiltonian path as list of vertices, or empty list if none exists
    """
    n = len(graph)
    path = []
    visited = [False] * n
    
    def backtrack(vertex: int) -> bool:
        path.append(vertex)
        visited[vertex] = True
        
        # Base case: path includes all vertices
        if len(path) == n:
            return True
        
        # Try all unvisited neighbors
        for neighbor in graph[vertex]:
            if not visited[neighbor]:
                if backtrack(neighbor):
                    return True
        
        # Backtrack
        path.pop()
        visited[vertex] = False
        return False
    
    # Try starting from each vertex
    for start_vertex in range(n):
        if backtrack(start_vertex):
            return path
    
    return []


def hamiltonian_cycle(graph: List[List[int]]) -> List[int]:
    """
    Find a Hamiltonian cycle in a graph (visits each vertex exactly once and returns to start).
    
    Time Complexity: O(N! * N)
    Space Complexity: O(N)
    
    Args:
        graph: Adjacency list representation of graph
        
    Returns:
        Hamiltonian cycle as list of vertices, or empty list if none exists
    """
    n = len(graph)
    path = []
    visited = [False] * n
    
    def backtrack(vertex: int, start: int) -> bool:
        path.append(vertex)
        visited[vertex] = True
        
        # Base case: path includes all vertices
        if len(path) == n:
            # Check if last vertex connects back to start
            if start in graph[vertex]:
                return True
            else:
                path.pop()
                visited[vertex] = False
                return False
        
        # Try all unvisited neighbors
        for neighbor in graph[vertex]:
            if not visited[neighbor]:
                if backtrack(neighbor, start):
                    return True
        
        # Backtrack
        path.pop()
        visited[vertex] = False
        return False
    
    # Try starting from vertex 0
    if backtrack(0, 0):
        return path
    
    return []


def graph_coloring(graph: List[List[int]], k: int) -> List[int]:
    """
    Color vertices of graph using at most k colors such that no adjacent vertices have same color.
    
    Time Complexity: O(k^N) where N is number of vertices
    Space Complexity: O(N)
    
    Args:
        graph: Adjacency list representation of graph
        k: Maximum number of colors allowed
        
    Returns:
        List where index i represents color of vertex i, or empty list if impossible
    """
    n = len(graph)
    colors = [-1] * n  # -1 means uncolored
    
    def is_safe(vertex: int, color: int) -> bool:
        """Check if assigning color to vertex is safe."""
        for neighbor in graph[vertex]:
            if colors[neighbor] == color:
                return False
        return True
    
    def backtrack(vertex: int) -> bool:
        # Base case: all vertices colored
        if vertex == n:
            return True
        
        # Try each color
        for color in range(k):
            if is_safe(vertex, color):
                colors[vertex] = color
                if backtrack(vertex + 1):
                    return True
                colors[vertex] = -1  # Backtrack
        
        return False
    
    if backtrack(0):
        return colors
    
    return []


def knights_tour(n: int) -> List[List[int]]:
    """
    Solve Knight's Tour problem on n×n chessboard.
    
    Time Complexity: O(8^(N*N)) where N is board size
    Space Complexity: O(N*N)
    
    Args:
        n: Size of chessboard
        
    Returns:
        Board with numbers indicating move sequence, or empty list if no solution
    """
    # Knight moves: 8 possible L-shaped moves
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    
    # Initialize board with -1 (unvisited)
    board = [[-1 for _ in range(n)] for _ in range(n)]
    
    def is_safe(x: int, y: int) -> bool:
        """Check if position is safe to move to."""
        return 0 <= x < n and 0 <= y < n and board[x][y] == -1
    
    def backtrack(x: int, y: int, move_count: int) -> bool:
        """Backtracking function for Knight's Tour."""
        board[x][y] = move_count
        
        # Base case: all squares visited
        if move_count == n * n - 1:
            return True
        
        # Try all 8 possible moves
        for dx, dy in moves:
            next_x, next_y = x + dx, y + dy
            if is_safe(next_x, next_y):
                if backtrack(next_x, next_y, move_count + 1):
                    return True
        
        # Backtrack
        board[x][y] = -1
        return False
    
    # Start from position (0, 0)
    if backtrack(0, 0, 0):
        return board
    
    return []


def word_search(board: List[List[str]], word: str) -> bool:
    """
    Check if word exists in 2D grid of characters.
    
    Time Complexity: O(M * N * 4^L) where M,N are board dimensions, L is word length
    Space Complexity: O(L) for recursion stack
    
    Args:
        board: 2D grid of characters
        word: Word to search for
        
    Returns:
        True if word exists in grid, False otherwise
    """
    if not board or not board[0] or not word:
        return False
    
    rows, cols = len(board), len(board[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    
    def backtrack(row: int, col: int, index: int) -> bool:
        """Backtracking function to search for word."""
        # Base case: entire word found
        if index == len(word):
            return True
        
        # Check bounds and character match
        if (row < 0 or row >= rows or col < 0 or col >= cols or 
            board[row][col] != word[index]):
            return False
        
        # Mark current cell as visited
        temp = board[row][col]
        board[row][col] = '#'  # Use special character to mark visited
        
        # Try all 4 directions
        for dr, dc in directions:
            if backtrack(row + dr, col + dc, index + 1):
                board[row][col] = temp  # Restore before returning
                return True
        
        # Backtrack
        board[row][col] = temp
        return False
    
    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    
    return False


def word_search_ii(board: List[List[str]], words: List[str]) -> List[str]:
    """
    Find all words from list that exist in 2D grid.
    
    Time Complexity: O(M * N * 4^L * W) where W is number of words
    Space Complexity: O(W * L) for Trie + O(L) for recursion
    
    Args:
        board: 2D grid of characters
        words: List of words to search for
        
    Returns:
        List of words found in grid
    """
    if not board or not board[0] or not words:
        return []
    
    # Build Trie for efficient word search
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None
    
    def build_trie(words_list: List[str]) -> TrieNode:
        root = TrieNode()
        for word in words_list:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
        return root
    
    rows, cols = len(board), len(board[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    result = set()
    
    def backtrack(row: int, col: int, node: TrieNode):
        """Backtracking function with Trie."""
        char = board[row][col]
        
        # Check if character exists in Trie
        if char not in node.children:
            return
        
        # Move to next Trie node
        next_node = node.children[char]
        
        # Check if current path forms a complete word
        if next_node.word:
            result.add(next_node.word)
            next_node.word = None  # Avoid duplicates
        
        # Mark current cell as visited
        board[row][col] = '#'
        
        # Explore neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                board[new_row][new_col] != '#'):
                backtrack(new_row, new_col, next_node)
        
        # Backtrack
        board[row][col] = char
        
        # Optimization: remove leaf nodes
        if not next_node.children:
            del node.children[char]
    
    # Build Trie
    trie = build_trie(words)
    
    # Search from each cell
    for i in range(rows):
        for j in range(cols):
            backtrack(i, j, trie)
    
    return list(result)


def solve_maze(maze: List[List[int]]) -> List[List[int]]:
    """
    Solve maze using backtracking (find path from top-left to bottom-right).
    
    Time Complexity: O(2^(M*N)) in worst case
    Space Complexity: O(M*N)
    
    Args:
        maze: 2D grid where 1 represents path and 0 represents wall
        
    Returns:
        Solution path as 2D grid with 1s showing path, or empty list if no solution
    """
    if not maze or not maze[0] or maze[0][0] == 0:
        return []
    
    rows, cols = len(maze), len(maze[0])
    solution = [[0 for _ in range(cols)] for _ in range(rows)]
    
    def is_safe(x: int, y: int) -> bool:
        """Check if position is safe to move to."""
        return (0 <= x < rows and 0 <= y < cols and 
                maze[x][y] == 1 and solution[x][y] == 0)
    
    def backtrack(x: int, y: int) -> bool:
        """Backtracking function to solve maze."""
        # Base case: reached destination
        if x == rows - 1 and y == cols - 1:
            solution[x][y] = 1
            return True
        
        # Check if current position is safe
        if is_safe(x, y):
            # Mark current cell as part of solution
            solution[x][y] = 1
            
            # Try moving right
            if backtrack(x, y + 1):
                return True
            
            # Try moving down
            if backtrack(x + 1, y):
                return True
            
            # Try moving left
            if backtrack(x, y - 1):
                return True
            
            # Try moving up
            if backtrack(x - 1, y):
                return True
            
            # Backtrack: unmark current cell
            solution[x][y] = 0
            return False
        
        return False
    
    if backtrack(0, 0):
        return solution
    
    return []


def demo():
    """Demonstrate the graph problems with backtracking."""
    print("=== Graph Problems with Backtracking ===\n")
    
    # Test Hamiltonian Path
    graph = [[1, 2], [0, 2, 3], [0, 1, 3], [1, 2]]
    path = hamiltonian_path(graph)
    print(f"Hamiltonian Path in graph {graph}:")
    print(f"  Path: {path}")
    
    print("\n" + "="*50)
    
    # Test Graph Coloring
    graph = [[1, 2], [0, 2], [0, 1]]
    colors = graph_coloring(graph, 3)
    print(f"\nGraph Coloring (3 colors) for graph {graph}:")
    print(f"  Colors: {colors}")
    
    # Test with impossible case
    colors = graph_coloring(graph, 2)
    print(f"  Colors with 2 colors: {colors}")
    
    print("\n" + "="*50)
    
    # Test Word Search
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    word = "ABCCED"
    found = word_search(board, word)
    print(f"\nWord Search for '{word}': {found}")
    
    word = "SEE"
    found = word_search(board, word)
    print(f"Word Search for '{word}': {found}")
    
    word = "ABCB"
    found = word_search(board, word)
    print(f"Word Search for '{word}': {found}")
    
    print("\n" + "="*50)
    
    # Test Word Search II
    board = [
        ['o', 'a', 'a', 'n'],
        ['e', 't', 'a', 'e'],
        ['i', 'h', 'k', 'r'],
        ['i', 'f', 'l', 'v']
    ]
    words = ["oath", "pea", "eat", "rain"]
    found = word_search_ii(board, words)
    print(f"\nWord Search II for {words}:")
    print(f"  Found: {found}")
    
    print("\n" + "="*50)
    
    # Test Maze Solving
    maze = [
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 1, 0, 0],
        [1, 1, 1, 1]
    ]
    solution = solve_maze(maze)
    print(f"\nMaze Solution:")
    if solution:
        for row in solution:
            print(f"  {row}")
    else:
        print("  No solution exists")


if __name__ == "__main__":
    demo()