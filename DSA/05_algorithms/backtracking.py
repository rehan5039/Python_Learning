"""
Algorithm Design Techniques - Backtracking
====================================

This module provides implementations and examples of backtracking algorithms,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Backtracking paradigm
- Classic backtracking problems (N-Queens, Sudoku, Subset Sum)
- Constraint satisfaction
- Optimization techniques
- Applications in data science
"""

from typing import List, Tuple, Optional, Set
import copy

def n_queens(n: int) -> List[List[int]]:
    """
    Solve N-Queens problem using backtracking
    Time Complexity: O(N!)
    Space Complexity: O(N²)
    
    Args:
        n: Size of chessboard and number of queens
    
    Returns:
        List of solutions, each solution is a list representing column positions
    """
    def is_safe(board: List[int], row: int, col: int) -> bool:
        """Check if placing queen at (row, col) is safe"""
        for i in range(row):
            # Check column conflict
            if board[i] == col:
                return False
            # Check diagonal conflicts
            if abs(board[i] - col) == abs(i - row):
                return False
        return True
    
    def solve_n_queens(board: List[int], row: int, solutions: List[List[int]]) -> None:
        """Recursive backtracking function"""
        if row == n:
            solutions.append(board[:])  # Add copy of current solution
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve_n_queens(board, row + 1, solutions)
                # No need to explicitly backtrack as we overwrite board[row]
    
    solutions = []
    board = [-1] * n  # board[i] represents column position of queen in row i
    solve_n_queens(board, 0, solutions)
    return solutions

def sudoku_solver(board: List[List[int]]) -> Optional[List[List[int]]]:
    """
    Solve Sudoku puzzle using backtracking
    Time Complexity: O(9^(n²)) in worst case
    Space Complexity: O(n²)
    
    Args:
        board: 9x9 Sudoku board (0 represents empty cell)
    
    Returns:
        Solved board or None if no solution exists
    """
    def is_valid(board: List[List[int]], row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid"""
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def find_empty_cell(board: List[List[int]]) -> Optional[Tuple[int, int]]:
        """Find next empty cell"""
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None
    
    def solve_sudoku(board: List[List[int]]) -> bool:
        """Recursive backtracking function"""
        empty_cell = find_empty_cell(board)
        if not empty_cell:
            return True  # No empty cells, puzzle solved
        
        row, col = empty_cell
        
        for num in range(1, 10):
            if is_valid(board, row, col, num):
                board[row][col] = num
                
                if solve_sudoku(board):
                    return True
                
                # Backtrack
                board[row][col] = 0
        
        return False
    
    # Create a copy to avoid modifying original board
    solved_board = copy.deepcopy(board)
    
    if solve_sudoku(solved_board):
        return solved_board
    else:
        return None

def subset_sum(arr: List[int], target: int) -> List[List[int]]:
    """
    Find all subsets that sum to target using backtracking
    Time Complexity: O(2^n)
    Space Complexity: O(n)
    
    Args:
        arr: List of integers
        target: Target sum
    
    Returns:
        List of all subsets that sum to target
    """
    def find_subsets(arr: List[int], target: int, index: int, 
                    current_subset: List[int], all_subsets: List[List[int]]) -> None:
        """Recursive backtracking function"""
        # Base case: found target sum
        if target == 0:
            all_subsets.append(current_subset[:])
            return
        
        # Base case: exceeded target or no more elements
        if target < 0 or index >= len(arr):
            return
        
        # Include current element
        current_subset.append(arr[index])
        find_subsets(arr, target - arr[index], index + 1, current_subset, all_subsets)
        
        # Exclude current element (backtrack)
        current_subset.pop()
        find_subsets(arr, target, index + 1, current_subset, all_subsets)
    
    all_subsets = []
    find_subsets(arr, target, 0, [], all_subsets)
    return all_subsets

def graph_coloring(graph: List[List[int]], num_colors: int) -> Optional[List[int]]:
    """
    Solve graph coloring problem using backtracking
    Time Complexity: O(m^n) where m is number of colors
    Space Complexity: O(n)
    
    Args:
        graph: Adjacency matrix representation
        num_colors: Number of available colors
    
    Returns:
        List representing color assignment for each vertex, or None if impossible
    """
    def is_safe(graph: List[List[int]], colors: List[int], vertex: int, color: int) -> bool:
        """Check if assigning color to vertex is safe"""
        for i in range(len(graph)):
            if graph[vertex][i] == 1 and colors[i] == color:
                return False
        return True
    
    def graph_coloring_util(graph: List[List[int]], num_colors: int, 
                          colors: List[int], vertex: int) -> bool:
        """Recursive backtracking function"""
        if vertex == len(graph):
            return True  # All vertices colored
        
        for color in range(1, num_colors + 1):
            if is_safe(graph, colors, vertex, color):
                colors[vertex] = color
                
                if graph_coloring_util(graph, num_colors, colors, vertex + 1):
                    return True
                
                # Backtrack
                colors[vertex] = 0
        
        return False
    
    colors = [0] * len(graph)  # 0 represents no color assigned
    
    if graph_coloring_util(graph, num_colors, colors, 0):
        return colors
    else:
        return None

def hamiltonian_cycle(graph: List[List[int]]) -> Optional[List[int]]:
    """
    Find Hamiltonian cycle in graph using backtracking
    Time Complexity: O(N!)
    Space Complexity: O(N)
    
    Args:
        graph: Adjacency matrix representation
    
    Returns:
        List representing Hamiltonian cycle, or None if doesn't exist
    """
    def is_safe(graph: List[List[int]], path: List[int], vertex: int, pos: int) -> bool:
        """Check if adding vertex to path[pos] is safe"""
        # Check if vertex is adjacent to last vertex in path
        if graph[path[pos - 1]][vertex] == 0:
            return False
        
        # Check if vertex is already in path
        if vertex in path:
            return False
        
        return True
    
    def hamiltonian_cycle_util(graph: List[List[int]], path: List[int], pos: int) -> bool:
        """Recursive backtracking function"""
        if pos == len(graph):
            # Check if last vertex connects back to first vertex
            if graph[path[pos - 1]][path[0]]:
                return True
            else:
                return False
        
        # Try different vertices as next candidate
        for vertex in range(1, len(graph)):
            if is_safe(graph, path, vertex, pos):
                path[pos] = vertex
                
                if hamiltonian_cycle_util(graph, path, pos + 1):
                    return True
                
                # Backtrack
                path[pos] = -1
        
        return False
    
    path = [-1] * len(graph)
    path[0] = 0  # Start from vertex 0
    
    if hamiltonian_cycle_util(graph, path, 1):
        return path
    else:
        return None

def backtracking_demo():
    """
    Demonstrate backtracking algorithms
    """
    print("=== Backtracking Demo ===")
    
    # N-Queens problem
    print("1. N-Queens Problem:")
    n = 4
    solutions = n_queens(n)
    print(f"   Board size: {n}x{n}")
    print(f"   Number of solutions: {len(solutions)}")
    if solutions:
        print(f"   First solution: {solutions[0]}")
        # Visualize first solution
        board = [["." for _ in range(n)] for _ in range(n)]
        for row, col in enumerate(solutions[0]):
            board[row][col] = "Q"
        print("   Visual representation:")
        for row in board:
            print(f"   {' '.join(row)}")
    
    # Subset Sum problem
    print("\n2. Subset Sum Problem:")
    arr = [3, 34, 4, 12, 5, 2]
    target = 9
    subsets = subset_sum(arr, target)
    print(f"   Array: {arr}")
    print(f"   Target sum: {target}")
    print(f"   Subsets that sum to target: {subsets}")
    
    # Graph Coloring problem
    print("\n3. Graph Coloring Problem:")
    # Simple graph: triangle (3 vertices, all connected)
    graph = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    num_colors = 3
    coloring = graph_coloring(graph, num_colors)
    print(f"   Graph adjacency matrix:")
    for row in graph:
        print(f"   {row}")
    print(f"   Number of colors: {num_colors}")
    if coloring:
        print(f"   Valid coloring: {coloring}")
    else:
        print("   No valid coloring exists")

def sudoku_demo():
    """
    Demonstrate Sudoku solver
    """
    print("\n=== Sudoku Solver Demo ===")
    
    # Example Sudoku puzzle (0 represents empty cells)
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    print("   Original puzzle:")
    for row in puzzle:
        print(f"   {row}")
    
    solution = sudoku_solver(puzzle)
    
    if solution:
        print("\n   Solved puzzle:")
        for row in solution:
            print(f"   {row}")
    else:
        print("\n   No solution exists!")

def backtracking_optimization():
    """
    Demonstrate backtracking optimization techniques
    """
    print("\n=== Backtracking Optimization Techniques ===")
    
    # 1. Constraint propagation
    print("1. Constraint Propagation:")
    print("   Reduce search space by propagating constraints")
    print("   Example: In Sudoku, eliminate impossible values early")
    
    # 2. Heuristics for variable ordering
    print("\n2. Variable Ordering Heuristics:")
    print("   Choose variables with minimum remaining values (MRV)")
    print("   Example: In Sudoku, fill cells with fewest possibilities first")
    
    # 3. Value ordering heuristics
    print("\n3. Value Ordering Heuristics:")
    print("   Choose values that are most constrained")
    print("   Example: In graph coloring, try colors used by neighbors first")
    
    # 4. Early termination
    print("\n4. Early Termination:")
    print("   Stop when first solution is found (if only one needed)")
    print("   Prune branches that cannot lead to valid solutions")

def backtracking_applications():
    """
    Demonstrate applications of backtracking
    """
    print("\n=== Backtracking Applications ===")
    
    # 1. Puzzle solving
    print("1. Puzzle Solving:")
    print("   - Sudoku, N-Queens, Crossword puzzles")
    print("   - Logic puzzles, Constraint satisfaction problems")
    
    # 2. Combinatorial optimization
    print("\n2. Combinatorial Optimization:")
    print("   - Traveling Salesman Problem (TSP)")
    print("   - Knapsack problem variants")
    print("   - Job scheduling with constraints")
    
    # 3. Game playing
    print("\n3. Game Playing:")
    print("   - Chess, Checkers move generation")
    print("   - Game tree search with pruning")
    print("   - Puzzle game solvers")
    
    # 4. Pattern matching
    print("\n4. Pattern Matching:")
    print("   - Regular expression matching")
    print("   - String matching with wildcards")
    print("   - Sequence alignment")

def data_science_applications():
    """
    Examples of backtracking in data science
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Feature selection
    print("1. Feature Selection:")
    print("   Backtracking for optimal feature subset selection")
    print("   Explore different combinations of features")
    print("   Evaluate model performance for each subset")
    
    # 2. Hyperparameter tuning
    print("\n2. Hyperparameter Tuning:")
    print("   Systematic search through hyperparameter space")
    print("   Early stopping when performance degrades")
    print("   Constraint-based pruning of search space")
    
    # 3. Decision tree construction
    print("\n3. Decision Tree Construction:")
    print("   Backtracking to find optimal splits")
    print("   Pruning branches that don't improve accuracy")
    print("   Constraint satisfaction for tree depth")
    
    # 4. Clustering validation
    print("\n4. Clustering Validation:")
    print("   Backtracking to find optimal cluster assignments")
    print("   Constraint-based clustering with must-link/cannot-link")
    print("   Validation of clustering results")

def performance_comparison():
    """
    Compare performance of backtracking with optimizations
    """
    print("\n=== Performance Comparison ===")
    
    import time
    
    # Test N-Queens with different board sizes
    sizes = [4, 6, 8]
    
    for size in sizes:
        print(f"\nTesting N-Queens with {size}x{size} board:")
        
        start = time.time()
        solutions = n_queens(size)
        solve_time = time.time() - start
        
        print(f"   Number of solutions: {len(solutions)}")
        print(f"   Time taken: {solve_time:.6f}s")
        
        if solutions:
            print(f"   First solution: {solutions[0]}")

# Example usage and testing
if __name__ == "__main__":
    # Backtracking demo
    backtracking_demo()
    print("\n" + "="*50 + "\n")
    
    # Sudoku demo
    sudoku_demo()
    print("\n" + "="*50 + "\n")
    
    # Optimization techniques
    backtracking_optimization()
    print("\n" + "="*50 + "\n")
    
    # Applications
    backtracking_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Classic backtracking problems (N-Queens, Sudoku, Subset Sum)")
    print("2. Graph problems using backtracking (Graph Coloring, Hamiltonian Cycle)")
    print("3. Optimization techniques for backtracking algorithms")
    print("4. Applications in computer science and data science")
    print("5. Performance characteristics of backtracking algorithms")
    print("\nKey takeaways:")
    print("- Backtracking systematically searches for solutions")
    print("- It's useful for constraint satisfaction problems")
    print("- Can be optimized with pruning and heuristics")
    print("- Exponential time complexity in worst case")
    print("- Essential for many puzzle-solving and optimization problems")