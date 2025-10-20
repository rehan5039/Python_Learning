"""
N-Queens Problem

This module solves the classic N-Queens problem using backtracking:
- Place N queens on an N×N chessboard so that no two queens attack each other
- Queens can attack horizontally, vertically, and diagonally
"""

from typing import List


def solve_n_queens(n: int) -> List[List[str]]:
    """
    Solve the N-Queens problem and return all distinct solutions.
    
    Time Complexity: O(N!) in worst case
    Space Complexity: O(N^2) for board + O(N) for recursion stack
    
    Args:
        n: Size of chessboard and number of queens
        
    Returns:
        List of all valid board configurations
    """
    def is_safe(board: List[int], row: int, col: int) -> bool:
        """
        Check if placing a queen at (row, col) is safe.
        
        Args:
            board: List where board[i] represents column of queen in row i
            row: Current row
            col: Current column
            
        Returns:
            True if position is safe, False otherwise
        """
        # Check column
        for i in range(row):
            if board[i] == col:
                return False
        
        # Check diagonals
        for i in range(row):
            # Check both diagonals: |row - i| == |col - board[i]|
            if abs(row - i) == abs(col - board[i]):
                return False
        
        return True
    
    def backtrack(board: List[int], row: int, solutions: List[List[str]]):
        """
        Backtracking function to find all solutions.
        
        Args:
            board: Current board state
            row: Current row to place queen
            solutions: List to store valid solutions
        """
        # Base case: all queens placed
        if row == n:
            # Convert board representation to string format
            solution = []
            for i in range(n):
                row_str = ""
                for j in range(n):
                    if board[i] == j:
                        row_str += "Q"
                    else:
                        row_str += "."
                solution.append(row_str)
            solutions.append(solution)
            return
        
        # Try placing queen in each column of current row
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col  # Place queen
                backtrack(board, row + 1, solutions)
                board[row] = -1   # Backtrack
    
    # Initialize board: board[i] = column of queen in row i
    board = [-1] * n
    solutions = []
    backtrack(board, 0, solutions)
    return solutions


def total_n_queens(n: int) -> int:
    """
    Count the total number of distinct solutions to N-Queens problem.
    
    Time Complexity: O(N!)
    Space Complexity: O(N) for recursion stack
    
    Args:
        n: Size of chessboard and number of queens
        
    Returns:
        Total number of distinct solutions
    """
    def is_safe(board: List[int], row: int, col: int) -> bool:
        for i in range(row):
            if board[i] == col or abs(row - i) == abs(col - board[i]):
                return False
        return True
    
    def backtrack(board: List[int], row: int) -> int:
        if row == n:
            return 1
        
        count = 0
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                count += backtrack(board, row + 1)
                board[row] = -1
        
        return count
    
    board = [-1] * n
    return backtrack(board, 0)


def solve_n_queens_optimized(n: int) -> List[List[str]]:
    """
    Optimized solution using bit manipulation for faster constraint checking.
    
    Time Complexity: O(N!)
    Space Complexity: O(N) for recursion stack
    
    Args:
        n: Size of chessboard and number of queens
        
    Returns:
        List of all valid board configurations
    """
    def backtrack(row: int, columns: List[int], diag1: int, diag2: int, solutions: List[List[str]]):
        if row == n:
            # Convert to string representation
            solution = []
            for col in columns:
                row_str = "." * col + "Q" + "." * (n - col - 1)
                solution.append(row_str)
            solutions.append(solution)
            return
        
        # Calculate available positions
        available = ((1 << n) - 1) & (~(columns[row] | diag1 | diag2))
        
        while available:
            # Get the rightmost available position
            position = available & (-available)
            available &= available - 1  # Remove this position
            
            # Calculate column index
            col = bin(position - 1).count('1')
            
            # Update constraints
            new_columns = columns[:]
            new_columns[row] = col
            
            backtrack(row + 1, new_columns, (diag1 | position) << 1, (diag2 | position) >> 1, solutions)
    
    solutions = []
    columns = [0] * n
    backtrack(0, columns, 0, 0, solutions)
    return solutions


def is_valid_n_queens(board: List[str]) -> bool:
    """
    Validate if a given board configuration is a valid N-Queens solution.
    
    Time Complexity: O(N^2)
    Space Complexity: O(N)
    
    Args:
        board: List of strings representing board configuration
        
    Returns:
        True if valid solution, False otherwise
    """
    n = len(board)
    if n == 0:
        return False
    
    # Check each row has exactly one queen
    queens = []
    for i in range(n):
        row = board[i]
        if len(row) != n:
            return False
        
        queen_count = 0
        for j in range(n):
            if row[j] == 'Q':
                queen_count += 1
                queens.append((i, j))
        
        if queen_count != 1:
            return False
    
    # Check if queens attack each other
    for i in range(len(queens)):
        for j in range(i + 1, len(queens)):
            row1, col1 = queens[i]
            row2, col2 = queens[j]
            
            # Check same row (already handled above)
            # Check same column
            if col1 == col2:
                return False
            
            # Check diagonals
            if abs(row1 - row2) == abs(col1 - col2):
                return False
    
    return True


def n_queens_all_distances(n: int) -> List[int]:
    """
    Calculate distances between queens for all solutions.
    
    Time Complexity: O(N! * N^2)
    Space Complexity: O(N! * N)
    
    Args:
        n: Size of chessboard
        
    Returns:
        List of distances between all pairs of queens across all solutions
    """
    solutions = solve_n_queens(n)
    all_distances = []
    
    for solution in solutions:
        queens = []
        for i in range(n):
            for j in range(n):
                if solution[i][j] == 'Q':
                    queens.append((i, j))
        
        # Calculate distances between all pairs
        for i in range(len(queens)):
            for j in range(i + 1, len(queens)):
                row1, col1 = queens[i]
                row2, col2 = queens[j]
                distance = ((row1 - row2) ** 2 + (col1 - col2) ** 2) ** 0.5
                all_distances.append(distance)
    
    return all_distances


def demo():
    """Demonstrate the N-Queens problem solutions."""
    print("=== N-Queens Problem ===\n")
    
    # Test 4-Queens
    n = 4
    solutions = solve_n_queens(n)
    print(f"4-Queens Problem:")
    print(f"  Total solutions: {len(solutions)}")
    
    # Display first solution
    if solutions:
        print(f"  First solution:")
        for row in solutions[0]:
            print(f"    {row}")
    
    print("\n" + "="*40)
    
    # Test Total N Queens
    for i in range(1, 6):
        count = total_n_queens(i)
        print(f"  {i}-Queens: {count} solutions")
    
    print("\n" + "="*40)
    
    # Test Validation
    valid_board = [
        ".Q..",
        "...Q",
        "Q...",
        "..Q."
    ]
    is_valid = is_valid_n_queens(valid_board)
    print(f"\nValidation of valid board: {is_valid}")
    
    invalid_board = [
        "QQ..",
        "...Q",
        "Q...",
        "..Q."
    ]
    is_valid = is_valid_n_queens(invalid_board)
    print(f"Validation of invalid board: {is_valid}")
    
    print("\n" + "="*40)
    
    # Test Distances
    distances = n_queens_all_distances(4)
    print(f"\nDistances between queens in 4-Queens solutions:")
    print(f"  Total distances calculated: {len(distances)}")
    if distances:
        print(f"  Sample distances: {distances[:5]}")


if __name__ == "__main__":
    demo()