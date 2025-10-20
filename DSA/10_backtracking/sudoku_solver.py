"""
Sudoku Solver

This module solves Sudoku puzzles using backtracking algorithm:
- Fill empty cells with digits 1-9
- Ensure no row, column, or 3x3 box has duplicate digits
"""

from typing import List


def solve_sudoku(board: List[List[str]]) -> bool:
    """
    Solve Sudoku puzzle using backtracking.
    
    Time Complexity: O(9^(n*n)) in worst case
    Space Complexity: O(n*n) for board + O(n*n) for recursion stack
    
    Args:
        board: 9x9 Sudoku board where '.' represents empty cell
        
    Returns:
        True if puzzle is solvable, False otherwise
    """
    def is_valid(board: List[List[str]], row: int, col: int, num: str) -> bool:
        """
        Check if placing num at (row, col) is valid.
        
        Args:
            board: Current Sudoku board
            row: Row index
            col: Column index
            num: Number to place ('1'-'9')
            
        Returns:
            True if placement is valid, False otherwise
        """
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def backtrack() -> bool:
        """
        Backtracking function to solve Sudoku.
        
        Returns:
            True if solution found, False otherwise
        """
        # Find empty cell
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    # Try digits 1-9
                    for num in '123456789':
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'  # Backtrack
                    return False  # No valid number found
        return True  # All cells filled
    
    return backtrack()


def solve_sudoku_optimized(board: List[List[str]]) -> bool:
    """
    Optimized Sudoku solver using constraint propagation.
    
    Time Complexity: O(9^(n*n)) but with better average performance
    Space Complexity: O(n*n) + O(n*n)
    
    Args:
        board: 9x9 Sudoku board
        
    Returns:
        True if puzzle is solvable, False otherwise
    """
    # Precompute possible values for each cell
    rows = [set('123456789') for _ in range(9)]
    cols = [set('123456789') for _ in range(9)]
    boxes = [set('123456789') for _ in range(9)]
    
    # Initialize constraints based on initial board
    empty_cells = []
    for i in range(9):
        for j in range(9):
            if board[i][j] != '.':
                num = board[i][j]
                rows[i].discard(num)
                cols[j].discard(num)
                boxes[(i // 3) * 3 + (j // 3)].discard(num)
            else:
                empty_cells.append((i, j))
    
    def get_possible_values(row: int, col: int) -> set:
        """Get possible values for a cell."""
        box_index = (row // 3) * 3 + (col // 3)
        return rows[row] & cols[col] & boxes[box_index]
    
    def backtrack_optimized() -> bool:
        """Optimized backtracking with constraint propagation."""
        # Find cell with minimum possible values (heuristic)
        min_cell = None
        min_values = None
        min_count = 10
        
        for row, col in empty_cells:
            if board[row][col] == '.':
                values = get_possible_values(row, col)
                if len(values) < min_count:
                    min_count = len(values)
                    min_cell = (row, col)
                    min_values = values
                
                # If no possible values, this path is invalid
                if min_count == 0:
                    return False
        
        # If no empty cells, puzzle is solved
        if min_cell is None:
            return True
        
        row, col = min_cell
        
        # Try each possible value
        for num in min_values:
            # Update constraints
            rows[row].discard(num)
            cols[col].discard(num)
            boxes[(row // 3) * 3 + (col // 3)].discard(num)
            board[row][col] = num
            
            # Recurse
            if backtrack_optimized():
                return True
            
            # Backtrack
            rows[row].add(num)
            cols[col].add(num)
            boxes[(row // 3) * 3 + (col // 3)].add(num)
            board[row][col] = '.'
        
        return False
    
    return backtrack_optimized()


def is_valid_sudoku(board: List[List[str]]) -> bool:
    """
    Validate if a Sudoku board is valid (doesn't violate rules).
    
    Time Complexity: O(n*n) where n=9
    Space Complexity: O(n)
    
    Args:
        board: 9x9 Sudoku board
        
    Returns:
        True if board is valid, False otherwise
    """
    # Check rows
    for i in range(9):
        seen = set()
        for j in range(9):
            if board[i][j] != '.':
                if board[i][j] in seen:
                    return False
                seen.add(board[i][j])
    
    # Check columns
    for j in range(9):
        seen = set()
        for i in range(9):
            if board[i][j] != '.':
                if board[i][j] in seen:
                    return False
                seen.add(board[i][j])
    
    # Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            seen = set()
            for i in range(box_row * 3, box_row * 3 + 3):
                for j in range(box_col * 3, box_col * 3 + 3):
                    if board[i][j] != '.':
                        if board[i][j] in seen:
                            return False
                        seen.add(board[i][j])
    
    return True


def generate_sudoku(difficulty: str = "medium") -> List[List[str]]:
    """
    Generate a Sudoku puzzle with specified difficulty.
    
    Time Complexity: O(9^(n*n)) for generation
    Space Complexity: O(n*n)
    
    Args:
        difficulty: "easy", "medium", or "hard"
        
    Returns:
        Generated Sudoku puzzle
    """
    import random
    
    # Start with empty board
    board = [['.' for _ in range(9)] for _ in range(9)]
    
    # Fill diagonal boxes (independent)
    def fill_diagonal():
        for i in range(0, 9, 3):
            fill_box(i, i)
    
    def fill_box(row_start: int, col_start: int):
        nums = list('123456789')
        random.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[row_start + i][col_start + j] = nums[i * 3 + j]
    
    def find_unassigned_location():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    return i, j
        return -1, -1
    
    def solve_temp():
        row, col = find_unassigned_location()
        if row == -1:
            return True
        
        nums = list('123456789')
        random.shuffle(nums)
        for num in nums:
            if is_valid(board, row, col, num):
                board[row][col] = num
                if solve_temp():
                    return True
                board[row][col] = '.'
        return False
    
    def is_valid(b, r, c, num):
        # Check row
        for j in range(9):
            if b[r][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if b[i][c] == num:
                return False
        
        # Check box
        box_row = (r // 3) * 3
        box_col = (c // 3) * 3
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if b[i][j] == num:
                    return False
        
        return True
    
    # Generate complete solution
    fill_diagonal()
    solve_temp()
    
    # Remove numbers based on difficulty
    cells_to_remove = {"easy": 40, "medium": 50, "hard": 60}
    remove_count = cells_to_remove.get(difficulty, 50)
    
    # Remove cells
    removed = 0
    while removed < remove_count:
        row = random.randint(0, 8)
        col = random.randint(0, 8)
        if board[row][col] != '.':
            board[row][col] = '.'
            removed += 1
    
    return board


def print_board(board: List[List[str]]):
    """
    Print Sudoku board in readable format.
    
    Args:
        board: 9x9 Sudoku board
    """
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("| ", end="")
            print(board[i][j] + " ", end="")
        print()


def demo():
    """Demonstrate the Sudoku solver."""
    print("=== Sudoku Solver ===\n")
    
    # Test Sudoku puzzle
    board = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"]
    ]
    
    print("Original Sudoku puzzle:")
    print_board(board)
    
    # Solve the puzzle
    if solve_sudoku(board):
        print("\nSolved Sudoku:")
        print_board(board)
    else:
        print("\nNo solution exists!")
    
    print("\n" + "="*40)
    
    # Test validation
    valid = is_valid_sudoku(board)
    print(f"\nIs the solved board valid? {valid}")
    
    # Test with invalid board
    invalid_board = [
        ["5", "3", "5", ".", "7", ".", ".", ".", "."],  # Duplicate in row
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"]
    ]
    
    valid = is_valid_sudoku(invalid_board)
    print(f"Is the invalid board valid? {valid}")


if __name__ == "__main__":
    demo()