# Practice Problems - Backtracking

This folder contains practice problems to reinforce your understanding of Backtracking algorithms. Each problem is designed to help you master different aspects of backtracking techniques.

## Problem Sets

### Fundamentals
1. **Subsets** - Generate all possible subsets of a set
2. **Subsets with Duplicates** - Generate unique subsets when input has duplicates
3. **Permutations** - Generate all possible permutations of a set
4. **Permutations with Duplicates** - Generate unique permutations when input has duplicates
5. **Combination Sum** - Find combinations that sum to target (with repetition)
6. **Combination Sum II** - Find combinations that sum to target (without repetition)

### Combinatorial Search
7. **Generate Parentheses** - Generate all valid combinations of parentheses
8. **Letter Combinations** - Generate letter combinations for phone digits
9. **Combinations** - Generate all combinations of k numbers from 1 to n
10. **Combination Sum III** - Find combinations of k numbers that sum to n
11. **Partition Equal Subset Sum** - Determine if array can be partitioned into equal sum subsets
12. **Palindrome Partitioning** - Partition string into palindromic substrings

### Classic Problems
13. **N-Queens** - Place N queens on chessboard without attacking each other
14. **Sudoku Solver** - Solve Sudoku puzzle
15. **Word Search** - Find word in 2D grid of characters
16. **Word Search II** - Find multiple words in 2D grid

### Graph Problems
17. **Hamiltonian Path** - Find path visiting each vertex exactly once
18. **Graph Coloring** - Color graph vertices with minimum colors
19. **Knight's Tour** - Find path for knight visiting each square exactly once
20. **Maze Solving** - Find path through maze

## Difficulty Levels

### Beginner (Problems 1-6)
- Focus on basic backtracking concepts
- Simple state management
- Direct implementation of classic algorithms
- Linear recursion patterns

### Intermediate (Problems 7-12)
- Combinatorial search problems
- More complex constraint handling
- Pruning and optimization techniques
- Memoization and caching

### Advanced (Problems 13-20)
- Classic backtracking problems
- Complex state space trees
- Advanced pruning techniques
- Real-world applications

## Solution Approach

For each problem, follow these steps:

1. **Problem Understanding**
   - Identify what needs to be generated or searched
   - Determine constraints and inputs/outputs
   - Look for examples and edge cases

2. **Backtracking Framework**
   - Define the state representation
   - Identify the choice points
   - Determine the constraints
   - Establish the base case

3. **Implementation**
   - Code the recursive backtracking function
   - Handle state changes and backtracking
   - Apply pruning techniques
   - Optimize with heuristics

4. **Testing and Analysis**
   - Test with provided examples
   - Analyze time and space complexity
   - Verify correctness with edge cases

## Tips for Success

1. **Identify Choice Points** - Determine what decisions need to be made at each step
2. **Define State** - Clearly represent the current state of the solution
3. **Apply Constraints Early** - Check constraints as soon as possible to prune search space
4. **Use Appropriate Data Structures** - Choose data structures that support efficient operations
5. **Optimize with Pruning** - Eliminate branches that cannot lead to valid solutions
6. **Handle Duplicates** - Use sorting and careful indexing to avoid duplicate solutions

## Common Backtracking Patterns

1. **Subset Generation** - For each element, choose to include or exclude
2. **Permutation Generation** - For each position, choose an unused element
3. **Combination Generation** - Choose elements in increasing order to avoid duplicates
4. **Grid Search** - Explore neighbors in 2D grid with backtracking
5. **Constraint Satisfaction** - Systematically try values while maintaining constraints

## Solutions

Refer to [problems.py](problems.py) for complete solutions with explanations.