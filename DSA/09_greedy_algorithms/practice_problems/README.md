# Practice Problems - Greedy Algorithms

This folder contains practice problems to reinforce your understanding of Greedy Algorithms concepts. Each problem is designed to help you master different aspects of greedy approaches.

## Problem Sets

### Fundamentals
1. **Activity Selection** - Select maximum number of non-overlapping activities
2. **Fractional Knapsack** - Maximize value with fractional item selection
3. **Huffman Coding** - Generate optimal prefix codes for data compression
4. **Job Sequencing** - Schedule jobs to maximize profit with deadlines

### Scheduling and Intervals
5. **Meeting Rooms** - Find minimum number of meeting rooms required
6. **Interval Scheduling** - Maximize non-overlapping intervals
7. **Task Scheduling** - Schedule tasks with deadlines and profits
8. **Remove Overlapping Intervals** - Minimize intervals to remove for non-overlapping set

### Graph Algorithms
9. **Kruskal's MST** - Find minimum spanning tree using Union-Find
10. **Prim's MST** - Find minimum spanning tree using priority queue
11. **Dijkstra's Shortest Path** - Find shortest paths from source vertex
12. **Maximum Spanning Tree** - Find maximum spanning tree

### Advanced Applications
13. **Minimum Cost Connecting Points** - Connect all points with minimum cost
14. **Network Delay Time** - Find time for all nodes to receive signal
15. **Cheapest Flights with K Stops** - Find cheapest route with stop constraints
16. **Gas Station** - Find starting gas station to complete circuit

## Difficulty Levels

### Beginner (Problems 1-4)
- Focus on basic greedy concepts
- Simple greedy choice strategies
- Direct implementation of classic algorithms
- Linear or near-linear time complexity

### Intermediate (Problems 5-8)
- Interval and scheduling problems
- More complex problem modeling
- Multiple data structures usage
- Sorting and optimization techniques

### Advanced (Problems 9-16)
- Graph algorithms with greedy approaches
- Real-world optimization problems
- Complex constraints and variations
- Advanced data structures (Union-Find, Priority Queues)

## Solution Approach

For each problem, follow these steps:

1. **Problem Understanding**
   - Identify what needs to be optimized
   - Determine constraints and inputs/outputs
   - Look for examples and edge cases

2. **Greedy Strategy Identification**
   - Determine greedy choice property
   - Verify optimal substructure
   - Prove correctness if possible

3. **Algorithm Design**
   - Choose appropriate data structures
   - Handle edge cases
   - Consider time and space complexity

4. **Implementation**
   - Code the greedy approach
   - Test with provided examples
   - Verify correctness

5. **Testing and Analysis**
   - Test edge cases
   - Analyze time and space complexity
   - Compare with alternative approaches

## Tips for Success

1. **Identify Greedy Properties** - Check if problem has greedy choice and optimal substructure
2. **Sort Strategically** - Often the key to greedy algorithms is proper sorting
3. **Prove Correctness** - Try to prove why greedy choice leads to global optimum
4. **Handle Edge Cases** - Consider empty inputs, single elements, and boundary conditions
5. **Optimize Data Structures** - Use appropriate data structures (heaps, Union-Find) for efficiency

## Common Greedy Patterns

1. **Sorting + Greedy Choice** - Sort by some criteria and make locally optimal choices
2. **Priority Queue** - Use heap to always select the best available option
3. **Two Pointers** - Efficiently process sorted data
4. **Union-Find** - For connectivity and cycle detection problems

## Solutions

Refer to [problems.py](problems.py) for complete solutions with explanations.