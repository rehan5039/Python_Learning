# Chapter 9: Greedy Algorithms

## Overview
Greedy algorithms make the locally optimal choice at each step with the hope of finding a global optimum. While not all problems can be solved optimally with greedy approaches, many important problems can be solved efficiently using this technique. This chapter explores when and how to apply greedy algorithms effectively.

## Topics Covered
- Greedy algorithm principles and characteristics
- Activity selection problem
- Fractional knapsack problem
- Huffman coding
- Minimum spanning tree (Prim's and Kruskal's algorithms)
- Dijkstra's shortest path algorithm
- Job sequencing problem
- Coin change problem (greedy variant)
- Interval scheduling optimization

## Learning Objectives
By the end of this chapter, you should be able to:
- Identify problems suitable for greedy algorithms
- Understand the difference between greedy and dynamic programming approaches
- Implement classic greedy algorithms
- Analyze correctness and complexity of greedy solutions
- Apply greedy techniques to real-world scenarios
- Recognize when greedy algorithms fail to produce optimal solutions

## Prerequisites
- Understanding of basic data structures (arrays, lists, priority queues)
- Knowledge of sorting algorithms
- Familiarity with graph representations
- Basic Python programming skills
- Understanding of algorithm analysis (Big O notation)

## Content Files
- [greedy_fundamentals.py](greedy_fundamentals.py) - Introduction to greedy concepts
- [activity_selection.py](activity_selection.py) - Activity selection and scheduling problems
- [minimum_spanning_tree.py](minimum_spanning_tree.py) - MST algorithms (Prim's, Kruskal's)
- [shortest_path.py](shortest_path.py) - Dijkstra's algorithm and variants
- [practice_problems/](practice_problems/) - Exercises to reinforce learning
  - [problems.py](practice_problems/problems.py) - Practice problems with solutions
  - [README.md](practice_problems/README.md) - Practice problem descriptions

## Real-World Applications
- **Resource Allocation**: CPU scheduling, memory management
- **Network Design**: Routing protocols, bandwidth allocation
- **Data Compression**: Huffman coding for file compression
- **Financial Systems**: Portfolio optimization, investment strategies
- **Supply Chain**: Inventory management, distribution optimization
- **Telecommunications**: Frequency allocation, network routing
- **Transportation**: Route planning, traffic optimization

## Key Characteristics of Greedy Algorithms
1. **Greedy Choice Property**: A global optimum can be arrived at by making locally optimal choices
2. **Optimal Substructure**: An optimal solution to the problem contains optimal solutions to subproblems

## Example: Activity Selection Problem
```python
def activity_selection(activities):
    """
    Select maximum number of activities that don't overlap.
    
    Args:
        activities: List of tuples (start_time, end_time, activity_name)
        
    Returns:
        List of selected activities
    """
    # Sort activities by end time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_end_time = activities[0][1]
    
    for i in range(1, len(activities)):
        # If current activity starts after last selected activity ends
        if activities[i][0] >= last_end_time:
            selected.append(activities[i])
            last_end_time = activities[i][1]
    
    return selected

# Example usage
activities = [
    (1, 4, "Activity A"),
    (3, 5, "Activity B"),
    (0, 6, "Activity C"),
    (5, 7, "Activity D"),
    (3, 9, "Activity E"),
    (5, 9, "Activity F"),
    (6, 10, "Activity G"),
    (8, 11, "Activity H"),
    (8, 12, "Activity I"),
    (2, 14, "Activity J"),
    (12, 16, "Activity K")
]

selected = activity_selection(activities)
print("Selected activities:", selected)
```

## When Greedy Algorithms Work
Greedy algorithms work when:
1. The problem has the greedy choice property
2. The problem has optimal substructure
3. Local optimal choices lead to global optimum

## When Greedy Algorithms Don't Work
Greedy algorithms fail for problems like:
- 0/1 Knapsack Problem (requires dynamic programming)
- Longest Path Problem (requires exhaustive search)
- Graph Coloring (requires backtracking)

## Next Chapter
[Chapter 10: Backtracking](../10_backtracking/)