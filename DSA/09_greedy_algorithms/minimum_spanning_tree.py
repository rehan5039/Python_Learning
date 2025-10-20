"""
Minimum Spanning Tree Algorithms

This module covers Minimum Spanning Tree (MST) algorithms:
- Kruskal's Algorithm
- Prim's Algorithm
- Applications and variations
"""

import heapq
from typing import List, Tuple, Dict


class UnionFind:
    """Union-Find (Disjoint Set) data structure for Kruskal's algorithm."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union x and y. Returns True if they were different sets."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True


def kruskal_mst(vertices: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    Find Minimum Spanning Tree using Kruskal's algorithm.
    
    Time Complexity: O(E log E) where E is number of edges
    Space Complexity: O(V) where V is number of vertices
    
    Args:
        vertices: Number of vertices in the graph
        edges: List of tuples (vertex1, vertex2, weight)
        
    Returns:
        Tuple of (total_weight, list_of_mst_edges)
    """
    if not edges:
        return 0, []
    
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    # Initialize Union-Find
    uf = UnionFind(vertices)
    
    mst_edges = []
    total_weight = 0
    
    # Process edges in order of increasing weight
    for u, v, weight in edges:
        # If including this edge doesn't create a cycle
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            
            # If we have V-1 edges, we have a spanning tree
            if len(mst_edges) == vertices - 1:
                break
    
    return total_weight, mst_edges


def prims_mst(vertices: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    Find Minimum Spanning Tree using Prim's algorithm.
    
    Time Complexity: O(E log V) where E is edges and V is vertices
    Space Complexity: O(V + E)
    
    Args:
        vertices: Number of vertices in the graph
        edges: List of tuples (vertex1, vertex2, weight)
        
    Returns:
        Tuple of (total_weight, list_of_mst_edges)
    """
    if not edges or vertices <= 0:
        return 0, []
    
    # Build adjacency list representation
    graph = [[] for _ in range(vertices)]
    for u, v, weight in edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))
    
    # Initialize data structures
    visited = [False] * vertices
    min_heap = [(0, 0, -1)]  # (weight, vertex, parent)
    mst_edges = []
    total_weight = 0
    
    while min_heap and len(mst_edges) < vertices - 1:
        weight, u, parent = heapq.heappop(min_heap)
        
        # Skip if already visited
        if visited[u]:
            continue
        
        # Mark as visited
        visited[u] = True
        total_weight += weight
        
        # Add edge to MST (except for the initial vertex)
        if parent != -1:
            mst_edges.append((parent, u, weight))
        
        # Add adjacent vertices to heap
        for v, edge_weight in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (edge_weight, v, u))
    
    return total_weight, mst_edges


def maximum_spanning_tree(vertices: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    Find Maximum Spanning Tree by negating weights and using MST algorithm.
    
    Time Complexity: O(E log E)
    Space Complexity: O(V)
    
    Args:
        vertices: Number of vertices in the graph
        edges: List of tuples (vertex1, vertex2, weight)
        
    Returns:
        Tuple of (total_weight, list_of_mst_edges)
    """
    # Negate all weights and find MST
    negated_edges = [(u, v, -weight) for u, v, weight in edges]
    neg_total, neg_edges = kruskal_mst(vertices, negated_edges)
    
    # Convert back to positive weights
    max_edges = [(u, v, -weight) for u, v, weight in neg_edges]
    max_total = -neg_total
    
    return max_total, max_edges


def second_best_mst(vertices: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    Find Second Best Minimum Spanning Tree.
    
    Time Complexity: O(E log E + V * E)
    Space Complexity: O(V + E)
    
    Args:
        vertices: Number of vertices in the graph
        edges: List of tuples (vertex1, vertex2, weight)
        
    Returns:
        Tuple of (total_weight, list_of_mst_edges)
    """
    # Find the MST first
    mst_weight, mst_edges = kruskal_mst(vertices, edges)
    
    if not mst_edges:
        return 0, []
    
    # Create a set of MST edges for quick lookup
    mst_edge_set = set()
    for u, v, weight in mst_edges:
        mst_edge_set.add((min(u, v), max(u, v)))
    
    second_best_weight = float('inf')
    second_best_edges = []
    
    # Try removing each MST edge and find new MST
    for remove_u, remove_v, remove_weight in mst_edges:
        # Create edge list without the removed edge
        filtered_edges = []
        for u, v, weight in edges:
            if (min(u, v), max(u, v)) != (min(remove_u, remove_v), max(remove_u, remove_v)):
                filtered_edges.append((u, v, weight))
        
        # Find MST of the filtered graph
        new_weight, new_edges = kruskal_mst(vertices, filtered_edges)
        
        # Update second best if this is better
        if new_weight < second_best_weight and len(new_edges) == vertices - 1:
            second_best_weight = new_weight
            second_best_edges = new_edges
    
    return second_best_weight, second_best_edges


def minimum_cost_connecting_points(points: List[List[int]]) -> int:
    """
    Find minimum cost to connect all points (Minimum Spanning Tree on complete graph).
    
    Time Complexity: O(n^2 log n) where n is number of points
    Space Complexity: O(n^2)
    
    Args:
        points: List of [x, y] coordinates
        
    Returns:
        Minimum cost to connect all points
    """
    if len(points) <= 1:
        return 0
    
    n = len(points)
    
    # Generate all edges with Manhattan distances
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # Manhattan distance
            distance = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            edges.append((i, j, distance))
    
    # Find MST
    total_cost, _ = kruskal_mst(n, edges)
    return total_cost


def demo():
    """Demonstrate the MST algorithms."""
    print("=== Minimum Spanning Tree Algorithms ===\n")
    
    # Test Kruskal's Algorithm
    vertices = 4
    edges = [
        (0, 1, 10),
        (0, 2, 6),
        (0, 3, 5),
        (1, 3, 15),
        (2, 3, 4)
    ]
    
    weight, mst_edges = kruskal_mst(vertices, edges)
    print("Kruskal's Algorithm:")
    print(f"  Vertices: {vertices}")
    print(f"  Edges: {edges}")
    print(f"  MST Weight: {weight}")
    print(f"  MST Edges: {mst_edges}")
    
    print("\n" + "="*50)
    
    # Test Prim's Algorithm
    weight, mst_edges = prims_mst(vertices, edges)
    print("\nPrim's Algorithm:")
    print(f"  Vertices: {vertices}")
    print(f"  Edges: {edges}")
    print(f"  MST Weight: {weight}")
    print(f"  MST Edges: {mst_edges}")
    
    print("\n" + "="*50)
    
    # Test Maximum Spanning Tree
    max_weight, max_edges = maximum_spanning_tree(vertices, edges)
    print(f"\nMaximum Spanning Tree:")
    print(f"  Vertices: {vertices}")
    print(f"  Edges: {edges}")
    print(f"  Maximum Weight: {max_weight}")
    print(f"  MST Edges: {max_edges}")
    
    print("\n" + "="*50)
    
    # Test Minimum Cost Connecting Points
    points = [[0, 0], [2, 2], [3, 10], [5, 2], [7, 0]]
    min_cost = minimum_cost_connecting_points(points)
    print(f"\nMinimum Cost Connecting Points:")
    print(f"  Points: {points}")
    print(f"  Minimum Cost: {min_cost}")
    
    print("\n" + "="*50)
    
    # Test Second Best MST
    second_weight, second_edges = second_best_mst(vertices, edges)
    print(f"\nSecond Best MST:")
    print(f"  Vertices: {vertices}")
    print(f"  Edges: {edges}")
    print(f"  Second Best Weight: {second_weight}")
    print(f"  MST Edges: {second_edges}")


if __name__ == "__main__":
    demo()