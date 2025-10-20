"""
Algorithm Design Techniques - Greedy Algorithms
========================================

This module provides implementations and examples of greedy algorithms,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Greedy algorithm paradigm
- Classic greedy problems (activity selection, Huffman coding)
- Greedy vs optimal solutions
- Problem-solving strategies
- Applications in data science
"""

from typing import List, Tuple, Dict, Set
import heapq

def activity_selection(activities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Select maximum number of non-overlapping activities
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    
    Args:
        activities: List of (start_time, end_time, name) tuples
    
    Returns:
        List of selected activities
    """
    # Sort activities by end time
    sorted_activities = sorted(activities, key=lambda x: x[1])
    
    selected = []
    last_end_time = 0
    
    for start, end, name in sorted_activities:
        # If activity starts after last selected activity ends
        if start >= last_end_time:
            selected.append((start, end, name))
            last_end_time = end
    
    return selected

def fractional_knapsack(weights: List[int], values: List[int], capacity: int) -> float:
    """
    Solve fractional knapsack problem using greedy approach
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Args:
        weights: List of item weights
        values: List of item values
        capacity: Knapsack capacity
    
    Returns:
        Maximum value that can be obtained
    """
    # Calculate value-to-weight ratios
    items = [(values[i] / weights[i], weights[i], values[i]) for i in range(len(weights))]
    
    # Sort by value-to-weight ratio in descending order
    items.sort(reverse=True)
    
    total_value = 0.0
    remaining_capacity = capacity
    
    for ratio, weight, value in items:
        if remaining_capacity >= weight:
            # Take entire item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction of item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            break
    
    return total_value

def huffman_coding(frequencies: Dict[str, int]) -> Dict[str, str]:
    """
    Generate Huffman codes for characters based on frequencies
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Args:
        frequencies: Dictionary mapping characters to their frequencies
    
    Returns:
        Dictionary mapping characters to their Huffman codes
    """
    import heapq
    
    class Node:
        def __init__(self, char: str = None, freq: int = 0, left=None, right=None):
            self.char = char
            self.freq = freq
            self.left = left
            self.right = right
        
        def __lt__(self, other):
            return self.freq < other.freq
    
    # Create leaf nodes for each character
    heap = [Node(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        # Extract two nodes with minimum frequency
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create internal node with combined frequency
        merged = Node(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    # Generate Huffman codes
    def generate_codes(node: Node, code: str = "", codes: Dict[str, str] = None) -> Dict[str, str]:
        if codes is None:
            codes = {}
        
        if node.char:  # Leaf node
            codes[node.char] = code if code else "0"  # Handle single character case
        else:
            generate_codes(node.left, code + "0", codes)
            generate_codes(node.right, code + "1", codes)
        
        return codes
    
    # Get root node and generate codes
    root = heap[0] if heap else None
    return generate_codes(root) if root else {}

def dijkstra_shortest_path(graph: Dict[str, List[Tuple[str, int]]], start: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Find shortest paths from start vertex using Dijkstra's greedy algorithm
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)
    
    Args:
        graph: Adjacency list representation {vertex: [(neighbor, weight), ...]}
        start: Starting vertex
    
    Returns:
        Tuple of (distances, previous_vertices)
    """
    # Initialize distances and previous vertices
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    previous = {vertex: None for vertex in graph}
    
    # Priority queue: (distance, vertex)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        # Skip if already processed
        if current_vertex in visited:
            continue
        
        visited.add(current_vertex)
        
        # Check neighbors
        for neighbor, weight in graph.get(current_vertex, []):
            if neighbor not in visited:
                new_distance = current_distance + weight
                
                # Found shorter path
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_vertex
                    heapq.heappush(pq, (new_distance, neighbor))
    
    return distances, previous

def kruskal_mst(edges: List[Tuple[int, int, int]], num_vertices: int) -> List[Tuple[int, int, int]]:
    """
    Find Minimum Spanning Tree using Kruskal's greedy algorithm
    Time Complexity: O(E log E)
    Space Complexity: O(V)
    
    Args:
        edges: List of (vertex1, vertex2, weight) tuples
        num_vertices: Number of vertices in graph
    
    Returns:
        List of edges in MST
    """
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])  # Path compression
            return self.parent[x]
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            # Union by rank
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True
    
    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda x: x[2])
    
    # Initialize Union-Find
    uf = UnionFind(num_vertices)
    
    mst = []
    for u, v, weight in sorted_edges:
        # If including this edge doesn't create a cycle
        if uf.union(u, v):
            mst.append((u, v, weight))
            # Stop when we have V-1 edges
            if len(mst) == num_vertices - 1:
                break
    
    return mst

def greedy_algorithms_demo():
    """
    Demonstrate greedy algorithms and their applications
    """
    print("=== Greedy Algorithms Demo ===")
    
    # Activity selection
    print("1. Activity Selection:")
    activities = [
        (1, 4, "A1"), (3, 5, "A2"), (0, 6, "A3"),
        (5, 7, "A4"), (3, 9, "A5"), (5, 9, "A6"),
        (6, 10, "A7"), (8, 11, "A8"), (8, 12, "A9"),
        (2, 14, "A10"), (12, 16, "A11")
    ]
    print(f"   Activities (start, end, name): {activities}")
    selected = activity_selection(activities)
    print(f"   Selected activities: {selected}")
    print(f"   Number of activities selected: {len(selected)}")
    
    # Fractional knapsack
    print("\n2. Fractional Knapsack:")
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    print(f"   Weights: {weights}")
    print(f"   Values: {values}")
    print(f"   Capacity: {capacity}")
    max_value = fractional_knapsack(weights, values, capacity)
    print(f"   Maximum value: {max_value}")
    
    # Huffman coding
    print("\n3. Huffman Coding:")
    frequencies = {'a': 45, 'b': 13, 'c': 12, 'd': 16, 'e': 9, 'f': 5}
    print(f"   Character frequencies: {frequencies}")
    codes = huffman_coding(frequencies)
    print(f"   Huffman codes: {codes}")
    
    # Calculate average code length
    total_chars = sum(frequencies.values())
    avg_length = sum(len(codes[char]) * freq for char, freq in frequencies.items()) / total_chars
    print(f"   Average code length: {avg_length:.2f} bits per character")

def graph_algorithms_demo():
    """
    Demonstrate graph algorithms using greedy approach
    """
    print("\n=== Graph Algorithms Demo ===")
    
    # Dijkstra's algorithm
    print("1. Dijkstra's Shortest Path:")
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 1), ('D', 5)],
        'C': [('D', 8), ('E', 10)],
        'D': [('E', 2)],
        'E': []
    }
    print(f"   Graph: {graph}")
    distances, previous = dijkstra_shortest_path(graph, 'A')
    print(f"   Shortest distances from A: {distances}")
    
    # Reconstruct path to E
    path = []
    current = 'E'
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    print(f"   Shortest path to E: {' -> '.join(path)}")
    
    # Kruskal's MST
    print("\n2. Kruskal's Minimum Spanning Tree:")
    edges = [
        (0, 1, 10), (0, 2, 6), (0, 3, 5),
        (1, 3, 15), (2, 3, 4)
    ]
    num_vertices = 4
    print(f"   Edges (u, v, weight): {edges}")
    mst = kruskal_mst(edges, num_vertices)
    print(f"   MST edges: {mst}")
    total_weight = sum(weight for _, _, weight in mst)
    print(f"   Total MST weight: {total_weight}")

def greedy_vs_optimal():
    """
    Compare greedy solutions with optimal solutions
    """
    print("\n=== Greedy vs Optimal Solutions ===")
    
    # 0/1 Knapsack (greedy doesn't work optimally)
    print("1. 0/1 Knapsack Problem:")
    print("   Greedy approach (by value/weight ratio) doesn't guarantee optimal solution")
    print("   Example:")
    print("   Items: [(weight=10, value=60), (weight=20, value=100), (weight=30, value=120)]")
    print("   Capacity: 50")
    print("   Greedy (fractional): Value = 240")
    print("   Optimal (0/1): Value = 220 (take items 2 and 3)")
    print("   Note: For 0/1 knapsack, dynamic programming is needed for optimal solution")
    
    # Activity selection (greedy works optimally)
    print("\n2. Activity Selection Problem:")
    print("   Greedy approach (select by earliest end time) is optimal")
    print("   Proof: Greedy choice property and optimal substructure")
    
    # Huffman coding (greedy works optimally)
    print("\n3. Huffman Coding:")
    print("   Greedy approach produces optimal prefix codes")
    print("   Minimizes expected code length")

def greedy_applications():
    """
    Demonstrate applications of greedy algorithms
    """
    print("\n=== Greedy Algorithm Applications ===")
    
    # 1. Job scheduling
    print("1. Job Scheduling:")
    print("   Minimize maximum lateness")
    print("   Schedule jobs by earliest deadline first")
    
    # 2. Data compression
    print("\n2. Data Compression:")
    print("   Huffman coding for lossless compression")
    print("   Used in ZIP files, JPEG images")
    
    # 3. Network design
    print("\n3. Network Design:")
    print("   Minimum Spanning Tree for connecting all nodes with minimum cost")
    print("   Used in telecommunications, transportation networks")
    
    # 4. Resource allocation
    print("\n4. Resource Allocation:")
    print("   Allocate resources to maximize total value")
    print("   Used in cloud computing, project management")

def data_science_applications():
    """
    Examples of greedy algorithms in data science
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Feature selection
    print("1. Feature Selection:")
    print("   Greedy forward/backward selection")
    print("   Add/remove features based on improvement in model performance")
    
    # 2. Clustering initialization
    print("\n2. Clustering Initialization:")
    print("   K-means++ uses greedy approach to initialize centroids")
    print("   Selects centroids to be far apart")
    
    # 3. Decision tree splitting
    print("\n3. Decision Tree Splitting:")
    print("   Greedy approach to select best split at each node")
    print("   Choose feature and threshold that maximize information gain")
    
    # 4. Recommendation systems
    print("\n4. Recommendation Systems:")
    print("   Greedy algorithms for top-k recommendations")
    print("   Select items with highest predicted ratings")

def performance_comparison():
    """
    Compare performance of different greedy algorithms
    """
    print("\n=== Performance Comparison ===")
    
    import time
    import random
    
    # Test with different problem sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nTesting with {size} elements:")
        
        # Activity selection
        activities = [(random.randint(0, 100), random.randint(101, 200), f"A{i}") for i in range(size)]
        start = time.time()
        selected = activity_selection(activities)
        activity_time = time.time() - start
        print(f"   Activity Selection: {activity_time:.6f}s, Selected: {len(selected)}")
        
        # Fractional knapsack
        weights = [random.randint(1, 100) for _ in range(size)]
        values = [random.randint(1, 100) for _ in range(size)]
        capacity = sum(weights) // 2
        start = time.time()
        max_value = fractional_knapsack(weights, values, capacity)
        knapsack_time = time.time() - start
        print(f"   Fractional Knapsack: {knapsack_time:.6f}s")
        
        # Huffman coding
        frequencies = {chr(65 + i): random.randint(1, 100) for i in range(min(size, 26))}
        start = time.time()
        codes = huffman_coding(frequencies)
        huffman_time = time.time() - start
        print(f"   Huffman Coding: {huffman_time:.6f}s")

# Example usage and testing
if __name__ == "__main__":
    # Greedy algorithms demo
    greedy_algorithms_demo()
    print("\n" + "="*50 + "\n")
    
    # Graph algorithms demo
    graph_algorithms_demo()
    print("\n" + "="*50 + "\n")
    
    # Greedy vs optimal
    greedy_vs_optimal()
    print("\n" + "="*50 + "\n")
    
    # Applications
    greedy_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Classic greedy algorithms (activity selection, knapsack, Huffman coding)")
    print("2. Graph algorithms using greedy approach (Dijkstra, Kruskal)")
    print("3. When greedy algorithms work optimally vs suboptimally")
    print("4. Applications in computer science and data science")
    print("5. Performance characteristics of greedy algorithms")
    print("\nKey takeaways:")
    print("- Greedy algorithms make locally optimal choices")
    print("- They work optimally for some problems (activity selection, MST, Huffman)")
    print("- They may not work optimally for others (0/1 knapsack)")
    print("- Easy to implement and often efficient")
    print("- Require mathematical proof for optimality")