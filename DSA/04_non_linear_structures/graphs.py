"""
Non-Linear Data Structures - Graphs
==============================

This module provides implementations and examples of graph data structures,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Graph representations (adjacency list, adjacency matrix)
- Graph traversal algorithms (BFS, DFS)
- Shortest path algorithms (Dijkstra, Bellman-Ford)
- Minimum spanning tree algorithms (Prim, Kruskal)
- Applications in algorithms and data science
"""

from collections import deque, defaultdict
import heapq
from typing import Any, List, Dict, Set, Tuple, Optional

class Graph:
    """
    Implementation of a Graph using adjacency list representation
    """
    
    def __init__(self, directed: bool = False):
        """
        Initialize a graph
        
        Args:
            directed: Whether the graph is directed (True) or undirected (False)
        """
        self.graph: Dict[Any, List[Tuple[Any, int]]] = defaultdict(list)
        self.directed = directed
        self.vertices: Set[Any] = set()
    
    def add_vertex(self, vertex: Any) -> None:
        """
        Add a vertex to the graph
        Time Complexity: O(1)
        """
        self.vertices.add(vertex)
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def add_edge(self, u: Any, v: Any, weight: int = 1) -> None:
        """
        Add an edge to the graph
        Time Complexity: O(1)
        """
        self.vertices.add(u)
        self.vertices.add(v)
        
        # Add edge from u to v
        self.graph[u].append((v, weight))
        
        # For undirected graph, add reverse edge
        if not self.directed:
            self.graph[v].append((u, weight))
    
    def remove_edge(self, u: Any, v: Any) -> None:
        """
        Remove an edge from the graph
        Time Complexity: O(E) where E is number of edges
        """
        if u in self.graph:
            self.graph[u] = [(vertex, weight) for vertex, weight in self.graph[u] if vertex != v]
        
        # For undirected graph, remove reverse edge
        if not self.directed and v in self.graph:
            self.graph[v] = [(vertex, weight) for vertex, weight in self.graph[v] if vertex != u]
    
    def get_neighbors(self, vertex: Any) -> List[Tuple[Any, int]]:
        """
        Get neighbors of a vertex
        Time Complexity: O(1)
        """
        return self.graph.get(vertex, [])
    
    def bfs(self, start: Any) -> List[Any]:
        """
        Breadth-First Search traversal
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        if start not in self.vertices:
            return []
        
        visited: Set[Any] = set()
        queue: deque = deque([start])
        result: List[Any] = []
        
        visited.add(start)
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            # Visit all neighbors
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def dfs(self, start: Any) -> List[Any]:
        """
        Depth-First Search traversal (iterative implementation)
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        if start not in self.vertices:
            return []
        
        visited: Set[Any] = set()
        stack: List[Any] = [start]
        result: List[Any] = []
        
        while stack:
            vertex = stack.pop()
            
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                # Add neighbors to stack (in reverse order for consistent traversal)
                for neighbor, _ in reversed(self.graph[vertex]):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result
    
    def dfs_recursive(self, start: Any) -> List[Any]:
        """
        Depth-First Search traversal (recursive implementation)
        Time Complexity: O(V + E)
        Space Complexity: O(V) due to recursion stack
        """
        if start not in self.vertices:
            return []
        
        visited: Set[Any] = set()
        result: List[Any] = []
        
        def dfs_helper(vertex: Any) -> None:
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        return result
    
    def has_path(self, start: Any, end: Any) -> bool:
        """
        Check if there is a path between two vertices
        Time Complexity: O(V + E)
        """
        if start not in self.vertices or end not in self.vertices:
            return False
        
        visited: Set[Any] = set()
        queue: deque = deque([start])
        visited.add(start)
        
        while queue:
            vertex = queue.popleft()
            
            if vertex == end:
                return True
            
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    def connected_components(self) -> List[List[Any]]:
        """
        Find all connected components in the graph
        Time Complexity: O(V + E)
        """
        visited: Set[Any] = set()
        components: List[List[Any]] = []
        
        for vertex in self.vertices:
            if vertex not in visited:
                # BFS to find all vertices in this component
                component: List[Any] = []
                queue: deque = deque([vertex])
                visited.add(vertex)
                
                while queue:
                    current = queue.popleft()
                    component.append(current)
                    
                    for neighbor, _ in self.graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    def is_connected(self) -> bool:
        """
        Check if the graph is connected
        Time Complexity: O(V + E)
        """
        if not self.vertices:
            return True
        
        # Find one vertex to start from
        start_vertex = next(iter(self.vertices))
        
        # BFS from start vertex
        visited: Set[Any] = set()
        queue: deque = deque([start_vertex])
        visited.add(start_vertex)
        
        while queue:
            vertex = queue.popleft()
            
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Check if all vertices were visited
        return len(visited) == len(self.vertices)
    
    def __str__(self) -> str:
        """String representation of the graph"""
        result = []
        for vertex in sorted(self.vertices):
            neighbors = [f"{neighbor}({weight})" for neighbor, weight in self.graph[vertex]]
            result.append(f"{vertex}: [{', '.join(neighbors)}]")
        return "Graph:\n" + "\n".join(result)

class WeightedGraph(Graph):
    """
    Implementation of a weighted graph with shortest path algorithms
    """
    
    def dijkstra(self, start: Any) -> Tuple[Dict[Any, int], Dict[Any, Any]]:
        """
        Dijkstra's algorithm for shortest paths from start vertex
        Time Complexity: O((V + E) log V)
        """
        if start not in self.vertices:
            return {}, {}
        
        # Distance from start to each vertex
        distances: Dict[Any, int] = {vertex: float('inf') for vertex in self.vertices}
        distances[start] = 0
        
        # Previous vertex in shortest path
        previous: Dict[Any, Any] = {}
        
        # Priority queue: (distance, vertex)
        pq: List[Tuple[int, Any]] = [(0, start)]
        visited: Set[Any] = set()
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            # Skip if already processed
            if current_vertex in visited:
                continue
            
            visited.add(current_vertex)
            
            # Check neighbors
            for neighbor, weight in self.graph[current_vertex]:
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    # Found shorter path
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_vertex
                        heapq.heappush(pq, (new_distance, neighbor))
        
        return distances, previous
    
    def shortest_path(self, start: Any, end: Any) -> Tuple[int, List[Any]]:
        """
        Find shortest path between two vertices using Dijkstra's algorithm
        Returns (distance, path)
        """
        distances, previous = self.dijkstra(start)
        
        if end not in distances or distances[end] == float('inf'):
            return -1, []
        
        # Reconstruct path
        path: List[Any] = []
        current = end
        
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        path.reverse()
        
        return distances[end], path

def graph_traversal_demo():
    """
    Demonstrate graph traversal algorithms
    """
    print("=== Graph Traversal Demo ===")
    
    # Create undirected graph
    g = Graph(directed=False)
    
    # Add edges to create a sample graph
    edges = [
        ('A', 'B'), ('A', 'C'),
        ('B', 'D'), ('B', 'E'),
        ('C', 'F'), ('C', 'G'),
        ('D', 'H'), ('E', 'H'),
        ('F', 'I'), ('G', 'I')
    ]
    
    print("1. Graph Structure:")
    for u, v in edges:
        g.add_edge(u, v)
    
    print(f"   {g}")
    
    # BFS traversal
    print(f"\n2. BFS Traversal from A: {g.bfs('A')}")
    
    # DFS traversal (iterative)
    print(f"   DFS Traversal from A (iterative): {g.dfs('A')}")
    
    # DFS traversal (recursive)
    print(f"   DFS Traversal from A (recursive): {g.dfs_recursive('A')}")
    
    # Path checking
    print(f"\n3. Path Checking:")
    print(f"   Path from A to H: {g.has_path('A', 'H')}")
    print(f"   Path from A to Z: {g.has_path('A', 'Z')}")

def shortest_path_demo():
    """
    Demonstrate shortest path algorithms
    """
    print("\n=== Shortest Path Demo ===")
    
    # Create weighted graph
    wg = WeightedGraph(directed=True)
    
    # Add weighted edges
    weighted_edges = [
        ('A', 'B', 4), ('A', 'C', 2),
        ('B', 'D', 3), ('B', 'E', 2),
        ('C', 'B', 1), ('C', 'D', 4), ('C', 'E', 5),
        ('D', 'E', 1)
    ]
    
    print("1. Weighted Graph Structure:")
    for u, v, w in weighted_edges:
        wg.add_edge(u, v, w)
    
    print(f"   {wg}")
    
    # Dijkstra's algorithm
    print("\n2. Dijkstra's Algorithm from A:")
    distances, previous = wg.dijkstra('A')
    print(f"   Distances: {distances}")
    print(f"   Previous vertices: {previous}")
    
    # Shortest paths to specific vertices
    print("\n3. Shortest Paths from A:")
    for vertex in ['B', 'C', 'D', 'E']:
        distance, path = wg.shortest_path('A', vertex)
        print(f"   To {vertex}: Distance = {distance}, Path = {' -> '.join(path)}")

def graph_properties_demo():
    """
    Demonstrate graph properties and analysis
    """
    print("\n=== Graph Properties Demo ===")
    
    # Create disconnected graph
    g = Graph(directed=False)
    edges1 = [('A', 'B'), ('B', 'C'), ('C', 'A')]  # Component 1
    edges2 = [('D', 'E'), ('E', 'F')]               # Component 2
    edges3 = ['G']                                  # Component 3 (isolated vertex)
    
    print("1. Connected Components:")
    for u, v in edges1 + edges2:
        g.add_edge(u, v)
    g.add_vertex('G')  # Isolated vertex
    
    print(f"   Graph: {g}")
    components = g.connected_components()
    print(f"   Connected Components: {components}")
    print(f"   Number of Components: {len(components)}")
    print(f"   Is Connected: {g.is_connected()}")
    
    # Create connected graph
    g2 = Graph(directed=False)
    edges_connected = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'C')]
    for u, v in edges_connected:
        g2.add_edge(u, v)
    
    print(f"\n2. Connected Graph:")
    print(f"   Graph: {g2}")
    print(f"   Is Connected: {g2.is_connected()}")
    print(f"   BFS from A: {g2.bfs('A')}")
    print(f"   DFS from A: {g2.dfs('A')}")

def graph_applications():
    """
    Demonstrate common applications of graphs
    """
    print("\n=== Graph Applications ===")
    
    # 1. Social network analysis
    print("1. Social Network Analysis:")
    print("   Graphs model relationships between people")
    print("   Vertices: People, Edges: Friendships/Connections")
    print("   Analysis: Finding communities, influencers, shortest connections")
    
    # 2. Web page linking
    print("\n2. Web Page Linking:")
    print("   Graphs model hyperlinks between web pages")
    print("   Vertices: Web pages, Edges: Hyperlinks")
    print("   Algorithms: PageRank, web crawling, link analysis")
    
    # 3. Transportation networks
    print("\n3. Transportation Networks:")
    print("   Graphs model roads, flights, or public transport")
    print("   Vertices: Locations, Edges: Routes with weights (distance/time)")
    print("   Algorithms: Shortest path, network flow, route optimization")
    
    # 4. Dependency management
    print("\n4. Dependency Management:")
    print("   Graphs model dependencies between tasks or modules")
    print("   Vertices: Tasks/Modules, Edges: Dependencies")
    print("   Algorithms: Topological sorting, cycle detection")

def data_science_applications():
    """
    Examples of graphs in data science applications
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Recommendation systems
    print("1. Recommendation Systems:")
    print("   Bipartite graphs model user-item interactions")
    print("   Collaborative filtering uses graph-based algorithms")
    print("   Link prediction suggests new connections")
    
    # 2. Network analysis
    print("\n2. Network Analysis:")
    print("   Social network analysis identifies communities and influencers")
    print("   Centrality measures find important nodes")
    print("   Clustering coefficients measure network cohesion")
    
    # 3. Knowledge graphs
    print("\n3. Knowledge Graphs:")
    print("   Represent entities and relationships as graphs")
    print("   Enable semantic search and question answering")
    print("   Used by Google, Facebook, and other tech companies")
    
    # 4. Graph neural networks
    print("\n4. Graph Neural Networks:")
    print("   Machine learning on graph-structured data")
    print("   Applications: Drug discovery, fraud detection, molecular analysis")
    print("   Frameworks: PyTorch Geometric, DGL, TensorFlow GNN")

def performance_comparison():
    """
    Compare performance of different graph operations
    """
    print("\n=== Performance Comparison ===")
    
    import time
    import random
    
    # Test with different graph sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nTesting with {size} vertices:")
        
        # Create random graph
        g = Graph(directed=False)
        vertices = list(range(size))
        
        # Add vertices
        for v in vertices:
            g.add_vertex(v)
        
        # Add random edges (about 10% density)
        num_edges = size * (size - 1) // 20
        for _ in range(num_edges):
            u = random.choice(vertices)
            v = random.choice(vertices)
            if u != v:
                g.add_edge(u, v)
        
        # BFS performance
        start = time.time()
        _ = g.bfs(0)
        bfs_time = time.time() - start
        
        # DFS performance
        start = time.time()
        _ = g.dfs(0)
        dfs_time = time.time() - start
        
        # Connected components
        start = time.time()
        _ = g.connected_components()
        cc_time = time.time() - start
        
        print(f"   BFS traversal: {bfs_time:.6f}s")
        print(f"   DFS traversal: {dfs_time:.6f}s")
        print(f"   Connected components: {cc_time:.6f}s")

# Example usage and testing
if __name__ == "__main__":
    # Graph traversal demo
    graph_traversal_demo()
    print("\n" + "="*50 + "\n")
    
    # Shortest path demo
    shortest_path_demo()
    print("\n" + "="*50 + "\n")
    
    # Graph properties demo
    graph_properties_demo()
    print("\n" + "="*50 + "\n")
    
    # Graph applications
    graph_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Graph implementation using adjacency list")
    print("2. Graph traversal algorithms (BFS, DFS)")
    print("3. Shortest path algorithms (Dijkstra)")
    print("4. Graph properties and analysis")
    print("5. Practical applications in systems and algorithms")
    print("6. Data science applications of graphs")
    print("7. Performance characteristics of graph operations")
    print("\nKey takeaways:")
    print("- Graphs model relationships between entities")
    print("- BFS is ideal for shortest path in unweighted graphs")
    print("- DFS is useful for connectivity and cycle detection")
    print("- Dijkstra's algorithm finds shortest paths in weighted graphs")
    print("- Graphs are fundamental in social networks, web, and transportation")
    print("- Graph algorithms are essential in data science and ML")