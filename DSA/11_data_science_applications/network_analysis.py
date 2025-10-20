"""
Network Analysis with Graph Algorithms

This module demonstrates how to apply graph algorithms to network analysis in data science:
- Social network analysis
- Recommendation systems
- Community detection
- Centrality measures
- Pathfinding algorithms
- Network visualization
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
from collections import defaultdict, deque
import heapq


class Graph:
    """
    Optimized Graph implementation for network analysis.
    """
    
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adjacency_list = defaultdict(list)
        self.nodes = set()
    
    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """Add an edge to the graph."""
        self.adjacency_list[u].append((v, weight))
        self.nodes.add(u)
        self.nodes.add(v)
        
        if not self.directed:
            self.adjacency_list[v].append((u, weight))
    
    def add_node(self, node: int) -> None:
        """Add a node to the graph."""
        self.nodes.add(node)
    
    def get_neighbors(self, node: int) -> List[Tuple[int, float]]:
        """Get neighbors of a node."""
        return self.adjacency_list[node]
    
    def get_nodes(self) -> Set[int]:
        """Get all nodes in the graph."""
        return self.nodes
    
    def get_edges(self) -> List[Tuple[int, int, float]]:
        """Get all edges in the graph."""
        edges = []
        for u in self.nodes:
            for v, weight in self.adjacency_list[u]:
                if not self.directed and u > v:
                    continue  # Avoid duplicate edges in undirected graph
                edges.append((u, v, weight))
        return edges


def breadth_first_search(graph: Graph, start: int) -> Dict[int, int]:
    """
    BFS for finding shortest paths in unweighted graphs.
    
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V)
    """
    distances = {node: float('inf') for node in graph.get_nodes()}
    distances[start] = 0
    queue = deque([start])
    visited = {start}
    
    while queue:
        current = queue.popleft()
        
        for neighbor, _ in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    
    return distances


def dijkstra_shortest_path(graph: Graph, start: int) -> Dict[int, float]:
    """
    Dijkstra's algorithm for shortest paths in weighted graphs.
    
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)
    """
    distances = {node: float('inf') for node in graph.get_nodes()}
    distances[start] = 0
    visited = set()
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        for neighbor, weight in graph.get_neighbors(current_node):
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances


def pagerank_optimization(graph: Graph, damping_factor: float = 0.85, 
                         max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[int, float]:
    """
    Optimized PageRank algorithm for node importance ranking.
    
    Time Complexity: O(V * E * max_iterations)
    Space Complexity: O(V)
    """
    nodes = list(graph.get_nodes())
    n = len(nodes)
    
    # Initialize PageRank values
    pagerank = {node: 1.0 / n for node in nodes}
    
    # Build reverse adjacency list for efficient computation
    reverse_adjacency = defaultdict(list)
    for u in nodes:
        for v, _ in graph.get_neighbors(u):
            reverse_adjacency[v].append(u)
    
    for iteration in range(max_iterations):
        new_pagerank = {}
        dangling_sum = 0
        
        # Calculate sum of PageRanks for dangling nodes (no outgoing edges)
        for node in nodes:
            if not graph.get_neighbors(node):
                dangling_sum += pagerank[node]
        
        # Update PageRank for each node
        for node in nodes:
            rank_sum = 0
            for incoming_node in reverse_adjacency[node]:
                out_degree = len(graph.get_neighbors(incoming_node))
                if out_degree > 0:
                    rank_sum += pagerank[incoming_node] / out_degree
            
            new_pagerank[node] = (1 - damping_factor) / n + \
                               damping_factor * (rank_sum + dangling_sum / n)
        
        # Check for convergence
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in nodes)
        pagerank = new_pagerank
        
        if diff < tolerance:
            break
    
    return pagerank


def community_detection_optimization(graph: Graph, max_iterations: int = 100) -> Dict[int, int]:
    """
    Optimized community detection using modularity optimization.
    
    Time Complexity: O(V * E * max_iterations)
    Space Complexity: O(V)
    """
    # Initialize each node to its own community
    communities = {node: i for i, node in enumerate(graph.get_nodes())}
    nodes = list(graph.get_nodes())
    
    def calculate_modularity(communities_dict: Dict[int, int]) -> float:
        """Calculate modularity of current community assignment."""
        m = sum(weight for _, _, weight in graph.get_edges())
        if m == 0:
            return 0
        
        modularity = 0
        for u in nodes:
            for v, weight in graph.get_neighbors(u):
                if communities_dict[u] == communities_dict[v]:
                    # Calculate expected edge probability
                    k_u = sum(w for _, w in graph.get_neighbors(u))
                    k_v = sum(w for _, w in graph.get_neighbors(v))
                    expected = k_u * k_v / (2 * m)
                    modularity += weight - expected
        
        return modularity / (2 * m)
    
    current_modularity = calculate_modularity(communities)
    
    # Greedy optimization
    for _ in range(max_iterations):
        improved = False
        
        # Try moving each node to a different community
        for node in nodes:
            current_community = communities[node]
            best_community = current_community
            best_modularity = current_modularity
            
            # Try moving to each neighbor's community
            for neighbor, _ in graph.get_neighbors(node):
                if communities[neighbor] != current_community:
                    # Try move
                    communities[node] = communities[neighbor]
                    new_modularity = calculate_modularity(communities)
                    
                    if new_modularity > best_modularity:
                        best_modularity = new_modularity
                        best_community = communities[neighbor]
                        improved = True
                    
                    # Revert move
                    communities[node] = current_community
            
            # Make best move
            if best_community != current_community:
                communities[node] = best_community
                current_modularity = best_modularity
        
        if not improved:
            break
    
    return communities


def centrality_measures_optimization(graph: Graph) -> Dict[str, Dict[int, float]]:
    """
    Calculate various centrality measures for network analysis.
    
    Time Complexity: O(V * (V + E)) for betweenness, O(V + E) for others
    Space Complexity: O(V)
    """
    nodes = list(graph.get_nodes())
    n = len(nodes)
    
    # Degree centrality
    degree_centrality = {}
    for node in nodes:
        degree_centrality[node] = len(graph.get_neighbors(node)) / (n - 1) if n > 1 else 0
    
    # Closeness centrality
    closeness_centrality = {}
    for node in nodes:
        distances = dijkstra_shortest_path(graph, node)
        total_distance = sum(distances.values())
        closeness_centrality[node] = (n - 1) / total_distance if total_distance > 0 else 0
    
    # Betweenness centrality (simplified version)
    betweenness_centrality = {node: 0 for node in nodes}
    # Full implementation would require all-pairs shortest paths
    
    return {
        'degree': degree_centrality,
        'closeness': closeness_centrality,
        'betweenness': betweenness_centrality
    }


def network_analysis_pipeline(edges: List[Tuple[int, int, float]], 
                            analysis_type: str = 'all') -> Dict:
    """
    Complete network analysis pipeline.
    
    Time Complexity: O(V * E * iterations) depending on analysis type
    Space Complexity: O(V + E)
    """
    # Create graph
    graph = Graph(directed=False)
    for u, v, weight in edges:
        graph.add_edge(u, v, weight)
    
    results = {}
    
    if analysis_type in ['all', 'shortest_path']:
        # Shortest path analysis
        sample_nodes = list(graph.get_nodes())[:min(5, len(graph.get_nodes()))]
        shortest_paths = {}
        for node in sample_nodes:
            shortest_paths[node] = dijkstra_shortest_path(graph, node)
        results['shortest_paths'] = shortest_paths
    
    if analysis_type in ['all', 'pagerank']:
        # PageRank analysis
        pagerank_scores = pagerank_optimization(graph)
        results['pagerank'] = pagerank_scores
    
    if analysis_type in ['all', 'communities']:
        # Community detection
        communities = community_detection_optimization(graph)
        results['communities'] = communities
    
    if analysis_type in ['all', 'centrality']:
        # Centrality measures
        centrality = centrality_measures_optimization(graph)
        results['centrality'] = centrality
    
    # Basic statistics
    results['statistics'] = {
        'nodes': len(graph.get_nodes()),
        'edges': len(graph.get_edges()),
        'density': len(graph.get_edges()) / (len(graph.get_nodes()) * (len(graph.get_nodes()) - 1) / 2) if len(graph.get_nodes()) > 1 else 0
    }
    
    return results


def performance_comparison():
    """Compare performance of different network analysis techniques."""
    print("=== Network Analysis Performance Comparison ===\n")
    
    # Create sample network
    np.random.seed(42)
    n_nodes = 1000
    n_edges = 5000
    
    # Generate random edges
    edges = []
    for _ in range(n_edges):
        u = np.random.randint(0, n_nodes)
        v = np.random.randint(0, n_nodes)
        if u != v:
            weight = np.random.rand()
            edges.append((u, v, weight))
    
    # Test shortest path algorithms
    print("1. Shortest Path Algorithms:")
    import time
    
    # Create smaller graph for testing
    small_edges = edges[:1000]
    graph = Graph(directed=False)
    for u, v, weight in small_edges:
        graph.add_edge(u, v, weight)
    
    start_time = time.time()
    distances = dijkstra_shortest_path(graph, 0)
    dijkstra_time = time.time() - start_time
    print(f"   Dijkstra time: {dijkstra_time:.6f} seconds")
    print(f"   Nodes reached: {len([d for d in distances.values() if d != float('inf')])}")
    
    # Test PageRank
    print("\n2. PageRank Optimization:")
    start_time = time.time()
    pagerank = pagerank_optimization(graph)
    pagerank_time = time.time() - start_time
    print(f"   PageRank time: {pagerank_time:.6f} seconds")
    print(f"   Top 3 nodes by PageRank: {sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Test community detection
    print("\n3. Community Detection:")
    start_time = time.time()
    communities = community_detection_optimization(graph)
    community_time = time.time() - start_time
    print(f"   Community detection time: {community_time:.6f} seconds")
    print(f"   Number of communities: {len(set(communities.values()))}")
    
    # Test centrality measures
    print("\n4. Centrality Measures:")
    start_time = time.time()
    centrality = centrality_measures_optimization(graph)
    centrality_time = time.time() - start_time
    print(f"   Centrality time: {centrality_time:.6f} seconds")
    print(f"   Top 3 by degree centrality: {sorted(centrality['degree'].items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Test complete pipeline
    print("\n5. Complete Analysis Pipeline:")
    start_time = time.time()
    results = network_analysis_pipeline(small_edges[:500], analysis_type='all')
    pipeline_time = time.time() - start_time
    print(f"   Pipeline time: {pipeline_time:.6f} seconds")
    print(f"   Network statistics: {results['statistics']}")


def demo():
    """Demonstrate network analysis techniques."""
    print("=== Network Analysis with Graph Algorithms ===\n")
    
    # Create sample social network
    edges = [
        (0, 1, 1.0), (0, 2, 0.8), (1, 2, 0.9), (1, 3, 0.7),
        (2, 4, 0.6), (3, 4, 0.8), (3, 5, 0.9), (4, 5, 0.7),
        (5, 6, 0.8), (6, 7, 0.9), (6, 8, 0.7), (7, 8, 0.8)
    ]
    
    print("Sample network edges:")
    for u, v, w in edges:
        print(f"  {u} --({w})-- {v}")
    
    # Create graph
    graph = Graph(directed=False)
    for u, v, weight in edges:
        graph.add_edge(u, v, weight)
    
    # Test shortest paths
    print("\n1. Shortest Paths from node 0:")
    distances = dijkstra_shortest_path(graph, 0)
    for node, distance in sorted(distances.items()):
        print(f"  Node {node}: distance {distance:.2f}")
    
    # Test PageRank
    print("\n2. PageRank Scores:")
    pagerank = pagerank_optimization(graph)
    for node, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
        print(f"  Node {node}: {score:.4f}")
    
    # Test community detection
    print("\n3. Community Detection:")
    communities = community_detection_optimization(graph)
    community_groups = defaultdict(list)
    for node, community in communities.items():
        community_groups[community].append(node)
    
    for community, nodes in community_groups.items():
        print(f"  Community {community}: {sorted(nodes)}")
    
    # Test centrality measures
    print("\n4. Centrality Measures:")
    centrality = centrality_measures_optimization(graph)
    print("  Degree Centrality:")
    for node, score in sorted(centrality['degree'].items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"    Node {node}: {score:.4f}")
    
    # Test complete pipeline
    print("\n5. Complete Network Analysis Pipeline:")
    results = network_analysis_pipeline(edges)
    print(f"  Network Statistics:")
    for key, value in results['statistics'].items():
        print(f"    {key}: {value}")
    
    # Performance comparison
    print("\n" + "="*60)
    performance_comparison()


if __name__ == "__main__":
    demo()