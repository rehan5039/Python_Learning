"""
Shortest Path Algorithms

This module covers shortest path algorithms using greedy approaches:
- Dijkstra's Algorithm
- Variations and applications
"""

import heapq
from typing import List, Tuple, Dict, Set


def dijkstra(graph: List[List[Tuple[int, int]]], start: int) -> Tuple[List[int], List[int]]:
    """
    Find shortest paths from start vertex to all other vertices using Dijkstra's algorithm.
    
    Time Complexity: O((V + E) log V) where V is vertices and E is edges
    Space Complexity: O(V)
    
    Args:
        graph: Adjacency list representation where graph[u] = [(v, weight), ...]
        start: Starting vertex
        
    Returns:
        Tuple of (distances, previous_vertices) arrays
    """
    n = len(graph)
    distances = [float('inf')] * n
    previous = [-1] * n
    visited = [False] * n
    
    # Distance to start vertex is 0
    distances[start] = 0
    
    # Priority queue: (distance, vertex)
    pq = [(0, start)]
    
    while pq:
        # Get vertex with minimum distance
        current_dist, u = heapq.heappop(pq)
        
        # Skip if already processed
        if visited[u]:
            continue
        
        # Mark as visited
        visited[u] = True
        
        # Update distances to neighbors
        for v, weight in graph[u]:
            if not visited[v]:
                new_dist = current_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    previous[v] = u
                    heapq.heappush(pq, (new_dist, v))
    
    return distances, previous


def dijkstra_path(graph: List[List[Tuple[int, int]]], start: int, end: int) -> Tuple[int, List[int]]:
    """
    Find shortest path from start to end vertex using Dijkstra's algorithm.
    
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)
    
    Args:
        graph: Adjacency list representation
        start: Starting vertex
        end: Target vertex
        
    Returns:
        Tuple of (shortest_distance, path_vertices)
    """
    distances, previous = dijkstra(graph, start)
    
    # Reconstruct path
    path = []
    current = end
    
    # Backtrack from end to start
    while current != -1:
        path.append(current)
        current = previous[current]
    
    # Reverse path to get start->end order
    path.reverse()
    
    # Check if path exists
    if distances[end] == float('inf'):
        return float('inf'), []
    
    # Check if path is valid (starts with start vertex)
    if path[0] != start:
        return float('inf'), []
    
    return distances[end], path


def dijkstra_all_pairs(graph: List[List[Tuple[int, int]]]) -> List[List[int]]:
    """
    Find shortest paths between all pairs of vertices.
    
    Time Complexity: O(V * (V + E) log V)
    Space Complexity: O(V^2)
    
    Args:
        graph: Adjacency list representation
        
    Returns:
        2D array where result[i][j] is shortest distance from i to j
    """
    n = len(graph)
    distances = []
    
    # Run Dijkstra from each vertex
    for i in range(n):
        dist, _ = dijkstra(graph, i)
        distances.append(dist)
    
    return distances


def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """
    Find the time it takes for all nodes to receive a signal starting from node k.
    
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V + E)
    
    Args:
        times: List of [source, target, time] representing directed edges
        n: Number of nodes
        k: Starting node
        
    Returns:
        Maximum time for all nodes to receive signal, or -1 if not all can
    """
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for source, target, time in times:
        graph[source - 1].append((target - 1, time))  # Convert to 0-indexed
    
    # Run Dijkstra from node k
    distances, _ = dijkstra(graph, k - 1)  # Convert to 0-indexed
    
    # Find maximum distance
    max_time = max(distances)
    
    # If any node is unreachable, return -1
    if max_time == float('inf'):
        return -1
    
    return int(max_time)


def cheapest_flights_within_k_stops(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    """
    Find cheapest price from src to dst with at most k stops.
    
    Time Complexity: O(E * K) where E is flights and K is stops
    Space Complexity: O(V)
    
    Args:
        n: Number of cities
        flights: List of [from, to, price] flights
        src: Source city
        dst: Destination city
        k: Maximum stops allowed
        
    Returns:
        Cheapest price or -1 if no valid route
    """
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for from_city, to_city, price in flights:
        graph[from_city].append((to_city, price))
    
    # Use modified BFS with stops constraint
    # Queue: (city, cost, stops)
    queue = [(src, 0, 0)]
    min_cost = [float('inf')] * n
    min_cost[src] = 0
    
    while queue:
        city, cost, stops = queue.pop(0)
        
        # If we've used more stops than allowed, skip
        if stops > k:
            continue
        
        # Explore neighbors
        for next_city, price in graph[city]:
            new_cost = cost + price
            
            # If we found a cheaper path to next_city
            if new_cost < min_cost[next_city]:
                min_cost[next_city] = new_cost
                queue.append((next_city, new_cost, stops + 1))
    
    return min_cost[dst] if min_cost[dst] != float('inf') else -1


def find_cheapest_price(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    """
    Alternative implementation using Bellman-Ford approach for limited stops.
    
    Time Complexity: O(E * K)
    Space Complexity: O(V)
    """
    # Initialize distances
    distances = [float('inf')] * n
    distances[src] = 0
    
    # Relax edges up to k+1 times
    for i in range(k + 1):
        # Create copy of current distances
        temp = distances[:]
        
        # Relax all edges
        for from_city, to_city, price in flights:
            if distances[from_city] != float('inf'):
                temp[to_city] = min(temp[to_city], distances[from_city] + price)
        
        distances = temp
    
    return distances[dst] if distances[dst] != float('inf') else -1


def demo():
    """Demonstrate the shortest path algorithms."""
    print("=== Shortest Path Algorithms ===\n")
    
    # Test Dijkstra's Algorithm
    # Graph representation: adjacency list
    # 0 -> [(1, 4), (2, 1)]
    # 1 -> [(3, 1)]
    # 2 -> [(1, 2), (3, 5)]
    # 3 -> []
    graph = [
        [(1, 4), (2, 1)],  # vertex 0
        [(3, 1)],          # vertex 1
        [(1, 2), (3, 5)],  # vertex 2
        []                 # vertex 3
    ]
    
    start = 0
    distances, previous = dijkstra(graph, start)
    print("Dijkstra's Algorithm:")
    print(f"  Graph: {graph}")
    print(f"  Start vertex: {start}")
    print(f"  Distances: {distances}")
    print(f"  Previous vertices: {previous}")
    
    # Find path from 0 to 3
    distance, path = dijkstra_path(graph, 0, 3)
    print(f"  Shortest path 0->3: {path} with distance {distance}")
    
    print("\n" + "="*50)
    
    # Test Network Delay Time
    times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    n = 4
    k = 2
    delay = network_delay_time(times, n, k)
    print(f"\nNetwork Delay Time:")
    print(f"  Times: {times}")
    print(f"  Nodes: {n}, Start: {k}")
    print(f"  Maximum delay: {delay}")
    
    times = [[1, 2, 1], [2, 3, 2], [1, 3, 4]]
    n = 3
    k = 1
    delay = network_delay_time(times, n, k)
    print(f"  Times: {times}")
    print(f"  Nodes: {n}, Start: {k}")
    print(f"  Maximum delay: {delay}")
    
    print("\n" + "="*50)
    
    # Test Cheapest Flights Within K Stops
    flights = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
    n = 3
    src, dst, k = 0, 2, 1
    cheapest = cheapest_flights_within_k_stops(n, flights, src, dst, k)
    print(f"\nCheapest Flights Within K Stops:")
    print(f"  Flights: {flights}")
    print(f"  Cities: {n}, From: {src}, To: {dst}, Stops: {k}")
    print(f"  Cheapest price: {cheapest}")
    
    flights = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
    src, dst, k = 0, 2, 0
    cheapest = cheapest_flights_within_k_stops(n, flights, src, dst, k)
    print(f"  Flights: {flights}")
    print(f"  Cities: {n}, From: {src}, To: {dst}, Stops: {k}")
    print(f"  Cheapest price: {cheapest}")
    
    print("\n" + "="*50)
    
    # Test All Pairs Shortest Path
    all_distances = dijkstra_all_pairs(graph)
    print(f"\nAll Pairs Shortest Path:")
    print(f"  Graph: {graph}")
    print(f"  Distance matrix:")
    for i, row in enumerate(all_distances):
        print(f"    From {i}: {row}")


if __name__ == "__main__":
    demo()