"""
Practice Problems for Greedy Algorithms

This file contains solutions to the practice problems with detailed explanations.
"""

import heapq
from typing import List, Tuple


class UnionFind:
    """Union-Find data structure for Kruskal's algorithm."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True


def activity_selection(activities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Problem 1: Activity Selection"""
    if not activities:
        return []
    activities.sort(key=lambda x: x[1])
    selected = [activities[0]]
    last_end = activities[0][1]
    for i in range(1, len(activities)):
        if activities[i][0] >= last_end:
            selected.append(activities[i])
            last_end = activities[i][1]
    return selected


def fractional_knapsack(weights: List[int], values: List[int], capacity: int) -> float:
    """Problem 2: Fractional Knapsack"""
    items = [(values[i]/weights[i], weights[i], values[i]) for i in range(len(weights))]
    items.sort(reverse=True)
    total_value = 0.0
    remaining = capacity
    for ratio, weight, value in items:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            total_value += value * (remaining / weight)
            break
    return total_value


def huffman_coding(frequencies: dict) -> dict:
    """Problem 3: Huffman Coding"""
    if not frequencies:
        return {}
    heap = [[freq, [char, ""]] for char, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        low = heapq.heappop(heap)
        high = heapq.heappop(heap)
        for pair in low[1:]:
            pair[1] = '0' + pair[1]
        for pair in high[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])
    return dict(heap[0][1:])


def job_sequencing(jobs: List[Tuple[int, int, int]]) -> Tuple[int, List[int]]:
    """Problem 4: Job Sequencing"""
    jobs.sort(key=lambda x: x[2], reverse=True)
    max_deadline = max(job[1] for job in jobs)
    slots = [-1] * (max_deadline + 1)
    selected = []
    total_profit = 0
    for job_id, deadline, profit in jobs:
        for j in range(min(max_deadline, deadline), 0, -1):
            if slots[j] == -1:
                slots[j] = job_id
                selected.append(job_id)
                total_profit += profit
                break
    return total_profit, selected


def meeting_rooms(intervals: List[Tuple[int, int]]) -> int:
    """Problem 5: Meeting Rooms"""
    if not intervals:
        return 0
    start_times = sorted([interval[0] for interval in intervals])
    end_times = sorted([interval[1] for interval in intervals])
    rooms = 0
    max_rooms = 0
    start_ptr = end_ptr = 0
    while start_ptr < len(start_times):
        if start_times[start_ptr] < end_times[end_ptr]:
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            start_ptr += 1
        else:
            rooms -= 1
            end_ptr += 1
    return max_rooms


def interval_scheduling(intervals: List[Tuple[int, int]]) -> int:
    """Problem 6: Interval Scheduling"""
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    count = 1
    last_end = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] >= last_end:
            count += 1
            last_end = intervals[i][1]
    return count


def task_scheduling(tasks: List[Tuple[int, int, int]]) -> Tuple[int, List[int]]:
    """Problem 7: Task Scheduling"""
    tasks.sort(key=lambda x: x[2], reverse=True)
    max_deadline = max(task[1] for task in tasks)
    slots = [-1] * (max_deadline + 1)
    scheduled = []
    total_profit = 0
    for task_id, deadline, profit in tasks:
        for j in range(min(max_deadline, deadline), 0, -1):
            if slots[j] == -1:
                slots[j] = task_id
                scheduled.append(task_id)
                total_profit += profit
                break
    return total_profit, scheduled


def remove_overlapping(intervals: List[List[int]]) -> int:
    """Problem 8: Remove Overlapping Intervals"""
    if len(intervals) <= 1:
        return 0
    intervals.sort(key=lambda x: x[1])
    count = 0
    last_end = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] < last_end:
            count += 1
        else:
            last_end = intervals[i][1]
    return count


def kruskal_mst(vertices: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Problem 9: Kruskal's MST"""
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(vertices)
    mst_edges = []
    total_weight = 0
    for u, v, weight in edges:
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            if len(mst_edges) == vertices - 1:
                break
    return total_weight, mst_edges


def prims_mst(vertices: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Problem 10: Prim's MST"""
    graph = [[] for _ in range(vertices)]
    for u, v, weight in edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))
    visited = [False] * vertices
    min_heap = [(0, 0, -1)]
    mst_edges = []
    total_weight = 0
    while min_heap and len(mst_edges) < vertices - 1:
        weight, u, parent = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += weight
        if parent != -1:
            mst_edges.append((parent, u, weight))
        for v, edge_weight in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (edge_weight, v, u))
    return total_weight, mst_edges


def dijkstra(graph: List[List[Tuple[int, int]]], start: int) -> List[int]:
    """Problem 11: Dijkstra's Shortest Path"""
    n = len(graph)
    distances = [float('inf')] * n
    visited = [False] * n
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        for v, weight in graph[u]:
            if not visited[v]:
                new_dist = current_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
    return distances


def maximum_spanning_tree(vertices: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Problem 12: Maximum Spanning Tree"""
    negated_edges = [(u, v, -weight) for u, v, weight in edges]
    neg_weight, neg_edges = kruskal_mst(vertices, negated_edges)
    max_edges = [(u, v, -weight) for u, v, weight in neg_edges]
    return -neg_weight, max_edges


def min_cost_connecting_points(points: List[List[int]]) -> int:
    """Problem 13: Minimum Cost Connecting Points"""
    if len(points) <= 1:
        return 0
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            distance = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            edges.append((i, j, distance))
    weight, _ = kruskal_mst(n, edges)
    return weight


def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """Problem 14: Network Delay Time"""
    graph = [[] for _ in range(n)]
    for source, target, time in times:
        graph[source - 1].append((target - 1, time))
    distances = dijkstra(graph, k - 1)
    max_time = max(distances)
    return int(max_time) if max_time != float('inf') else -1


def cheapest_flights(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    """Problem 15: Cheapest Flights with K Stops"""
    distances = [float('inf')] * n
    distances[src] = 0
    for i in range(k + 1):
        temp = distances[:]
        for from_city, to_city, price in flights:
            if distances[from_city] != float('inf'):
                temp[to_city] = min(temp[to_city], distances[from_city] + price)
        distances = temp
    return distances[dst] if distances[dst] != float('inf') else -1


def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """Problem 16: Gas Station"""
    total_gas = sum(gas)
    total_cost = sum(cost)
    if total_gas < total_cost:
        return -1
    
    current_gas = 0
    start = 0
    for i in range(len(gas)):
        current_gas += gas[i] - cost[i]
        if current_gas < 0:
            start = i + 1
            current_gas = 0
    return start


def run_all_problems():
    """Run all practice problems to verify solutions."""
    print("=== Running All Practice Problems ===\n")
    
    # Problem 1: Activity Selection
    activities = [(1, 4, "A"), (3, 5, "B"), (0, 6, "C"), (5, 7, "D")]
    selected = activity_selection(activities)
    print(f"1. Activity Selection: {len(selected)} activities selected")
    
    # Problem 2: Fractional Knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    max_value = fractional_knapsack(weights, values, capacity)
    print(f"2. Fractional Knapsack: Max value = {max_value}")
    
    # Problem 3: Huffman Coding
    frequencies = {'a': 5, 'b': 9, 'c': 12}
    codes = huffman_coding(frequencies)
    print(f"3. Huffman Coding: {len(codes)} codes generated")
    
    # Problem 4: Job Sequencing
    jobs = [(1, 2, 100), (2, 1, 19), (3, 2, 27)]
    profit, selected = job_sequencing(jobs)
    print(f"4. Job Sequencing: Profit = {profit}, Jobs = {selected}")
    
    # Problem 5: Meeting Rooms
    intervals = [(0, 30), (5, 10), (15, 20)]
    rooms = meeting_rooms(intervals)
    print(f"5. Meeting Rooms: {rooms} rooms needed")
    
    # Problem 6: Interval Scheduling
    intervals = [(1, 2), (3, 4), (0, 6), (5, 7)]
    count = interval_scheduling(intervals)
    print(f"6. Interval Scheduling: {count} intervals selected")
    
    # Problem 9: Kruskal's MST
    vertices = 4
    edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
    weight, mst = kruskal_mst(vertices, edges)
    print(f"9. Kruskal's MST: Weight = {weight}")
    
    # Problem 11: Dijkstra's Algorithm
    graph = [[(1, 4), (2, 1)], [(3, 1)], [(1, 2), (3, 5)], []]
    distances = dijkstra(graph, 0)
    print(f"11. Dijkstra's: Distances = {distances}")
    
    # Problem 13: Minimum Cost Connecting Points
    points = [[0, 0], [2, 2], [3, 10]]
    cost = min_cost_connecting_points(points)
    print(f"13. Min Cost Connecting Points: Cost = {cost}")
    
    # Problem 16: Gas Station
    gas = [1, 2, 3, 4, 5]
    cost = [3, 4, 5, 1, 2]
    start = can_complete_circuit(gas, cost)
    print(f"16. Gas Station: Start at station {start}")


if __name__ == "__main__":
    run_all_problems()