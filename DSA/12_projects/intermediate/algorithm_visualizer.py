"""
Intermediate Project: Algorithm Visualizer

This project creates a visualization tool for sorting and searching algorithms,
demonstrating how different algorithms work step-by-step.

Concepts covered:
- Algorithm visualization techniques
- Animation and graphics programming
- Performance comparison of algorithms
- Object-oriented design
- Event handling and user interaction
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import time
from typing import List, Generator, Tuple


class SortingAlgorithms:
    """
    Collection of sorting algorithms with step-by-step visualization support.
    """
    
    def __init__(self):
        self.steps = []
        self.comparisons = 0
        self.swaps = 0
    
    def reset_stats(self):
        """Reset algorithm statistics."""
        self.steps = []
        self.comparisons = 0
        self.swaps = 0
    
    def bubble_sort(self, arr: List[int]) -> Generator[List[int], None, None]:
        """
        Bubble sort with visualization steps.
        
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        self.reset_stats()
        arr = arr.copy()
        n = len(arr)
        
        self.steps.append(arr.copy())
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                self.comparisons += 1
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.swaps += 1
                    swapped = True
                    self.steps.append(arr.copy())
            
            if not swapped:
                break
        
        return arr
    
    def quick_sort(self, arr: List[int]) -> Generator[List[int], None, None]:
        """
        Quick sort with visualization steps.
        
        Time Complexity: O(n log n) average, O(n^2) worst case
        Space Complexity: O(log n)
        """
        self.reset_stats()
        arr = arr.copy()
        
        def quick_sort_helper(low: int, high: int):
            if low < high:
                self.steps.append(arr.copy())
                pi = partition(low, high)
                quick_sort_helper(low, pi - 1)
                quick_sort_helper(pi + 1, high)
        
        def partition(low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1
            
            for j in range(low, high):
                self.comparisons += 1
                if arr[j] <= pivot:
                    i += 1
                    if i != j:
                        arr[i], arr[j] = arr[j], arr[i]
                        self.swaps += 1
                        self.steps.append(arr.copy())
            
            if i + 1 != high:
                arr[i + 1], arr[high] = arr[high], arr[i + 1]
                self.swaps += 1
                self.steps.append(arr.copy())
            
            return i + 1
        
        quick_sort_helper(0, len(arr) - 1)
        return arr
    
    def merge_sort(self, arr: List[int]) -> Generator[List[int], None, None]:
        """
        Merge sort with visualization steps.
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        self.reset_stats()
        arr = arr.copy()
        
        def merge_sort_helper(left: int, right: int):
            if left < right:
                mid = (left + right) // 2
                merge_sort_helper(left, mid)
                merge_sort_helper(mid + 1, right)
                merge(left, mid, right)
        
        def merge(left: int, mid: int, right: int):
            # Create temporary arrays
            left_arr = arr[left:mid + 1]
            right_arr = arr[mid + 1:right + 1]
            
            i = j = 0
            k = left
            
            # Merge the temporary arrays back
            while i < len(left_arr) and j < len(right_arr):
                self.comparisons += 1
                if left_arr[i] <= right_arr[j]:
                    arr[k] = left_arr[i]
                    i += 1
                else:
                    arr[k] = right_arr[j]
                    j += 1
                self.steps.append(arr.copy())
                k += 1
            
            # Copy remaining elements
            while i < len(left_arr):
                arr[k] = left_arr[i]
                self.steps.append(arr.copy())
                i += 1
                k += 1
            
            while j < len(right_arr):
                arr[k] = right_arr[j]
                self.steps.append(arr.copy())
                j += 1
                k += 1
        
        merge_sort_helper(0, len(arr) - 1)
        return arr
    
    def heap_sort(self, arr: List[int]) -> Generator[List[int], None, None]:
        """
        Heap sort with visualization steps.
        
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        self.reset_stats()
        arr = arr.copy()
        n = len(arr)
        
        def heapify(n: int, i: int):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n:
                self.comparisons += 1
                if arr[left] > arr[largest]:
                    largest = left
            
            if right < n:
                self.comparisons += 1
                if arr[right] > arr[largest]:
                    largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.swaps += 1
                self.steps.append(arr.copy())
                heapify(n, largest)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(n, i)
        
        # Extract elements from heap
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.swaps += 1
            self.steps.append(arr.copy())
            heapify(i, 0)
        
        return arr


class SearchingAlgorithms:
    """
    Collection of searching algorithms with visualization support.
    """
    
    def __init__(self):
        self.steps = []
        self.comparisons = 0
    
    def reset_stats(self):
        """Reset algorithm statistics."""
        self.steps = []
        self.comparisons = 0
    
    def binary_search_visual(self, arr: List[int], target: int) -> Tuple[int, List[Tuple[int, int, int]]]:
        """
        Binary search with visualization steps.
        
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        self.reset_stats()
        steps = []
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            steps.append((left, right, mid))
            self.comparisons += 1
            
            if arr[mid] == target:
                return mid, steps
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1, steps  # Not found


class AlgorithmVisualizer:
    """
    Visualization tool for algorithms using matplotlib.
    """
    
    def __init__(self):
        self.sorting_algorithms = SortingAlgorithms()
        self.searching_algorithms = SearchingAlgorithms()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.bars = None
    
    def visualize_sorting(self, algorithm_name: str, arr: List[int], 
                         interval: int = 100, save_animation: bool = False):
        """
        Visualize sorting algorithm.
        
        Args:
            algorithm_name: Name of algorithm ('bubble', 'quick', 'merge', 'heap')
            arr: Array to sort
            interval: Animation interval in milliseconds
            save_animation: Whether to save animation to file
        """
        # Get the appropriate algorithm
        if algorithm_name == 'bubble':
            algorithm = self.sorting_algorithms.bubble_sort
        elif algorithm_name == 'quick':
            algorithm = self.sorting_algorithms.quick_sort
        elif algorithm_name == 'merge':
            algorithm = self.sorting_algorithms.merge_sort
        elif algorithm_name == 'heap':
            algorithm = self.sorting_algorithms.heap_sort
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Generate steps
        list(algorithm(arr))  # Execute algorithm to generate steps
        steps = self.sorting_algorithms.steps
        
        # Create animation
        def animate(frame):
            if frame < len(steps):
                self.ax.clear()
                bars = self.ax.bar(range(len(steps[frame])), steps[frame], 
                                 color='skyblue', edgecolor='navy', alpha=0.7)
                self.ax.set_title(f'{algorithm_name.title()} Sort - Step {frame + 1}')
                self.ax.set_xlabel('Index')
                self.ax.set_ylabel('Value')
                
                # Add statistics
                stats_text = f'Comparisons: {self.sorting_algorithms.comparisons}\n'
                stats_text += f'Swaps: {self.sorting_algorithms.swaps}'
                self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        anim = animation.FuncAnimation(self.fig, animate, frames=len(steps), 
                                     interval=interval, repeat=False, blit=False)
        
        if save_animation:
            anim.save(f'{algorithm_name}_sort.gif', writer='pillow')
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def visualize_searching(self, arr: List[int], target: int):
        """
        Visualize binary search algorithm.
        """
        # Sort array first (required for binary search)
        sorted_arr = sorted(arr)
        
        # Perform binary search
        index, steps = self.searching_algorithms.binary_search_visual(sorted_arr, target)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the array
        bars = ax.bar(range(len(sorted_arr)), sorted_arr, color='lightblue')
        ax.set_title(f'Binary Search for {target}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        
        # Highlight search steps
        for i, (left, right, mid) in enumerate(steps):
            # Reset colors
            for bar in bars:
                bar.set_color('lightblue')
            
            # Highlight search range
            for j in range(left, right + 1):
                bars[j].set_color('yellow')
            
            # Highlight middle element
            bars[mid].set_color('red')
            
            # Add text
            ax.text(mid, sorted_arr[mid] + 1, f'{sorted_arr[mid]}', 
                   ha='center', va='bottom', fontweight='bold')
            
            plt.pause(1)  # Pause for visualization
        
        # Final result
        if index != -1:
            bars[index].set_color('green')
            ax.set_title(f'Found {target} at index {index}')
        else:
            ax.set_title(f'{target} not found')
        
        plt.tight_layout()
        plt.show()


def performance_comparison():
    """Compare performance of different sorting algorithms."""
    print("=== Algorithm Performance Comparison ===\n")
    
    # Generate test data
    sizes = [100, 500, 1000, 2000]
    algorithms = {
        'Bubble Sort': SortingAlgorithms().bubble_sort,
        'Quick Sort': SortingAlgorithms().quick_sort,
        'Merge Sort': SortingAlgorithms().merge_sort,
        'Heap Sort': SortingAlgorithms().heap_sort
    }
    
    for size in sizes:
        print(f"Array size: {size}")
        arr = [random.randint(1, 1000) for _ in range(size)]
        
        for name, algorithm in algorithms.items():
            # Create new instance for each test
            sorter = SortingAlgorithms()
            
            start_time = time.time()
            list(algorithm(arr.copy()))
            end_time = time.time()
            
            execution_time = end_time - start_time
            comparisons = sorter.comparisons
            swaps = sorter.swaps
            
            print(f"  {name}: {execution_time:.6f}s, {comparisons} comparisons, {swaps} swaps")
        
        print()


def demonstrate_visualizer():
    """Demonstrate algorithm visualizer functionality."""
    print("=== Algorithm Visualizer Demo ===\n")
    
    # Create sample data
    arr = [64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42]
    print(f"Original array: {arr}")
    
    # Test sorting algorithms
    print("\n1. Sorting Algorithm Performance:")
    sorter = SortingAlgorithms()
    
    # Test bubble sort
    start_time = time.time()
    list(sorter.bubble_sort(arr.copy()))
    bubble_time = time.time() - start_time
    print(f"  Bubble Sort: {bubble_time:.6f}s, {sorter.comparisons} comparisons, {sorter.swaps} swaps")
    
    # Test quick sort
    start_time = time.time()
    list(sorter.quick_sort(arr.copy()))
    quick_time = time.time() - start_time
    print(f"  Quick Sort: {quick_time:.6f}s, {sorter.comparisons} comparisons, {sorter.swaps} swaps")
    
    # Test merge sort
    start_time = time.time()
    list(sorter.merge_sort(arr.copy()))
    merge_time = time.time() - start_time
    print(f"  Merge Sort: {merge_time:.6f}s, {sorter.comparisons} comparisons, {sorter.swaps} swaps")
    
    # Test heap sort
    start_time = time.time()
    list(sorter.heap_sort(arr.copy()))
    heap_time = time.time() - start_time
    print(f"  Heap Sort: {heap_time:.6f}s, {sorter.comparisons} comparisons, {sorter.swaps} swaps")
    
    # Test searching algorithm
    print("\n2. Searching Algorithm:")
    searcher = SearchingAlgorithms()
    sorted_arr = sorted(arr)
    target = 25
    
    index, steps = searcher.binary_search_visual(sorted_arr, target)
    print(f"  Binary search for {target} in {sorted_arr}")
    print(f"  Found at index: {index}")
    print(f"  Steps taken: {len(steps)}")
    print(f"  Comparisons: {searcher.comparisons}")


if __name__ == "__main__":
    demonstrate_visualizer()
    performance_comparison()
    
    # Uncomment the following lines to run visualizations (requires matplotlib)
    # visualizer = AlgorithmVisualizer()
    # sample_arr = [64, 34, 25, 12, 22, 11, 90]
    # visualizer.visualize_sorting('bubble', sample_arr, interval=500)