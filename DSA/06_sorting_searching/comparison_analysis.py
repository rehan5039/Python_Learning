"""
Sorting and Searching - Comparison and Analysis
=========================================

This module provides comprehensive comparison and analysis of sorting and searching algorithms,
with performance benchmarks and real-world applications.

Topics Covered:
- Algorithm performance comparison
- Benchmarking techniques
- Real-world scenario analysis
- Choosing the right algorithm
- Optimization strategies
"""

import time
import random
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple, Dict
import sys

# Import sorting algorithms
from sorting_algorithms import (
    bubble_sort, selection_sort, insertion_sort,
    merge_sort, quick_sort, heap_sort,
    counting_sort, radix_sort
)

# Import searching algorithms
from searching_algorithms import (
    linear_search, binary_search,
    interpolation_search, exponential_search,
    jump_search
)

def benchmark_sorting_algorithms(sizes: List[int]) -> Dict[str, List[float]]:
    """
    Benchmark sorting algorithms with different input sizes
    """
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort": merge_sort,
        "Quick Sort": quick_sort,
        "Heap Sort": heap_sort,
        "Counting Sort": counting_sort,
        "Radix Sort": radix_sort
    }
    
    results = {name: [] for name in algorithms}
    
    for size in sizes:
        print(f"Benchmarking with size {size}...")
        
        # Generate random data
        data = [random.randint(1, 1000) for _ in range(size)]
        
        for name, func in algorithms.items():
            # Skip slow algorithms for large sizes
            if size > 5000 and name in ["Bubble Sort", "Selection Sort"]:
                results[name].append(float('inf'))
                continue
            
            if size > 10000 and name == "Insertion Sort":
                results[name].append(float('inf'))
                continue
            
            start_time = time.time()
            try:
                _ = func(data)
                elapsed_time = time.time() - start_time
                results[name].append(elapsed_time)
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name].append(float('inf'))
    
    return results

def benchmark_searching_algorithms(sizes: List[int]) -> Dict[str, List[float]]:
    """
    Benchmark searching algorithms with different input sizes
    """
    algorithms = {
        "Linear Search": linear_search,
        "Binary Search": binary_search,
        "Interpolation Search": interpolation_search,
        "Exponential Search": exponential_search,
        "Jump Search": jump_search
    }
    
    results = {name: [] for name in algorithms}
    
    for size in sizes:
        print(f"Benchmarking searching with size {size}...")
        
        # Generate sorted data for searching
        data = list(range(size))
        target = size // 2  # Search for middle element
        
        for name, func in algorithms.items():
            start_time = time.time()
            try:
                if name == "Linear Search":
                    _ = func(data, target)
                else:
                    _ = func(data, target)
                elapsed_time = time.time() - start_time
                results[name].append(elapsed_time)
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name].append(float('inf'))
    
    return results

def analyze_algorithm_complexity():
    """
    Analyze and display algorithm complexity information
    """
    print("=== Algorithm Complexity Analysis ===")
    
    sorting_complexity = {
        "Bubble Sort": {
            "Best": "O(n)", "Average": "O(n²)", "Worst": "O(n²)",
            "Space": "O(1)", "Stable": "Yes", "In-place": "Yes"
        },
        "Selection Sort": {
            "Best": "O(n²)", "Average": "O(n²)", "Worst": "O(n²)",
            "Space": "O(1)", "Stable": "No", "In-place": "Yes"
        },
        "Insertion Sort": {
            "Best": "O(n)", "Average": "O(n²)", "Worst": "O(n²)",
            "Space": "O(1)", "Stable": "Yes", "In-place": "Yes"
        },
        "Merge Sort": {
            "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n log n)",
            "Space": "O(n)", "Stable": "Yes", "In-place": "No"
        },
        "Quick Sort": {
            "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n²)",
            "Space": "O(log n)", "Stable": "No", "In-place": "Yes"
        },
        "Heap Sort": {
            "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n log n)",
            "Space": "O(1)", "Stable": "No", "In-place": "Yes"
        },
        "Counting Sort": {
            "Best": "O(n + k)", "Average": "O(n + k)", "Worst": "O(n + k)",
            "Space": "O(k)", "Stable": "Yes", "In-place": "No"
        },
        "Radix Sort": {
            "Best": "O(d(n + k))", "Average": "O(d(n + k))", "Worst": "O(d(n + k))",
            "Space": "O(n + k)", "Stable": "Yes", "In-place": "No"
        }
    }
    
    searching_complexity = {
        "Linear Search": {
            "Best": "O(1)", "Average": "O(n)", "Worst": "O(n)",
            "Space": "O(1)", "Prerequisite": "None"
        },
        "Binary Search": {
            "Best": "O(1)", "Average": "O(log n)", "Worst": "O(log n)",
            "Space": "O(1)", "Prerequisite": "Sorted array"
        },
        "Interpolation Search": {
            "Best": "O(1)", "Average": "O(log log n)", "Worst": "O(n)",
            "Space": "O(1)", "Prerequisite": "Sorted, uniform data"
        },
        "Exponential Search": {
            "Best": "O(1)", "Average": "O(log n)", "Worst": "O(log n)",
            "Space": "O(1)", "Prerequisite": "Sorted array"
        },
        "Jump Search": {
            "Best": "O(1)", "Average": "O(√n)", "Worst": "O(√n)",
            "Space": "O(1)", "Prerequisite": "Sorted array"
        }
    }
    
    print("\nSorting Algorithms Complexity:")
    for name, complexity in sorting_complexity.items():
        print(f"\n{name}:")
        for metric, value in complexity.items():
            print(f"   {metric}: {value}")
    
    print("\nSearching Algorithms Complexity:")
    for name, complexity in searching_complexity.items():
        print(f"\n{name}:")
        for metric, value in complexity.items():
            print(f"   {metric}: {value}")

def real_world_scenarios():
    """
    Analyze algorithm choices for real-world scenarios
    """
    print("\n=== Real-World Scenario Analysis ===")
    
    scenarios = {
        "Small Dataset (< 50 elements)": {
            "Sorting": "Insertion Sort - Simple, adaptive, low overhead",
            "Searching": "Linear Search - Simple, no preprocessing needed"
        },
        "Nearly Sorted Data": {
            "Sorting": "Insertion Sort - Takes advantage of existing order",
            "Searching": "Binary Search - After initial sort"
        },
        "Large Dataset with Limited Range": {
            "Sorting": "Counting Sort or Radix Sort - Linear time possible",
            "Searching": "Binary Search - After initial sort"
        },
        "Memory-Constrained Environment": {
            "Sorting": "Heap Sort - O(1) space complexity",
            "Searching": "Binary Search - O(1) space complexity"
        },
        "Frequent Searches on Static Data": {
            "Sorting": "Any O(n log n) algorithm - Sort once",
            "Searching": "Binary Search - O(log n) queries"
        },
        "Dynamic Data with Frequent Updates": {
            "Sorting": "Maintain sorted structure with balanced trees",
            "Searching": "Hash tables for O(1) average case"
        },
        "Data Science Preprocessing": {
            "Sorting": "Merge Sort or Python's Timsort - Stable, efficient",
            "Searching": "Binary Search for ordered operations"
        }
    }
    
    for scenario, recommendations in scenarios.items():
        print(f"\n{scenario}:")
        print(f"   Sorting: {recommendations['Sorting']}")
        print(f"   Searching: {recommendations['Searching']}")

def optimization_strategies():
    """
    Discuss optimization strategies for algorithms
    """
    print("\n=== Optimization Strategies ===")
    
    strategies = {
        "Sorting Optimizations": [
            "Hybrid algorithms (e.g., Timsort combines merge and insertion sort)",
            "Early termination for nearly sorted data",
            "Adaptive algorithms that change behavior based on data",
            "Parallel sorting for multi-core systems",
            "External sorting for data larger than memory"
        ],
        "Searching Optimizations": [
            "Indexing and caching for repeated searches",
            "Interpolation search for uniformly distributed data",
            "Hash tables for exact match queries",
            "B-trees for disk-based data",
            "Approximate search algorithms for large datasets"
        ],
        "General Optimizations": [
            "Choose appropriate data structures",
            "Preprocess data when beneficial",
            "Use built-in optimized implementations",
            "Consider memory hierarchy effects",
            "Profile and benchmark actual use cases"
        ]
    }
    
    for category, items in strategies.items():
        print(f"\n{category}:")
        for item in items:
            print(f"   • {item}")

def generate_performance_report():
    """
    Generate a comprehensive performance report
    """
    print("\n=== Performance Report ===")
    
    # Test sizes
    sizes = [100, 500, 1000, 2000, 5000]
    
    print("Running sorting algorithm benchmarks...")
    sorting_results = benchmark_sorting_algorithms(sizes)
    
    print("\nRunning searching algorithm benchmarks...")
    searching_results = benchmark_searching_algorithms(sizes)
    
    print("\nSorting Algorithm Performance (seconds):")
    print(f"{'Size':<8}", end="")
    for name in sorting_results:
        if name not in ["Bubble Sort", "Selection Sort"]:  # Skip slow algorithms
            print(f"{name:<15}", end="")
    print()
    
    for i, size in enumerate(sizes):
        print(f"{size:<8}", end="")
        for name, times in sorting_results.items():
            if name not in ["Bubble Sort", "Selection Sort"]:
                time_val = times[i] if i < len(times) and times[i] != float('inf') else "N/A"
                print(f"{time_val:<15.6f}" if isinstance(time_val, float) else f"{time_val:<15}", end="")
        print()
    
    print("\nSearching Algorithm Performance (seconds):")
    print(f"{'Size':<8}", end="")
    for name in searching_results:
        print(f"{name:<20}", end="")
    print()
    
    for i, size in enumerate(sizes):
        print(f"{size:<8}", end="")
        for name, times in searching_results.items():
            time_val = times[i] if i < len(times) and times[i] != float('inf') else "N/A"
            print(f"{time_val:<20.8f}" if isinstance(time_val, float) else f"{time_val:<20}", end="")
        print()

def data_science_case_studies():
    """
    Case studies of algorithm applications in data science
    """
    print("\n=== Data Science Case Studies ===")
    
    case_studies = {
        "1. E-commerce Product Search": {
            "Challenge": "Fast search through millions of products",
            "Solution": "Combination of indexing, hash tables, and ranking algorithms",
            "Algorithms": "Binary search on indices, hash lookups, merge sort for ranking",
            "Performance": "Sub-millisecond response times for search queries"
        },
        "2. Financial Market Data Analysis": {
            "Challenge": "Real-time processing of stock price streams",
            "Solution": "Streaming algorithms with sliding windows",
            "Algorithms": "Quick select for percentiles, merge sort for ordered analysis",
            "Performance": "Microsecond latency for critical calculations"
        },
        "3. Genomic Sequence Analysis": {
            "Challenge": "Matching DNA sequences in large databases",
            "Solution": "Specialized string matching algorithms",
            "Algorithms": "Suffix trees, Burrows-Wheeler transform, binary search",
            "Performance": "Efficient matching of billion-character sequences"
        },
        "4. Recommendation Systems": {
            "Challenge": "Finding similar users/items in large datasets",
            "Solution": "Nearest neighbor search with indexing",
            "Algorithms": "Hash-based similarity, binary search on sorted features",
            "Performance": "Real-time recommendations for millions of users"
        }
    }
    
    for case, details in case_studies.items():
        print(f"\n{case}:")
        print(f"   Challenge: {details['Challenge']}")
        print(f"   Solution: {details['Solution']}")
        print(f"   Key Algorithms: {details['Algorithms']}")
        print(f"   Performance: {details['Performance']}")

# Example usage and testing
if __name__ == "__main__":
    # Algorithm complexity analysis
    analyze_algorithm_complexity()
    print("\n" + "="*60 + "\n")
    
    # Real-world scenarios
    real_world_scenarios()
    print("\n" + "="*60 + "\n")
    
    # Optimization strategies
    optimization_strategies()
    print("\n" + "="*60 + "\n")
    
    # Data science case studies
    data_science_case_studies()
    print("\n" + "="*60 + "\n")
    
    # Generate performance report
    generate_performance_report()
    
    print("\n=== Summary ===")
    print("This analysis covered:")
    print("1. Complexity analysis of sorting and searching algorithms")
    print("2. Real-world scenario recommendations")
    print("3. Optimization strategies for better performance")
    print("4. Data science applications and case studies")
    print("5. Performance benchmarking results")
    print("\nKey takeaways:")
    print("- Algorithm choice depends heavily on data characteristics")
    print("- Preprocessing can dramatically improve search performance")
    print("- Hybrid approaches often provide the best real-world performance")
    print("- Profiling with actual data is crucial for optimization")
    print("- Modern libraries provide highly optimized implementations")