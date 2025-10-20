"""
Large Scale Data Processing with DSA Principles

This module demonstrates how to handle large-scale data processing using efficient algorithms:
- Memory management techniques
- Streaming algorithms
- Distributed processing concepts
- Efficient data structures for big data
- Performance optimization strategies
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Iterator, Generator
import heapq
import hashlib
from collections import deque, defaultdict
import time


class ExternalSort:
    """
    External sorting for datasets larger than memory using merge sort principles.
    """
    
    def __init__(self, chunk_size: int = 1000000):
        self.chunk_size = chunk_size
    
    def sort_large_array(self, data_generator: Iterator[List[float]]) -> Generator[List[float], None, None]:
        """
        Sort large array that doesn't fit in memory using external merge sort.
        
        Time Complexity: O(n log n) where n is total elements
        Space Complexity: O(chunk_size)
        """
        # Phase 1: Sort chunks and write to temporary files
        temp_files = []
        chunk_count = 0
        
        for chunk in data_generator:
            # Sort chunk in memory
            chunk.sort()
            
            # Write sorted chunk to temporary file
            temp_filename = f"temp_chunk_{chunk_count}.txt"
            with open(temp_filename, 'w') as f:
                for item in chunk:
                    f.write(f"{item}\n")
            temp_files.append(temp_filename)
            chunk_count += 1
        
        # Phase 2: Merge sorted chunks
        yield from self._merge_sorted_files(temp_files)
        
        # Cleanup temporary files
        import os
        for temp_file in temp_files:
            os.remove(temp_file)
    
    def _merge_sorted_files(self, file_names: List[str]) -> Generator[List[float], None, None]:
        """Merge multiple sorted files."""
        # Open all files
        file_handles = [open(fname, 'r') for fname in file_names]
        heap = []
        
        # Initialize heap with first element from each file
        for i, fh in enumerate(file_handles):
            line = fh.readline()
            if line:
                heapq.heappush(heap, (float(line.strip()), i))
        
        # Merge process
        result_chunk = []
        while heap:
            value, file_index = heapq.heappop(heap)
            result_chunk.append(value)
            
            # Yield chunk when it reaches chunk_size
            if len(result_chunk) >= self.chunk_size:
                yield result_chunk
                result_chunk = []
            
            # Read next element from the same file
            line = file_handles[file_index].readline()
            if line:
                heapq.heappush(heap, (float(line.strip()), file_index))
        
        # Yield remaining elements
        if result_chunk:
            yield result_chunk
        
        # Close all files
        for fh in file_handles:
            fh.close()


class StreamingStatistics:
    """
    Calculate statistics on streaming data using efficient algorithms.
    """
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from mean
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, value: float) -> None:
        """Update statistics with new value using Welford's online algorithm."""
        self.count += 1
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # Welford's online algorithm for variance
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics."""
        if self.count < 2:
            variance = 0.0
        else:
            variance = self.m2 / (self.count - 1)
        
        return {
            'count': self.count,
            'mean': self.mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'min': self.min_val,
            'max': self.max_val
        }


class BloomFilter:
    """
    Probabilistic data structure for fast membership testing.
    """
    
    def __init__(self, capacity: int = 1000000, error_rate: float = 0.01):
        import math
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Calculate optimal size and number of hash functions
        self.bit_array_size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.hash_count = int(self.bit_array_size * math.log(2) / capacity)
        
        # Initialize bit array
        self.bit_array = [0] * self.bit_array_size
    
    def _hash(self, item: str, seed: int) -> int:
        """Generate hash using double hashing."""
        hash1 = int(hashlib.md5((item + str(seed)).encode()).hexdigest(), 16)
        hash2 = int(hashlib.sha256((item + str(seed)).encode()).hexdigest(), 16)
        return (hash1 + seed * hash2) % self.bit_array_size
    
    def add(self, item: str) -> None:
        """Add item to Bloom filter."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = 1
    
    def contains(self, item: str) -> bool:
        """Check if item might be in set (false positives possible)."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True


class ReservoirSampling:
    """
    Reservoir sampling for uniform random sampling from large datasets.
    """
    
    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
        self.reservoir = []
        self.count = 0
    
    def add(self, item) -> None:
        """Add item to reservoir sample."""
        self.count += 1
        
        if len(self.reservoir) < self.sample_size:
            self.reservoir.append(item)
        else:
            # Replace elements with gradually decreasing probability
            j = np.random.randint(0, self.count)
            if j < self.sample_size:
                self.reservoir[j] = item
    
    def get_sample(self) -> List:
        """Get current reservoir sample."""
        return self.reservoir.copy()


def memory_efficient_groupby(data_generator: Iterator[Tuple], 
                           group_key_index: int, 
                           agg_function: str = 'sum') -> Dict:
    """
    Memory-efficient groupby operation using hash tables.
    
    Time Complexity: O(n) where n is number of records
    Space Complexity: O(k) where k is number of unique groups
    """
    groups = defaultdict(list)
    
    # Process data in chunks
    for record in data_generator:
        key = record[group_key_index]
        groups[key].append(record)
    
    # Apply aggregation
    results = {}
    for key, records in groups.items():
        if agg_function == 'sum':
            results[key] = sum(record[1] for record in records)  # Assuming value at index 1
        elif agg_function == 'count':
            results[key] = len(records)
        elif agg_function == 'mean':
            values = [record[1] for record in records]
            results[key] = sum(values) / len(values)
    
    return results


def sliding_window_optimization(data: List[float], window_size: int) -> List[float]:
    """
    Optimize sliding window calculations using deque data structure.
    
    Time Complexity: O(n) where n is data length
    Space Complexity: O(window_size)
    """
    if len(data) < window_size:
        return []
    
    window = deque()
    results = []
    
    for i, value in enumerate(data):
        # Add current value to window
        window.append(value)
        
        # Remove values outside window
        if len(window) > window_size:
            window.popleft()
        
        # Calculate window average when window is full
        if len(window) == window_size:
            results.append(sum(window) / window_size)
    
    return results


def data_stream_processing_pipeline(data_stream: Iterator[List[float]], 
                                 operations: List[str]) -> Dict:
    """
    Complete data stream processing pipeline with multiple operations.
    
    Time Complexity: O(n * m) where n is data size, m is number of operations
    Space Complexity: O(k) where k is the largest intermediate result
    """
    results = {}
    
    # Initialize streaming algorithms
    stats = StreamingStatistics()
    bloom_filter = BloomFilter()
    reservoir = ReservoirSampling(sample_size=100)
    
    # Process data stream
    total_records = 0
    for chunk in data_stream:
        for value in chunk:
            # Update streaming statistics
            stats.update(value)
            
            # Add to Bloom filter (convert to string for demonstration)
            bloom_filter.add(str(value))
            
            # Add to reservoir sample
            reservoir.add(value)
            
            total_records += 1
    
    # Store results
    results['statistics'] = stats.get_statistics()
    results['sample'] = reservoir.get_sample()
    results['total_records'] = total_records
    
    return results


def performance_comparison():
    """Compare performance of different large-scale processing techniques."""
    print("=== Large Scale Processing Performance Comparison ===\n")
    
    # Create sample large dataset generator
    def data_generator(chunk_size: int = 100000, num_chunks: int = 10):
        np.random.seed(42)
        for _ in range(num_chunks):
            yield np.random.randn(chunk_size).tolist()
    
    # Test external sorting
    print("1. External Sorting:")
    start_time = time.time()
    external_sort = ExternalSort(chunk_size=50000)
    # For demonstration, we'll just process one chunk
    sample_data = [np.random.randn(10000).tolist()]
    sorted_chunks = list(external_sort.sort_large_array(iter(sample_data)))
    sort_time = time.time() - start_time
    print(f"   External sort time: {sort_time:.6f} seconds")
    print(f"   Number of sorted chunks: {len(sorted_chunks)}")
    if sorted_chunks:
        print(f"   First chunk size: {len(sorted_chunks[0])}")
    
    # Test streaming statistics
    print("\n2. Streaming Statistics:")
    start_time = time.time()
    stats = StreamingStatistics()
    for chunk in data_generator(chunk_size=50000, num_chunks=5):
        for value in chunk:
            stats.update(value)
    streaming_time = time.time() - start_time
    final_stats = stats.get_statistics()
    print(f"   Streaming time: {streaming_time:.6f} seconds")
    print(f"   Records processed: {final_stats['count']}")
    print(f"   Mean: {final_stats['mean']:.4f}")
    print(f"   Std: {final_stats['std']:.4f}")
    
    # Test Bloom filter
    print("\n3. Bloom Filter:")
    start_time = time.time()
    bloom = BloomFilter(capacity=100000)
    test_data = [str(i) for i in range(50000)]
    for item in test_data:
        bloom.add(item)
    
    # Test membership (all should return True)
    false_positives = 0
    for i in range(50000, 60000):
        if bloom.contains(str(i)):  # These shouldn't be in the set
            false_positives += 1
    
    bloom_time = time.time() - start_time
    print(f"   Bloom filter time: {bloom_time:.6f} seconds")
    print(f"   False positive rate: {false_positives / 10000:.4f}")
    
    # Test reservoir sampling
    print("\n4. Reservoir Sampling:")
    start_time = time.time()
    reservoir = ReservoirSampling(sample_size=1000)
    for chunk in data_generator(chunk_size=30000, num_chunks=3):
        for value in chunk:
            reservoir.add(value)
    reservoir_time = time.time() - start_time
    sample = reservoir.get_sample()
    print(f"   Reservoir sampling time: {reservoir_time:.6f} seconds")
    print(f"   Sample size: {len(sample)}")
    print(f"   Sample mean: {np.mean(sample):.4f}")
    
    # Test sliding window
    print("\n5. Sliding Window Optimization:")
    window_data = np.random.randn(10000).tolist()
    start_time = time.time()
    window_results = sliding_window_optimization(window_data, window_size=100)
    window_time = time.time() - start_time
    print(f"   Sliding window time: {window_time:.6f} seconds")
    print(f"   Number of window averages: {len(window_results)}")
    print(f"   First average: {window_results[0]:.4f}")


def demo():
    """Demonstrate large-scale data processing techniques."""
    print("=== Large Scale Data Processing with DSA ===\n")
    
    # Create sample data stream
    def sample_data_stream():
        np.random.seed(42)
        for i in range(5):
            yield np.random.randn(1000).tolist()
    
    print("Sample data stream created with 5 chunks of 1000 random numbers each.")
    
    # Test external sorting (simplified)
    print("\n1. External Sorting:")
    external_sort = ExternalSort(chunk_size=500)
    # Process small sample
    sample_data = [np.random.randn(100).tolist()]
    sorted_chunks = list(external_sort.sort_large_array(iter(sample_data)))
    print(f"  Number of sorted chunks: {len(sorted_chunks)}")
    if sorted_chunks:
        print(f"  First 10 elements of first chunk: {sorted_chunks[0][:10]}")
    
    # Test streaming statistics
    print("\n2. Streaming Statistics:")
    stats = StreamingStatistics()
    for chunk in sample_data_stream():
        for value in chunk:
            stats.update(value)
    final_stats = stats.get_statistics()
    print(f"  Count: {final_stats['count']}")
    print(f"  Mean: {final_stats['mean']:.4f}")
    print(f"  Std: {final_stats['std']:.4f}")
    print(f"  Min: {final_stats['min']:.4f}")
    print(f"  Max: {final_stats['max']:.4f}")
    
    # Test Bloom filter
    print("\n3. Bloom Filter:")
    bloom = BloomFilter(capacity=1000)
    test_items = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    for item in test_items:
        bloom.add(item)
    
    # Test membership
    test_queries = ['apple', 'banana', 'grape', 'kiwi']
    for item in test_queries:
        result = bloom.contains(item)
        print(f"  '{item}' in filter: {result}")
    
    # Test reservoir sampling
    print("\n4. Reservoir Sampling:")
    reservoir = ReservoirSampling(sample_size=10)
    for chunk in sample_data_stream():
        for value in chunk:
            reservoir.add(value)
    sample = reservoir.get_sample()
    print(f"  Reservoir sample size: {len(sample)}")
    print(f"  Sample values: {sample[:5]}...")
    
    # Test sliding window
    print("\n5. Sliding Window:")
    window_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window_results = sliding_window_optimization(window_data, window_size=3)
    print(f"  Data: {window_data}")
    print(f"  Window averages (size 3): {window_results}")
    
    # Test complete pipeline
    print("\n6. Complete Data Stream Processing Pipeline:")
    pipeline_results = data_stream_processing_pipeline(sample_data_stream(), ['mean', 'sum'])
    print(f"  Pipeline results keys: {list(pipeline_results.keys())}")
    print(f"  Statistics: {pipeline_results['statistics']}")
    
    # Performance comparison
    print("\n" + "="*60)
    performance_comparison()


if __name__ == "__main__":
    demo()