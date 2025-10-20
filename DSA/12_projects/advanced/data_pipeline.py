"""
Advanced Project: Data Pipeline Optimizer

This project implements an optimized data processing pipeline that demonstrates
advanced DSA concepts in a real-world data science context.

Concepts covered:
- Pipeline architecture and design patterns
- Memory-efficient data processing
- Parallel processing and concurrency
- Performance optimization techniques
- Error handling and fault tolerance
- Streaming data processing
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Iterator, Generator, Callable, Optional
from collections import deque, defaultdict
import threading
import queue
import time
import hashlib
from abc import ABC, abstractmethod


class DataProcessor(ABC):
    """
    Abstract base class for data processors in the pipeline.
    """
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process data and return result.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get processor name for logging and identification."""
        pass


class DataPipeline:
    """
    Optimized data processing pipeline with multiple stages.
    """
    
    def __init__(self, name: str = "DataPipeline"):
        self.name = name
        self.processors: List[DataProcessor] = []
        self.stats = defaultdict(int)
        self.errors = []
    
    def add_processor(self, processor: DataProcessor) -> 'DataPipeline':
        """
        Add processor to pipeline.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.processors.append(processor)
        return self
    
    def process_single(self, data: Any) -> Any:
        """
        Process single data item through all processors.
        
        Time Complexity: O(p) where p is number of processors
        Space Complexity: O(1)
        """
        start_time = time.time()
        
        try:
            for processor in self.processors:
                processor_start = time.time()
                data = processor.process(data)
                processor_time = time.time() - processor_start
                self.stats[f"{processor.get_name()}_time"] += processor_time
                self.stats[f"{processor.get_name()}_count"] += 1
        except Exception as e:
            self.errors.append(f"Error in {processor.get_name()}: {str(e)}")
            raise
        
        total_time = time.time() - start_time
        self.stats["total_processing_time"] += total_time
        self.stats["items_processed"] += 1
        
        return data
    
    def process_batch(self, data_list: List[Any]) -> List[Any]:
        """
        Process batch of data items.
        
        Time Complexity: O(n * p) where n is items, p is processors
        Space Complexity: O(n)
        """
        results = []
        for data in data_list:
            try:
                result = self.process_single(data)
                results.append(result)
            except Exception as e:
                self.errors.append(f"Batch processing error: {str(e)}")
                results.append(None)  # Placeholder for failed item
        
        return results
    
    def process_stream(self, data_stream: Iterator[Any]) -> Generator[Any, None, None]:
        """
        Process streaming data.
        
        Time Complexity: O(1) per item
        Space Complexity: O(1)
        """
        for data in data_stream:
            try:
                yield self.process_single(data)
            except Exception as e:
                self.errors.append(f"Stream processing error: {str(e)}")
                yield None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return dict(self.stats)
    
    def get_errors(self) -> List[str]:
        """Get processing errors."""
        return self.errors.copy()


class ParallelDataPipeline:
    """
    Parallel data processing pipeline using threading for concurrent processing.
    """
    
    def __init__(self, name: str = "ParallelDataPipeline", num_workers: int = 4):
        self.name = name
        self.num_workers = num_workers
        self.processors: List[DataProcessor] = []
        self.stats = defaultdict(int)
        self.errors = []
        self.lock = threading.Lock()
    
    def add_processor(self, processor: DataProcessor) -> 'ParallelDataPipeline':
        """Add processor to pipeline."""
        self.processors.append(processor)
        return self
    
    def _worker(self, work_queue: queue.Queue, result_queue: queue.Queue):
        """Worker thread function."""
        while True:
            try:
                item_id, data = work_queue.get(timeout=1)
                if data is None:  # Poison pill
                    break
                
                start_time = time.time()
                try:
                    for processor in self.processors:
                        processor_start = time.time()
                        data = processor.process(data)
                        processor_time = time.time() - processor_start
                        with self.lock:
                            self.stats[f"{processor.get_name()}_time"] += processor_time
                            self.stats[f"{processor.get_name()}_count"] += 1
                
                except Exception as e:
                    with self.lock:
                        self.errors.append(f"Error in {processor.get_name()}: {str(e)}")
                    result_queue.put((item_id, None))
                    work_queue.task_done()
                    continue
                
                total_time = time.time() - start_time
                with self.lock:
                    self.stats["total_processing_time"] += total_time
                    self.stats["items_processed"] += 1
                
                result_queue.put((item_id, data))
                work_queue.task_done()
                
            except queue.Empty:
                continue
    
    def process_batch_parallel(self, data_list: List[Any]) -> List[Any]:
        """
        Process batch of data items in parallel.
        
        Time Complexity: O(n * p / w) where n is items, p is processors, w is workers
        Space Complexity: O(n)
        """
        if not data_list:
            return []
        
        # Create queues
        work_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add work items to queue
        for i, data in enumerate(data_list):
            work_queue.put((i, data))
        
        # Add poison pills
        for _ in range(self.num_workers):
            work_queue.put((None, None))
        
        # Start worker threads
        threads = []
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._worker, args=(work_queue, result_queue))
            t.start()
            threads.append(t)
        
        # Wait for completion
        work_queue.join()
        
        # Collect results
        results = [None] * len(data_list)
        while not result_queue.empty():
            item_id, result = result_queue.get()
            if item_id is not None:
                results[item_id] = result
        
        # Wait for threads to finish
        for t in threads:
            t.join()
        
        return results


# Example processors for demonstration
class DataCleaner(DataProcessor):
    """Processor for cleaning data."""
    
    def process(self, data: Any) -> Any:
        if isinstance(data, str):
            return data.strip().lower()
        elif isinstance(data, (list, tuple)):
            return [self.process(item) for item in data]
        elif isinstance(data, dict):
            return {k: self.process(v) for k, v in data.items()}
        else:
            return data
    
    def get_name(self) -> str:
        return "DataCleaner"


class DataValidator(DataProcessor):
    """Processor for validating data."""
    
    def __init__(self, required_fields: List[str] = None):
        self.required_fields = required_fields or []
    
    def process(self, data: Any) -> Any:
        if isinstance(data, dict):
            # Check required fields
            for field in self.required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate data types
            for key, value in data.items():
                if key == "age" and not isinstance(value, (int, float)):
                    raise ValueError(f"Invalid age type: {type(value)}")
                elif key == "email" and not isinstance(value, str):
                    raise ValueError(f"Invalid email type: {type(value)}")
        
        return data
    
    def get_name(self) -> str:
        return "DataValidator"


class DataTransformer(DataProcessor):
    """Processor for transforming data."""
    
    def __init__(self, transformations: Dict[str, Callable] = None):
        self.transformations = transformations or {}
    
    def process(self, data: Any) -> Any:
        if isinstance(data, dict):
            result = data.copy()
            for field, transform_func in self.transformations.items():
                if field in result:
                    result[field] = transform_func(result[field])
            return result
        else:
            return data
    
    def get_name(self) -> str:
        return "DataTransformer"


class DataAggregator(DataProcessor):
    """Processor for aggregating data."""
    
    def __init__(self):
        self.aggregates = defaultdict(list)
    
    def process(self, data: Any) -> Any:
        if isinstance(data, dict):
            # Aggregate by category
            category = data.get("category", "unknown")
            self.aggregates[category].append(data)
            
            # Return summary statistics
            return {
                "category": category,
                "count": len(self.aggregates[category]),
                "latest_item": data
            }
        else:
            return data
    
    def get_name(self) -> str:
        return "DataAggregator"


def generate_sample_data(n_records: int = 10000) -> List[Dict[str, Any]]:
    """Generate sample data for testing."""
    np.random.seed(42)
    data = []
    
    categories = ["A", "B", "C", "D"]
    for i in range(n_records):
        record = {
            "id": i,
            "name": f"User {i}",
            "age": np.random.randint(18, 80),
            "category": np.random.choice(categories),
            "score": np.random.randn(),
            "email": f"user{i}@example.com"
        }
        data.append(record)
    
    return data


def demonstrate_data_pipeline():
    """Demonstrate data pipeline functionality."""
    print("=== Data Pipeline Optimizer Demo ===\n")
    
    # Generate sample data
    sample_data = generate_sample_data(1000)
    print(f"Generated {len(sample_data)} sample records")
    
    # Create sequential pipeline
    print("\n1. Sequential Pipeline:")
    sequential_pipeline = DataPipeline("SequentialPipeline")
    sequential_pipeline.add_processor(DataCleaner()) \
                      .add_processor(DataValidator(required_fields=["id", "name", "age"])) \
                      .add_processor(DataTransformer({
                          "age": lambda x: max(0, x),  # Ensure non-negative age
                          "email": lambda x: x.lower()  # Normalize email
                      }))
    
    # Process sample data
    start_time = time.time()
    results = sequential_pipeline.process_batch(sample_data[:100])
    sequential_time = time.time() - start_time
    
    print(f"  Processed 100 records in {sequential_time:.4f} seconds")
    print(f"  Success rate: {sum(1 for r in results if r is not None)}/100")
    
    stats = sequential_pipeline.get_stats()
    print(f"  Total processing time: {stats.get('total_processing_time', 0):.4f} seconds")
    
    # Create parallel pipeline
    print("\n2. Parallel Pipeline:")
    parallel_pipeline = ParallelDataPipeline("ParallelPipeline", num_workers=4)
    parallel_pipeline.add_processor(DataCleaner()) \
                     .add_processor(DataValidator(required_fields=["id", "name", "age"])) \
                     .add_processor(DataTransformer({
                         "age": lambda x: max(0, x),
                         "email": lambda x: x.lower()
                     }))
    
    # Process sample data in parallel
    start_time = time.time()
    parallel_results = parallel_pipeline.process_batch_parallel(sample_data[:100])
    parallel_time = time.time() - start_time
    
    print(f"  Processed 100 records in {parallel_time:.4f} seconds")
    print(f"  Success rate: {sum(1 for r in parallel_results if r is not None)}/100")
    print(f"  Speedup: {sequential_time / parallel_time:.2f}x")
    
    stats = parallel_pipeline.get_stats()
    print(f"  Total processing time: {stats.get('total_processing_time', 0):.4f} seconds")
    
    # Test streaming processing
    print("\n3. Streaming Processing:")
    def data_stream():
        for record in sample_data[:50]:
            yield record
    
    stream_pipeline = DataPipeline("StreamPipeline")
    stream_pipeline.add_processor(DataCleaner()) \
                   .add_processor(DataValidator(required_fields=["id", "name"])) \
                   .add_processor(DataTransformer({
                       "name": lambda x: x.upper()
                   }))
    
    start_time = time.time()
    stream_results = list(stream_pipeline.process_stream(data_stream()))
    stream_time = time.time() - start_time
    
    print(f"  Processed 50 records in {stream_time:.4f} seconds")
    print(f"  Success rate: {sum(1 for r in stream_results if r is not None)}/50")
    
    # Test aggregation
    print("\n4. Data Aggregation:")
    agg_pipeline = DataPipeline("AggregationPipeline")
    aggregator = DataAggregator()
    agg_pipeline.add_processor(aggregator)
    
    # Process data through aggregator
    agg_results = agg_pipeline.process_batch(sample_data[:200])
    
    print(f"  Processed 200 records for aggregation")
    print(f"  Categories found: {list(aggregator.aggregates.keys())}")
    for category, records in list(aggregator.aggregates.items())[:3]:
        print(f"  Category {category}: {len(records)} records")


def performance_comparison():
    """Compare performance of different pipeline approaches."""
    print("\n=== Performance Comparison ===\n")
    
    # Generate larger dataset
    large_data = generate_sample_data(5000)
    
    # Test sequential processing
    print("1. Sequential Processing Performance:")
    sequential_pipeline = DataPipeline("Sequential")
    sequential_pipeline.add_processor(DataCleaner()) \
                      .add_processor(DataValidator()) \
                      .add_processor(DataTransformer())
    
    start_time = time.time()
    seq_results = sequential_pipeline.process_batch(large_data)
    sequential_time = time.time() - start_time
    
    success_count = sum(1 for r in seq_results if r is not None)
    print(f"   Processed {len(large_data)} records in {sequential_time:.4f} seconds")
    print(f"   Success rate: {success_count}/{len(large_data)} ({success_count/len(large_data)*100:.1f}%)")
    print(f"   Throughput: {len(large_data)/sequential_time:.1f} records/second")
    
    # Test parallel processing with different worker counts
    worker_counts = [1, 2, 4, 8]
    for workers in worker_counts:
        print(f"\n2. Parallel Processing with {workers} Workers:")
        parallel_pipeline = ParallelDataPipeline("Parallel", num_workers=workers)
        parallel_pipeline.add_processor(DataCleaner()) \
                         .add_processor(DataValidator()) \
                         .add_processor(DataTransformer())
        
        start_time = time.time()
        par_results = parallel_pipeline.process_batch_parallel(large_data)
        parallel_time = time.time() - start_time
        
        success_count = sum(1 for r in par_results if r is not None)
        print(f"   Processed {len(large_data)} records in {parallel_time:.4f} seconds")
        print(f"   Success rate: {success_count}/{len(large_data)} ({success_count/len(large_data)*100:.1f}%)")
        print(f"   Throughput: {len(large_data)/parallel_time:.1f} records/second")
        print(f"   Speedup vs sequential: {sequential_time/parallel_time:.2f}x")


if __name__ == "__main__":
    demonstrate_data_pipeline()
    performance_comparison()