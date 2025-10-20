"""
Pandas Optimization with DSA Principles

This module demonstrates how to optimize Pandas operations using Data Structures and Algorithms:
- Efficient data manipulation techniques
- Memory optimization strategies
- Performance improvement methods
- Best practices for large datasets
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by selecting appropriate data types.
    
    Time Complexity: O(n * m) where n is rows and m is columns
    Space Complexity: O(1) additional space
    
    Args:
        df: Input DataFrame
        
    Returns:
        Optimized DataFrame with reduced memory usage
    """
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type != object:
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            
            # Optimize integer columns
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
                # Keep as int64 if needed
            
            # Optimize float columns
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
        
        # Optimize object columns (categorical)
        else:
            # If number of unique values is less than 50% of total values
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
    
    return optimized_df


def efficient_merge(df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
    """
    Efficiently merge two DataFrames using sorting and binary search principles.
    
    Time Complexity: O(n log n + m log m) for sorting
    Space Complexity: O(n + m)
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        on: Column name to merge on
        
    Returns:
        Merged DataFrame
    """
    # Sort both DataFrames on merge column for efficient merging
    df1_sorted = df1.sort_values(on).reset_index(drop=True)
    df2_sorted = df2.sort_values(on).reset_index(drop=True)
    
    # Perform merge (Pandas internally uses efficient algorithms)
    return pd.merge(df1_sorted, df2_sorted, on=on)


def groupby_optimization(df: pd.DataFrame, group_col: str, agg_col: str) -> pd.DataFrame:
    """
    Optimize groupby operations using hash table principles.
    
    Time Complexity: O(n) average case
    Space Complexity: O(k) where k is number of groups
    
    Args:
        df: Input DataFrame
        group_col: Column to group by
        agg_col: Column to aggregate
        
    Returns:
        Aggregated DataFrame
    """
    # Use vectorized operations instead of apply
    return df.groupby(group_col)[agg_col].agg(['mean', 'sum', 'count'])


def efficient_filtering(df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    """
    Efficiently filter DataFrame using multiple conditions.
    
    Time Complexity: O(n) where n is number of rows
    Space Complexity: O(m) where m is number of filtered rows
    
    Args:
        df: Input DataFrame
        conditions: Dictionary of column-value pairs for filtering
        
    Returns:
        Filtered DataFrame
    """
    # Combine all conditions using boolean indexing
    mask = pd.Series([True] * len(df))
    
    for col, value in conditions.items():
        if isinstance(value, (list, tuple, set)):
            mask &= df[col].isin(value)
        else:
            mask &= (df[col] == value)
    
    return df[mask]


def vectorized_string_operations(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Perform vectorized string operations for better performance.
    
    Time Complexity: O(n) where n is number of rows
    Space Complexity: O(n)
    
    Args:
        df: Input DataFrame
        col: Column name for string operations
        
    Returns:
        DataFrame with processed string column
    """
    result_df = df.copy()
    
    # Use vectorized string methods instead of apply
    result_df[f'{col}_lower'] = df[col].str.lower()
    result_df[f'{col}_length'] = df[col].str.len()
    result_df[f'{col}_words'] = df[col].str.split().str.len()
    
    return result_df


def efficient_missing_value_handling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently handle missing values using appropriate strategies.
    
    Time Complexity: O(n * m) where n is rows and m is columns
    Space Complexity: O(1) additional space
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with handled missing values
    """
    result_df = df.copy()
    
    # For numerical columns, use median (less sensitive to outliers)
    numerical_cols = result_df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if result_df[col].isnull().any():
            median_value = result_df[col].median()
            result_df[col].fillna(median_value, inplace=True)
    
    # For categorical columns, use mode
    categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if result_df[col].isnull().any():
            mode_value = result_df[col].mode()
            if not mode_value.empty:
                result_df[col].fillna(mode_value[0], inplace=True)
    
    return result_df


def chunked_processing(df: pd.DataFrame, chunk_size: int = 10000) -> pd.DataFrame:
    """
    Process large DataFrames in chunks to manage memory usage.
    
    Time Complexity: O(n) where n is number of rows
    Space Complexity: O(chunk_size)
    
    Args:
        df: Input DataFrame
        chunk_size: Size of each chunk
        
    Returns:
        Processed DataFrame
    """
    # Process DataFrame in chunks
    processed_chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        # Apply processing to chunk
        processed_chunk = chunk.copy()  # Placeholder for actual processing
        processed_chunks.append(processed_chunk)
    
    # Combine chunks
    return pd.concat(processed_chunks, ignore_index=True)


def performance_comparison():
    """Compare performance of different approaches."""
    # Create sample data
    np.random.seed(42)
    size = 100000
    
    df1 = pd.DataFrame({
        'key': np.random.randint(0, 1000, size),
        'value1': np.random.randn(size),
        'category': np.random.choice(['A', 'B', 'C'], size)
    })
    
    df2 = pd.DataFrame({
        'key': np.random.randint(0, 1000, size//2),
        'value2': np.random.randn(size//2)
    })
    
    print("=== Pandas Optimization Performance Comparison ===\n")
    
    # Test memory optimization
    print("1. Memory Optimization:")
    original_memory = df1.memory_usage(deep=True).sum()
    optimized_df = optimize_dataframe_dtypes(df1)
    optimized_memory = optimized_df.memory_usage(deep=True).sum()
    memory_reduction = (original_memory - optimized_memory) / original_memory * 100
    print(f"   Original memory: {original_memory / 1024 / 1024:.2f} MB")
    print(f"   Optimized memory: {optimized_memory / 1024 / 1024:.2f} MB")
    print(f"   Memory reduction: {memory_reduction:.1f}%")
    
    # Test efficient merging
    print("\n2. Efficient Merging:")
    start_time = time.time()
    merged_df = efficient_merge(df1, df2, 'key')
    merge_time = time.time() - start_time
    print(f"   Merge time: {merge_time:.4f} seconds")
    print(f"   Result rows: {len(merged_df)}")
    
    # Test groupby optimization
    print("\n3. GroupBy Optimization:")
    start_time = time.time()
    grouped_df = groupby_optimization(df1, 'category', 'value1')
    groupby_time = time.time() - start_time
    print(f"   GroupBy time: {groupby_time:.4f} seconds")
    print(f"   Groups: {len(grouped_df)}")
    
    # Test efficient filtering
    print("\n4. Efficient Filtering:")
    start_time = time.time()
    filtered_df = efficient_filtering(df1, {'category': ['A', 'B']})
    filter_time = time.time() - start_time
    print(f"   Filter time: {filter_time:.4f} seconds")
    print(f"   Filtered rows: {len(filtered_df)}")


def demo():
    """Demonstrate pandas optimization techniques."""
    print("=== Pandas Optimization with DSA ===\n")
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'id': range(1000),
        'name': [f'Person_{i}' for i in range(1000)],
        'age': np.random.randint(18, 80, 1000),
        'salary': np.random.randint(30000, 150000, 1000),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 1000),
        'score': np.random.randn(1000)
    })
    
    print("Original DataFrame:")
    print(df.head())
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Optimize data types
    optimized_df = optimize_dataframe_dtypes(df)
    print(f"\nOptimized DataFrame memory: {optimized_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Efficient filtering
    filtered_df = efficient_filtering(optimized_df, {'department': 'IT'})
    print(f"\nFiltered DataFrame (IT department): {len(filtered_df)} rows")
    
    # GroupBy optimization
    grouped_df = groupby_optimization(optimized_df, 'department', 'salary')
    print(f"\nGroupBy results:")
    print(grouped_df)
    
    # Vectorized string operations
    string_df = vectorized_string_operations(optimized_df, 'name')
    print(f"\nString operations completed. New columns: {list(string_df.columns)}")
    
    # Performance comparison
    print("\n" + "="*60)
    performance_comparison()


if __name__ == "__main__":
    demo()