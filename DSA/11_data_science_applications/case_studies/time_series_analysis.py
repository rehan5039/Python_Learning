"""
Time Series Analysis Case Study

This case study demonstrates the application of Data Structures and Algorithms in time series analysis:
- Efficient time series storage and retrieval
- Pattern recognition and anomaly detection
- Forecasting algorithms optimization
- Sliding window techniques
- Memory-efficient processing of large time series
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from collections import deque
import heapq
import time


class TimeSeriesBuffer:
    """
    Efficient time series data structure using circular buffer for streaming data.
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = np.zeros(capacity)
        self.timestamps = np.zeros(capacity, dtype='datetime64[ns]')
        self.size = 0
        self.head = 0
        self.tail = 0
    
    def append(self, timestamp: np.datetime64, value: float) -> None:
        """
        Append new data point to time series.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.buffer[self.tail] = value
        self.timestamps[self.tail] = timestamp
        self.tail = (self.tail + 1) % self.capacity
        
        if self.size < self.capacity:
            self.size += 1
        else:
            # Buffer is full, move head
            self.head = (self.head + 1) % self.capacity
    
    def get_range(self, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data points in specified range.
        
        Time Complexity: O(k) where k is range size
        Space Complexity: O(k)
        """
        if start_idx < 0 or end_idx >= self.size or start_idx > end_idx:
            raise IndexError("Invalid range")
        
        actual_start = (self.head + start_idx) % self.capacity
        actual_end = (self.head + end_idx) % self.capacity
        
        if actual_start <= actual_end:
            values = self.buffer[actual_start:actual_end+1]
            timestamps = self.timestamps[actual_start:actual_end+1]
        else:
            # Wrap around case
            values = np.concatenate([
                self.buffer[actual_start:],
                self.buffer[:actual_end+1]
            ])
            timestamps = np.concatenate([
                self.timestamps[actual_start:],
                self.timestamps[:actual_end+1]
            ])
        
        return timestamps, values
    
    def get_last_n(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get last n data points."""
        if n > self.size:
            n = self.size
        start_idx = self.size - n
        return self.get_range(start_idx, self.size - 1)
    
    def __len__(self) -> int:
        return self.size


class SlidingWindowStatistics:
    """
    Efficient computation of statistics over sliding windows using incremental algorithms.
    """
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.window = deque()
        self.sum = 0.0
        self.sum_squares = 0.0
    
    def update(self, value: float) -> None:
        """
        Update window with new value.
        
        Time Complexity: O(1) amortized
        Space Complexity: O(window_size)
        """
        self.window.append(value)
        self.sum += value
        self.sum_squares += value * value
        
        # Remove old values if window is full
        if len(self.window) > self.window_size:
            old_value = self.window.popleft()
            self.sum -= old_value
            self.sum_squares -= old_value * old_value
    
    def get_mean(self) -> float:
        """Get current window mean."""
        if not self.window:
            return 0.0
        return self.sum / len(self.window)
    
    def get_variance(self) -> float:
        """Get current window variance."""
        if len(self.window) < 2:
            return 0.0
        mean = self.get_mean()
        return (self.sum_squares / len(self.window)) - (mean * mean)
    
    def get_std(self) -> float:
        """Get current window standard deviation."""
        return np.sqrt(self.get_variance())


class AnomalyDetector:
    """
    Anomaly detection using statistical methods and efficient data structures.
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.stats = SlidingWindowStatistics(window_size)
        self.anomalies = []
    
    def detect(self, timestamp: np.datetime64, value: float) -> bool:
        """
        Detect if current value is anomalous.
        
        Time Complexity: O(1)
        Space Complexity: O(window_size)
        """
        # Update statistics
        self.stats.update(value)
        
        # Check if we have enough data
        if len(self.stats.window) < self.window_size:
            return False
        
        # Calculate z-score
        mean = self.stats.get_mean()
        std = self.stats.get_std()
        
        if std == 0:
            return False
        
        z_score = abs(value - mean) / std
        
        # Detect anomaly
        is_anomaly = z_score > self.threshold
        
        if is_anomaly:
            self.anomalies.append((timestamp, value, z_score))
        
        return is_anomaly


class TimeSeriesForecaster:
    """
    Time series forecasting using efficient algorithms.
    """
    
    def __init__(self, method: str = 'moving_average'):
        self.method = method
        self.history = []
    
    def fit(self, timestamps: np.ndarray, values: np.ndarray) -> 'TimeSeriesForecaster':
        """
        Fit forecaster to historical data.
        
        Time Complexity: O(n) where n is data size
        Space Complexity: O(n)
        """
        self.history = list(zip(timestamps, values))
        return self
    
    def forecast_moving_average(self, n_periods: int, window_size: int = 10) -> List[float]:
        """
        Forecast using moving average.
        
        Time Complexity: O(window_size + n_periods)
        Space Complexity: O(window_size)
        """
        if len(self.history) < window_size:
            return []
        
        # Get last window_size values
        recent_values = [value for _, value in self.history[-window_size:]]
        avg = np.mean(recent_values)
        
        # Forecast constant value
        return [avg] * n_periods
    
    def forecast_exponential_smoothing(self, n_periods: int, alpha: float = 0.3) -> List[float]:
        """
        Forecast using exponential smoothing.
        
        Time Complexity: O(n + n_periods) where n is history size
        Space Complexity: O(1)
        """
        if not self.history:
            return []
        
        # Initialize with first value
        forecast = self.history[0][1]
        
        # Fit the model
        for _, value in self.history:
            forecast = alpha * value + (1 - alpha) * forecast
        
        # Forecast constant value
        return [forecast] * n_periods
    
    def forecast(self, n_periods: int, **kwargs) -> List[float]:
        """Forecast future values."""
        if self.method == 'moving_average':
            window_size = kwargs.get('window_size', 10)
            return self.forecast_moving_average(n_periods, window_size)
        elif self.method == 'exponential_smoothing':
            alpha = kwargs.get('alpha', 0.3)
            return self.forecast_exponential_smoothing(n_periods, alpha)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def generate_sample_time_series(n_points: int = 10000, frequency: str = '1H') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample time series data with trends and seasonality.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Generate timestamps
    start_time = np.datetime64('2023-01-01T00:00:00')
    timestamps = start_time + np.arange(n_points) * np.timedelta64(1, frequency)
    
    # Generate values with trend, seasonality, and noise
    trend = 0.1 * np.arange(n_points)  # Linear trend
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily seasonality
    noise = np.random.normal(0, 2, n_points)  # Random noise
    
    values = trend + seasonal + noise
    
    return timestamps, values


def performance_evaluation():
    """Evaluate performance of time series algorithms."""
    print("=== Time Series Analysis Performance Evaluation ===\n")
    
    # Generate sample data
    print("1. Generating Sample Time Series:")
    timestamps, values = generate_sample_time_series(n_points=50000, frequency='1T')
    print(f"   Generated {len(timestamps)} data points")
    print(f"   Time range: {timestamps[0]} to {timestamps[-1]}")
    print(f"   Value range: {np.min(values):.2f} to {np.max(values):.2f}")
    
    # Test TimeSeriesBuffer
    print("\n2. TimeSeriesBuffer Performance:")
    start_time = time.time()
    buffer = TimeSeriesBuffer(capacity=100000)
    for i in range(len(timestamps)):
        buffer.append(timestamps[i], values[i])
    buffer_time = time.time() - start_time
    print(f"   Buffer insertion time: {buffer_time:.4f} seconds")
    print(f"   Buffer size: {len(buffer)}")
    
    # Test range query
    start_time = time.time()
    range_timestamps, range_values = buffer.get_range(1000, 2000)
    range_time = time.time() - start_time
    print(f"   Range query time: {range_time:.6f} seconds")
    print(f"   Range size: {len(range_values)}")
    
    # Test SlidingWindowStatistics
    print("\n3. SlidingWindowStatistics Performance:")
    start_time = time.time()
    stats = SlidingWindowStatistics(window_size=1000)
    for value in values[:10000]:
        stats.update(value)
    stats_time = time.time() - start_time
    print(f"   Statistics update time: {stats_time:.4f} seconds")
    print(f"   Mean: {stats.get_mean():.4f}")
    print(f"   Std: {stats.get_std():.4f}")
    
    # Test AnomalyDetector
    print("\n4. AnomalyDetector Performance:")
    start_time = time.time()
    detector = AnomalyDetector(window_size=100, threshold=3.0)
    anomalies_found = 0
    for i in range(min(10000, len(timestamps))):
        is_anomaly = detector.detect(timestamps[i], values[i])
        if is_anomaly:
            anomalies_found += 1
    detector_time = time.time() - start_time
    print(f"   Anomaly detection time: {detector_time:.4f} seconds")
    print(f"   Anomalies found: {anomalies_found}")
    print(f"   Total anomalies recorded: {len(detector.anomalies)}")
    
    # Test TimeSeriesForecaster
    print("\n5. TimeSeriesForecaster Performance:")
    forecaster = TimeSeriesForecaster(method='moving_average')
    forecaster.fit(timestamps[:1000], values[:1000])
    
    start_time = time.time()
    ma_forecast = forecaster.forecast(n_periods=100, window_size=50)
    ma_time = time.time() - start_time
    print(f"   Moving average forecast time: {ma_time:.6f} seconds")
    print(f"   Forecast values: {ma_forecast[:5]}...")
    
    forecaster = TimeSeriesForecaster(method='exponential_smoothing')
    forecaster.fit(timestamps[:1000], values[:1000])
    
    start_time = time.time()
    es_forecast = forecaster.forecast(n_periods=100, alpha=0.3)
    es_time = time.time() - start_time
    print(f"   Exponential smoothing forecast time: {es_time:.6f} seconds")
    print(f"   Forecast values: {es_forecast[:5]}...")


def demo():
    """Demonstrate time series analysis case study."""
    print("=== Time Series Analysis Case Study ===\n")
    
    # Generate sample time series
    timestamps, values = generate_sample_time_series(n_points=1000, frequency='1H')
    print("Sample time series generated:")
    print(f"  Data points: {len(timestamps)}")
    print(f"  Time range: {timestamps[0]} to {timestamps[-1]}")
    print(f"  Value statistics: mean={np.mean(values):.2f}, std={np.std(values):.2f}")
    
    # Demonstrate TimeSeriesBuffer
    print("\n1. TimeSeriesBuffer:")
    buffer = TimeSeriesBuffer(capacity=2000)
    for i in range(len(timestamps)):
        buffer.append(timestamps[i], values[i])
    
    print(f"  Buffer size: {len(buffer)}")
    recent_timestamps, recent_values = buffer.get_last_n(5)
    print(f"  Last 5 values: {recent_values}")
    
    # Demonstrate SlidingWindowStatistics
    print("\n2. SlidingWindowStatistics:")
    stats = SlidingWindowStatistics(window_size=100)
    for value in values[:200]:
        stats.update(value)
    
    print(f"  Window mean: {stats.get_mean():.4f}")
    print(f"  Window std: {stats.get_std():.4f}")
    print(f"  Window size: {len(stats.window)}")
    
    # Demonstrate AnomalyDetector
    print("\n3. AnomalyDetector:")
    detector = AnomalyDetector(window_size=50, threshold=2.5)
    anomalies = []
    for i in range(min(300, len(timestamps))):
        is_anomaly = detector.detect(timestamps[i], values[i])
        if is_anomaly:
            anomalies.append((timestamps[i], values[i]))
    
    print(f"  Anomalies detected: {len(anomalies)}")
    if anomalies:
        print(f"  First anomaly: {anomalies[0]}")
    
    # Demonstrate TimeSeriesForecaster
    print("\n4. TimeSeriesForecaster:")
    forecaster = TimeSeriesForecaster(method='moving_average')
    forecaster.fit(timestamps[:200], values[:200])
    
    ma_forecast = forecaster.forecast(n_periods=10, window_size=20)
    print(f"  Moving average forecast: {ma_forecast[:5]}...")
    
    forecaster = TimeSeriesForecaster(method='exponential_smoothing')
    forecaster.fit(timestamps[:200], values[:200])
    
    es_forecast = forecaster.forecast(n_periods=10, alpha=0.3)
    print(f"  Exponential smoothing forecast: {es_forecast[:5]}...")
    
    # Performance evaluation
    print("\n" + "="*60)
    performance_evaluation()


if __name__ == "__main__":
    demo()