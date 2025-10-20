"""
Non-Linear Data Structures - Hash Tables
===================================

This module provides implementations and examples of hash table data structures,
with a focus on Python-specific implementations and data science applications.

Topics Covered:
- Hash table concepts and collision resolution
- Hash functions and their properties
- Hash table implementations (chaining, open addressing)
- Performance analysis and load factors
- Applications in algorithms and data science
"""

from typing import Any, List, Optional, Tuple, Union
import hashlib

class HashTable:
    """
    Implementation of a Hash Table using chaining for collision resolution
    """
    
    def __init__(self, initial_capacity: int = 16):
        """
        Initialize a hash table
        
        Args:
            initial_capacity: Initial size of the hash table
        """
        self.capacity = initial_capacity
        self.size = 0
        # Each bucket is a list of (key, value) pairs
        self.buckets: List[List[Tuple[Any, Any]]] = [[] for _ in range(self.capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, key: Any) -> int:
        """
        Hash function to compute index for a key
        Time Complexity: O(1) average, O(k) where k is key length
        """
        # Convert key to string and hash it
        key_str = str(key)
        hash_value = hash(key_str) % self.capacity
        return hash_value
    
    def _resize(self) -> None:
        """
        Resize the hash table when load factor exceeds threshold
        Time Complexity: O(n) where n is number of elements
        """
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all existing elements
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update a key-value pair
        Time Complexity: O(1) average, O(n) worst case
        """
        # Check if resize is needed
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        index = self._hash(key)
        bucket = self.buckets[index]
        
        # Check if key already exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Update existing
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self.size += 1
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value for a key
        Time Complexity: O(1) average, O(n) worst case
        """
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return None  # Key not found
    
    def delete(self, key: Any) -> bool:
        """
        Delete a key-value pair
        Time Complexity: O(1) average, O(n) worst case
        """
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return True
        
        return False  # Key not found
    
    def contains(self, key: Any) -> bool:
        """
        Check if key exists in hash table
        Time Complexity: O(1) average, O(n) worst case
        """
        return self.get(key) is not None
    
    def keys(self) -> List[Any]:
        """
        Get all keys in the hash table
        Time Complexity: O(n)
        """
        result = []
        for bucket in self.buckets:
            for key, _ in bucket:
                result.append(key)
        return result
    
    def values(self) -> List[Any]:
        """
        Get all values in the hash table
        Time Complexity: O(n)
        """
        result = []
        for bucket in self.buckets:
            for _, value in bucket:
                result.append(value)
        return result
    
    def items(self) -> List[Tuple[Any, Any]]:
        """
        Get all key-value pairs in the hash table
        Time Complexity: O(n)
        """
        result = []
        for bucket in self.buckets:
            for key, value in bucket:
                result.append((key, value))
        return result
    
    def load_factor(self) -> float:
        """
        Calculate current load factor
        Time Complexity: O(1)
        """
        return self.size / self.capacity
    
    def __str__(self) -> str:
        """String representation of the hash table"""
        items = self.items()
        return f"HashTable(Size: {self.size}, Capacity: {self.capacity}, Load Factor: {self.load_factor():.2f})"

class LinearProbingHashTable:
    """
    Implementation of a Hash Table using linear probing for collision resolution
    """
    
    def __init__(self, initial_capacity: int = 16):
        """
        Initialize a hash table with linear probing
        """
        self.capacity = initial_capacity
        self.size = 0
        # Each slot is either None, (key, value), or DELETED marker
        self.slots: List[Optional[Tuple[Any, Any]]] = [None] * self.capacity
        self.DELETED = object()  # Marker for deleted slots
        self.load_factor_threshold = 0.5  # Lower threshold for linear probing
    
    def _hash(self, key: Any) -> int:
        """
        Hash function to compute initial index for a key
        """
        key_str = str(key)
        return hash(key_str) % self.capacity
    
    def _find_slot(self, key: Any) -> Tuple[int, bool]:
        """
        Find slot for a key using linear probing
        Returns (index, found) where found indicates if key exists
        """
        index = self._hash(key)
        original_index = index
        
        while self.slots[index] is not None:
            if self.slots[index] != self.DELETED and self.slots[index][0] == key:
                return index, True  # Found existing key
            
            index = (index + 1) % self.capacity
            
            # Check if we've wrapped around completely
            if index == original_index:
                break
        
        return index, False  # Found empty slot
    
    def _resize(self) -> None:
        """
        Resize the hash table when load factor exceeds threshold
        """
        old_slots = self.slots
        self.capacity *= 2
        self.size = 0
        self.slots = [None] * self.capacity
        
        # Rehash all existing elements
        for slot in old_slots:
            if slot is not None and slot != self.DELETED:
                self.put(slot[0], slot[1])
    
    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update a key-value pair
        Time Complexity: O(1) average, O(n) worst case
        """
        # Check if resize is needed
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        index, found = self._find_slot(key)
        
        if found:
            # Update existing key
            self.slots[index] = (key, value)
        else:
            # Insert new key-value pair
            self.slots[index] = (key, value)
            self.size += 1
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value for a key
        Time Complexity: O(1) average, O(n) worst case
        """
        index, found = self._find_slot(key)
        if found:
            return self.slots[index][1]
        return None
    
    def delete(self, key: Any) -> bool:
        """
        Delete a key-value pair
        Time Complexity: O(1) average, O(n) worst case
        """
        index, found = self._find_slot(key)
        if found:
            self.slots[index] = self.DELETED
            self.size -= 1
            return True
        return False
    
    def contains(self, key: Any) -> bool:
        """
        Check if key exists in hash table
        Time Complexity: O(1) average, O(n) worst case
        """
        _, found = self._find_slot(key)
        return found
    
    def keys(self) -> List[Any]:
        """
        Get all keys in the hash table
        Time Complexity: O(n)
        """
        result = []
        for slot in self.slots:
            if slot is not None and slot != self.DELETED:
                result.append(slot[0])
        return result
    
    def values(self) -> List[Any]:
        """
        Get all values in the hash table
        Time Complexity: O(n)
        """
        result = []
        for slot in self.slots:
            if slot is not None and slot != self.DELETED:
                result.append(slot[1])
        return result
    
    def load_factor(self) -> float:
        """
        Calculate current load factor
        Time Complexity: O(1)
        """
        return self.size / self.capacity
    
    def __str__(self) -> str:
        """String representation of the hash table"""
        items = [(k, v) for k, v in zip(self.keys(), self.values())]
        return f"LinearProbingHashTable(Size: {self.size}, Capacity: {self.capacity}, Load Factor: {self.load_factor():.2f})"

def hash_table_demo():
    """
    Demonstrate hash table operations and their complexities
    """
    print("=== Hash Table Demo ===")
    
    # Chaining hash table
    print("1. Chaining Hash Table:")
    ht = HashTable()
    
    # Insert key-value pairs
    data = [("apple", 5), ("banana", 3), ("cherry", 8), ("date", 2), ("elderberry", 7)]
    print(f"   Inserting: {data}")
    
    for key, value in data:
        ht.put(key, value)
        print(f"   After inserting ({key}, {value}): {ht}")
        print(f"   Load factor: {ht.load_factor():.2f}")
    
    # Retrieve values
    print("\n2. Retrieving Values:")
    for key, _ in data:
        value = ht.get(key)
        print(f"   {key}: {value}")
    
    # Check containment
    print("\n3. Key Containment:")
    test_keys = ["apple", "grape", "cherry"]
    for key in test_keys:
        exists = ht.contains(key)
        print(f"   '{key}' exists: {exists}")
    
    # Delete operations
    print("\n4. Delete Operations:")
    delete_keys = ["banana", "date"]
    for key in delete_keys:
        deleted = ht.delete(key)
        print(f"   Delete '{key}': {deleted}")
        print(f"   After deletion: {ht}")
        print(f"   Load factor: {ht.load_factor():.2f}")

def linear_probing_demo():
    """
    Demonstrate linear probing hash table
    """
    print("\n=== Linear Probing Hash Table Demo ===")
    
    lht = LinearProbingHashTable()
    
    # Insert key-value pairs
    data = [("A", 1), ("B", 2), ("C", 3), ("D", 4), ("E", 5)]
    print(f"1. Inserting: {data}")
    
    for key, value in data:
        lht.put(key, value)
        print(f"   After inserting ({key}, {value}): {lht}")
        print(f"   Load factor: {lht.load_factor():.2f}")
    
    # Retrieve values
    print("\n2. Retrieving Values:")
    for key, _ in data:
        value = lht.get(key)
        print(f"   {key}: {value}")
    
    # Delete operations
    print("\n3. Delete Operations:")
    delete_keys = ["B", "D"]
    for key in delete_keys:
        deleted = lht.delete(key)
        print(f"   Delete '{key}': {deleted}")
        print(f"   After deletion: {lht}")
        print(f"   Load factor: {lht.load_factor():.2f}")

def hash_function_examples():
    """
    Demonstrate different hash functions and their properties
    """
    print("\n=== Hash Function Examples ===")
    
    # Simple hash function
    def simple_hash(key: str, table_size: int) -> int:
        """Simple hash function using sum of ASCII values"""
        return sum(ord(char) for char in key) % table_size
    
    # Better hash function (djb2)
    def djb2_hash(key: str, table_size: int) -> int:
        """djb2 hash function - better distribution"""
        hash_value = 5381
        for char in key:
            hash_value = ((hash_value << 5) + hash_value) + ord(char)
        return hash_value % table_size
    
    # Python's built-in hash
    def builtin_hash(key: str, table_size: int) -> int:
        """Python's built-in hash function"""
        return hash(key) % table_size
    
    # Test with sample data
    keys = ["apple", "banana", "cherry", "date", "elderberry"]
    table_size = 10
    
    print("1. Hash Function Comparison:")
    print(f"   Keys: {keys}")
    print(f"   Table size: {table_size}")
    
    print("\n   Simple hash:")
    for key in keys:
        hash_val = simple_hash(key, table_size)
        print(f"   '{key}' -> {hash_val}")
    
    print("\n   DJB2 hash:")
    for key in keys:
        hash_val = djb2_hash(key, table_size)
        print(f"   '{key}' -> {hash_val}")
    
    print("\n   Built-in hash:")
    for key in keys:
        hash_val = builtin_hash(key, table_size)
        print(f"   '{key}' -> {hash_val}")

def hash_table_applications():
    """
    Demonstrate common applications of hash tables
    """
    print("\n=== Hash Table Applications ===")
    
    # 1. Caching/Memoization
    print("1. Caching/Memoization:")
    print("   Hash tables store computed results for reuse")
    print("   Avoid redundant calculations in recursive algorithms")
    print("   Example: Fibonacci with memoization")
    
    def fibonacci_memo(n: int, memo: dict = None) -> int:
        if memo is None:
            memo = {}
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
        return memo[n]
    
    print(f"   Fibonacci(10) with memoization: {fibonacci_memo(10)}")
    
    # 2. Database indexing
    print("\n2. Database Indexing:")
    print("   Hash indexes provide O(1) average lookup time")
    print("   Used for exact match queries")
    print("   Trade-off: Fast lookups but no range queries")
    
    # 3. Counting and frequency analysis
    print("\n3. Counting and Frequency Analysis:")
    def word_frequency(text: str) -> dict:
        """Count frequency of words in text"""
        freq = {}
        words = text.lower().split()
        for word in words:
            # Remove punctuation
            clean_word = ''.join(char for char in word if char.isalnum())
            if clean_word:
                freq[clean_word] = freq.get(clean_word, 0) + 1
        return freq
    
    text = "the quick brown fox jumps over the lazy dog the dog was really lazy"
    freq = word_frequency(text)
    print(f"   Text: '{text}'")
    print(f"   Word frequencies: {freq}")
    
    # 4. Set operations
    print("\n4. Set Operations:")
    print("   Hash tables implement sets with O(1) average operations")
    set1 = {"apple", "banana", "cherry"}
    set2 = {"banana", "date", "elderberry"}
    print(f"   Set 1: {set1}")
    print(f"   Set 2: {set2}")
    print(f"   Union: {set1 | set2}")
    print(f"   Intersection: {set1 & set2}")
    print(f"   Difference: {set1 - set2}")

def data_science_applications():
    """
    Examples of hash tables in data science applications
    """
    print("\n=== Data Science Applications ===")
    
    # 1. Feature hashing
    print("1. Feature Hashing:")
    print("   Hash functions convert categorical features to numerical")
    print("   Reduces memory usage for high-cardinality features")
    print("   Used in machine learning pipelines")
    
    # 2. Data deduplication
    print("\n2. Data Deduplication:")
    print("   Hash tables efficiently identify duplicate records")
    print("   Useful for data cleaning and preprocessing")
    
    def remove_duplicates(data: List[Any]) -> List[Any]:
        """Remove duplicates while preserving order"""
        seen = set()
        result = []
        for item in data:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    
    data_with_duplicates = [1, 2, 3, 2, 4, 1, 5, 3]
    unique_data = remove_duplicates(data_with_duplicates)
    print(f"   Original: {data_with_duplicates}")
    print(f"   Deduplicated: {unique_data}")
    
    # 3. Join operations in data processing
    print("\n3. Join Operations:")
    print("   Hash joins use hash tables for efficient data merging")
    print("   Faster than nested loop joins for large datasets")
    
    # 4. Bloom filters
    print("\n4. Bloom Filters:")
    print("   Probabilistic data structure using multiple hash functions")
    print("   Space-efficient for membership testing")
    print("   Used in databases, caching systems, and network applications")

def performance_comparison():
    """
    Compare performance of different hash table implementations
    """
    print("\n=== Performance Comparison ===")
    
    import time
    import random
    import string
    
    # Generate test data
    def random_string(length: int) -> str:
        return ''.join(random.choices(string.ascii_letters, k=length))
    
    test_data = [(random_string(10), random.randint(1, 1000)) for _ in range(10000)]
    
    # Chaining hash table
    ht = HashTable()
    start = time.time()
    for key, value in test_data:
        ht.put(key, value)
    insert_time_ht = time.time() - start
    
    start = time.time()
    for key, _ in test_data[:1000]:  # Test first 1000 keys
        _ = ht.get(key)
    lookup_time_ht = time.time() - start
    
    # Linear probing hash table
    lht = LinearProbingHashTable()
    start = time.time()
    for key, value in test_data:
        lht.put(key, value)
    insert_time_lht = time.time() - start
    
    start = time.time()
    for key, _ in test_data[:1000]:  # Test first 1000 keys
        _ = lht.get(key)
    lookup_time_lht = time.time() - start
    
    # Python dict (built-in hash table)
    py_dict = {}
    start = time.time()
    for key, value in test_data:
        py_dict[key] = value
    insert_time_dict = time.time() - start
    
    start = time.time()
    for key, _ in test_data[:1000]:  # Test first 1000 keys
        _ = py_dict.get(key)
    lookup_time_dict = time.time() - start
    
    print("Performance with 10,000 insertions and 1,000 lookups:")
    print(f"Chaining Hash Table - Insert: {insert_time_ht:.6f}s, Lookup: {lookup_time_ht:.6f}s")
    print(f"Linear Probing Hash Table - Insert: {insert_time_lht:.6f}s, Lookup: {lookup_time_lht:.6f}s")
    print(f"Python dict - Insert: {insert_time_dict:.6f}s, Lookup: {lookup_time_dict:.6f}s")

# Example usage and testing
if __name__ == "__main__":
    # Hash table demo
    hash_table_demo()
    print("\n" + "="*50 + "\n")
    
    # Linear probing demo
    linear_probing_demo()
    print("\n" + "="*50 + "\n")
    
    # Hash function examples
    hash_function_examples()
    print("\n" + "="*50 + "\n")
    
    # Hash table applications
    hash_table_applications()
    print("\n" + "="*50 + "\n")
    
    # Data science applications
    data_science_applications()
    print("\n" + "="*50 + "\n")
    
    # Performance comparison
    performance_comparison()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Hash table implementation with chaining")
    print("2. Linear probing hash table implementation")
    print("3. Different hash functions and their properties")
    print("4. Common operations and their time complexities")
    print("5. Practical applications in systems and algorithms")
    print("6. Data science applications of hash tables")
    print("7. Performance comparison of implementations")
    print("\nKey takeaways:")
    print("- Hash tables provide O(1) average time complexity for insert, delete, and lookup")
    print("- Chaining handles collisions by storing multiple items in each bucket")
    print("- Linear probing stores items directly in the table, handling collisions by probing")
    print("- Load factor affects performance - resize when it gets too high")
    print("- Hash tables are fundamental in databases, caches, and data processing")
    print("- Python's dict is a highly optimized hash table implementation")