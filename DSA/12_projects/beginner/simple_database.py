"""
Beginner Project: Simple Database

This project implements a simple key-value database that demonstrates
hash table (dictionary) data structures and basic database operations.

Concepts covered:
- Hash table implementation
- Collision resolution techniques
- Memory management
- CRUD operations (Create, Read, Update, Delete)
- Data persistence (basic)
"""

import json
import hashlib
from typing import Any, Dict, List, Optional


class HashTable:
    """
    Hash table implementation using chaining for collision resolution.
    
    Time Complexities:
    - Insert: O(1) average, O(n) worst case
    - Search: O(1) average, O(n) worst case
    - Delete: O(1) average, O(n) worst case
    Space Complexity: O(n)
    """
    
    def __init__(self, initial_capacity: int = 16):
        self.capacity = initial_capacity
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, key: str) -> int:
        """Hash function using Python's built-in hash."""
        return hash(key) % self.capacity
    
    def _hash_sha256(self, key: str) -> int:
        """Alternative hash function using SHA-256."""
        hash_object = hashlib.sha256(key.encode())
        hash_hex = hash_object.hexdigest()
        return int(hash_hex, 16) % self.capacity
    
    def _resize(self) -> None:
        """Resize hash table when load factor exceeds threshold."""
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all existing key-value pairs
        for bucket in old_buckets:
            for key, value in bucket:
                self.insert(key, value)
    
    def insert(self, key: str, value: Any) -> None:
        """Insert key-value pair into hash table."""
        # Check if resize is needed
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        # Check if key already exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Update existing key
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self.size += 1
    
    def search(self, key: str) -> Optional[Any]:
        """Search for value by key."""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return None  # Key not found
    
    def delete(self, key: str) -> bool:
        """Delete key-value pair by key."""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return True
        
        return False  # Key not found
    
    def get_all_keys(self) -> List[str]:
        """Get all keys in hash table."""
        keys = []
        for bucket in self.buckets:
            for key, _ in bucket:
                keys.append(key)
        return keys
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hash table statistics."""
        bucket_lengths = [len(bucket) for bucket in self.buckets]
        return {
            'capacity': self.capacity,
            'size': self.size,
            'load_factor': self.size / self.capacity,
            'max_bucket_length': max(bucket_lengths),
            'avg_bucket_length': sum(bucket_lengths) / len(bucket_lengths),
            'empty_buckets': bucket_lengths.count(0)
        }


class SimpleDatabase:
    """
    Simple key-value database using hash table as storage engine.
    """
    
    def __init__(self, filename: Optional[str] = None):
        self.storage = HashTable()
        self.filename = filename
        if filename:
            self.load_from_file()
    
    def set(self, key: str, value: Any) -> None:
        """
        Set key-value pair in database.
        
        Time Complexity: O(1) average
        """
        self.storage.insert(key, value)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value by key from database.
        
        Time Complexity: O(1) average
        """
        return self.storage.search(key)
    
    def delete(self, key: str) -> bool:
        """
        Delete key-value pair from database.
        
        Time Complexity: O(1) average
        """
        return self.storage.delete(key)
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in database.
        
        Time Complexity: O(1) average
        """
        return self.storage.search(key) is not None
    
    def keys(self) -> List[str]:
        """
        Get all keys in database.
        
        Time Complexity: O(n)
        """
        return self.storage.get_all_keys()
    
    def flush(self) -> None:
        """
        Clear all data from database.
        
        Time Complexity: O(1)
        """
        self.storage = HashTable()
    
    def save_to_file(self, filename: Optional[str] = None) -> None:
        """
        Save database to file as JSON.
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        save_filename = filename or self.filename
        if not save_filename:
            raise ValueError("No filename specified")
        
        # Convert hash table to dictionary
        data = {}
        for key in self.keys():
            data[key] = self.get(key)
        
        # Save to file
        with open(save_filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: Optional[str] = None) -> None:
        """
        Load database from JSON file.
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        load_filename = filename or self.filename
        if not load_filename:
            raise ValueError("No filename specified")
        
        try:
            with open(load_filename, 'r') as f:
                data = json.load(f)
            
            # Load data into hash table
            self.flush()
            for key, value in data.items():
                self.set(key, value)
                
        except FileNotFoundError:
            # File doesn't exist, start with empty database
            pass
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {load_filename}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Time Complexity: O(1)
        """
        return self.storage.get_stats()


def demonstrate_database():
    """Demonstrate simple database functionality."""
    print("=== Simple Database Demo ===\n")
    
    # Create database
    db = SimpleDatabase()
    
    # Test basic operations
    print("1. Basic Database Operations:")
    
    # Set key-value pairs
    db.set("name", "Alice")
    db.set("age", 30)
    db.set("city", "New York")
    db.set("occupation", "Engineer")
    db.set("salary", 75000.50)
    
    print(f"  Set 5 key-value pairs")
    
    # Get values
    print(f"  Name: {db.get('name')}")
    print(f"  Age: {db.get('age')}")
    print(f"  City: {db.get('city')}")
    
    # Check existence
    print(f"  Does 'name' exist? {db.exists('name')}")
    print(f"  Does 'email' exist? {db.exists('email')}")
    
    # Update value
    db.set("age", 31)
    print(f"  Updated age: {db.get('age')}")
    
    # Delete key
    deleted = db.delete("city")
    print(f"  Deleted 'city': {deleted}")
    print(f"  City exists after deletion: {db.exists('city')}")
    
    # Get all keys
    all_keys = db.keys()
    print(f"  All keys: {all_keys}")
    
    # Test database statistics
    print("\n2. Database Statistics:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test file operations
    print("\n3. File Operations:")
    try:
        # Save database
        db.save_to_file("demo_database.json")
        print("  Database saved to 'demo_database.json'")
        
        # Create new database and load from file
        new_db = SimpleDatabase("demo_database.json")
        print(f"  Loaded database has {len(new_db.keys())} keys")
        print(f"  Name in loaded database: {new_db.get('name')}")
        
    except Exception as e:
        print(f"  File operation error: {e}")


def performance_comparison():
    """Compare performance of different database operations."""
    import time
    
    print("\n=== Performance Comparison ===\n")
    
    db = SimpleDatabase()
    
    # Test insertion performance
    print("1. Insertion Performance:")
    n_operations = 10000
    
    start_time = time.time()
    for i in range(n_operations):
        db.set(f"key_{i}", f"value_{i}")
    insert_time = time.time() - start_time
    print(f"   Time to insert {n_operations} key-value pairs: {insert_time:.6f} seconds")
    print(f"   Average time per insertion: {insert_time/n_operations:.8f} seconds")
    
    # Test search performance
    print("\n2. Search Performance:")
    start_time = time.time()
    for i in range(0, n_operations, 100):  # Search every 100th key
        value = db.get(f"key_{i}")
    search_time = time.time() - start_time
    search_count = n_operations // 100
    print(f"   Time to search {search_count} keys: {search_time:.6f} seconds")
    print(f"   Average time per search: {search_time/search_count:.8f} seconds")
    
    # Test deletion performance
    print("\n3. Deletion Performance:")
    start_time = time.time()
    for i in range(0, n_operations, 100):  # Delete every 100th key
        db.delete(f"key_{i}")
    delete_time = time.time() - start_time
    delete_count = n_operations // 100
    print(f"   Time to delete {delete_count} keys: {delete_time:.6f} seconds")
    print(f"   Average time per deletion: {delete_time/delete_count:.8f} seconds")
    
    # Test database statistics
    print("\n4. Database Statistics:")
    stats = db.get_stats()
    print(f"   Final database size: {stats['size']}")
    print(f"   Load factor: {stats['load_factor']:.4f}")
    print(f"   Max bucket length: {stats['max_bucket_length']}")


if __name__ == "__main__":
    demonstrate_database()
    performance_comparison()