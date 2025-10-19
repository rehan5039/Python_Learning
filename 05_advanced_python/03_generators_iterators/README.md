# 🔁 Generators and Iterators in Python

Generators and iterators are powerful features in Python that enable memory-efficient processing of large datasets and lazy evaluation. This guide will teach you how to create and use generators and iterators effectively.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Understand the difference between iterators and iterables
- Create custom iterators and iterables
- Implement generators using functions and expressions
- Use generator methods and coroutine features
- Apply generators for memory-efficient data processing
- Work with built-in iterator functions

## 🔁 What are Iterators and Iterables?

### Iterables
An iterable is any object that can be looped over. Examples include lists, tuples, strings, dictionaries, and sets.

### Iterators
An iterator is an object that implements the iterator protocol, which consists of the methods `__iter__()` and `__next__()`.

```python
# Basic iteration
my_list = [1, 2, 3, 4, 5]
for item in my_list:
    print(item)

# Manual iteration
iterator = iter(my_list)
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
```

## 🏗️ Creating Custom Iterators

### Basic Iterator Class

```python
class CountDown:
    """Iterator that counts down from a number"""
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# Usage
countdown = CountDown(5)
for num in countdown:
    print(num)

# Manual iteration
countdown2 = CountDown(3)
iterator = iter(countdown2)
print(next(iterator))  # 3
print(next(iterator))  # 2
print(next(iterator))  # 1
# print(next(iterator))  # Raises StopIteration
```

### Advanced Iterator Class

```python
class FibonacciIterator:
    """Iterator that generates Fibonacci numbers"""
    def __init__(self, max_count=None):
        self.max_count = max_count
        self.count = 0
        self.current = 0
        self.next_val = 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.max_count is not None and self.count >= self.max_count:
            raise StopIteration
        
        result = self.current
        self.current, self.next_val = self.next_val, self.current + self.next_val
        self.count += 1
        return result

# Usage
fib = FibonacciIterator(10)
for num in fib:
    print(num, end=" ")
print()

# Convert to list
fib_list = list(FibonacciIterator(15))
print(f"First 15 Fibonacci numbers: {fib_list}")
```

## ⚡ Creating Generators

Generators are a simpler way to create iterators using functions with the `yield` keyword.

### Basic Generator Function

```python
def count_up_to(max_num):
    """Generator that counts up to a maximum number"""
    count = 1
    while count <= max_num:
        yield count
        count += 1

# Usage
counter = count_up_to(5)
print(next(counter))  # 1
print(next(counter))  # 2
for num in counter:
    print(num)  # 3, 4, 5

# Convert to list
numbers = list(count_up_to(10))
print(numbers)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### Generator with Multiple Yields

```python
def my_generator():
    """Generator with multiple yield statements"""
    print("Starting generator")
    yield 1
    print("After first yield")
    yield 2
    print("After second yield")
    yield 3
    print("After third yield")

# Usage
gen = my_generator()
print("Created generator")
print(next(gen))  # Starting generator \n 1
print(next(gen))  # After first yield \n 2
print(next(gen))  # After second yield \n 3
# print(next(gen))  # After third yield \n StopIteration
```

### Generator Expressions

Generator expressions are similar to list comprehensions but use parentheses instead of brackets.

```python
# List comprehension (creates entire list in memory)
squares_list = [x**2 for x in range(10)]
print(squares_list)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Generator expression (creates generator object)
squares_gen = (x**2 for x in range(10))
print(squares_gen)   # <generator object <genexpr> at 0x...>
print(list(squares_gen))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Memory efficient processing
large_squares = (x**2 for x in range(1000000))
print(f"Generator object size: {large_squares.__sizeof__()} bytes")

# Only computes values when needed
first_five = [next(large_squares) for _ in range(5)]
print(f"First five squares: {first_five}")
```

## 🔄 Generator Methods and Coroutine Features

### Sending Values to Generators

```python
def echo_generator():
    """Generator that can receive values"""
    received = yield "Ready"
    while received is not None:
        received = yield f"Echo: {received}"

# Usage
echo_gen = echo_generator()
print(next(echo_gen))  # Ready
print(echo_gen.send("Hello"))  # Echo: Hello
print(echo_gen.send("World"))  # Echo: World
```

### Throwing Exceptions into Generators

```python
def exception_handling_generator():
    """Generator that handles exceptions"""
    try:
        yield 1
        yield 2
        yield 3
    except ValueError as e:
        yield f"Caught exception: {e}"
    yield 4

# Usage
gen = exception_handling_generator()
print(next(gen))  # 1
print(next(gen))  # 2
print(gen.throw(ValueError, "Something went wrong"))  # Caught exception: Something went wrong
print(next(gen))  # 4
```

### Closing Generators

```python
def closable_generator():
    """Generator that can be closed"""
    try:
        yield 1
        yield 2
        yield 3
    except GeneratorExit:
        print("Generator is being closed")
        raise

# Usage
gen = closable_generator()
print(next(gen))  # 1
print(next(gen))  # 2
gen.close()       # Generator is being closed
# print(next(gen))  # StopIteration
```

## 🧪 Practical Examples

### Example 1: File Processing Generator
```python
def read_large_file(file_path, chunk_size=1024):
    """Generator to read large files in chunks"""
    try:
        with open(file_path, 'r') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except FileNotFoundError:
        print(f"File {file_path} not found")
    except Exception as e:
        print(f"Error reading file: {e}")

def process_log_file_lines(file_path):
    """Generator to process log file lines"""
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                # Process each line (remove whitespace, etc.)
                processed_line = line.strip()
                if processed_line:  # Skip empty lines
                    yield line_num, processed_line
    except FileNotFoundError:
        print(f"File {file_path} not found")
    except Exception as e:
        print(f"Error processing file: {e}")

# Create a sample log file
sample_log = """2023-01-01 10:00:00 INFO Application started
2023-01-01 10:05:00 DEBUG Processing user data
2023-01-01 10:10:00 ERROR Database connection failed
2023-01-01 10:15:00 INFO User logged in
2023-01-01 10:20:00 WARNING Low disk space
"""

with open('sample.log', 'w') as f:
    f.write(sample_log)

# Usage
print("=== File Processing Generator ===")
print("Processing log file lines:")
for line_num, line in process_log_file_lines('sample.log'):
    print(f"Line {line_num}: {line}")

# Clean up
import os
os.remove('sample.log')
```

### Example 2: Data Pipeline with Generators
```python
def read_data():
    """Generator that reads raw data"""
    data = [
        "Alice,25,Engineer",
        "Bob,30,Designer",
        "Charlie,35,Manager",
        "Diana,28,Developer",
        "Eve,32,Analyst"
    ]
    for item in data:
        yield item

def parse_data(raw_data_generator):
    """Generator that parses raw data"""
    for raw_item in raw_data_generator:
        name, age, role = raw_item.split(',')
        yield {
            'name': name,
            'age': int(age),
            'role': role
        }

def filter_by_role(parsed_data_generator, role):
    """Generator that filters data by role"""
    for item in parsed_data_generator:
        if item['role'] == role:
            yield item

def add_experience(data_generator):
    """Generator that adds experience based on age"""
    for item in data_generator:
        # Simple experience calculation: age - 22 (assuming started work at 22)
        experience = max(0, item['age'] - 22)
        item['experience'] = experience
        yield item

# Usage
print("\n=== Data Pipeline with Generators ===")
print("Full pipeline:")
pipeline = add_experience(filter_by_role(parse_data(read_data()), "Developer"))
for person in pipeline:
    print(f"  {person}")

print("\nAll engineers:")
engineers = filter_by_role(parse_data(read_data()), "Engineer")
for person in engineers:
    print(f"  {person}")
```

### Example 3: Prime Number Generator
```python
def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def prime_generator(max_number=None):
    """Generator that yields prime numbers"""
    num = 2
    while max_number is None or num <= max_number:
        if is_prime(num):
            yield num
        num += 1

def prime_pairs_generator():
    """Generator that yields pairs of consecutive primes"""
    primes = prime_generator()
    prev_prime = next(primes)
    for current_prime in primes:
        yield (prev_prime, current_prime)
        prev_prime = current_prime

# Usage
print("\n=== Prime Number Generator ===")
print("First 20 prime numbers:")
primes = prime_generator()
first_20 = [next(primes) for _ in range(20)]
print(first_20)

print("\nPrime pairs (first 10):")
prime_pairs = prime_pairs_generator()
first_10_pairs = [next(prime_pairs) for _ in range(10)]
for pair in first_10_pairs:
    print(f"  {pair[0]} and {pair[1]} (difference: {pair[1] - pair[0]})")
```

### Example 4: Fibonacci Generator with Memoization
```python
def fibonacci_generator():
    """Generator that yields Fibonacci numbers"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def fibonacci_with_limit(max_value):
    """Generator that yields Fibonacci numbers up to a maximum value"""
    a, b = 0, 1
    while a <= max_value:
        yield a
        a, b = b, a + b

def fibonacci_with_index(max_index):
    """Generator that yields Fibonacci numbers with their indices"""
    a, b = 0, 1
    index = 0
    while index <= max_index:
        yield index, a
        a, b = b, a + b
        index += 1

# Usage
print("\n=== Fibonacci Generator ===")
print("First 15 Fibonacci numbers:")
fib_gen = fibonacci_generator()
first_15 = [next(fib_gen) for _ in range(15)]
print(first_15)

print("\nFibonacci numbers up to 100:")
fib_limited = list(fibonacci_with_limit(100))
print(fib_limited)

print("\nFibonacci numbers with indices (first 10):")
fib_indexed = list(fibonacci_with_index(9))
for index, value in fib_indexed:
    print(f"  F({index}) = {value}")
```

## 🛠 Built-in Iterator Functions

Python provides several built-in functions that work with iterators.

### itertools Module

```python
import itertools

# Infinite iterators
print("=== Infinite Iterators ===")
counter = itertools.count(start=10, step=2)
first_five = [next(counter) for _ in range(5)]
print(f"Count iterator: {first_five}")

cycle = itertools.cycle(['A', 'B', 'C'])
cycle_first_six = [next(cycle) for _ in range(6)]
print(f"Cycle iterator: {cycle_first_six}")

repeat = itertools.repeat('Hello', 3)
repeat_list = list(repeat)
print(f"Repeat iterator: {repeat_list}")

# Combinatoric iterators
print("\n=== Combinatoric Iterators ===")
data = ['A', 'B', 'C']

# Combinations
combinations = list(itertools.combinations(data, 2))
print(f"Combinations of {data} taken 2: {combinations}")

# Permutations
permutations = list(itertools.permutations(data, 2))
print(f"Permutations of {data} taken 2: {permutations}")

# Product
product = list(itertools.product(['X', 'Y'], data))
print(f"Product of ['X', 'Y'] and {data}: {product}")

# Terminating iterators
print("\n=== Terminating Iterators ===")
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Chain
chain_result = list(itertools.chain([1, 2, 3], [4, 5, 6], [7, 8, 9]))
print(f"Chain result: {chain_result}")

# Accumulate
accumulate_result = list(itertools.accumulate(numbers))
print(f"Accumulate result: {accumulate_result}")

# Groupby
data = [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('C', 5)]
grouped = itertools.groupby(data, key=lambda x: x[0])
print("Groupby result:")
for key, group in grouped:
    group_list = list(group)
    print(f"  {key}: {group_list}")
```

## ⚠️ Memory Efficiency Comparison

```python
import sys

# Memory comparison
print("=== Memory Efficiency Comparison ===")

# List comprehension (loads everything into memory)
squares_list = [x**2 for x in range(100000)]
list_size = sys.getsizeof(squares_list)
print(f"List size: {list_size} bytes")

# Generator expression (lazy evaluation)
squares_gen = (x**2 for x in range(100000))
gen_size = sys.getsizeof(squares_gen)
print(f"Generator size: {gen_size} bytes")

print(f"Generator uses {list_size / gen_size:.0f}x less memory")

# Processing large datasets efficiently
def process_large_dataset():
    """Process a large dataset using generators"""
    # Simulate a large dataset
    def large_dataset():
        for i in range(1000000):
            yield i * 2
    
    # Process in chunks
    total = 0
    count = 0
    dataset = large_dataset()
    
    for value in dataset:
        total += value
        count += 1
        if count % 100000 == 0:
            print(f"Processed {count} items, running total: {total}")
    
    return total / count if count > 0 else 0

# This would use minimal memory even for very large datasets
average = process_large_dataset()
print(f"Average: {average}")
```

## ⚠️ Common Mistakes and Best Practices

### 1. Exhausting Generators

```python
# Wrong - generators can only be consumed once
# def simple_generator():
#     yield 1
#     yield 2
#     yield 3
# 
# gen = simple_generator()
# list1 = list(gen)  # [1, 2, 3]
# list2 = list(gen)  # [] - generator is exhausted!

# Correct - create new generator instances
def simple_generator():
    yield 1
    yield 2
    yield 3

list1 = list(simple_generator())  # [1, 2, 3]
list2 = list(simple_generator())  # [1, 2, 3] - fresh generator
```

### 2. Not Handling StopIteration

```python
# Wrong - not handling StopIteration
# def bad_usage():
#     gen = (x for x in range(3))
#     while True:
#         print(next(gen))  # Will raise StopIteration

# Correct - handle StopIteration properly
def good_usage():
    gen = (x for x in range(3))
    try:
        while True:
            print(next(gen))
    except StopIteration:
        print("Generator exhausted")

good_usage()
```

### 3. Using Generators for Small Datasets

```python
# For small datasets, lists might be more appropriate
# Generators are beneficial for:
# - Large datasets that don't fit in memory
# - Expensive computations
# - Lazy evaluation scenarios

# For small datasets, this is fine:
small_list = [x**2 for x in range(10)]

# For large datasets, use generators:
large_gen = (x**2 for x in range(1000000))
```

## 📚 Next Steps

Now that you understand generators and iterators, you're ready to learn:

1. **Context Managers**: Resource management with `with` statements
2. **Metaclasses**: Classes that create classes
3. **Descriptors**: Custom attribute access
4. **Concurrency**: Threading and multiprocessing

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: What's the difference between a generator and a regular function?
A: A generator uses `yield` instead of `return` and maintains its state between calls.

### Q: When should I use generators instead of lists?
A: Use generators for large datasets, memory efficiency, or lazy evaluation. Use lists for small datasets or when you need random access.

### Q: Can I restart a generator?
A: No, generators can only be consumed once. Create a new generator instance to restart.

### Q: What's the difference between `yield` and `return`?
A: `return` terminates the function, while `yield` pauses it and returns a value, allowing the function to resume later.

---

**Practice creating different types of generators and iterators with various scenarios to build your expertise!** 🐍