# Generator and Iterator Examples

# Basic Iterator Concept
print("=== Basic Iterator Concept ===")

my_list = [1, 2, 3, 4, 5]
for item in my_list:
    print(item)

# Manual iteration
iterator = iter(my_list)
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3

# Creating Custom Iterators
print("\n=== Creating Custom Iterators ===")

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

# Advanced Iterator Class
print("\n=== Advanced Iterator Class ===")

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

# Creating Generators
print("\n=== Creating Generators ===")

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

# Generator with Multiple Yields
print("\n=== Generator with Multiple Yields ===")

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

# Generator Expressions
print("\n=== Generator Expressions ===")

# List comprehension
squares_list = [x**2 for x in range(10)]
print(f"List comprehension: {squares_list}")

# Generator expression
squares_gen = (x**2 for x in range(10))
print(f"Generator expression: {squares_gen}")
print(f"Generator to list: {list(squares_gen)}")

# Memory efficient processing
large_squares = (x**2 for x in range(1000000))
print(f"Generator object size: {large_squares.__sizeof__()} bytes")

# Generator Methods and Coroutine Features
print("\n=== Generator Methods and Coroutine Features ===")

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

# Practical Example: File Processing Generator
print("\n=== Practical Example: File Processing Generator ===")

def process_log_file_lines(sample_data):
    """Generator to process log file lines"""
    lines = sample_data.strip().split('\n')
    for line_num, line in enumerate(lines, 1):
        processed_line = line.strip()
        if processed_line:
            yield line_num, processed_line

# Sample log data
sample_log = """2023-01-01 10:00:00 INFO Application started
2023-01-01 10:05:00 DEBUG Processing user data
2023-01-01 10:10:00 ERROR Database connection failed
2023-01-01 10:15:00 INFO User logged in
2023-01-01 10:20:00 WARNING Low disk space"""

print("Processing log file lines:")
for line_num, line in process_log_file_lines(sample_log):
    print(f"Line {line_num}: {line}")

# Practical Example: Data Pipeline with Generators
print("\n=== Practical Example: Data Pipeline with Generators ===")

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

# Usage
print("All developers:")
developers = filter_by_role(parse_data(read_data()), "Developer")
for person in developers:
    print(f"  {person}")

# Prime Number Generator
print("\n=== Prime Number Generator ===")

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

# Usage
print("First 20 prime numbers:")
primes = prime_generator()
first_20 = [next(primes) for _ in range(20)]
print(first_20)

# Fibonacci Generator
print("\n=== Fibonacci Generator ===")

def fibonacci_generator():
    """Generator that yields Fibonacci numbers"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Usage
print("First 15 Fibonacci numbers:")
fib_gen = fibonacci_generator()
first_15 = [next(fib_gen) for _ in range(15)]
print(first_15)

# Built-in Iterator Functions
print("\n=== Built-in Iterator Functions ===")

import itertools

# Infinite iterators
print("Infinite iterators:")
counter = itertools.count(start=10, step=2)
first_five = [next(counter) for _ in range(5)]
print(f"Count iterator: {first_five}")

cycle = itertools.cycle(['A', 'B', 'C'])
cycle_first_six = [next(cycle) for _ in range(6)]
print(f"Cycle iterator: {cycle_first_six}")

# Combinatoric iterators
print("\nCombinatoric iterators:")
data = ['A', 'B', 'C']

combinations = list(itertools.combinations(data, 2))
print(f"Combinations: {combinations}")

permutations = list(itertools.permutations(data, 2))
print(f"Permutations: {permutations}")

# Memory Efficiency Comparison
print("\n=== Memory Efficiency Comparison ===")

import sys

# List comprehension
squares_list = [x**2 for x in range(10000)]
list_size = sys.getsizeof(squares_list)
print(f"List size: {list_size} bytes")

# Generator expression
squares_gen = (x**2 for x in range(10000))
gen_size = sys.getsizeof(squares_gen)
print(f"Generator size: {gen_size} bytes")

print(f"Generator uses {list_size / gen_size:.0f}x less memory")

# Common Mistakes Demonstration
print("\n=== Common Mistakes Demonstration ===")

# Correct way - generators can only be consumed once
def simple_generator():
    yield 1
    yield 2
    yield 3

gen = simple_generator()
list1 = list(gen)  # [1, 2, 3]
print(f"First consumption: {list1}")

# Create new generator for second consumption
gen2 = simple_generator()
list2 = list(gen2)  # [1, 2, 3]
print(f"Second consumption: {list2}")

# Correct way to handle StopIteration
def good_usage():
    gen = (x for x in range(3))
    try:
        while True:
            print(next(gen))
    except StopIteration:
        print("Generator exhausted")

print("Handling StopIteration:")
good_usage()