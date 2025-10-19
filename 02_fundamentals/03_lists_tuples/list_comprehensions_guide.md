# 🧠 List Comprehensions: Python's Elegant Way to Create Lists

List comprehensions provide a concise way to create lists in Python. They're more readable and often faster than traditional loops, making them a powerful feature every Python developer should master.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Understand the syntax and structure of list comprehensions
- Convert traditional loops to list comprehensions
- Use conditional logic in list comprehensions
- Apply nested list comprehensions
- Recognize when to use (and when not to use) list comprehensions

## 📝 Basic Syntax

The basic syntax of a list comprehension is:
```python
[expression for item in iterable]
```

### Simple Example
```python
# Traditional approach
squares = []
for x in range(10):
    squares.append(x**2)

# List comprehension approach
squares = [x**2 for x in range(10)]

print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

## 🔧 Components of List Comprehensions

### 1. Expression
The expression is applied to each item in the iterable.

```python
# Simple expression
numbers = [x for x in range(5)]
print(numbers)  # [0, 1, 2, 3, 4]

# Complex expression
formatted = [f"Number: {x}" for x in range(3)]
print(formatted)  # ['Number: 0', 'Number: 1', 'Number: 2']

# Expression with function
words = ["hello", "world", "python"]
lengths = [len(word) for word in words]
print(lengths)  # [5, 5, 6]
```

### 2. Iterable
Any iterable object can be used.

```python
# List
doubled = [x * 2 for x in [1, 2, 3]]
print(doubled)  # [2, 4, 6]

# String (iterable of characters)
chars = [char.upper() for char in "hello"]
print(chars)  # ['H', 'E', 'L', 'L', 'O']

# Tuple
tripled = [x * 3 for x in (1, 2, 3)]
print(tripled)  # [3, 6, 9]

# Set
unique_doubled = [x * 2 for x in {1, 2, 3}]
print(unique_doubled)  # [2, 4, 6] (order may vary)

# Dictionary keys
keys_squared = [k**2 for k in {'a': 1, 'b': 2, 'c': 3}]
print(keys_squared)  # [1, 4, 9] (order may vary)
```

## 🎯 Conditional Logic in List Comprehensions

### 1. Filtering with `if`

```python
# Filter even numbers
numbers = range(10)
evens = [x for x in numbers if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# Filter strings by length
words = ["apple", "banana", "cherry", "date", "elderberry"]
long_words = [word for word in words if len(word) > 5]
print(long_words)  # ['banana', 'cherry', 'elderberry']

# Filter with multiple conditions
numbers = range(20)
filtered = [x for x in numbers if x % 2 == 0 and x > 10]
print(filtered)  # [12, 14, 16, 18]
```

### 2. Conditional Expressions (Ternary Operator)

```python
# Apply different transformations based on condition
numbers = range(10)
processed = [x if x % 2 == 0 else -x for x in numbers]
print(processed)  # [0, -1, 2, -3, 4, -5, 6, -7, 8, -9]

# Categorize numbers
numbers = range(1, 11)
categories = ["even" if x % 2 == 0 else "odd" for x in numbers]
print(categories)  # ['odd', 'even', 'odd', 'even', 'odd', 'even', 'odd', 'even', 'odd', 'even']
```

### 3. Complex Filtering

```python
# Filter dictionaries
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78},
    {"name": "Diana", "grade": 96}
]

high_performers = [student["name"] for student in students if student["grade"] > 90]
print(high_performers)  # ['Bob', 'Diana']

# Filter and transform
grade_messages = [f"{student['name']}: A" for student in students if student["grade"] >= 90]
print(grade_messages)  # ['Bob: A', 'Diana: A']
```

## 🔄 Nested List Comprehensions

### 1. Flattening Nested Lists

```python
# Traditional approach
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = []
for row in matrix:
    for item in row:
        flattened.append(item)

# List comprehension approach
flattened = [item for row in matrix for item in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 2. Creating Nested Structures

```python
# Create multiplication table
multiplication_table = [[i * j for j in range(1, 11)] for i in range(1, 11)]
print(multiplication_table[2])  # [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

# Create coordinate pairs
coordinates = [(x, y) for x in range(3) for y in range(3)]
print(coordinates)  # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
```

## 🎨 Advanced Examples

### 1. Dictionary Comprehensions

```python
# Create dictionary from two lists
keys = ["name", "age", "city"]
values = ["Alice", 25, "New York"]
person = {k: v for k, v in zip(keys, values)}
print(person)  # {'name': 'Alice', 'age': 25, 'city': 'New York'}

# Transform dictionary
original = {"a": 1, "b": 2, "c": 3}
squared = {k: v**2 for k, v in original.items()}
print(squared)  # {'a': 1, 'b': 4, 'c': 9}

# Filter dictionary
filtered = {k: v for k, v in original.items() if v > 1}
print(filtered)  # {'b': 2, 'c': 3}
```

### 2. Set Comprehensions

```python
# Create set of unique squares
numbers = [1, 2, 2, 3, 3, 4, 4, 5]
unique_squares = {x**2 for x in numbers}
print(unique_squares)  # {1, 4, 9, 16, 25}

# Filter and transform to set
words = ["hello", "world", "python", "programming"]
unique_lengths = {len(word) for word in words}
print(unique_lengths)  # {5, 6, 11} (order may vary)
```

### 3. Generator Expressions

```python
# Memory-efficient alternative to list comprehensions
# Use parentheses instead of square brackets
squares_gen = (x**2 for x in range(10))

# Generator is lazy - computes values on demand
print(next(squares_gen))  # 0
print(next(squares_gen))  # 1
print(list(squares_gen))  # [4, 9, 16, 25, 36, 49, 64, 81]

# Memory usage comparison
import sys
list_comp = [x**2 for x in range(1000)]
gen_exp = (x**2 for x in range(1000))

print(f"List comprehension size: {sys.getsizeof(list_comp)} bytes")
print(f"Generator expression size: {sys.getsizeof(gen_exp)} bytes")
```

## 🚀 Performance Comparison

```python
import time

# Traditional loop
def traditional_approach():
    result = []
    for x in range(100000):
        if x % 2 == 0:
            result.append(x * 2)
    return result

# List comprehension
def comprehension_approach():
    return [x * 2 for x in range(100000) if x % 2 == 0]

# Time comparison
start = time.time()
traditional_result = traditional_approach()
traditional_time = time.time() - start

start = time.time()
comprehension_result = comprehension_approach()
comprehension_time = time.time() - start

print(f"Traditional approach: {traditional_time:.4f} seconds")
print(f"List comprehension: {comprehension_time:.4f} seconds")
print(f"List comprehension is {traditional_time/comprehension_time:.2f}x faster")
```

## ⚠️ When NOT to Use List Comprehensions

### 1. Complex Logic

```python
# Bad: Complex logic in list comprehension
# Hard to read and debug
result = [complex_function(x) for x in data if condition1(x) and condition2(x) and condition3(x)]

# Good: Traditional loop for complex logic
result = []
for x in data:
    if condition1(x) and condition2(x) and condition3(x):
        processed = complex_function(x)
        result.append(processed)
```

### 2. Side Effects

```python
# Bad: List comprehension with side effects
# Hard to understand and debug
results = [print(x) for x in range(5)]  # Prints values but creates list of None

# Good: Traditional loop for side effects
for x in range(5):
    print(x)
```

### 3. Very Long Comprehensions

```python
# Bad: Too long and hard to read
result = [x**2 if x % 2 == 0 else x**3 for x in range(100) if x > 10 and x < 50 and some_complex_condition(x)]

# Good: Break into multiple steps
filtered = [x for x in range(100) if x > 10 and x < 50 and some_complex_condition(x)]
result = [x**2 if x % 2 == 0 else x**3 for x in filtered]
```

## 🧪 Practice Problems

### Problem 1: Temperature Conversion
Convert a list of temperatures from Celsius to Fahrenheit using list comprehension.

```python
celsius_temps = [0, 20, 30, 40, 100]
# Your code here
# Expected: [32.0, 68.0, 86.0, 104.0, 212.0]
```

### Problem 2: Word Processing
Given a list of sentences, create a list of all words that are longer than 4 characters.

```python
sentences = [
    "Python is a powerful language",
    "List comprehensions are elegant",
    "Programming should be fun"
]
# Your code here
# Expected: ['Python', 'powerful', 'language', 'comprehensions', 'elegant', 'Programming']
```

### Problem 3: Matrix Operations
Given a 3x3 matrix, create a new matrix where each element is doubled.

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
# Your code here
# Expected: [[2, 4, 6], [8, 10, 12], [14, 16, 18]]
```

## 🎯 Solutions

### Solution 1: Temperature Conversion
```python
celsius_temps = [0, 20, 30, 40, 100]
fahrenheit_temps = [(c * 9/5) + 32 for c in celsius_temps]
print(fahrenheit_temps)  # [32.0, 68.0, 86.0, 104.0, 212.0]
```

### Solution 2: Word Processing
```python
sentences = [
    "Python is a powerful language",
    "List comprehensions are elegant",
    "Programming should be fun"
]
long_words = [word for sentence in sentences for word in sentence.split() if len(word) > 4]
print(long_words)  # ['Python', 'powerful', 'language', 'comprehensions', 'elegant', 'Programming']
```

### Solution 3: Matrix Operations
```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
doubled_matrix = [[element * 2 for element in row] for row in matrix]
print(doubled_matrix)  # [[2, 4, 6], [8, 10, 12], [14, 16, 18]]
```

## 📚 Best Practices

1. **Keep it readable**: If a list comprehension becomes too complex, use a traditional loop
2. **Use meaningful variable names**: `[student.name for student in students]` is clearer than `[s.name for s in students]`
3. **Consider memory usage**: Use generator expressions for large datasets
4. **Profile your code**: List comprehensions are often faster, but always measure performance
5. **Avoid side effects**: List comprehensions should focus on creating lists, not performing actions

## 🎯 Summary

List comprehensions are a powerful Python feature that:
- Provide a concise way to create lists
- Are often more readable than traditional loops
- Can be faster than equivalent loop-based code
- Support filtering and complex transformations
- Can be extended to dictionary and set comprehensions

Use them when:
- Creating new lists from existing iterables
- Applying simple transformations
- Filtering data
- The logic is straightforward and readable

Avoid them when:
- The logic becomes complex
- You need side effects
- Readability suffers
- Memory usage is a concern (consider generators)

---

**Master list comprehensions and write more Pythonic code!** 🐍