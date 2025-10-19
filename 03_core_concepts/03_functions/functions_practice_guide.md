# 🧠 Functions Practice Guide: Mastering Python Functions

Functions are one of the most important concepts in Python programming. This guide provides comprehensive practice problems with detailed solutions to help you master functions.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Create functions with different types of parameters
- Use return statements effectively
- Implement recursive functions
- Apply function best practices
- Solve complex problems using functions

## 📝 Function Basics Review

### Simple Function
```python
def greet():
    """Simple function that prints a greeting"""
    print("Hello, World!")

greet()  # Hello, World!
```

### Function with Parameters
```python
def greet_person(name):
    """Function that greets a specific person"""
    print(f"Hello, {name}!")

greet_person("Alice")  # Hello, Alice!
```

### Function with Return Value
```python
def add_numbers(a, b):
    """Function that adds two numbers and returns the result"""
    return a + b

result = add_numbers(5, 3)
print(result)  # 8
```

## 🧪 Practice Problems

### Problem Set 1: Basic Functions

#### Problem 1: Area Calculator
Create functions to calculate the area of different shapes:
- Rectangle: length × width
- Circle: π × radius²
- Triangle: ½ × base × height

```python
import math

def rectangle_area(length, width):
    """Calculate the area of a rectangle"""
    # Your code here
    pass

def circle_area(radius):
    """Calculate the area of a circle"""
    # Your code here
    pass

def triangle_area(base, height):
    """Calculate the area of a triangle"""
    # Your code here
    pass

# Test your functions
print(f"Rectangle area: {rectangle_area(5, 3)}")  # 15
print(f"Circle area: {circle_area(4):.2f}")      # 50.27
print(f"Triangle area: {triangle_area(6, 8)}")    # 24.0
```

#### Problem 2: String Utilities
Create functions for common string operations:
- Count vowels in a string
- Reverse a string
- Check if a string is a palindrome

```python
def count_vowels(text):
    """Count the number of vowels in a string"""
    # Your code here
    pass

def reverse_string(text):
    """Reverse a string"""
    # Your code here
    pass

def is_palindrome(text):
    """Check if a string is a palindrome (ignoring case and spaces)"""
    # Your code here
    pass

# Test your functions
print(f"Vowels in 'Hello World': {count_vowels('Hello World')}")  # 3
print(f"Reverse of 'Python': {reverse_string('Python')}")          # nohtyP
print(f"Is 'racecar' a palindrome? {is_palindrome('racecar')}")    # True
```

### Problem Set 2: Advanced Functions

#### Problem 3: Default Parameters and Keyword Arguments
Create a function that generates a personalized greeting with default values.

```python
def create_greeting(name, greeting="Hello", punctuation="!"):
    """Create a personalized greeting with customizable greeting and punctuation"""
    # Your code here
    pass

# Test with different combinations
print(create_greeting("Alice"))                           # Hello, Alice!
print(create_greeting("Bob", "Hi"))                      # Hi, Bob!
print(create_greeting("Charlie", "Hey", "."))            # Hey, Charlie.
print(create_greeting("David", punctuation="?"))         # Hello, David?
print(create_greeting("Eve", greeting="Good morning"))   # Good morning, Eve!
```

#### Problem 4: Variable Arguments
Create functions that can accept a variable number of arguments.

```python
def calculate_average(*numbers):
    """Calculate the average of any number of arguments"""
    # Your code here
    pass

def create_profile(name, **details):
    """Create a profile dictionary with name and additional details"""
    # Your code here
    pass

# Test your functions
print(f"Average of 1, 2, 3, 4, 5: {calculate_average(1, 2, 3, 4, 5)}")  # 3.0
print(f"Average of 10, 20: {calculate_average(10, 20)}")                 # 15.0

profile = create_profile("Alice", age=25, city="New York", occupation="Engineer")
print(profile)  # {'name': 'Alice', 'age': 25, 'city': 'New York', 'occupation': 'Engineer'}
```

### Problem Set 3: Recursive Functions

#### Problem 5: Factorial Calculator
Implement factorial calculation using recursion.

```python
def factorial(n):
    """Calculate factorial of n using recursion"""
    # Your code here
    pass

# Test your function
print(f"5! = {factorial(5)}")  # 120
print(f"0! = {factorial(0)}")  # 1
```

#### Problem 6: Fibonacci Sequence
Generate Fibonacci numbers using recursion.

```python
def fibonacci(n):
    """Calculate the nth Fibonacci number using recursion"""
    # Your code here
    pass

# Test your function
print("First 10 Fibonacci numbers:")
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
```

### Problem Set 4: Higher-Order Functions

#### Problem 7: Function as Parameter
Create a function that applies another function to a list of numbers.

```python
def apply_function(numbers, func):
    """Apply a function to each number in a list"""
    # Your code here
    pass

def square(x):
    """Square a number"""
    return x ** 2

def double(x):
    """Double a number"""
    return x * 2

# Test your function
numbers = [1, 2, 3, 4, 5]
print(f"Original: {numbers}")
print(f"Squared: {apply_function(numbers, square)}")  # [1, 4, 9, 16, 25]
print(f"Doubled: {apply_function(numbers, double)}")   # [2, 4, 6, 8, 10]
```

#### Problem 8: Decorator Practice
Create a simple decorator that measures function execution time.

```python
import time
import functools

def timing_decorator(func):
    """Decorator that measures function execution time"""
    # Your code here
    pass

@timing_decorator
def slow_function():
    """Function that takes some time to execute"""
    time.sleep(1)
    return "Done!"

# Test your decorator
result = slow_function()
print(result)
```

## 🎯 Solutions

### Solution 1: Area Calculator
```python
import math

def rectangle_area(length, width):
    """Calculate the area of a rectangle"""
    return length * width

def circle_area(radius):
    """Calculate the area of a circle"""
    return math.pi * radius ** 2

def triangle_area(base, height):
    """Calculate the area of a triangle"""
    return 0.5 * base * height

# Test your functions
print(f"Rectangle area: {rectangle_area(5, 3)}")  # 15
print(f"Circle area: {circle_area(4):.2f}")      # 50.27
print(f"Triangle area: {triangle_area(6, 8)}")    # 24.0
```

### Solution 2: String Utilities
```python
def count_vowels(text):
    """Count the number of vowels in a string"""
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

def reverse_string(text):
    """Reverse a string"""
    return text[::-1]

def is_palindrome(text):
    """Check if a string is a palindrome (ignoring case and spaces)"""
    # Remove spaces and convert to lowercase
    cleaned = text.replace(" ", "").lower()
    return cleaned == cleaned[::-1]

# Test your functions
print(f"Vowels in 'Hello World': {count_vowels('Hello World')}")  # 3
print(f"Reverse of 'Python': {reverse_string('Python')}")          # nohtyP
print(f"Is 'racecar' a palindrome? {is_palindrome('racecar')}")    # True
print(f"Is 'A man a plan a canal Panama' a palindrome? {is_palindrome('A man a plan a canal Panama')}")  # True
```

### Solution 3: Default Parameters and Keyword Arguments
```python
def create_greeting(name, greeting="Hello", punctuation="!"):
    """Create a personalized greeting with customizable greeting and punctuation"""
    return f"{greeting}, {name}{punctuation}"

# Test with different combinations
print(create_greeting("Alice"))                           # Hello, Alice!
print(create_greeting("Bob", "Hi"))                      # Hi, Bob!
print(create_greeting("Charlie", "Hey", "."))            # Hey, Charlie.
print(create_greeting("David", punctuation="?"))         # Hello, David?
print(create_greeting("Eve", greeting="Good morning"))   # Good morning, Eve!
```

### Solution 4: Variable Arguments
```python
def calculate_average(*numbers):
    """Calculate the average of any number of arguments"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def create_profile(name, **details):
    """Create a profile dictionary with name and additional details"""
    profile = {"name": name}
    profile.update(details)
    return profile

# Test your functions
print(f"Average of 1, 2, 3, 4, 5: {calculate_average(1, 2, 3, 4, 5)}")  # 3.0
print(f"Average of 10, 20: {calculate_average(10, 20)}")                 # 15.0

profile = create_profile("Alice", age=25, city="New York", occupation="Engineer")
print(profile)  # {'name': 'Alice', 'age': 25, 'city': 'New York', 'occupation': 'Engineer'}
```

### Solution 5: Factorial Calculator
```python
def factorial(n):
    """Calculate factorial of n using recursion"""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Test your function
print(f"5! = {factorial(5)}")  # 120
print(f"0! = {factorial(0)}")  # 1
```

### Solution 6: Fibonacci Sequence
```python
def fibonacci(n):
    """Calculate the nth Fibonacci number using recursion"""
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

# Test your function
print("First 10 Fibonacci numbers:")
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
# Output:
# F(0) = 0
# F(1) = 1
# F(2) = 1
# F(3) = 2
# F(4) = 3
# F(5) = 5
# F(6) = 8
# F(7) = 13
# F(8) = 21
# F(9) = 34
```

### Solution 7: Function as Parameter
```python
def apply_function(numbers, func):
    """Apply a function to each number in a list"""
    return [func(num) for num in numbers]

def square(x):
    """Square a number"""
    return x ** 2

def double(x):
    """Double a number"""
    return x * 2

# Test your function
numbers = [1, 2, 3, 4, 5]
print(f"Original: {numbers}")
print(f"Squared: {apply_function(numbers, square)}")  # [1, 4, 9, 16, 25]
print(f"Doubled: {apply_function(numbers, double)}")   # [2, 4, 6, 8, 10]
```

### Solution 8: Decorator Practice
```python
import time
import functools

def timing_decorator(func):
    """Decorator that measures function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    """Function that takes some time to execute"""
    time.sleep(1)
    return "Done!"

# Test your decorator
result = slow_function()
print(result)
# Output:
# slow_function executed in 1.0041 seconds
# Done!
```

## 🎯 Advanced Practice Problems

### Problem 9: Function Composition
Create a function that composes two functions (applies one after another).

```python
def compose(f, g):
    """Compose two functions: returns a function that applies g then f"""
    # Your code here
    pass

def add_one(x):
    return x + 1

def multiply_by_two(x):
    return x * 2

# Create composed function
add_one_then_multiply = compose(multiply_by_two, add_one)
print(add_one_then_multiply(5))  # Should be 12 (5+1=6, 6*2=12)
```

### Problem 10: Memoization
Implement a memoization decorator to cache function results.

```python
def memoize(func):
    """Decorator that caches function results"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        # Your code here
        pass
    
    return wrapper

@memoize
def fibonacci_memo(n):
    """Fibonacci with memoization"""
    if n < 2:
        return n
    return fibonacci_memo(n-1) + fibonacci_memo(n-2)

# Test the performance difference
import time

# Without memoization (will be slow for large n)
start = time.time()
result1 = fibonacci(30)
time1 = time.time() - start

# With memoization (should be fast)
start = time.time()
result2 = fibonacci_memo(30)
time2 = time.time() - start

print(f"Without memoization: {result1} in {time1:.4f} seconds")
print(f"With memoization: {result2} in {time2:.4f} seconds")
```

## 📚 Best Practices

1. **Use descriptive function names**: `calculate_area()` is better than `calc()`
2. **Keep functions focused**: Each function should do one thing well
3. **Use docstrings**: Document what your functions do
4. **Handle edge cases**: Consider what happens with invalid inputs
5. **Use type hints**: Make your code more readable and catch errors early
6. **Avoid global variables**: Pass data as parameters instead
7. **Return early**: Use early returns to avoid deep nesting
8. **Test your functions**: Write tests to ensure they work correctly

## 🎯 Summary

Functions are fundamental to Python programming because they:
- Promote code reuse
- Improve readability
- Make debugging easier
- Enable modular design
- Support functional programming paradigms

Master these concepts and you'll be well on your way to writing clean, efficient Python code!

---

**Practice makes perfect! Keep coding and experimenting with functions.** 🐍