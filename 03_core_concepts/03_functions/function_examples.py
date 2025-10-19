# Function Examples

# Basic Function Definition
print("=== Basic Function Definition ===")
def greet():
    print("Hello, World!")

greet()  # Output: Hello, World!

# Function with Parameters
print("\n=== Function with Parameters ===")
def greet_person(name):
    print(f"Hello, {name}!")

greet_person("Alice")  # Output: Hello, Alice!
greet_person("Bob")    # Output: Hello, Bob!

# Function with Return Values
print("\n=== Function with Return Values ===")
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)  # Output: 8
print(add_numbers(10, 20))  # Output: 30

# Function Parameters and Arguments
print("\n=== Function Parameters and Arguments ===")

# Positional Arguments
def introduce(name, age, city):
    print(f"Hi, I'm {name}, {age} years old, from {city}.")

introduce("Alice", 25, "New York")

# Keyword Arguments
introduce(city="Boston", name="Bob", age=30)

# Default Parameters
def greet_person(name, greeting="Hello", punctuation="!"):
    print(f"{greeting}, {name}{punctuation}")

greet_person("Alice")  # Output: Hello, Alice!
greet_person("Bob", "Hi")  # Output: Hi, Bob!
greet_person("Charlie", "Hey", "?")  # Output: Hey, Charlie?

# Variable Number of Arguments
print("\n=== Variable Number of Arguments ===")

# *args - variable positional arguments
def sum_numbers(*args):
    total = 0
    for num in args:
        total += num
    return total

print(sum_numbers(1, 2, 3))        # Output: 6
print(sum_numbers(1, 2, 3, 4, 5))  # Output: 15

# **kwargs - variable keyword arguments
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="New York")

# Combining *args and **kwargs
def flexible_function(*args, **kwargs):
    print("Positional arguments:", args)
    print("Keyword arguments:", kwargs)

flexible_function(1, 2, 3, name="Alice", age=25)

# Return Values
print("\n=== Return Values ===")

# Single Return Value
def square(number):
    return number ** 2

print(square(5))  # Output: 25

# Multiple Return Values
def get_name_age():
    return "Alice", 25

name, age = get_name_age()
print(f"Name: {name}, Age: {age}")  # Output: Name: Alice, Age: 25

# Early Returns
def check_number(number):
    if number > 0:
        return "Positive"
    elif number < 0:
        return "Negative"
    else:
        return "Zero"

print(check_number(5))   # Output: Positive
print(check_number(-3))  # Output: Negative
print(check_number(0))   # Output: Zero

# Recursion
print("\n=== Recursion ===")

# Factorial using recursion
def factorial(n):
    # Base case
    if n == 0 or n == 1:
        return 1
    # Recursive case
    else:
        return n * factorial(n - 1)

print(f"Factorial of 5: {factorial(5)}")  # Output: 120

# Fibonacci sequence using recursion
def fibonacci(n):
    # Base cases
    if n <= 1:
        return n
    # Recursive case
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

print("First 10 Fibonacci numbers:")
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")

# Function Best Practices
print("\n=== Function Best Practices ===")

# Docstrings
def calculate_area(length, width):
    """
    Calculate the area of a rectangle.
    
    Args:
        length (float): The length of the rectangle
        width (float): The width of the rectangle
    
    Returns:
        float: The area of the rectangle
    
    Raises:
        ValueError: If length or width is negative
    """
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    return length * width

print(calculate_area.__doc__)

# Function Annotations
def greet_person(name: str, age: int) -> str:
    """
    Greet a person with their name and age.
    
    Args:
        name (str): The person's name
        age (int): The person's age
    
    Returns:
        str: A greeting message
    """
    return f"Hello, {name}! You are {age} years old."

message = greet_person("Alice", 25)
print(message)  # Output: Hello, Alice! You are 25 years old.

# Lambda Functions
print("\n=== Lambda Functions ===")

# Lambda functions (anonymous functions)
square = lambda x: x ** 2
print(f"Square of 5: {square(5)}")  # Output: 25

# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(f"Squared numbers: {squared}")  # Output: [1, 4, 9, 16, 25]

# Filtering with lambda
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {even_numbers}")  # Output: [2, 4]

# Practical Example: Calculator Functions
print("\n=== Practical Example: Calculator Functions ===")

def add(a, b):
    """Add two numbers"""
    return a + b

def subtract(a, b):
    """Subtract b from a"""
    return a - b

def multiply(a, b):
    """Multiply two numbers"""
    return a * b

def divide(a, b):
    """Divide a by b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def calculator(operation, a, b):
    """Perform calculation based on operation"""
    operations = {
        '+': add,
        '-': subtract,
        '*': multiply,
        '/': divide
    }
    
    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return operations[operation](a, b)

# Usage
try:
    print(f"5 + 3 = {calculator('+', 5, 3)}")
    print(f"10 - 4 = {calculator('-', 10, 4)}")
    print(f"6 * 7 = {calculator('*', 6, 7)}")
    print(f"15 / 3 = {calculator('/', 15, 3)}")
except ValueError as e:
    print(f"Error: {e}")

# Practical Example: Text Processing Functions
print("\n=== Practical Example: Text Processing Functions ===")

def count_words(text):
    """Count the number of words in text"""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    return len(text.split())

def count_characters(text, include_spaces=True):
    """Count characters in text"""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(" ", ""))

sample_text = "python is a powerful programming language"
print(f"Original: {sample_text}")
print(f"Word count: {count_words(sample_text)}")
print(f"Character count (with spaces): {count_characters(sample_text)}")
print(f"Character count (without spaces): {count_characters(sample_text, False)}")

# Common Mistakes Demonstration
print("\n=== Common Mistakes Demonstration ===")

# Mutable default arguments - correct approach
def add_item_correct(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item_correct("apple"))      # ['apple']
print(add_item_correct("banana"))     # ['banana']

# Function with proper documentation
def calculate_area_proper(length, width):
    """Calculate rectangle area"""
    return length * width

result = calculate_area_proper(5, 3)
print(f"Area: {result}")  # Output: Area: 15