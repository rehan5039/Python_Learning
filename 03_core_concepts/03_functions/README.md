# 📞 Functions in Python

Functions are reusable blocks of code that perform specific tasks. They help organize code, reduce repetition, and make programs more modular and maintainable. This guide will teach you how to create and use functions effectively.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Define and call functions
- Use function parameters and return values
- Apply different types of function arguments
- Implement recursion
- Write clean and reusable function code

## 📝 What are Functions?

Functions are named blocks of code that perform a specific task and can be called multiple times throughout a program. They promote code reusability and modularity.

### Basic Function Definition

```python
# Basic function definition
def greet():
    print("Hello, World!")

# Calling a function
greet()  # Output: Hello, World!
```

### Function with Parameters

```python
# Function with parameters
def greet_person(name):
    print(f"Hello, {name}!")

# Calling function with arguments
greet_person("Alice")  # Output: Hello, Alice!
greet_person("Bob")    # Output: Hello, Bob!
```

### Function with Return Values

```python
# Function that returns a value
def add_numbers(a, b):
    return a + b

# Using the return value
result = add_numbers(5, 3)
print(result)  # Output: 8

# Using return value directly
print(add_numbers(10, 20))  # Output: 30
```

## 🔧 Function Parameters and Arguments

### Positional Arguments

```python
def introduce(name, age, city):
    print(f"Hi, I'm {name}, {age} years old, from {city}.")

# Arguments must be in the correct order
introduce("Alice", 25, "New York")
# Output: Hi, I'm Alice, 25 years old, from New York.
```

### Keyword Arguments

```python
# Using keyword arguments (order doesn't matter)
introduce(city="Boston", name="Bob", age=30)
# Output: Hi, I'm Bob, 30 years old, from Boston.

# Mixing positional and keyword arguments
introduce("Charlie", city="Chicago", age=35)
# Output: Hi, I'm Charlie, 35 years old, from Chicago.
```

### Default Parameters

```python
# Function with default parameters
def greet_person(name, greeting="Hello", punctuation="!"):
    print(f"{greeting}, {name}{punctuation}")

# Using default values
greet_person("Alice")  # Output: Hello, Alice!

# Overriding default values
greet_person("Bob", "Hi")  # Output: Hi, Bob!
greet_person("Charlie", "Hey", "?")  # Output: Hey, Charlie?
```

### Variable Number of Arguments

```python
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
# Output:
# name: Alice
# age: 25
# city: New York

# Combining *args and **kwargs
def flexible_function(*args, **kwargs):
    print("Positional arguments:", args)
    print("Keyword arguments:", kwargs)

flexible_function(1, 2, 3, name="Alice", age=25)
# Output:
# Positional arguments: (1, 2, 3)
# Keyword arguments: {'name': 'Alice', 'age': 25}
```

## 🔄 Return Values

### Single Return Value

```python
def square(number):
    return number ** 2

result = square(5)
print(result)  # Output: 25
```

### Multiple Return Values

```python
# Returning multiple values as a tuple
def get_name_age():
    return "Alice", 25

# Unpacking returned values
name, age = get_name_age()
print(f"Name: {name}, Age: {age}")  # Output: Name: Alice, Age: 25

# Returning multiple values as a dictionary
def get_person_info():
    return {"name": "Bob", "age": 30, "city": "Boston"}

info = get_person_info()
print(info)  # Output: {'name': 'Bob', 'age': 30, 'city': 'Boston'}
```

### Early Returns

```python
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
```

## 🔁 Recursion

Recursion is when a function calls itself to solve a problem.

### Basic Recursion Example

```python
# Factorial using recursion
def factorial(n):
    # Base case
    if n == 0 or n == 1:
        return 1
    # Recursive case
    else:
        return n * factorial(n - 1)

print(factorial(5))  # Output: 120
```

### Fibonacci Sequence

```python
# Fibonacci sequence using recursion
def fibonacci(n):
    # Base cases
    if n <= 1:
        return n
    # Recursive case
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Print first 10 Fibonacci numbers
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
```

### Recursive Factorial with Error Handling

```python
def safe_factorial(n):
    # Input validation
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    # Base case
    if n == 0 or n == 1:
        return 1
    # Recursive case
    else:
        return n * safe_factorial(n - 1)

try:
    print(safe_factorial(5))   # Output: 120
    print(safe_factorial(-1))  # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")
```

## 🛠 Function Best Practices

### Docstrings

```python
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
    
    Example:
        >>> calculate_area(5, 3)
        15
    """
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    return length * width

# Accessing docstring
print(calculate_area.__doc__)
```

### Function Annotations

```python
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

# Using annotations
message = greet_person("Alice", 25)
print(message)  # Output: Hello, Alice! You are 25 years old.
```

### Lambda Functions

```python
# Lambda functions (anonymous functions)
square = lambda x: x ** 2
print(square(5))  # Output: 25

# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # Output: [1, 4, 9, 16, 25]

# Filtering with lambda
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # Output: [2, 4]
```

## 🧪 Practical Examples

### Example 1: Calculator Functions
```python
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
```

### Example 2: Text Processing Functions
```python
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

def reverse_words(text):
    """Reverse the order of words in text"""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    return " ".join(text.split()[::-1])

def capitalize_words(text):
    """Capitalize the first letter of each word"""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    return " ".join(word.capitalize() for word in text.split())

# Usage
sample_text = "python is a powerful programming language"
print(f"Original: {sample_text}")
print(f"Word count: {count_words(sample_text)}")
print(f"Character count (with spaces): {count_characters(sample_text)}")
print(f"Character count (without spaces): {count_characters(sample_text, False)}")
print(f"Reversed words: {reverse_words(sample_text)}")
print(f"Capitalized words: {capitalize_words(sample_text)}")
```

### Example 3: Data Validation Functions
```python
def validate_email(email):
    """Validate email format"""
    if not isinstance(email, str):
        return False
    
    if "@" not in email or "." not in email:
        return False
    
    local, domain = email.split("@", 1)
    if not local or not domain:
        return False
    
    if "." not in domain:
        return False
    
    return True

def validate_password(password):
    """Validate password strength"""
    if not isinstance(password, str):
        return False
    
    if len(password) < 8:
        return False
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()-_" for c in password)
    
    return has_upper and has_lower and has_digit and has_special

def validate_user_data(name, email, password):
    """Validate user registration data"""
    errors = []
    
    if not name or len(name.strip()) < 2:
        errors.append("Name must be at least 2 characters long")
    
    if not validate_email(email):
        errors.append("Invalid email format")
    
    if not validate_password(password):
        errors.append("Password must be at least 8 characters with uppercase, lowercase, digit, and special character")
    
    return len(errors) == 0, errors

# Usage
test_cases = [
    ("Alice", "alice@example.com", "Password123!"),
    ("", "invalid-email", "weak"),
    ("Bob", "bob@example.com", "StrongPass123@")
]

for name, email, password in test_cases:
    is_valid, errors = validate_user_data(name, email, password)
    print(f"\nName: {name}, Email: {email}, Password: {password}")
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
```

## ⚠️ Common Mistakes and Tips

### 1. Mutable Default Arguments
```python
# Wrong - mutable default argument
# def add_item_wrong(item, items=[]):
#     items.append(item)
#     return items

# Correct - use None as default
def add_item_correct(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

# Testing
print(add_item_correct("apple"))      # ['apple']
print(add_item_correct("banana"))     # ['banana'] (not ['apple', 'banana'])
```

### 2. Modifying Global Variables
```python
# Avoid modifying global variables inside functions
counter = 0

# Wrong approach
# def increment_wrong():
#     global counter
#     counter += 1
#     return counter

# Better approach
def increment_correct(counter):
    return counter + 1

counter = increment_correct(counter)
print(counter)  # 1
```

### 3. Returning vs Printing
```python
# Wrong - mixing return and print
# def calculate_wrong(a, b):
#     print(a + b)  # This prints but doesn't return

# Correct - separate concerns
def calculate_correct(a, b):
    result = a + b
    print(f"Calculation: {a} + {b} = {result}")
    return result

result = calculate_correct(5, 3)
# Can use result for further calculations
```

### 4. Function Naming
```python
# Good naming practices
def calculate_area(length, width):
    """Descriptive name, clear parameters"""
    return length * width

def is_even(number):
    """Boolean function with 'is_' prefix"""
    return number % 2 == 0

def get_user_name():
    """Getter function with 'get_' prefix"""
    return "Alice"
```

## 📚 Next Steps

Now that you understand functions, you're ready to learn:

1. **Exception Handling**: Managing errors gracefully
2. **File I/O**: Reading and writing files
3. **Object-Oriented Programming**: Advanced programming concepts
4. **Modules and Packages**: Organizing code in larger projects

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: What's the difference between parameters and arguments?
A: Parameters are the variables defined in the function signature, while arguments are the actual values passed when calling the function.

### Q: When should I use `*args` and `**kwargs`?
A: Use `*args` when you want to accept a variable number of positional arguments, and `**kwargs` for variable keyword arguments.

### Q: What is recursion used for?
A: Recursion is useful for problems that can be broken down into smaller, similar subproblems (like tree traversal, factorial calculation, etc.).

### Q: Are lambda functions better than regular functions?
A: Lambda functions are good for simple, one-line operations. For complex logic, regular functions with proper documentation are better.

---

**Practice writing different types of functions with various scenarios to build your skills!** 🐍