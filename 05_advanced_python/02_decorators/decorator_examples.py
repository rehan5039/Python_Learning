# Decorator Examples

# Basic Decorator Concept
print("=== Basic Decorator Concept ===")

# Without decorator
def greet():
    return "Hello!"

# Manually applying decorator concept
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        result = func()
        print("Something is happening after the function is called.")
        return result
    return wrapper

# Applying decorator manually
greet = my_decorator(greet)
print(greet())

# Using the @ syntax
print("\n=== Using the @ Syntax ===")

def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        result = func()
        print("Something is happening after the function is called.")
        return result
    return wrapper

@my_decorator
def greet():
    return "Hello!"

print(greet())

# Function Decorators
print("\n=== Function Decorators ===")

def uppercase_decorator(func):
    def wrapper():
        result = func()
        return result.upper()
    return wrapper

@uppercase_decorator
def greet():
    return "hello, world!"

print(greet())

# Decorator Preserving Function Metadata
print("\n=== Decorator Preserving Function Metadata ===")

import functools

def uppercase_decorator(func):
    @functools.wraps(func)
    def wrapper():
        result = func()
        return result.upper()
    return wrapper

@uppercase_decorator
def greet():
    """Return a greeting message"""
    return "hello, world!"

print(greet())
print(greet.__name__)
print(greet.__doc__)

# Decorator with Function Arguments
print("\n=== Decorator with Function Arguments ===")

def debug_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

@debug_decorator
def add_numbers(a, b):
    """Add two numbers"""
    return a + b

@debug_decorator
def greet_person(name, greeting="Hello"):
    """Greet a person"""
    return f"{greeting}, {name}!"

print(add_numbers(3, 5))
print(greet_person("Alice", greeting="Hi"))

# Decorator Factories
print("\n=== Decorator Factories ===")

def repeat(num_times):
    """Decorator factory that repeats a function num_times"""
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator_repeat

@repeat(3)
def greet():
    print("Hello!")

greet()

# Timing Decorator
print("\n=== Timing Decorator ===")

import time

def timer(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return result
    return wrapper

@timer
def slow_function():
    """A function that takes some time to execute"""
    time.sleep(0.1)
    return "Done!"

result = slow_function()

# Logging Decorator
print("\n=== Logging Decorator ===")

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_calls(func):
    """Decorator to log function calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.info(f"Calling {func.__name__}({signature})")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} returned {result!r}")
            return result
        except Exception as e:
            logger.exception(f"Exception raised in {func.__name__}. exception: {e!r}")
            raise
    return wrapper

@log_calls
def calculate_area(length, width):
    """Calculate the area of a rectangle"""
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    return length * width

# Usage
try:
    area = calculate_area(5, 3)
    print(f"Area: {area}")
    
    area = calculate_area(-1, 3)
except ValueError as e:
    print(f"Error: {e}")

# Caching Decorator
print("\n=== Caching Decorator ===")

def memoize(func):
    """Decorator to cache function results"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        if key in cache:
            print(f"Cache hit for {func.__name__} with args {args}")
            return cache[key]
        
        print(f"Cache miss for {func.__name__} with args {args}")
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    
    # Add cache info method
    wrapper.cache_info = lambda: f"Cache size: {len(cache)}"
    wrapper.cache_clear = lambda: cache.clear()
    
    return wrapper

@memoize
def fibonacci(n):
    """Calculate the nth Fibonacci number"""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Usage
print(fibonacci(10))
print(fibonacci.cache_info())

# Second call - should use cache
print(fibonacci(10))
print(fibonacci.cache_info())

# Class Decorators
print("\n=== Class Decorators ===")

def add_str_method(cls):
    """Class decorator to add a __str__ method"""
    def __str__(self):
        attrs = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return f"{cls.__name__}({', '.join(attrs)})"
    
    cls.__str__ = __str__
    return cls

@add_str_method
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 25)
print(person)

# Built-in Decorators
print("\n=== Built-in Decorators ===")

class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        """Get the radius"""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Set the radius with validation"""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        """Calculate the area (read-only)"""
        import math
        return math.pi * self._radius ** 2

# Usage
circle = Circle(5)
print(f"Radius: {circle.radius}")
print(f"Area: {circle.area:.2f}")

circle.radius = 7
print(f"New radius: {circle.radius}")

# Practical Example: Authentication Decorator
print("\n=== Practical Example: Authentication Decorator ===")

# Simulate user authentication
class User:
    def __init__(self, username: str, role: str):
        self.username = username
        self.role = role

# Global user context
current_user = None

def login_user(username: str, role: str):
    """Simulate user login"""
    global current_user
    current_user = User(username, role)
    print(f"Logged in as {username} ({role})")

def logout_user():
    """Simulate user logout"""
    global current_user
    current_user = None
    print("Logged out")

def requires_auth(roles=None):
    """Decorator factory for authentication"""
    def decorator_auth(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if current_user is None:
                raise PermissionError("Authentication required")
            
            if roles and current_user.role not in roles:
                raise PermissionError(f"Access denied. Required roles: {roles}")
            
            print(f"User {current_user.username} authorized to call {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator_auth

# Application functions
@requires_auth()
def view_profile():
    """View user profile"""
    return f"Profile data for {current_user.username}"

@requires_auth(roles=["admin"])
def delete_user(username: str):
    """Delete a user (admin only)"""
    return f"User {username} deleted by {current_user.username}"

# Usage
try:
    view_profile()
except PermissionError as e:
    print(f"Error: {e}")

login_user("alice", "user")
print(view_profile())

try:
    print(delete_user("bob"))
except PermissionError as e:
    print(f"Error: {e}")

logout_user()
login_user("admin", "admin")
print(delete_user("bob"))

logout_user()

# Common Mistakes Demonstration
print("\n=== Common Mistakes Demonstration ===")

# Correct decorator with functools.wraps
def good_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@good_decorator
def example_function():
    """This is an example function"""
    pass

print(f"Function name: {example_function.__name__}")
print(f"Function doc: {example_function.__doc__}")