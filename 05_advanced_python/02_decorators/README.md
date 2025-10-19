# 🎨 Decorators in Python

Decorators are a powerful and elegant feature in Python that allow you to modify or extend the behavior of functions and classes without permanently modifying their code. This guide will teach you how to create and use decorators effectively.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Understand what decorators are and how they work
- Create function decorators with and without arguments
- Implement class decorators
- Use built-in decorators like `@property`, `@staticmethod`, and `@classmethod`
- Apply decorators for logging, timing, and caching
- Create decorator factories and nested decorators

## 🎨 What are Decorators?

Decorators are functions that modify the behavior of other functions or classes. They provide a clean and readable way to add functionality to existing code.

### Basic Decorator Concept

```python
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
```

### Using the `@` Syntax

```python
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
```

## 🔧 Function Decorators

### Basic Function Decorator

```python
def uppercase_decorator(func):
    def wrapper():
        result = func()
        return result.upper()
    return wrapper

@uppercase_decorator
def greet():
    return "hello, world!"

print(greet())  # HELLO, WORLD!
```

### Decorator Preserving Function Metadata

```python
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

print(greet())           # HELLO, WORLD!
print(greet.__name__)    # greet (not wrapper)
print(greet.__doc__)     # Return a greeting message
```

### Decorator with Function Arguments

```python
import functools

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
```

## 🏭 Decorator Factories

Decorator factories are functions that return decorators, allowing you to pass arguments to decorators.

### Basic Decorator Factory

```python
import functools

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
```

### Advanced Decorator Factory

```python
import functools
import time

def retry(max_attempts=3, delay=1):
    """Decorator factory for retrying function calls"""
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator_retry

@retry(max_attempts=3, delay=0.5)
def unreliable_function():
    """A function that sometimes fails"""
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ValueError("Random failure occurred")
    return "Success!"

# Usage
try:
    result = unreliable_function()
    print(result)
except ValueError as e:
    print(f"Function failed after all retries: {e}")
```

## 🕐 Practical Decorators

### Timing Decorator

```python
import functools
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
    time.sleep(1)
    return "Done!"

result = slow_function()
```

### Logging Decorator

```python
import functools
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
    
    area = calculate_area(-1, 3)  # This will raise an exception
except ValueError as e:
    print(f"Error: {e}")
```

### Caching Decorator

```python
import functools

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
```

##  deep dive into Class Decorators

### Basic Class Decorator

```python
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
print(person)  # Person(name='Alice', age=25)
```

### Advanced Class Decorator

```python
def singleton(cls):
    """Class decorator to make a class a singleton"""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        self.connection_id = id(self)
        print(f"Creating database connection {self.connection_id}")
    
    def query(self, sql):
        return f"Executing {sql} on connection {self.connection_id}"

# Usage
db1 = DatabaseConnection()  # Creating database connection
db2 = DatabaseConnection()  # No output - same instance
print(db1 is db2)  # True
print(db1.query("SELECT * FROM users"))
```

## 🎯 Built-in Decorators

### @property Decorator

```python
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
```

### @staticmethod and @classmethod Decorators

```python
class MathUtils:
    pi = 3.14159
    
    @staticmethod
    def add(x, y):
        """Add two numbers"""
        return x + y
    
    @classmethod
    def circle_area(cls, radius):
        """Calculate circle area using class variable"""
        return cls.pi * radius ** 2

# Usage
print(MathUtils.add(5, 3))           # 8
print(MathUtils.circle_area(5))      # 78.53975
```

## 🧪 Practical Examples

### Example 1: Authentication Decorator
```python
import functools
from typing import Callable, Any

# Simulate user authentication
class User:
    def __init__(self, username: str, role: str):
        self.username = username
        self.role = role

# Global user context (in real apps, this would be session-based)
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
    def decorator_auth(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if current_user is None:
                raise PermissionError("Authentication required")
            
            if roles and current_user.role not in roles:
                raise PermissionError(f"Access denied. Required roles: {roles}")
            
            print(f"User {current_user.username} authorized to call {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator_auth

# Application functions with different permission requirements
@requires_auth()
def view_profile():
    """View user profile"""
    return f"Profile data for {current_user.username}"

@requires_auth(roles=["admin", "moderator"])
def delete_user(username: str):
    """Delete a user (admin/moderator only)"""
    return f"User {username} deleted by {current_user.username}"

@requires_auth(roles=["admin"])
def create_admin_report():
    """Create admin report (admin only)"""
    return f"Admin report generated by {current_user.username}"

# Usage demonstration
print("=== Authentication Decorator Demo ===")

# Try to access without login
try:
    view_profile()
except PermissionError as e:
    print(f"Error: {e}")

# Login as regular user
login_user("alice", "user")

# Regular user can view profile
try:
    print(view_profile())
except PermissionError as e:
    print(f"Error: {e}")

# Regular user cannot delete users
try:
    print(delete_user("bob"))
except PermissionError as e:
    print(f"Error: {e}")

# Login as admin
logout_user()
login_user("admin", "admin")

# Admin can do everything
try:
    print(view_profile())
    print(delete_user("bob"))
    print(create_admin_report())
except PermissionError as e:
    print(f"Error: {e}")

logout_user()
```

### Example 2: Rate Limiting Decorator
```python
import functools
import time
from collections import defaultdict, deque
from typing import Callable, Any

class RateLimitError(Exception):
    """Raised when rate limit is exceeded"""
    pass

def rate_limit(calls: int, period: int):
    """Decorator factory for rate limiting function calls"""
    def decorator_rate_limit(func: Callable) -> Callable:
        # Store call history for each function
        call_history = defaultdict(deque)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            now = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Remove old calls outside the time window
            while call_history[func_name] and call_history[func_name][0] < now - period:
                call_history[func_name].popleft()
            
            # Check if we're within the limit
            if len(call_history[func_name]) >= calls:
                oldest_call = call_history[func_name][0]
                wait_time = period - (now - oldest_call)
                raise RateLimitError(
                    f"Rate limit exceeded for {func_name}. "
                    f"Try again in {wait_time:.1f} seconds."
                )
            
            # Record this call
            call_history[func_name].append(now)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator_rate_limit

# Simulate API calls with rate limiting
@rate_limit(calls=3, period=10)  # 3 calls per 10 seconds
def api_call(endpoint: str) -> str:
    """Simulate an API call"""
    return f"Response from {endpoint}"

# Usage demonstration
print("=== Rate Limiting Decorator Demo ===")

def test_rate_limiting():
    for i in range(5):
        try:
            result = api_call(f"/api/data/{i}")
            print(f"Call {i+1}: {result}")
        except RateLimitError as e:
            print(f"Call {i+1}: {e}")
        time.sleep(1)  # Wait 1 second between calls

test_rate_limiting()

print("\nWaiting 10 seconds to reset rate limit...")
time.sleep(10)

print("\nTrying again after reset:")
test_rate_limiting()
```

### Example 3: Validation Decorator
```python
import functools
from typing import Callable, Any, Dict, List

class ValidationError(Exception):
    """Raised when validation fails"""
    pass

def validate_args(**validators):
    """Decorator factory for argument validation"""
    def decorator_validate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Validation failed for parameter '{param_name}': {value}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator_validate

# Validation functions
def is_positive(value):
    return isinstance(value, (int, float)) and value > 0

def is_email(value):
    return isinstance(value, str) and "@" in value and "." in value

def is_non_empty(value):
    return isinstance(value, str) and len(value.strip()) > 0

# Functions with validation
@validate_args(amount=is_positive)
def withdraw_money(amount: float) -> str:
    """Withdraw money from account"""
    return f"Withdrew ${amount:.2f}"

@validate_args(email=is_email, name=is_non_empty)
def register_user(name: str, email: str) -> str:
    """Register a new user"""
    return f"Registered user {name} with email {email}"

@validate_args(length=is_positive, width=is_positive)
def calculate_area(length: float, width: float) -> float:
    """Calculate rectangle area"""
    return length * width

# Usage demonstration
print("=== Validation Decorator Demo ===")

# Valid calls
try:
    print(withdraw_money(100.50))
    print(register_user("Alice", "alice@example.com"))
    print(f"Area: {calculate_area(5.0, 3.0)}")
except ValidationError as e:
    print(f"Validation error: {e}")

# Invalid calls
print("\nTesting invalid inputs:")

try:
    withdraw_money(-50)  # Negative amount
except ValidationError as e:
    print(f"Error: {e}")

try:
    register_user("", "invalid-email")  # Empty name, invalid email
except ValidationError as e:
    print(f"Error: {e}")

try:
    calculate_area(-5, 3)  # Negative length
except ValidationError as e:
    print(f"Error: {e}")
```

## ⚠️ Common Mistakes and Best Practices

### 1. Forgetting to Use `@functools.wraps`

```python
import functools

# Wrong - metadata not preserved
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Correct - metadata preserved
def good_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@good_decorator
def example_function():
    """This is an example function"""
    pass

print(example_function.__name__)  # example_function
print(example_function.__doc__)   # This is an example function
```

### 2. Not Handling Function Arguments Properly

```python
import functools

# Wrong - doesn't handle arguments
# def bad_timer(func):
#     def wrapper():
#         import time
#         start = time.time()
#         result = func()
#         end = time.time()
#         print(f"{func.__name__} took {end - start} seconds")
#         return result
#     return wrapper

# Correct - handles all arguments
def good_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@good_timer
def slow_add(a, b):
    import time
    time.sleep(0.1)
    return a + b

print(slow_add(3, 4))
```

### 3. Creating Decorators That Are Hard to Debug

```python
import functools

def debuggable_decorator(func):
    """Decorator that's easy to debug"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[DEBUG] Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            print(f"[DEBUG] {func.__name__} returned {result}")
            return result
        except Exception as e:
            print(f"[DEBUG] {func.__name__} raised {type(e).__name__}: {e}")
            raise
    return wrapper

@debuggable_decorator
def divide(a, b):
    return a / b

# Usage
try:
    print(divide(10, 2))
    print(divide(10, 0))  # This will raise an exception
except ZeroDivisionError as e:
    print(f"Caught exception: {e}")
```

## 📚 Next Steps

Now that you understand decorators, you're ready to learn:

1. **Generators and Iterators**: Memory-efficient data processing
2. **Context Managers**: Resource management with `with` statements
3. **Metaclasses**: Classes that create classes
4. **Descriptors**: Custom attribute access

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: What's the difference between `@functools.wraps` and not using it?
A: `@functools.wraps` preserves the original function's metadata (name, docstring, etc.), making debugging easier.

### Q: Can I stack multiple decorators?
A: Yes, you can stack decorators. They are applied from bottom to top (inside out).

### Q: How do I pass arguments to decorators?
A: Use decorator factories - functions that return decorators.

### Q: When should I use class decorators vs function decorators?
A: Use class decorators when you want to modify class behavior, and function decorators for function behavior.

---

**Practice creating different types of decorators with various scenarios to build your expertise!** 🐍