# 🚨 Exception Handling in Python

Exception handling is a crucial aspect of robust programming that allows you to gracefully handle errors and unexpected situations in your code. This guide will teach you how to use try-except blocks, create custom exceptions, and implement proper error handling strategies.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Use try-except blocks to handle exceptions
- Implement finally and else clauses
- Create and raise custom exceptions
- Understand exception hierarchy
- Apply best practices for error handling
- Use context managers for resource management

## 🚫 What are Exceptions?

Exceptions are events that disrupt the normal flow of a program's execution. They occur when an error or unexpected condition is encountered during program execution.

### Basic Exception Handling

```python
# Basic try-except block
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Handling multiple exceptions
try:
    number = int(input("Enter a number: "))
    result = 10 / number
    print(f"Result: {result}")
except ValueError:
    print("Invalid input! Please enter a valid number.")
except ZeroDivisionError:
    print("Cannot divide by zero!")
```

### Exception Hierarchy

Python has a hierarchy of built-in exceptions. Understanding this hierarchy helps you catch exceptions appropriately.

```python
# Exception hierarchy example
try:
    # This will raise a ValueError
    number = int("not_a_number")
except ValueError as e:
    print(f"ValueError: {e}")
except ArithmeticError as e:
    print(f"ArithmeticError: {e}")
except Exception as e:
    print(f"General Exception: {e}")
```

## 🛠 Try-Except Blocks

### Basic Structure

```python
try:
    # Code that might raise an exception
    pass
except ExceptionType:
    # Code to handle the exception
    pass
else:
    # Code that runs if no exception occurred
    pass
finally:
    # Code that always runs
    pass
```

### Detailed Exception Handling

```python
def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError as e:
        print(f"Error: Cannot divide by zero - {e}")
        return None
    except TypeError as e:
        print(f"Error: Invalid types for division - {e}")
        return None
    else:
        print("Division successful!")
        return result
    finally:
        print("Division operation completed.")

# Usage
print(divide_numbers(10, 2))   # 5.0
print(divide_numbers(10, 0))   # Error: Cannot divide by zero
print(divide_numbers(10, "2")) # Error: Invalid types for division
```

### Catching Multiple Exceptions

```python
def process_data(data):
    try:
        # Convert to integer
        number = int(data)
        # Perform calculation
        result = 100 / number
        return result
    except (ValueError, ZeroDivisionError) as e:
        print(f"Error processing data: {type(e).__name__} - {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
print(process_data("10"))    # 10.0
print(process_data("0"))     # Error processing data: ZeroDivisionError
print(process_data("abc"))   # Error processing data: ValueError
print(process_data(None))    # Unexpected error: int() argument must be a string...
```

## 🎯 Else and Finally Clauses

### The Else Clause

The else clause runs only if no exception was raised in the try block.

```python
def read_file(filename):
    try:
        file = open(filename, 'r')
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    else:
        # This runs only if no exception occurred
        content = file.read()
        file.close()
        print(f"Successfully read {len(content)} characters from {filename}")
        return content
    finally:
        # This always runs
        print("File operation completed.")

# Usage
content = read_file("nonexistent.txt")  # File nonexistent.txt not found.
content = read_file("example.txt")      # Successfully read X characters...
```

### The Finally Clause

The finally clause always executes, regardless of whether an exception occurred.

```python
def database_operation():
    connection = None
    try:
        print("Connecting to database...")
        # Simulate database connection
        connection = "database_connection"
        # Simulate an operation that might fail
        raise ValueError("Database operation failed")
    except ValueError as e:
        print(f"Database error: {e}")
    finally:
        # This always runs, even if an exception occurred
        if connection:
            print("Closing database connection...")
        print("Database operation cleanup completed.")

# Usage
database_operation()
```

## 🚀 Raising Exceptions

You can raise exceptions manually using the `raise` statement.

### Basic Exception Raising

```python
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    elif age > 150:
        raise ValueError("Age seems unrealistic")
    else:
        return f"Valid age: {age}"

# Usage
try:
    print(validate_age(25))   # Valid age: 25
    print(validate_age(-5))   # Raises ValueError
except ValueError as e:
    print(f"Invalid age: {e}")
```

### Re-raising Exceptions

```python
def process_data(data):
    try:
        # Some processing that might fail
        result = int(data) * 2
        return result
    except ValueError:
        print("Logging the error...")
        # Re-raise the exception
        raise

# Usage
try:
    result = process_data("invalid")
except ValueError:
    print("Handled in main code")
```

## 🏷️ Custom Exceptions

Creating custom exceptions makes your code more readable and allows for specific error handling.

### Basic Custom Exception

```python
class CustomError(Exception):
    """Base class for custom exceptions"""
    pass

class ValidationError(CustomError):
    """Raised when data validation fails"""
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code

class ProcessingError(CustomError):
    """Raised when data processing fails"""
    pass

def validate_email(email):
    if "@" not in email:
        raise ValidationError("Invalid email format", error_code="INVALID_FORMAT")
    if len(email) < 5:
        raise ValidationError("Email too short", error_code="TOO_SHORT")

def process_user_data(email):
    try:
        validate_email(email)
        print(f"Processing email: {email}")
    except ValidationError as e:
        print(f"Validation failed: {e}")
        if hasattr(e, 'error_code'):
            print(f"Error code: {e.error_code}")

# Usage
process_user_data("invalid-email")      # Validation failed: Invalid email format
process_user_data("a@b")               # Validation failed: Email too short
process_user_data("valid@example.com") # Processing email: valid@example.com
```

### Advanced Custom Exception

```python
class BankError(Exception):
    """Base exception for banking operations"""
    pass

class InsufficientFundsError(BankError):
    """Raised when account has insufficient funds"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Insufficient funds: Balance ${balance}, Attempted ${amount}")

class InvalidAmountError(BankError):
    """Raised when transaction amount is invalid"""
    def __init__(self, amount):
        self.amount = amount
        super().__init__(f"Invalid amount: ${amount}")

class BankAccount:
    def __init__(self, account_holder, initial_balance=0):
        self.account_holder = account_holder
        self.balance = initial_balance
    
    def deposit(self, amount):
        if amount <= 0:
            raise InvalidAmountError(amount)
        self.balance += amount
        return f"Deposited ${amount}. New balance: ${self.balance}"
    
    def withdraw(self, amount):
        if amount <= 0:
            raise InvalidAmountError(amount)
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        return f"Withdrew ${amount}. New balance: ${self.balance}"

# Usage
account = BankAccount("Alice", 100)

try:
    print(account.deposit(50))
    print(account.withdraw(200))  # Raises InsufficientFundsError
except InsufficientFundsError as e:
    print(f"Transaction failed: {e}")
    print(f"Available balance: ${e.balance}")
except InvalidAmountError as e:
    print(f"Invalid transaction: {e}")
    print(f"Amount attempted: ${e.amount}")
```

## 🔄 Context Managers

Context managers provide a clean way to handle resource management using the `with` statement.

### Built-in Context Managers

```python
# File handling with context manager
try:
    with open('example.txt', 'w') as file:
        file.write("Hello, World!")
        # File is automatically closed even if an exception occurs
except IOError as e:
    print(f"File operation failed: {e}")
```

### Custom Context Managers

```python
class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connected = False
    
    def __enter__(self):
        print(f"Connecting to database at {self.host}:{self.port}")
        # Simulate connection
        self.connected = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing database connection")
        self.connected = False
        
        # Handle exceptions
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
            # Return True to suppress the exception
            # Return False or None to propagate it
            return False
    
    def execute_query(self, query):
        if not self.connected:
            raise RuntimeError("Not connected to database")
        print(f"Executing query: {query}")
        # Simulate query execution
        if "error" in query.lower():
            raise ValueError("Query execution failed")

# Usage
try:
    with DatabaseConnection("localhost", 5432) as db:
        db.execute_query("SELECT * FROM users")
        db.execute_query("This will cause an error")
except ValueError as e:
    print(f"Query failed: {e}")
```

## 🧪 Practical Examples

### Example 1: Robust File Processor
```python
import json
import csv
from typing import List, Dict, Any

class FileProcessingError(Exception):
    """Custom exception for file processing errors"""
    pass

class DataProcessor:
    @staticmethod
    def read_json_file(filename: str) -> Dict[str, Any]:
        """Read and parse JSON file with error handling"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileProcessingError(f"File not found: {filename}")
        except json.JSONDecodeError as e:
            raise FileProcessingError(f"Invalid JSON in {filename}: {e}")
        except PermissionError:
            raise FileProcessingError(f"Permission denied: {filename}")
        except Exception as e:
            raise FileProcessingError(f"Unexpected error reading {filename}: {e}")
    
    @staticmethod
    def read_csv_file(filename: str) -> List[Dict[str, str]]:
        """Read and parse CSV file with error handling"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                return list(reader)
        except FileNotFoundError:
            raise FileProcessingError(f"File not found: {filename}")
        except csv.Error as e:
            raise FileProcessingError(f"CSV parsing error in {filename}: {e}")
        except PermissionError:
            raise FileProcessingError(f"Permission denied: {filename}")
        except Exception as e:
            raise FileProcessingError(f"Unexpected error reading {filename}: {e}")
    
    @staticmethod
    def write_json_file(filename: str, data: Dict[str, Any]) -> None:
        """Write data to JSON file with error handling"""
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
        except PermissionError:
            raise FileProcessingError(f"Permission denied: {filename}")
        except Exception as e:
            raise FileProcessingError(f"Error writing to {filename}: {e}")

# Usage example
def process_user_data():
    processor = DataProcessor()
    
    # Create sample data
    sample_data = {
        "users": [
            {"name": "Alice", "age": 25, "email": "alice@example.com"},
            {"name": "Bob", "age": 30, "email": "bob@example.com"}
        ]
    }
    
    # Write sample data
    try:
        processor.write_json_file("users.json", sample_data)
        print("Data written successfully")
    except FileProcessingError as e:
        print(f"Error writing data: {e}")
        return
    
    # Read and process data
    try:
        data = processor.read_json_file("users.json")
        print("Data read successfully:")
        for user in data["users"]:
            print(f"  {user['name']} ({user['age']}) - {user['email']}")
    except FileProcessingError as e:
        print(f"Error reading data: {e}")

# Create sample file for demonstration
sample_content = """name,age,email
Alice,25,alice@example.com
Bob,30,bob@example.com"""

try:
    with open("users.csv", "w") as f:
        f.write(sample_content)
except Exception as e:
    print(f"Error creating sample file: {e}")

# Process the data
process_user_data()
```

### Example 2: Network Request Handler
```python
import time
import random
from typing import Optional

class NetworkError(Exception):
    """Base exception for network-related errors"""
    pass

class ConnectionError(NetworkError):
    """Raised when connection fails"""
    pass

class TimeoutError(NetworkError):
    """Raised when request times out"""
    pass

class HTTPError(NetworkError):
    """Raised when HTTP request fails"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")

class NetworkClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session_active = False
    
    def __enter__(self):
        print("Starting network session")
        self.session_active = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Ending network session")
        self.session_active = False
        if exc_type is not None:
            print(f"Session ended with exception: {exc_type.__name__}")
        return False
    
    def make_request(self, endpoint: str, retries: int = 3) -> dict:
        """Make a network request with retry logic"""
        if not self.session_active:
            raise ConnectionError("No active session")
        
        for attempt in range(retries + 1):
            try:
                return self._perform_request(endpoint)
            except (ConnectionError, TimeoutError) as e:
                if attempt < retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
            except HTTPError:
                # Don't retry on HTTP errors
                raise
    
    def _perform_request(self, endpoint: str) -> dict:
        """Simulate network request"""
        print(f"Making request to {self.base_url}/{endpoint}")
        
        # Simulate network issues
        if random.random() < 0.1:  # 10% chance of connection error
            raise ConnectionError("Failed to establish connection")
        
        if random.random() < 0.05:  # 5% chance of timeout
            raise TimeoutError("Request timed out")
        
        # Simulate different HTTP responses
        if endpoint == "users":
            if random.random() < 0.1:  # 10% chance of 404
                raise HTTPError(404, "User not found")
            return {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        
        elif endpoint == "posts":
            if random.random() < 0.05:  # 5% chance of 403
                raise HTTPError(403, "Access forbidden")
            return {"posts": [{"id": 1, "title": "Hello World"}]}
        
        else:
            raise HTTPError(404, "Endpoint not found")

# Usage
def fetch_user_data():
    try:
        with NetworkClient("https://api.example.com") as client:
            # Fetch users
            users = client.make_request("users")
            print("Users:", users)
            
            # Fetch posts
            posts = client.make_request("posts")
            print("Posts:", posts)
            
            # Try invalid endpoint
            client.make_request("invalid")
            
    except ConnectionError as e:
        print(f"Connection failed: {e}")
    except TimeoutError as e:
        print(f"Request timed out: {e}")
    except HTTPError as e:
        print(f"HTTP error {e.status_code}: {e}")
    except NetworkError as e:
        print(f"Network error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Run the example
fetch_user_data()
```

### Example 3: Configuration Manager with Validation
```python
import json
import os
from typing import Dict, Any, Optional

class ConfigError(Exception):
    """Base exception for configuration errors"""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails"""
    pass

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as file:
                    self.config = json.load(file)
                self._validate_config()
            else:
                print(f"Config file {self.config_file} not found. Using defaults.")
                self.config = self._get_default_config()
                self.save_config()
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in config file: {e}")
        except PermissionError:
            raise ConfigError(f"Permission denied reading {self.config_file}")
        except ConfigValidationError:
            raise
        except Exception as e:
            raise ConfigError(f"Error loading config: {e}")
    
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as file:
                json.dump(self.config, file, indent=2)
        except PermissionError:
            raise ConfigError(f"Permission denied writing {self.config_file}")
        except Exception as e:
            raise ConfigError(f"Error saving config: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "app_name": "My Application",
            "version": "1.0.0",
            "debug": False,
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "mydb",
                "username": "user",
                "password": "password"
            },
            "api": {
                "timeout": 30,
                "retries": 3
            }
        }
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        required_sections = ["app_name", "version", "database", "api"]
        for section in required_sections:
            if section not in self.config:
                raise ConfigValidationError(f"Missing required section: {section}")
        
        # Validate database config
        db_config = self.config.get("database", {})
        required_db_fields = ["host", "port", "name", "username", "password"]
        for field in required_db_fields:
            if field not in db_config:
                raise ConfigValidationError(f"Missing database field: {field}")
        
        # Validate port number
        port = db_config.get("port")
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ConfigValidationError(f"Invalid database port: {port}")
        
        # Validate API config
        api_config = self.config.get("api", {})
        timeout = api_config.get("timeout", 30)
        retries = api_config.get("retries", 3)
        
        if not isinstance(timeout, int) or timeout <= 0:
            raise ConfigValidationError(f"Invalid API timeout: {timeout}")
        
        if not isinstance(retries, int) or retries < 0:
            raise ConfigValidationError(f"Invalid API retries: {retries}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        try:
            self._validate_config()
        except ConfigValidationError:
            # Revert the change if validation fails
            self.load_config()  # Reload from file
            raise

# Usage
def demonstrate_config_manager():
    # Create a sample config file
    sample_config = {
        "app_name": "Test App",
        "version": "2.0.0",
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "testdb",
            "username": "testuser",
            "password": "testpass"
        },
        "api": {
            "timeout": 60,
            "retries": 5
        }
    }
    
    try:
        with open("test_config.json", "w") as f:
            json.dump(sample_config, f, indent=2)
    except Exception as e:
        print(f"Error creating sample config: {e}")
        return
    
    try:
        # Load and use configuration
        config_manager = ConfigManager("test_config.json")
        
        print("Configuration loaded successfully:")
        print(f"App Name: {config_manager.get('app_name')}")
        print(f"Database Host: {config_manager.get('database.host')}")
        print(f"API Timeout: {config_manager.get('api.timeout')}")
        
        # Try to set a valid value
        config_manager.set('app_name', 'Updated App')
        print(f"Updated App Name: {config_manager.get('app_name')}")
        
        # Try to set an invalid value (this should raise an exception)
        try:
            config_manager.set('database.port', 99999)  # Invalid port
        except ConfigValidationError as e:
            print(f"Validation error caught: {e}")
            print(f"Port remains: {config_manager.get('database.port')}")
            
    except ConfigError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")
            print("Cleaned up test config file")

# Run the demonstration
demonstrate_config_manager()
```

## ⚠️ Best Practices and Common Mistakes

### 1. Be Specific with Exception Handling

```python
# Wrong - catching too broad exceptions
# try:
#     result = 10 / int(input("Enter number: "))
# except Exception:  # Too broad
#     print("Something went wrong")

# Correct - catch specific exceptions
try:
    result = 10 / int(input("Enter number: "))
except ValueError:
    print("Please enter a valid number")
except ZeroDivisionError:
    print("Cannot divide by zero")
```

### 2. Don't Ignore Exceptions

```python
# Wrong - ignoring exceptions
# try:
#     file = open("data.txt")
# except FileNotFoundError:
#     pass  # Ignoring the error

# Correct - handle or log exceptions
try:
    file = open("data.txt")
except FileNotFoundError:
    print("Data file not found, using default values")
    # Handle the error appropriately
```

### 3. Use Finally for Cleanup

```python
# Good practice - using finally for cleanup
file = None
try:
    file = open("data.txt")
    # Process file
except FileNotFoundError:
    print("File not found")
finally:
    if file:
        file.close()  # Ensure file is closed

# Even better - use context managers
try:
    with open("data.txt") as file:
        # Process file
        pass
except FileNotFoundError:
    print("File not found")
# File is automatically closed
```

### 4. Create Meaningful Custom Exceptions

```python
# Good - meaningful custom exceptions
class ValidationError(Exception):
    """Raised when data validation fails"""
    pass

class ProcessingError(Exception):
    """Raised when data processing fails"""
    pass

# Use them appropriately
def validate_user_data(data):
    if not data.get('email'):
        raise ValidationError("Email is required")
    if '@' not in data['email']:
        raise ValidationError("Invalid email format")

def process_user_data(data):
    try:
        validate_user_data(data)
        # Process data
    except ValidationError as e:
        print(f"Validation failed: {e}")
        raise ProcessingError("Unable to process user data") from e
```

## 📚 Next Steps

Now that you understand exception handling, you're ready to learn:

1. **Advanced Python Features**: Decorators, generators, and more
2. **Testing**: Unit testing and test-driven development
3. **Performance Optimization**: Profiling and optimization techniques
4. **Design Patterns**: Common solutions to design problems

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: What's the difference between `except Exception` and `except BaseException`?
A: `Exception` catches most exceptions but not system-exiting ones like `KeyboardInterrupt`. `BaseException` catches everything, including system exits.

### Q: When should I use `else` clause in try-except?
A: Use the `else` clause for code that should run only when no exception occurs in the try block.

### Q: What's the purpose of `finally` clause?
A: The `finally` clause runs regardless of whether an exception occurred, making it perfect for cleanup operations.

### Q: How do I chain exceptions?
A: Use `raise NewException() from OriginalException` to chain exceptions and preserve the original traceback.

---

**Practice implementing exception handling in different scenarios to build robust applications!** 🐍