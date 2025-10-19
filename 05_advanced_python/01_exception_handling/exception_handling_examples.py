# Exception Handling Examples

# Basic Exception Handling
print("=== Basic Exception Handling ===")

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

# Exception Hierarchy
print("\n=== Exception Hierarchy ===")

try:
    number = int("not_a_number")
except ValueError as e:
    print(f"ValueError: {e}")
except ArithmeticError as e:
    print(f"ArithmeticError: {e}")
except Exception as e:
    print(f"General Exception: {e}")

# Try-Except Blocks
print("\n=== Try-Except Blocks ===")

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
print(divide_numbers(10, 2))
print(divide_numbers(10, 0))
print(divide_numbers(10, "2"))

# Catching Multiple Exceptions
print("\n=== Catching Multiple Exceptions ===")

def process_data(data):
    try:
        number = int(data)
        result = 100 / number
        return result
    except (ValueError, ZeroDivisionError) as e:
        print(f"Error processing data: {type(e).__name__} - {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
print(process_data("10"))
print(process_data("0"))
print(process_data("abc"))
print(process_data(None))

# Else and Finally Clauses
print("\n=== Else and Finally Clauses ===")

def read_file(filename):
    try:
        file = open(filename, 'r')
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    else:
        content = file.read()
        file.close()
        print(f"Successfully read {len(content)} characters from {filename}")
        return content
    finally:
        print("File operation completed.")

# Create a test file
with open('example.txt', 'w') as f:
    f.write("Hello, World!")

# Usage
content = read_file("nonexistent.txt")
content = read_file("example.txt")

# Raising Exceptions
print("\n=== Raising Exceptions ===")

def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    elif age > 150:
        raise ValueError("Age seems unrealistic")
    else:
        return f"Valid age: {age}"

# Usage
try:
    print(validate_age(25))
    print(validate_age(-5))
except ValueError as e:
    print(f"Invalid age: {e}")

# Custom Exceptions
print("\n=== Custom Exceptions ===")

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
process_user_data("invalid-email")
process_user_data("a@b")
process_user_data("valid@example.com")

# Context Managers
print("\n=== Context Managers ===")

class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connected = False
    
    def __enter__(self):
        print(f"Connecting to database at {self.host}:{self.port}")
        self.connected = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing database connection")
        self.connected = False
        
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
        return False
    
    def execute_query(self, query):
        if not self.connected:
            raise RuntimeError("Not connected to database")
        print(f"Executing query: {query}")

# Usage
try:
    with DatabaseConnection("localhost", 5432) as db:
        db.execute_query("SELECT * FROM users")
except RuntimeError as e:
    print(f"Database error: {e}")

# Practical Example: Robust File Processor
print("\n=== Practical Example: Robust File Processor ===")

import json
import csv

class FileProcessingError(Exception):
    """Custom exception for file processing errors"""
    pass

class DataProcessor:
    @staticmethod
    def read_json_file(filename: str) -> dict:
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
    def write_json_file(filename: str, data: dict) -> None:
        """Write data to JSON file with error handling"""
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
        except PermissionError:
            raise FileProcessingError(f"Permission denied: {filename}")
        except Exception as e:
            raise FileProcessingError(f"Error writing to {filename}: {e}")

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

# Run the example
process_user_data()

# Common Mistakes Demonstration
print("\n=== Common Mistakes Demonstration ===")

# Correct exception handling
try:
    result = 10 / int(input("Enter number: "))
except ValueError:
    print("Please enter a valid number")
except ZeroDivisionError:
    print("Cannot divide by zero")

# Cleanup - remove created files
import os
files_to_remove = ['example.txt', 'users.json']
for filename in files_to_remove:
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed {filename}")