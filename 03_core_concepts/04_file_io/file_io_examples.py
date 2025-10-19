# File I/O Examples

# Basic File Reading
print("=== Basic File Reading ===")

# Create a sample file for demonstration
with open('sample.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("This is the second line.\n")
    file.write("This is the third line.\n")

# Reading entire file
print("Reading entire file:")
try:
    with open('sample.txt', 'r') as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print("File not found!")

# Reading with context manager (recommended)
print("\nReading with context manager:")
with open('sample.txt', 'r') as file:
    content = file.read()
    print(content)

# Reading line by line
print("\nReading line by line:")
with open('sample.txt', 'r') as file:
    for line in file:
        print(line.strip())

# File Writing
print("\n=== File Writing ===")

# Writing to a file
with open('output.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("This is a new line.\n")

# Reading the written file
print("Reading written file:")
with open('output.txt', 'r') as file:
    print(file.read())

# Appending to files
print("\nAppending to file:")
with open('output.txt', 'a') as file:
    file.write("This line is appended.\n")

# Reading the file after appending
print("Reading file after appending:")
with open('output.txt', 'r') as file:
    print(file.read())

# File Modes
print("\n=== File Modes ===")

# Writing with different modes
with open('test_modes.txt', 'w') as file:
    file.write("Original content\n")

# Appending
with open('test_modes.txt', 'a') as file:
    file.write("Appended content\n")

# Reading
with open('test_modes.txt', 'r') as file:
    print("File content:")
    print(file.read())

# Reading with Error Handling
print("\n=== Reading with Error Handling ===")

def read_file_safely(filename):
    """Safely read a file with error handling"""
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied to read '{filename}'.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Usage
content = read_file_safely('sample.txt')
if content:
    print("Content of sample.txt:")
    print(content)

# File Position and Seeking
print("\n=== File Position and Seeking ===")

with open('sample.txt', 'r') as file:
    # Read first 10 characters
    first_part = file.read(10)
    print(f"First 10 characters: '{first_part}'")
    
    # Get current position
    position = file.tell()
    print(f"Current position: {position}")
    
    # Move to beginning
    file.seek(0)
    beginning = file.read(5)
    print(f"First 5 characters: '{beginning}'")

# Working with Different File Types
print("\n=== Working with Different File Types ===")

# CSV-like data
data = [
    ["Name", "Age", "City"],
    ["Alice", "25", "New York"],
    ["Bob", "30", "Boston"],
    ["Charlie", "35", "Chicago"]
]

# Writing CSV data
with open('people.csv', 'w') as file:
    for row in data:
        file.write(','.join(row) + '\n')

# Reading CSV data
print("Reading CSV data:")
with open('people.csv', 'r') as file:
    for line in file:
        fields = line.strip().split(',')
        print(fields)

# JSON-like data (manual approach)
import json

# Writing JSON-like data
person_data = {
    "name": "Alice",
    "age": 25,
    "city": "New York",
    "hobbies": ["reading", "swimming", "coding"]
}

with open('data.json', 'w') as file:
    json.dump(person_data, file, indent=2)

# Reading JSON-like data
print("\nReading JSON data:")
with open('data.json', 'r') as file:
    loaded_data = json.load(file)
    print(loaded_data)

# Binary files
print("\n=== Binary Files ===")

# Writing binary data
data = b"Binary data \x00\x01\x02\x03"
with open('binary_file.bin', 'wb') as file:
    file.write(data)

# Reading binary data
with open('binary_file.bin', 'rb') as file:
    binary_data = file.read()
    print(f"Binary data: {binary_data}")

# Practical Example: Log File Processor
print("\n=== Practical Example: Log File Processor ===")

import datetime

class LogFileProcessor:
    def __init__(self, filename):
        self.filename = filename
    
    def write_log(self, message, level="INFO"):
        """Write a log entry with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        try:
            with open(self.filename, 'a') as file:
                file.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def read_logs(self, level_filter=None):
        """Read log entries, optionally filtered by level"""
        try:
            with open(self.filename, 'r') as file:
                logs = []
                for line in file:
                    if level_filter is None or level_filter in line:
                        logs.append(line.strip())
                return logs
        except FileNotFoundError:
            print(f"Log file {self.filename} not found.")
            return []
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []

# Usage
logger = LogFileProcessor('app.log')
logger.write_log("Application started")
logger.write_log("User logged in", "INFO")
logger.write_log("Low disk space", "WARNING")
logger.write_log("Database connection failed", "ERROR")

print("All logs:")
for log in logger.read_logs():
    print(log)

print("\nError logs only:")
for log in logger.read_logs("ERROR"):
    print(log)

# Practical Example: Text File Analyzer
print("\n=== Practical Example: Text File Analyzer ===")

# Create a sample text file for analysis
sample_text = """
Python is a high-level, interpreted programming language. Python is known for its simplicity and readability.
Many developers love Python because Python is versatile and powerful. Python can be used for web development,
data science, artificial intelligence, automation, and more. The Python community is large and supportive.
Python has a rich ecosystem of libraries and frameworks that make development faster and easier.
Learning Python is a great investment for any aspiring programmer or data scientist.
"""

with open('sample_text.txt', 'w') as file:
    file.write(sample_text)

# Simple analysis
with open('sample_text.txt', 'r') as file:
    content = file.read()
    word_count = len(content.split())
    char_count = len(content)
    line_count = content.count('\n') + 1
    
    print(f"Text Analysis:")
    print(f"  Words: {word_count}")
    print(f"  Characters: {char_count}")
    print(f"  Lines: {line_count}")

# Common Mistakes Demonstration
print("\n=== Common Mistakes Demonstration ===")

# Correct way using context manager
print("Correct way - using context manager:")
try:
    with open('sample.txt', 'r') as file:
        content = file.read()
        print("File read successfully!")
except FileNotFoundError:
    print("File not found!")

# Handling encoding issues
print("\nHandling encoding:")
with open('sample.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    print("File read with UTF-8 encoding!")

# Cleanup - remove created files
import os

files_to_remove = ['sample.txt', 'output.txt', 'test_modes.txt', 'people.csv', 'data.json', 'binary_file.bin', 'app.log', 'sample_text.txt']
for filename in files_to_remove:
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed {filename}")