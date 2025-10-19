# 📁 File Input/Output in Python

File I/O (Input/Output) operations allow you to read from and write to files on your computer. This is essential for storing data permanently, processing large datasets, and working with external files. This guide will teach you how to work with files effectively in Python.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Open, read, and write files
- Use different file modes
- Handle file exceptions properly
- Work with different file types
- Use context managers for safe file handling
- Process large files efficiently

## 📂 File Operations Basics

### Opening Files

Python uses the `open()` function to work with files. The basic syntax is:
```python
file_object = open(filename, mode)
```

### File Modes

| Mode | Description |
|------|-------------|
| `'r'` | Read only (default) |
| `'w'` | Write only (overwrites existing file) |
| `'a'` | Append only |
| `'r+'` | Read and write |
| `'w+'` | Write and read (overwrites existing file) |
| `'a+'` | Append and read |
| `'b'` | Binary mode (e.g., `'rb'`, `'wb'`) |
| `'t'` | Text mode (default) |

### Basic File Reading

```python
# Reading an entire file
try:
    file = open('example.txt', 'r')
    content = file.read()
    print(content)
    file.close()
except FileNotFoundError:
    print("File not found!")
```

### Basic File Writing

```python
# Writing to a file
file = open('output.txt', 'w')
file.write("Hello, World!\n")
file.write("This is a new line.")
file.close()
```

## 🛠 Context Managers (with statement)

The `with` statement ensures files are properly closed even if an error occurs.

### Reading Files with Context Manager

```python
# Reading with context manager (recommended)
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
# File is automatically closed here
```

### Writing Files with Context Manager

```python
# Writing with context manager (recommended)
with open('output.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("This is a new line.")
# File is automatically closed here
```

## 📖 Reading Files

### Reading Entire File

```python
# Method 1: read() - reads entire file as string
with open('data.txt', 'r') as file:
    content = file.read()
    print(content)

# Method 2: readlines() - reads all lines into a list
with open('data.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        print(line.strip())  # strip() removes newline characters
```

### Reading Line by Line

```python
# Method 1: readline() - reads one line at a time
with open('data.txt', 'r') as file:
    line = file.readline()
    while line:
        print(line.strip())
        line = file.readline()

# Method 2: Iterating directly over file object (most efficient)
with open('data.txt', 'r') as file:
    for line in file:
        print(line.strip())
```

### Reading with Error Handling

```python
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
content = read_file_safely('example.txt')
if content:
    print(content)
```

## ✍️ Writing Files

### Writing Text to Files

```python
# Writing a single string
with open('output.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("This is the second line.\n")

# Writing multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open('output.txt', 'w') as file:
    file.writelines(lines)
```

### Appending to Files

```python
# Appending to existing file
with open('log.txt', 'a') as file:
    file.write("New log entry\n")
    file.write("Another log entry\n")
```

### Writing with Different Encodings

```python
# Writing with specific encoding
with open('unicode.txt', 'w', encoding='utf-8') as file:
    file.write("Hello, 世界!\n")
    file.write("Python 🐍 Programming\n")
```

## 🔄 File Position and Seeking

```python
# Working with file positions
with open('data.txt', 'r') as file:
    # Read first 10 characters
    first_part = file.read(10)
    print(f"First 10 characters: {first_part}")
    
    # Get current position
    position = file.tell()
    print(f"Current position: {position}")
    
    # Move to beginning
    file.seek(0)
    beginning = file.read(5)
    print(f"First 5 characters: {beginning}")
```

## 📊 Working with Different File Types

### CSV Files

```python
# Writing CSV data
data = [
    ["Name", "Age", "City"],
    ["Alice", "25", "New York"],
    ["Bob", "30", "Boston"],
    ["Charlie", "35", "Chicago"]
]

with open('people.csv', 'w', newline='') as file:
    for row in data:
        file.write(','.join(row) + '\n')

# Reading CSV data
with open('people.csv', 'r') as file:
    for line in file:
        fields = line.strip().split(',')
        print(fields)
```

### JSON Files

```python
import json

# Writing JSON data
data = {
    "name": "Alice",
    "age": 25,
    "city": "New York",
    "hobbies": ["reading", "swimming", "coding"]
}

with open('data.json', 'w') as file:
    json.dump(data, file, indent=2)

# Reading JSON data
with open('data.json', 'r') as file:
    loaded_data = json.load(file)
    print(loaded_data)
```

### Binary Files

```python
# Writing binary data
data = b"Binary data \x00\x01\x02\x03"
with open('binary_file.bin', 'wb') as file:
    file.write(data)

# Reading binary data
with open('binary_file.bin', 'rb') as file:
    binary_data = file.read()
    print(binary_data)
```

## 🧪 Practical Examples

### Example 1: Log File Processor
```python
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
    
    def count_log_levels(self):
        """Count occurrences of each log level"""
        levels = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        try:
            with open(self.filename, 'r') as file:
                for line in file:
                    for level in levels:
                        if level in line:
                            levels[level] += 1
            return levels
        except FileNotFoundError:
            return levels
        except Exception as e:
            print(f"Error counting log levels: {e}")
            return levels

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

print(f"\nLog level counts: {logger.count_log_levels()}")
```

### Example 2: Configuration File Manager
```python
import json
import os

class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as file:
                    return json.load(file)
            except json.JSONDecodeError:
                print("Invalid JSON in config file. Using default config.")
                return self.default_config()
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.default_config()
        else:
            print("Config file not found. Creating default config.")
            self.save_config(self.default_config())
            return self.default_config()
    
    def default_config(self):
        """Return default configuration"""
        return {
            "app_name": "My Application",
            "version": "1.0.0",
            "debug": False,
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "mydb"
            },
            "features": {
                "enable_logging": True,
                "max_connections": 100
            }
        }
    
    def save_config(self, config=None):
        """Save configuration to file"""
        config_to_save = config if config is not None else self.config
        try:
            with open(self.config_file, 'w') as file:
                json.dump(config_to_save, file, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

# Usage
config = ConfigManager()
print(f"App name: {config.get('app_name')}")
print(f"Database host: {config.get('database.host')}")
print(f"Non-existent key: {config.get('non.existent', 'default_value')}")

# Modify config
config.set('app_name', 'New Application Name')
config.set('database.port', 3306)
config.save_config()

print("Updated config saved.")
```

### Example 3: Text File Analyzer
```python
import string
from collections import Counter

class TextFileAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.content = self.read_file()
    
    def read_file(self):
        """Read file content"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"File {self.filename} not found.")
            return ""
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""
    
    def word_count(self):
        """Count total words"""
        if not self.content:
            return 0
        return len(self.content.split())
    
    def character_count(self, include_spaces=True):
        """Count characters"""
        if not self.content:
            return 0
        if include_spaces:
            return len(self.content)
        else:
            return len(self.content.replace(" ", ""))
    
    def line_count(self):
        """Count lines"""
        if not self.content:
            return 0
        return self.content.count('\n') + 1
    
    def most_common_words(self, n=10):
        """Get n most common words"""
        if not self.content:
            return []
        
        # Clean and split text
        text = self.content.translate(str.maketrans('', '', string.punctuation)).lower()
        words = [word for word in text.split() if len(word) > 3]  # Words longer than 3 characters
        word_counter = Counter(words)
        return word_counter.most_common(n)
    
    def average_word_length(self):
        """Calculate average word length"""
        if not self.content:
            return 0
        
        words = self.content.split()
        if not words:
            return 0
        
        total_length = sum(len(word.strip(string.punctuation)) for word in words)
        return total_length / len(words)
    
    def analyze(self):
        """Perform complete analysis"""
        print(f"=== Analysis of {self.filename} ===")
        print(f"Words: {self.word_count()}")
        print(f"Characters (with spaces): {self.character_count()}")
        print(f"Characters (without spaces): {self.character_count(False)}")
        print(f"Lines: {self.line_count()}")
        print(f"Average word length: {self.average_word_length():.2f}")
        
        print(f"\nMost common words:")
        for word, count in self.most_common_words(10):
            print(f"  {word}: {count}")

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

# Analyze the file
analyzer = TextFileAnalyzer('sample_text.txt')
analyzer.analyze()
```

## ⚠️ Common Mistakes and Tips

### 1. Forgetting to Close Files

```python
# Wrong - file may not be closed properly
# file = open('data.txt', 'r')
# content = file.read()
# # Forgot to close file!

# Correct - using context manager
with open('data.txt', 'r') as file:
    content = file.read()
# File is automatically closed
```

### 2. Not Handling File Exceptions

```python
# Wrong - no error handling
# with open('nonexistent.txt', 'r') as file:
#     content = file.read()

# Correct - with error handling
try:
    with open('nonexistent.txt', 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("File not found!")
except PermissionError:
    print("Permission denied!")
```

### 3. Using Wrong File Mode

```python
# Wrong - trying to read from write-only file
# with open('data.txt', 'w') as file:
#     content = file.read()  # Error!

# Correct - using appropriate mode
with open('data.txt', 'r') as file:
    content = file.read()
```

### 4. Not Handling Encoding Issues

```python
# When dealing with non-ASCII characters
with open('unicode_file.txt', 'r', encoding='utf-8') as file:
    content = file.read()
```

## 📚 Next Steps

Now that you understand file I/O operations, you're ready to learn:

1. **Exception Handling**: Managing errors gracefully
2. **Object-Oriented Programming**: Advanced programming concepts
3. **Modules and Packages**: Organizing code in larger projects
4. **Working with Databases**: Storing and retrieving data

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: Why use the `with` statement for file operations?
A: The `with` statement ensures files are properly closed even if an error occurs, making your code more robust.

### Q: What's the difference between `read()`, `readline()`, and `readlines()`?
A: `read()` reads the entire file, `readline()` reads one line, and `readlines()` reads all lines into a list.

### Q: When should I use binary mode?
A: Use binary mode (`'rb'`, `'wb'`) when working with non-text files like images, executables, or when you need precise control over bytes.

### Q: How do I handle large files efficiently?
A: For large files, read them line by line or in chunks rather than loading the entire file into memory.

---

**Practice file operations with different file types and scenarios to build your skills!** 🐍