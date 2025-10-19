# 🔤 Strings in Python

Strings are one of the most important data types in Python. This comprehensive guide will teach you everything you need to know about working with text in Python.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Create and manipulate strings
- Use string indexing and slicing
- Apply string methods
- Work with escape sequences
- Format strings effectively

## 📝 What are Strings?

Strings are sequences of characters used to store and manipulate text. In Python, strings are immutable, meaning once created, they cannot be changed.

### Creating Strings

You can create strings using single quotes, double quotes, or triple quotes:

```python
# Single quotes
string1 = 'Hello, World!'

# Double quotes
string2 = "Python Programming"

# Triple quotes (for multi-line strings)
string3 = """This is a
multi-line
string"""

# All of these are valid strings
print(string1)
print(string2)
print(string3)
```

### When to Use Which Quotes?

```python
# Use double quotes when your string contains single quotes
message = "It's a beautiful day!"

# Use single quotes when your string contains double quotes
quote = 'He said, "Hello!"'

# Use triple quotes for multi-line strings or strings with both single and double quotes
paragraph = """She said, "It's a beautiful day!" and smiled."""
```

## 🔢 String Indexing

Each character in a string has a position called an index. Python uses zero-based indexing.

```python
text = "Python"
# Index: 012345

print(text[0])  # P
print(text[1])  # y
print(text[5])  # n

# Negative indexing (from the end)
print(text[-1])  # n
print(text[-2])  # o
```

### Indexing Rules
- First character is at index 0
- Last character is at index `len(string) - 1`
- Negative indices count from the end (-1 is the last character)

## ✂️ String Slicing

Slicing allows you to extract parts of a string using the syntax `[start:end:step]`.

```python
text = "Python Programming"

# Basic slicing [start:end] - end is not included
print(text[0:6])    # Python
print(text[7:18])   # Programming

# Omitting start (starts from beginning)
print(text[:6])     # Python

# Omitting end (goes to the end)
print(text[7:])     # Programming

# Using negative indices
print(text[-11:-1]) # Programmin

# Using step [start:end:step]
print(text[::2])    # Pto rgamn (every 2nd character)
print(text[::-1])   # gnimmargorP nohtyP (reversed)
```

## 🛠 String Methods

Python provides many built-in methods for string manipulation:

### Case Conversion
```python
text = "Hello, World!"

print(text.upper())     # HELLO, WORLD!
print(text.lower())     # hello, world!
print(text.capitalize()) # Hello, world!
print(text.title())     # Hello, World!
```

### Searching and Replacing
```python
text = "Python is awesome, Python is powerful"

# Find substring
print(text.find("Python"))     # 0 (first occurrence)
print(text.find("Java"))       # -1 (not found)
print(text.count("Python"))    # 2

# Replace substring
print(text.replace("Python", "Java"))  # Java is awesome, Java is powerful
```

### Checking String Properties
```python
text = "Python123"

print(text.isalpha())   # False (contains numbers)
print(text.isdigit())   # False (contains letters)
print(text.isalnum())   # True (alphanumeric)
print(text.isspace())   # False
```

### Splitting and Joining
```python
# Splitting
sentence = "Python is awesome"
words = sentence.split()  # ['Python', 'is', 'awesome']
print(words)

# Joining
separator = "-"
joined = separator.join(words)  # Python-is-awesome
print(joined)

# Custom separator
data = "apple,banana,orange"
fruits = data.split(",")  # ['apple', 'banana', 'orange']
print(fruits)
```

### Stripping Whitespace
```python
text = "   Hello, World!   "

print(text.strip())   # "Hello, World!" (removes both ends)
print(text.lstrip())  # "Hello, World!   " (removes left)
print(text.rstrip())  # "   Hello, World!" (removes right)
```

## 🎭 Escape Sequences

Escape sequences allow you to include special characters in strings:

```python
# New line
print("Hello\nWorld")
# Output:
# Hello
# World

# Tab
print("Name:\tAlice")
# Output: Name:    Alice

# Backslash
print("Path: C:\\Users\\Alice")
# Output: Path: C:\Users\Alice

# Single quote
print('It\'s a beautiful day!')
# Output: It's a beautiful day!

# Double quote
print("She said, \"Hello!\"")
# Output: She said, "Hello!"

# Raw strings (prefix with r)
path = r"C:\Users\Alice\Documents"
print(path)  # C:\Users\Alice\Documents
```

## 🎨 String Formatting

There are several ways to format strings in Python:

### 1. f-strings (Recommended - Python 3.6+)
```python
name = "Alice"
age = 25
height = 5.6

message = f"Hello, {name}! You are {age} years old and {height} feet tall."
print(message)

# Expressions in f-strings
print(f"Next year you will be {age + 1} years old.")
print(f"{10 * 5 = }")  # 10 * 5 = 50
```

### 2. format() method
```python
message = "Hello, {}! You are {} years old.".format(name, age)
print(message)

# Named placeholders
message = "Hello, {name}! You are {age} years old.".format(name=name, age=age)
print(message)

# Positional placeholders
message = "Hello, {0}! You are {1} years old.".format(name, age)
print(message)
```

### 3. % formatting (Old style)
```python
message = "Hello, %s! You are %d years old." % (name, age)
print(message)
```

## 🔍 String Operations

### Concatenation
```python
first_name = "Alice"
last_name = "Smith"

# Using + operator
full_name = first_name + " " + last_name
print(full_name)  # Alice Smith

# Using += operator
greeting = "Hello, "
greeting += first_name
print(greeting)  # Hello, Alice
```

### Repetition
```python
text = "Python "
repeated = text * 3
print(repeated)  # Python Python Python 
```

### Membership Testing
```python
text = "Python Programming"

print("Python" in text)     # True
print("Java" in text)       # False
print("python" in text)     # False (case-sensitive)
print("python" not in text) # True
```

## 🧪 Practical Examples

### Example 1: Email Validator
```python
def validate_email(email):
    """Simple email validation"""
    if "@" in email and "." in email:
        at_index = email.find("@")
        dot_index = email.rfind(".")
        if at_index < dot_index and at_index > 0 and dot_index < len(email) - 1:
            return True
    return False

# Test the function
emails = ["alice@example.com", "invalid.email", "test@.com", "@example.com"]
for email in emails:
    print(f"{email}: {validate_email(email)}")
```

### Example 2: Text Analyzer
```python
def analyze_text(text):
    """Analyze text properties"""
    print(f"Text: {text}")
    print(f"Length: {len(text)} characters")
    print(f"Words: {len(text.split())}")
    print(f"Uppercase: {text.upper()}")
    print(f"Lowercase: {text.lower()}")
    print(f"Title Case: {text.title()}")

# Test the function
sample_text = "Python is an amazing programming language"
analyze_text(sample_text)
```

### Example 3: Password Strength Checker
```python
def check_password_strength(password):
    """Check password strength"""
    score = 0
    feedback = []
    
    if len(password) >= 8:
        score += 1
    else:
        feedback.append("Password should be at least 8 characters long")
    
    if any(c.isupper() for c in password):
        score += 1
    else:
        feedback.append("Password should contain uppercase letters")
    
    if any(c.islower() for c in password):
        score += 1
    else:
        feedback.append("Password should contain lowercase letters")
    
    if any(c.isdigit() for c in password):
        score += 1
    else:
        feedback.append("Password should contain numbers")
    
    if any(c in "!@#$%^&*()-_" for c in password):
        score += 1
    else:
        feedback.append("Password should contain special characters")
    
    strength = ["Very Weak", "Weak", "Fair", "Good", "Strong"][min(score, 4)]
    
    return {
        "strength": strength,
        "score": score,
        "feedback": feedback
    }

# Test the function
passwords = ["123", "Password", "Password123", "P@ssw0rd123"]
for pwd in passwords:
    result = check_password_strength(pwd)
    print(f"\nPassword: {pwd}")
    print(f"Strength: {result['strength']} ({result['score']}/5)")
    if result['feedback']:
        print("Feedback:", ", ".join(result['feedback']))
```

## ⚠️ Common Mistakes and Tips

### 1. String Index Out of Range
```python
# Wrong
text = "Python"
# print(text[10])  # IndexError

# Correct
if len(text) > 10:
    print(text[10])
else:
    print("Index out of range")
```

### 2. Immutable Nature of Strings
```python
# Wrong - strings are immutable
text = "Python"
# text[0] = "J"  # TypeError

# Correct - create a new string
text = "J" + text[1:]
print(text)  # Jython
```

### 3. Case Sensitivity
```python
text = "Python"
print(text == "python")  # False
print(text.lower() == "python")  # True
```

## 📚 Next Steps

Now that you've mastered strings, you're ready to learn:

1. **Lists and Tuples**: Working with collections of data
2. **Dictionaries and Sets**: More complex data structures
3. **Functions**: Creating reusable code blocks
4. **File I/O**: Reading and writing files

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: What's the difference between `find()` and `index()`?
A: `find()` returns -1 if not found, while `index()` raises a ValueError.

### Q: When should I use f-strings vs format()?
A: Use f-strings for simplicity and readability. Use format() when you need more complex formatting.

### Q: Are strings mutable in Python?
A: No, strings are immutable. You must create a new string to make changes.

### Q: How do I handle special characters in strings?
A: Use escape sequences or raw strings (prefix with `r`).

---

**Practice string manipulation with different examples to build your skills!** 🐍