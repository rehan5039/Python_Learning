# 📦 Variables and Data Types in Python

Understanding variables and data types is fundamental to programming in Python. This guide will explain these core concepts with practical examples.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Understand what variables are and how to use them
- Identify different data types in Python
- Perform type conversion
- Take user input
- Use the `type()` function to check data types

## 📝 What are Variables?

Variables are named storage locations in memory that hold data. Think of them as containers that can hold different values.

### Creating Variables

In Python, you create a variable by assigning a value to a name using the `=` operator:

```python
name = "Alice"
age = 25
height = 5.6
is_student = True
```

### Variable Naming Rules

1. Must start with a letter (a-z, A-Z) or underscore (_)
2. Can contain letters, numbers, and underscores
3. Cannot be a Python keyword (like `if`, `else`, `for`, etc.)
4. Case-sensitive (`name` and `Name` are different variables)

```python
# Valid variable names
my_variable = 10
_myVariable = 20
variable123 = 30

# Invalid variable names
# 123variable = 10  # Cannot start with a number
# my-variable = 20  # Hyphens are not allowed
# class = 30        # Cannot use Python keywords
```

## 🧮 Python Data Types

Python has several built-in data types. Here are the most common ones:

### 1. Numeric Types

#### Integer (`int`)
Whole numbers without decimal points:
```python
age = 25
temperature = -10
count = 1000
```

#### Float (`float`)
Numbers with decimal points:
```python
price = 19.99
pi = 3.14159
temperature = -5.5
```

#### Complex (`complex`)
Numbers with a real and imaginary part:
```python
complex_number = 3 + 4j
```

### 2. Text Type

#### String (`str`)
Sequences of characters enclosed in quotes:
```python
name = "Alice"
message = 'Hello, World!'
paragraph = """This is a
multi-line string"""
```

### 3. Boolean Type

#### Boolean (`bool`)
Represents True or False values:
```python
is_raining = True
is_sunny = False
```

### 4. None Type

#### NoneType (`None`)
Represents the absence of a value:
```python
result = None
```

## 🔍 Checking Data Types

Use the `type()` function to check the data type of a variable:

```python
age = 25
name = "Alice"
price = 19.99
is_active = True

print(type(age))      # <class 'int'>
print(type(name))     # <class 'str'>
print(type(price))    # <class 'float'>
print(type(is_active)) # <class 'bool'>
```

## 🔁 Type Conversion

Python allows you to convert between data types:

### Implicit Conversion
Python automatically converts types when needed:
```python
a = 5      # int
b = 3.2    # float
c = a + b  # Python converts 'a' to float
print(c)   # 8.2
print(type(c))  # <class 'float'>
```

### Explicit Conversion
You can manually convert types using functions:
```python
# Converting to integer
x = int(3.7)     # 3
y = int("10")    # 10

# Converting to float
a = float(5)     # 5.0
b = float("3.14") # 3.14

# Converting to string
name = str(123)  # "123"
age = str(25)    # "25"

# Converting to boolean
is_true = bool(1)    # True
is_false = bool(0)   # False
```

## 🖥️ Taking User Input

Use the `input()` function to get input from users:

```python
# Basic input
name = input("Enter your name: ")
print("Hello, " + name)

# Converting input to appropriate type
age = int(input("Enter your age: "))
height = float(input("Enter your height in meters: "))

# Displaying the information
print(f"Name: {name}")
print(f"Age: {age}")
print(f"Height: {height} meters")
```

## 🧪 Practical Examples

### Example 1: Simple Calculator
```python
# Taking two numbers as input
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Performing operations
sum_result = num1 + num2
difference = num1 - num2
product = num1 * num2
quotient = num1 / num2

# Displaying results
print(f"Sum: {sum_result}")
print(f"Difference: {difference}")
print(f"Product: {product}")
print(f"Quotient: {quotient}")
```

### Example 2: Personal Information Form
```python
# Collecting user information
name = input("Enter your name: ")
age = int(input("Enter your age: "))
height = float(input("Enter your height (in meters): "))
is_student = input("Are you a student? (yes/no): ").lower() == "yes"

# Displaying information
print("\n--- Personal Information ---")
print(f"Name: {name}")
print(f"Age: {age} years old")
print(f"Height: {height} meters")
print(f"Student: {'Yes' if is_student else 'No'}")
```

## ⚠️ Common Mistakes and Tips

### 1. Forgetting to Convert Input
```python
# Wrong - this will cause an error
# age = input("Enter your age: ")
# next_year = age + 1  # Error: can't add string and int

# Correct - convert to integer
age = int(input("Enter your age: "))
next_year = age + 1
print(f"Next year you will be {next_year} years old")
```

### 2. Invalid Type Conversion
```python
# This will cause an error
# number = int("hello")  # ValueError

# Always validate input when possible
try:
    number = int(input("Enter a number: "))
    print(f"You entered: {number}")
except ValueError:
    print("That's not a valid number!")
```

### 3. Case Sensitivity
```python
# These are different variables
Name = "Alice"
name = "Bob"
NAME = "Charlie"

print(Name)  # Alice
print(name)  # Bob
print(NAME)  # Charlie
```

## 🧠 Memory and Variables

In Python, variables are references to objects in memory:

```python
# When you create a variable, Python creates an object in memory
x = 10

# When you assign one variable to another, both point to the same object
y = x

# Changing one variable doesn't affect the other if they're immutable
x = 20
print(y)  # Still 10
```

## 📚 Next Steps

Now that you understand variables and data types, you're ready to learn:

1. **Operators**: Mathematical and logical operations
2. **Strings**: Advanced string manipulation
3. **Lists and Tuples**: Working with collections of data
4. **Dictionaries and Sets**: More complex data structures

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: Do I need to declare variable types in Python?
A: No, Python is dynamically typed. The type is determined automatically based on the value assigned.

### Q: What's the difference between `int()` and `float()`?
A: `int()` converts to whole numbers, while `float()` converts to decimal numbers.

### Q: Why does `input()` always return a string?
A: For safety and flexibility. You can convert the string to any type you need.

### Q: What happens if I try to convert invalid data?
A: Python raises a `ValueError`. Always validate input when necessary.

---

**Practice makes perfect! Try creating variables with different data types and experiment with type conversion.** 🐍