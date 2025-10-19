# Lesson 02: Variables & Data Types 📊

Welcome to Lesson 02! Now that you can write basic Python programs, let's learn how to store and work with data using variables.

## 📚 What You'll Learn

- What are variables and how to create them
- Naming rules and conventions
- Basic data types (int, float, string, boolean)
- Type checking and conversion
- User input
- String operations and formatting

---

## 💾 What Are Variables?

A **variable** is like a labeled box that stores data. You can put data in, take it out, and change it.

```python
# Creating a variable
name = "Python"
age = 32
pi = 3.14159
is_awesome = True
```

### Variable Assignment

```python
x = 5              # x stores the integer 5
name = "Alice"     # name stores the string "Alice"
price = 19.99      # price stores the float 19.99
```

---

## 📏 Variable Naming Rules

### ✅ ALLOWED:
```python
name = "John"           # Lowercase
first_name = "John"     # Snake case (preferred in Python)
firstName = "John"      # Camel case
_private = "secret"     # Starting with underscore
age2 = 25              # Numbers allowed (not at start)
```

### ❌ NOT ALLOWED:
```python
2age = 25              # Can't start with number
first-name = "John"    # No hyphens
first name = "John"    # No spaces
class = "Python"       # Can't use keywords
```

### Python Keywords (Reserved Words)
You cannot use these as variable names:
```
False, None, True, and, as, assert, async, await, break, class, 
continue, def, del, elif, else, except, finally, for, from, global, 
if, import, in, is, lambda, nonlocal, not, or, pass, raise, return, 
try, while, with, yield
```

### Naming Conventions (Best Practices)

```python
# ✅ Good naming
user_name = "Alice"
total_price = 99.99
is_active = True
MAX_SIZE = 100          # Constants in UPPERCASE

# ❌ Poor naming
x = "Alice"             # Not descriptive
usrnm = "Alice"         # Hard to read
UserName = "Alice"      # Should be lowercase
```

---

## 🎨 Data Types

Python has several built-in data types:

### 1. Integer (`int`)
Whole numbers, positive or negative.

```python
age = 25
temperature = -5
population = 1000000
```

### 2. Float (`float`)
Decimal numbers.

```python
pi = 3.14159
price = 19.99
temperature = 98.6
```

### 3. String (`str`)
Text data, enclosed in quotes.

```python
name = "Alice"
message = 'Hello, World!'
paragraph = """This is a
multi-line string"""
```

### 4. Boolean (`bool`)
True or False values.

```python
is_student = True
is_raining = False
has_license = True
```

### 5. NoneType
Represents absence of value.

```python
result = None
```

---

## 🔍 Checking Data Types

Use `type()` to check the data type:

```python
x = 5
print(type(x))  # <class 'int'>

y = 3.14
print(type(y))  # <class 'float'>

name = "Python"
print(type(name))  # <class 'str'>

flag = True
print(type(flag))  # <class 'bool'>
```

---

## 🔄 Type Conversion (Casting)

Convert between different types:

```python
# To integer
x = int(3.8)        # x = 3
y = int("10")       # y = 10

# To float
a = float(5)        # a = 5.0
b = float("3.14")   # b = 3.14

# To string
s = str(100)        # s = "100"
t = str(3.14)       # t = "3.14"

# To boolean
bool(1)             # True
bool(0)             # False
bool("")            # False
bool("Hello")       # True
```

---

## ⌨️ Getting User Input

Use `input()` to get data from users:

```python
name = input("Enter your name: ")
print("Hello, " + name)

age = input("Enter your age: ")
# Note: input() always returns a string!

# Convert to integer
age = int(input("Enter your age: "))
```

---

## 🔤 String Operations

### Concatenation
```python
first = "Hello"
last = "World"
message = first + " " + last  # "Hello World"
```

### String Formatting

#### 1. Old style (%)
```python
name = "Alice"
age = 25
print("My name is %s and I am %d years old" % (name, age))
```

#### 2. `.format()` method
```python
print("My name is {} and I am {} years old".format(name, age))
```

#### 3. F-strings (Modern, Preferred)
```python
name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old")
print(f"Next year I'll be {age + 1}")
```

### String Methods
```python
text = "Hello, World!"
print(text.upper())      # HELLO, WORLD!
print(text.lower())      # hello, world!
print(text.replace("Hello", "Hi"))  # Hi, World!
print(len(text))         # 13 (length)
```

---

## ✍️ Practice Exercises

### Exercise 1: Variable Creation
Create variables to store:
- Your name
- Your age
- Your height (in meters)
- Whether you like Python (True/False)
- Print all of them

### Exercise 2: Type Conversion
```python
# Given:
num_str = "42"
# Convert to integer, add 8, print result
```

### Exercise 3: User Profile
Create a program that:
1. Asks for user's name
2. Asks for user's age
3. Asks for user's city
4. Prints a formatted message using all inputs

### Exercise 4: Calculator with Input
Create a calculator that:
1. Asks user for two numbers
2. Calculates sum, difference, product, quotient
3. Prints all results

### Exercise 5: String Manipulation
```python
# Given:
quote = "python is awesome"
# 1. Convert to uppercase
# 2. Convert to title case
# 3. Count how many characters
# 4. Replace "awesome" with "amazing"
```

---

## 🎓 Key Takeaways

✅ Variables store data with meaningful names  
✅ Follow naming conventions (snake_case)  
✅ Python has multiple data types: int, float, str, bool  
✅ Use `type()` to check types, casting to convert  
✅ `input()` gets user data (always returns string)  
✅ F-strings are the modern way to format strings  

---

## 🚀 Next Steps

Excellent work! 🎉

**Next**: [Lesson 03: Operators](../03_Operators/) - Learn to perform operations on data.

---

## 📚 Additional Resources

- [Python Variables - Official Docs](https://docs.python.org/3/tutorial/introduction.html)
- [Python Data Types](https://realpython.com/python-data-types/)
- [String Formatting in Python](https://realpython.com/python-string-formatting/)

---

**Happy Coding! 🐍**
