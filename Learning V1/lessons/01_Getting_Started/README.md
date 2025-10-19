# Lesson 01: Getting Started with Python 🚀

Welcome to your first Python lesson! This lesson will introduce you to Python programming and help you write your first programs.

## 📚 What You'll Learn

- What is Python and why learn it?
- Python's history and philosophy
- Writing your first Python program
- Understanding the Python interpreter
- Comments and documentation
- Python syntax basics

---

## 🐍 What is Python?

Python is a **high-level**, **interpreted**, **general-purpose** programming language created by **Guido van Rossum** in **1991**.

### Why Python?

✅ **Easy to Learn**: Simple, readable syntax similar to English  
✅ **Versatile**: Web dev, data science, AI, automation, games, and more  
✅ **Powerful**: Extensive standard library and third-party packages  
✅ **Popular**: Used by Google, Netflix, NASA, Instagram, Spotify  
✅ **Community**: Large, active community with tons of resources  
✅ **Cross-platform**: Works on Windows, macOS, Linux  

### Python Philosophy: The Zen of Python

Open Python and type `import this` to see:

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Readability counts.
```

---

## 📝 Your First Python Program

### Example 1: Hello World

**File**: `01_hello_world.py`

```python
# This is your first Python program!
print("Hello, World!")
```

**Output**:
```
Hello, World!
```

### How to Run:

```bash
# Windows
python 01_hello_world.py

# macOS/Linux
python3 01_hello_world.py
```

---

## 💬 Comments in Python

Comments are notes for programmers that Python ignores when running code.

### Single-Line Comments

```python
# This is a single-line comment
print("This will run")  # This is an inline comment
```

### Multi-Line Comments

```python
"""
This is a multi-line comment.
It can span multiple lines.
Used for detailed explanations.
"""

'''
You can also use single quotes
for multi-line comments.
'''
```

---

## 🧮 Python as a Calculator

Python can perform mathematical operations directly!

**File**: `02_calculator_basics.py`

```python
# Python as a calculator
print(5 + 3)      # Addition: 8
print(10 - 4)     # Subtraction: 6
print(7 * 6)      # Multiplication: 42
print(20 / 4)     # Division: 5.0
print(17 // 5)    # Floor division: 3
print(17 % 5)     # Modulus (remainder): 2
print(2 ** 3)     # Exponentiation: 8
```

---

## 🎯 Understanding `print()`

The `print()` function displays output to the screen.

**File**: `03_print_function.py`

```python
# Basic printing
print("Hello, Python!")

# Printing multiple items
print("Python", "is", "awesome!")  # Output: Python is awesome!

# Using different separators
print("Python", "is", "awesome!", sep="-")  # Output: Python-is-awesome!

# Changing end character
print("Hello", end=" ")
print("World")  # Output: Hello World

# Printing numbers
print(42)
print(3.14)

# Printing expressions
print(5 + 3)  # Output: 8
```

---

## 🔍 Understanding the Python Interpreter

### Interactive Mode

You can run Python in **interactive mode** for quick testing:

```bash
# Start Python interactive mode
python

# Or on macOS/Linux
python3
```

Then type:
```python
>>> print("Hello!")
Hello!
>>> 5 + 3
8
>>> exit()  # To quit
```

### Script Mode

Write code in a `.py` file and run it:

```bash
python my_script.py
```

---

## ✍️ Practice Exercises

### Exercise 1: Personal Greeting
Create a program that prints a personalized greeting.

**Expected Output**:
```
Hello, my name is [Your Name]
I am learning Python!
Python is awesome! 🐍
```

### Exercise 2: ASCII Art
Create a simple ASCII art using print statements.

**Example**:
```
    *
   ***
  *****
 *******
*********
    |
```

### Exercise 3: Calculator Challenge
Write a program that:
- Adds 15 and 27
- Subtracts 50 from 100
- Multiplies 8 by 7
- Divides 100 by 4
- Prints all results

### Exercise 4: Comments Practice
Take any of your previous programs and add:
- A multi-line comment at the top explaining what it does
- Single-line comments explaining each line

---

## 🎓 Key Takeaways

✅ Python is easy to learn and powerful  
✅ Use `print()` to display output  
✅ Comments help document your code  
✅ Python can be used as a calculator  
✅ Two modes: Interactive (testing) and Script (programs)  

---

## 🚀 Next Steps

Great job completing Lesson 01! 🎉

**Next**: [Lesson 02: Variables & Data Types](../02_Variables_DataTypes/) - Learn how to store and work with data.

---

## 📚 Additional Resources

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Python for Beginners](https://www.python.org/about/gettingstarted/)
- [Real Python - Python Basics](https://realpython.com/learning-paths/python-basics/)

---

**Happy Coding! 🐍**
