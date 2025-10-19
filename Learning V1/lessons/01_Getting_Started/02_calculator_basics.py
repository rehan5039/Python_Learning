"""
Lesson 01: Python as a Calculator
===================================
Learn how to use Python for mathematical operations.

This program demonstrates:
- Basic arithmetic operations
- Mathematical operators in Python
- Order of operations
"""

print("=" * 50)
print("Python Calculator Demo")
print("=" * 50)

# Addition
print("\n📊 Addition:")
print("5 + 3 =", 5 + 3)
print("100 + 250 =", 100 + 250)

# Subtraction
print("\n📊 Subtraction:")
print("10 - 4 =", 10 - 4)
print("100 - 35 =", 100 - 35)

# Multiplication
print("\n📊 Multiplication:")
print("7 * 6 =", 7 * 6)
print("12 * 12 =", 12 * 12)

# Division (returns float)
print("\n📊 Division:")
print("20 / 4 =", 20 / 4)
print("15 / 2 =", 15 / 2)

# Floor Division (returns integer, rounds down)
print("\n📊 Floor Division:")
print("17 // 5 =", 17 // 5)  # 3, not 3.4
print("20 // 3 =", 20 // 3)  # 6, not 6.666...

# Modulus (remainder after division)
print("\n📊 Modulus (Remainder):")
print("17 % 5 =", 17 % 5)   # Remainder when 17 is divided by 5
print("20 % 3 =", 20 % 3)   # Remainder when 20 is divided by 3

# Exponentiation (power)
print("\n📊 Exponentiation:")
print("2 ** 3 =", 2 ** 3)   # 2 to the power of 3 = 8
print("5 ** 2 =", 5 ** 2)   # 5 squared = 25
print("10 ** 3 =", 10 ** 3) # 10 cubed = 1000

# Complex expressions
print("\n📊 Complex Calculations:")
print("(5 + 3) * 2 =", (5 + 3) * 2)
print("10 + 5 * 2 =", 10 + 5 * 2)  # Multiplication before addition
print("(10 + 5) * 2 =", (10 + 5) * 2)  # Parentheses first

# Real-world example: Calculate area of a rectangle
print("\n📐 Real-World Example: Rectangle Area")
length = 10
width = 5
area = length * width
print(f"Rectangle with length {length} and width {width}")
print(f"Area = {area} square units")

print("\n" + "=" * 50)
print("✅ Calculator demo complete!")
print("=" * 50)
