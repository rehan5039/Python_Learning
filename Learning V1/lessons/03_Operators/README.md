# Lesson 03: Operators in Python 🔢

Welcome to Lesson 03! Now you'll learn how to perform operations on data using various types of operators.

## 📚 What You'll Learn

- Arithmetic operators
- Comparison operators
- Logical operators
- Assignment operators
- Bitwise operators
- Identity and membership operators
- Operator precedence

---

## ➕ Arithmetic Operators

Perform mathematical calculations.

### Basic Arithmetic

```python
# Addition
print(10 + 5)    # 15

# Subtraction
print(10 - 5)    # 5

# Multiplication
print(10 * 5)    # 50

# Division (always returns float)
print(10 / 3)    # 3.3333...

# Floor Division (rounds down to integer)
print(10 // 3)   # 3

# Modulus (remainder)
print(10 % 3)    # 1

# Exponentiation (power)
print(2 ** 3)    # 8 (2 to the power of 3)
```

### Real-World Examples

```python
# Calculate total price with tax
price = 100
tax_rate = 0.08
total = price + (price * tax_rate)  # 108.0

# Calculate area of a circle
radius = 5
pi = 3.14159
area = pi * radius ** 2  # 78.53975

# Convert Celsius to Fahrenheit
celsius = 25
fahrenheit = (celsius * 9/5) + 32  # 77.0
```

---

## 🔍 Comparison Operators

Compare values and return True or False.

```python
# Equal to
print(5 == 5)    # True
print(5 == 3)    # False

# Not equal to
print(5 != 3)    # True

# Greater than
print(5 > 3)     # True

# Less than
print(5 < 3)     # False

# Greater than or equal to
print(5 >= 5)    # True

# Less than or equal to
print(5 <= 3)    # False
```

### Comparing Strings

```python
# Alphabetical comparison
print("apple" < "banana")    # True
print("apple" == "Apple")    # False (case-sensitive)

# Length comparison
name1 = "Alice"
name2 = "Bob"
print(len(name1) > len(name2))  # True
```

---

## 🧠 Logical Operators

Combine conditional statements.

### AND, OR, NOT

```python
# AND - Both conditions must be True
age = 25
has_license = True
print(age >= 18 and has_license)  # True

# OR - At least one condition must be True
is_weekend = True
is_holiday = False
print(is_weekend or is_holiday)   # True

# NOT - Reverses the boolean value
is_raining = False
print(not is_raining)  # True
```

### Complex Conditions

```python
# Voting eligibility
age = 20
is_citizen = True
can_vote = age >= 18 and is_citizen  # True

# Weekend or holiday discount
is_weekend = False
is_holiday = True
has_discount = is_weekend or is_holiday  # True

# Password validation
password = "SecurePass123"
is_long_enough = len(password) >= 8
has_numbers = any(char.isdigit() for char in password)
is_valid = is_long_enough and has_numbers  # True
```

---

## 📝 Assignment Operators

Assign and update variable values.

```python
# Basic assignment
x = 10

# Add and assign
x += 5   # Same as: x = x + 5 (now x = 15)

# Subtract and assign
x -= 3   # Same as: x = x - 3 (now x = 12)

# Multiply and assign
x *= 2   # Same as: x = x * 2 (now x = 24)

# Divide and assign
x /= 4   # Same as: x = x / 4 (now x = 6.0)

# Floor divide and assign
x //= 2  # Same as: x = x // 2 (now x = 3.0)

# Modulus and assign
x %= 2   # Same as: x = x % 2 (now x = 1.0)

# Exponent and assign
x **= 3  # Same as: x = x ** 3 (now x = 1.0)
```

### Practical Examples

```python
# Counter
count = 0
count += 1  # Increment by 1
count += 1
count += 1
print(count)  # 3

# Score accumulator
score = 100
score += 50   # Bonus points
score -= 20   # Penalty
print(score)  # 130

# Price with discount
price = 100
price *= 0.8  # 20% discount (multiply by 0.8)
print(price)  # 80.0
```

---

## 🔢 Bitwise Operators

Work with binary representations of numbers.

```python
# AND
print(5 & 3)   # 1 (Binary: 101 & 011 = 001)

# OR
print(5 | 3)   # 7 (Binary: 101 | 011 = 111)

# XOR
print(5 ^ 3)   # 6 (Binary: 101 ^ 011 = 110)

# NOT
print(~5)      # -6

# Left shift
print(5 << 1)  # 10 (Binary: 101 << 1 = 1010)

# Right shift
print(5 >> 1)  # 2 (Binary: 101 >> 1 = 10)
```

---

## 🎭 Identity Operators

Check if objects are the same object in memory.

```python
# is
x = [1, 2, 3]
y = [1, 2, 3]
z = x

print(x is z)   # True (same object)
print(x is y)   # False (different objects, same values)

# is not
print(x is not y)  # True

# == vs is
a = None
print(a is None)  # True (correct way to check None)
print(a == None)  # True (works but not recommended)
```

---

## 📦 Membership Operators

Check if a value exists in a sequence.

```python
# in
fruits = ["apple", "banana", "cherry"]
print("apple" in fruits)      # True
print("orange" in fruits)     # False

# not in
print("orange" not in fruits) # True

# With strings
text = "Python is awesome"
print("Python" in text)       # True
print("Java" not in text)     # True

# With dictionaries (checks keys)
person = {"name": "Alice", "age": 25}
print("name" in person)       # True
print("Alice" in person)      # False (value, not key)
```

---

## 🎯 Operator Precedence

Order in which operators are evaluated.

### Precedence Table (High to Low)

1. `()`           - Parentheses
2. `**`           - Exponentiation
3. `+x`, `-x`     - Unary plus/minus
4. `*`, `/`, `//`, `%` - Multiplication, Division
5. `+`, `-`       - Addition, Subtraction
6. `<<`, `>>`     - Bitwise shifts
7. `&`            - Bitwise AND
8. `^`            - Bitwise XOR
9. `|`            - Bitwise OR
10. `==`, `!=`, `<`, `>`, `<=`, `>=` - Comparisons
11. `is`, `is not` - Identity
12. `in`, `not in` - Membership
13. `not`         - Logical NOT
14. `and`         - Logical AND
15. `or`          - Logical OR

### Examples

```python
# Without parentheses
result = 10 + 5 * 2    # 20 (multiplication first)

# With parentheses
result = (10 + 5) * 2  # 30 (parentheses first)

# Complex expression
result = 10 + 5 * 2 ** 2   # 30
# Breakdown: 2 ** 2 = 4, then 5 * 4 = 20, then 10 + 20 = 30

# Using parentheses for clarity
result = 10 + (5 * (2 ** 2))  # Same as above but clearer
```

---

## ✍️ Practice Exercises

### Exercise 1: Basic Calculator
Create variables for two numbers and perform all arithmetic operations.

### Exercise 2: Age Checker
Ask user for age and check if they're:
- A child (< 13)
- A teenager (13-19)
- An adult (20-64)
- A senior (65+)

### Exercise 3: Grade Calculator
Calculate final grade from:
- Homework: 20%
- Midterm: 30%
- Final: 50%

### Exercise 4: Discount Calculator
```python
price = 150
# Apply 10% discount if price > 100
# Apply additional 5% if it's weekend
# Calculate final price
```

### Exercise 5: Leap Year Checker
Check if a year is a leap year:
- Divisible by 4
- But NOT divisible by 100
- Unless ALSO divisible by 400

---

## 🎓 Key Takeaways

✅ Arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`  
✅ Comparison operators: `==`, `!=`, `<`, `>`, `<=`, `>=`  
✅ Logical operators: `and`, `or`, `not`  
✅ Assignment operators: `=`, `+=`, `-=`, etc.  
✅ Membership: `in`, `not in`  
✅ Identity: `is`, `is not`  
✅ Use parentheses for clarity  

---

## 🚀 Next Steps

Great job! 🎉

**Next**: [Lesson 04: Control Flow](../04_Control_Flow/) - Learn to make decisions in your code.

---

**Happy Coding! 🐍**
