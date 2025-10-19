# 🔀 Conditional Statements in Python

Conditional statements allow your programs to make decisions and execute different code paths based on certain conditions. This guide will teach you how to use `if`, `elif`, and `else` statements effectively.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Use `if`, `elif`, and `else` statements
- Create complex conditional expressions
- Apply logical and comparison operators
- Implement nested conditions
- Write clean and readable conditional code

## 🤔 What are Conditional Statements?

Conditional statements execute different blocks of code based on whether certain conditions are true or false. They're essential for controlling the flow of your programs.

### Basic `if` Statement

```python
age = 18

if age >= 18:
    print("You are eligible to vote!")

# Output: You are eligible to vote!
```

### `if-else` Statement

```python
age = 16

if age >= 18:
    print("You are eligible to vote!")
else:
    print("You are not eligible to vote yet.")

# Output: You are not eligible to vote yet.
```

### `if-elif-else` Statement

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Your grade is: {grade}")

# Output: Your grade is: B
```

## 📊 Comparison Operators

Comparison operators compare two values and return a Boolean result (`True` or `False`).

```python
x = 10
y = 20

print(x == y)  # False (equal to)
print(x != y)  # True  (not equal to)
print(x < y)   # True  (less than)
print(x > y)   # False (greater than)
print(x <= y)  # True  (less than or equal to)
print(x >= y)  # False (greater than or equal to)
```

### Comparing Different Data Types

```python
# String comparison
name1 = "Alice"
name2 = "Bob"
print(name1 < name2)  # True (lexicographic comparison)

# Case sensitivity
print("Alice" == "alice")  # False

# Mixed types (be careful!)
print(5 == 5.0)    # True
print(5 == "5")    # False
```

## 🔗 Logical Operators

Logical operators combine multiple conditions.

```python
age = 25
has_license = True
has_car = False

# AND operator - both conditions must be true
if age >= 18 and has_license:
    print("You can drive!")

# OR operator - at least one condition must be true
if has_license or has_car:
    print("You have transportation options!")

# NOT operator - negates a condition
if not has_car:
    print("You don't have a car.")

# Complex combinations
if (age >= 18 and has_license) and not has_car:
    print("You can drive but don't have a car.")
```

### Operator Precedence

```python
# AND has higher precedence than OR
result1 = True or False and False  # True (evaluated as: True or (False and False))
result2 = (True or False) and False  # False

print(f"Without parentheses: {result1}")
print(f"With parentheses: {result2}")
```

## 🔄 Nested Conditions

You can nest conditional statements inside each other for complex logic.

```python
username = "admin"
password = "secret123"
is_logged_in = False

if username == "admin":
    if password == "secret123":
        is_logged_in = True
        print("Welcome, administrator!")
    else:
        print("Incorrect password!")
else:
    print("Unknown user!")

# Output: Welcome, administrator!
```

### Cleaner Nested Conditions

```python
# Instead of deep nesting, use logical operators
if username == "admin" and password == "secret123":
    is_logged_in = True
    print("Welcome, administrator!")
elif username == "admin":
    print("Incorrect password!")
else:
    print("Unknown user!")
```

## 🎯 Ternary Operator (Conditional Expression)

Python provides a concise way to write simple if-else statements.

```python
age = 20

# Traditional way
if age >= 18:
    status = "adult"
else:
    status = "minor"

# Ternary operator
status = "adult" if age >= 18 else "minor"

print(f"Status: {status}")

# More examples
temperature = 25
weather = "warm" if temperature > 20 else "cold"
print(f"Weather: {weather}")

# Chaining ternary operators (not recommended for readability)
grade = 85
letter_grade = "A" if grade >= 90 else "B" if grade >= 80 else "C" if grade >= 70 else "F"
print(f"Letter grade: {letter_grade}")
```

## 🧮 Membership and Identity Operators

### Membership Operators (`in`, `not in`)

```python
fruits = ["apple", "banana", "orange"]

if "apple" in fruits:
    print("Apple is in the list!")

if "grape" not in fruits:
    print("Grape is not in the list!")

# Works with strings
text = "Python Programming"
if "Python" in text:
    print("Found 'Python' in the text!")
```

### Identity Operators (`is`, `is not`)

```python
# is checks if two variables refer to the same object
list1 = [1, 2, 3]
list2 = [1, 2, 3]
list3 = list1

print(list1 == list2)  # True (same values)
print(list1 is list2)  # False (different objects)
print(list1 is list3)  # True (same object)

# Use is with None
value = None
if value is None:
    print("Value is not set!")
```

## 🧪 Practical Examples

### Example 1: User Authentication System
```python
class Authenticator:
    def __init__(self):
        self.users = {
            "admin": {"password": "admin123", "role": "administrator"},
            "user1": {"password": "pass456", "role": "user"},
            "guest": {"password": "guest789", "role": "guest"}
        }
    
    def login(self, username, password):
        if username in self.users:
            if self.users[username]["password"] == password:
                return {
                    "success": True,
                    "role": self.users[username]["role"],
                    "message": f"Welcome, {username}!"
                }
            else:
                return {
                    "success": False,
                    "message": "Incorrect password!"
                }
        else:
            return {
                "success": False,
                "message": "User not found!"
            }
    
    def has_permission(self, role, required_permission):
        permissions = {
            "administrator": ["read", "write", "delete", "manage_users"],
            "user": ["read", "write"],
            "guest": ["read"]
        }
        
        return required_permission in permissions.get(role, [])

# Usage
auth = Authenticator()
result = auth.login("admin", "admin123")
print(result["message"])

if result["success"]:
    if auth.has_permission(result["role"], "delete"):
        print("User has delete permission")
    else:
        print("User does not have delete permission")
```

### Example 2: Grade Calculator with Conditions
```python
def calculate_final_grade(assignments, midterm, final, participation):
    """Calculate final grade with various conditions"""
    
    # Validate inputs
    if not (0 <= assignments <= 100):
        return "Invalid assignment score"
    if not (0 <= midterm <= 100):
        return "Invalid midterm score"
    if not (0 <= final <= 100):
        return "Invalid final score"
    if not (0 <= participation <= 100):
        return "Invalid participation score"
    
    # Calculate weighted average
    final_score = (
        assignments * 0.3 +
        midterm * 0.3 +
        final * 0.3 +
        participation * 0.1
    )
    
    # Apply bonus for perfect participation
    if participation == 100:
        final_score = min(100, final_score + 2)  # Max 100
        print("Bonus points added for perfect participation!")
    
    # Determine letter grade
    if final_score >= 90:
        letter_grade = "A"
        status = "Excellent work!"
    elif final_score >= 80:
        letter_grade = "B"
        status = "Good job!"
    elif final_score >= 70:
        letter_grade = "C"
        status = "Satisfactory"
    elif final_score >= 60:
        letter_grade = "D"
        status = "Needs improvement"
    else:
        letter_grade = "F"
        status = "Failed - see instructor"
    
    return {
        "numeric_score": round(final_score, 2),
        "letter_grade": letter_grade,
        "status": status
    }

# Usage
result = calculate_final_grade(95, 87, 92, 100)
print(f"Final Grade: {result['numeric_score']}% ({result['letter_grade']})")
print(result['status'])
```

### Example 3: Weather-Based Activity Recommender
```python
def recommend_activity(temperature, is_raining, is_weekend):
    """Recommend activities based on weather and day"""
    
    # Temperature categories
    if temperature > 80:
        temp_category = "hot"
    elif temperature > 60:
        temp_category = "warm"
    elif temperature > 40:
        temp_category = "cool"
    else:
        temp_category = "cold"
    
    # Activity recommendations
    if is_raining:
        if temp_category in ["warm", "hot"] and is_weekend:
            return "Visit a museum or indoor pool"
        elif temp_category == "cold":
            return "Stay indoors and read a book"
        else:
            return "Go to a coffee shop or indoor mall"
    else:  # Not raining
        if temp_category == "hot":
            if is_weekend:
                return "Go to the beach or pool"
            else:
                return "Exercise early morning or evening"
        elif temp_category == "warm":
            if is_weekend:
                return "Go hiking or have a picnic"
            else:
                return "Take a walk during lunch break"
        elif temp_category == "cool":
            if is_weekend:
                return "Go apple picking or visit a park"
            else:
                return "Take a walk or exercise outdoors"
        else:  # cold
            if is_weekend:
                return "Go skiing or ice skating"
            else:
                return "Bundle up and take a short walk"

# Usage
activities = [
    (85, False, True),   # Hot, not raining, weekend
    (65, True, False),   # Warm, raining, weekday
    (45, False, True),   # Cool, not raining, weekend
    (30, True, False),   # Cold, raining, weekday
]

for temp, raining, weekend in activities:
    activity = recommend_activity(temp, raining, weekend)
    print(f"Temp: {temp}°F, Raining: {raining}, Weekend: {weekend}")
    print(f"Recommended activity: {activity}\n")
```

## ⚠️ Common Mistakes and Tips

### 1. Assignment vs Comparison
```python
# Wrong - using = instead of ==
age = 18
# if age = 18:  # SyntaxError
#     print("Age is 18")

# Correct - using ==
if age == 18:
    print("Age is 18")
```

### 2. Chaining Comparisons Incorrectly
```python
# Wrong - doesn't work as expected
age = 25
# if 18 <= age <= 65:  # This actually works in Python!
#     print("Working age")

# But this is wrong:
# if 18 <= age and <= 65:  # SyntaxError

# Correct chaining (Pythonic)
if 18 <= age <= 65:
    print("Working age")

# Traditional way
if age >= 18 and age <= 65:
    print("Working age")
```

### 3. Comparing with None
```python
# Wrong - not recommended
value = None
# if value == None:  # Works but not preferred

# Correct - use is
if value is None:
    print("Value is not set")
```

### 4. Empty Collections in Conditions
```python
# Pythonic way to check for empty collections
my_list = []
my_dict = {}
my_string = ""

# Instead of:
# if len(my_list) == 0:
# if my_list == []:

# Use:
if not my_list:
    print("List is empty")

if not my_dict:
    print("Dictionary is empty")

if not my_string:
    print("String is empty")
```

## 📚 Next Steps

Now that you understand conditional statements, you're ready to learn:

1. **Loops**: Repeating code with `for` and `while` loops
2. **Functions**: Creating reusable code blocks
3. **Exception Handling**: Managing errors gracefully
4. **File I/O**: Reading and writing files

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: What's the difference between `==` and `is`?
A: `==` compares values, while `is` checks if two variables refer to the same object.

### Q: Why use `elif` instead of multiple `if` statements?
A: `elif` is more efficient and prevents multiple conditions from being checked unnecessarily.

### Q: Can I have multiple `elif` statements?
A: Yes, you can have as many `elif` statements as needed.

### Q: What happens if I don't include an `else` clause?
A: If no conditions match, nothing happens (the code continues after the conditional block).

---

**Practice writing conditional statements with different scenarios to build your skills!** 🐍