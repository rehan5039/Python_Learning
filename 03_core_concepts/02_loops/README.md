# 🔁 Loops in Python

Loops allow you to execute blocks of code repeatedly, making them essential for automating repetitive tasks. This guide will teach you how to use `for` and `while` loops effectively.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Use `for` and `while` loops
- Apply loop control statements (`break`, `continue`, `pass`)
- Work with the `range()` function
- Implement nested loops
- Write efficient and readable loop code

## 🔁 While Loops

While loops execute a block of code as long as a condition is true.

### Basic While Loop

```python
# Basic while loop
count = 0
while count < 5:
    print(f"Count is: {count}")
    count += 1

# Output:
# Count is: 0
# Count is: 1
# Count is: 2
# Count is: 3
# Count is: 4
```

### Infinite Loops and Break Statements

```python
# Infinite loop with break
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == "quit":
        break
    print(f"You entered: {user_input}")

# Counter-controlled loop
number = 1
while number <= 10:
    if number % 2 == 0:
        print(f"{number} is even")
    else:
        print(f"{number} is odd")
    number += 1
```

### While Loop with Continue

```python
# Using continue to skip iterations
count = 0
while count < 10:
    count += 1
    if count % 2 == 0:
        continue  # Skip even numbers
    print(f"Odd number: {count}")
```

## 🔁 For Loops

For loops iterate over sequences (lists, tuples, strings, etc.) or other iterable objects.

### Basic For Loop

```python
# Iterating over a list
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(f"I like {fruit}")

# Iterating over a string
for char in "Python":
    print(char)

# Iterating over a tuple
coordinates = (10, 20, 30)
for coord in coordinates:
    print(f"Coordinate: {coord}")
```

### Using range() Function

```python
# range(stop)
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# range(start, stop)
for i in range(2, 8):
    print(i)  # 2, 3, 4, 5, 6, 7

# range(start, stop, step)
for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

# Reverse counting
for i in range(10, 0, -1):
    print(i)  # 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
```

### Enumerate Function

```python
# Getting index and value
fruits = ["apple", "banana", "orange"]
for index, fruit in enumerate(fruits):
    print(f"Index {index}: {fruit}")

# Starting enumeration from a different number
for index, fruit in enumerate(fruits, 1):
    print(f"Item {index}: {fruit}")
```

## 🔧 Loop Control Statements

### Break Statement

```python
# Breaking out of a loop
numbers = [1, 3, 5, 7, 8, 9, 11]
for num in numbers:
    if num % 2 == 0:
        print(f"First even number found: {num}")
        break
else:
    print("No even numbers found")

# Break in nested loops
for i in range(3):
    for j in range(3):
        if i == 1 and j == 1:
            print(f"Breaking at i={i}, j={j}")
            break
        print(f"i={i}, j={j}")
    else:
        continue
    break
```

### Continue Statement

```python
# Skipping iterations
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for num in numbers:
    if num % 2 == 0:
        continue  # Skip even numbers
    print(f"Odd number: {num}")

# Continue in nested loops
for i in range(3):
    for j in range(3):
        if j == 1:
            continue  # Skip j=1
        print(f"i={i}, j={j}")
```

### Pass Statement

```python
# Using pass as a placeholder
for i in range(5):
    if i == 2:
        pass  # Do nothing, placeholder
    else:
        print(i)

# Empty function placeholder
def todo_function():
    pass  # Will implement later

# Empty class placeholder
class TodoClass:
    pass  # Will implement later
```

## 🔄 For-Else and While-Else

Python has a unique feature where you can have an `else` clause with loops.

```python
# For-else: else executes if loop completes normally (no break)
numbers = [1, 3, 5, 7, 9]
for num in numbers:
    if num % 2 == 0:
        print(f"Found even number: {num}")
        break
else:
    print("No even numbers found in the list")

# While-else: same concept
count = 1
while count < 5:
    if count == 3:
        print("Found 3!")
        break
    count += 1
else:
    print("Loop completed without finding 3")
```

## 🔢 Nested Loops

Nested loops are loops inside other loops.

```python
# Nested for loops
for i in range(3):
    for j in range(2):
        print(f"i={i}, j={j}")

# Creating a multiplication table
print("Multiplication Table:")
for i in range(1, 6):
    for j in range(1, 6):
        print(f"{i*j:3}", end=" ")
    print()  # New line after each row

# Nested loops with lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for row in matrix:
    for element in row:
        print(element, end=" ")
    print()  # New line after each row
```

## 🛠 List Comprehensions

List comprehensions provide a concise way to create lists using loops.

```python
# Basic list comprehension
squares = [x**2 for x in range(1, 6)]
print(squares)  # [1, 4, 9, 16, 25]

# List comprehension with condition
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(even_squares)  # [4, 16, 36, 64, 100]

# List comprehension with transformation
fruits = ["apple", "banana", "cherry"]
uppercase_fruits = [fruit.upper() for fruit in fruits]
print(uppercase_fruits)  # ['APPLE', 'BANANA', 'CHERRY']

# Nested list comprehension
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
print(matrix)  # [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
```

## 🧪 Practical Examples

### Example 1: Number Guessing Game
```python
import random

def number_guessing_game():
    """A simple number guessing game"""
    secret_number = random.randint(1, 100)
    attempts = 0
    max_attempts = 7
    
    print("Welcome to the Number Guessing Game!")
    print(f"I'm thinking of a number between 1 and 100. You have {max_attempts} attempts.")
    
    while attempts < max_attempts:
        try:
            guess = int(input(f"\nAttempt {attempts + 1}: Enter your guess: "))
            attempts += 1
            
            if guess == secret_number:
                print(f"Congratulations! You guessed the number in {attempts} attempts!")
                return
            elif guess < secret_number:
                print("Too low! Try a higher number.")
            else:
                print("Too high! Try a lower number.")
                
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"You have {remaining} attempts left.")
                
        except ValueError:
            print("Please enter a valid number.")
            attempts -= 1  # Don't count invalid input as an attempt
    
    print(f"\nGame over! The number was {secret_number}.")

# Uncomment to play the game
# number_guessing_game()
```

### Example 2: Prime Number Finder
```python
def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes(limit):
    """Find all prime numbers up to a limit"""
    primes = []
    for num in range(2, limit + 1):
        if is_prime(num):
            primes.append(num)
    return primes

def display_primes(limit, per_line=10):
    """Display prime numbers in a formatted way"""
    primes = find_primes(limit)
    print(f"Prime numbers up to {limit}:")
    
    for i, prime in enumerate(primes):
        print(f"{prime:4}", end=" ")
        if (i + 1) % per_line == 0:
            print()  # New line after every 'per_line' numbers
    
    if len(primes) % per_line != 0:
        print()  # Final new line if needed
    
    print(f"\nTotal primes found: {len(primes)}")

# Usage
display_primes(100)
```

### Example 3: Text Analysis Tool
```python
def analyze_text(text):
    """Analyze text and provide statistics"""
    # Clean and process text
    import string
    cleaned_text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = cleaned_text.split()
    
    # Statistics
    stats = {
        "characters": len(text),
        "words": len(words),
        "lines": text.count('\n') + 1,
        "unique_words": len(set(words)),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
    
    # Word frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return stats, sorted_words[:10]  # Top 10 most frequent words

def display_text_analysis(text):
    """Display text analysis results"""
    stats, top_words = analyze_text(text)
    
    print("=== Text Analysis ===")
    print(f"Characters: {stats['characters']}")
    print(f"Words: {stats['words']}")
    print(f"Lines: {stats['lines']}")
    print(f"Unique words: {stats['unique_words']}")
    print(f"Average word length: {stats['average_word_length']:.2f}")
    
    print("\nTop 10 most frequent words:")
    for i, (word, count) in enumerate(top_words, 1):
        print(f"{i:2}. {word:<15} ({count} times)")

# Usage
sample_text = """
Python is a powerful programming language. Python is easy to learn and very versatile.
Many developers love Python because Python is readable and has a simple syntax.
Python can be used for web development, data science, automation, and more.
The Python community is large and supportive, making it easy to find help.
"""

display_text_analysis(sample_text)
```

### Example 4: Pattern Printing
```python
def print_patterns():
    """Print various patterns using loops"""
    
    print("1. Right Triangle:")
    for i in range(1, 6):
        print("*" * i)
    
    print("\n2. Left Triangle:")
    for i in range(1, 6):
        print(" " * (5 - i) + "*" * i)
    
    print("\n3. Pyramid:")
    for i in range(1, 6):
        print(" " * (5 - i) + "*" * (2 * i - 1))
    
    print("\n4. Diamond:")
    # Upper half
    for i in range(1, 4):
        print(" " * (4 - i) + "*" * (2 * i - 1))
    # Lower half
    for i in range(2, 0, -1):
        print(" " * (4 - i) + "*" * (2 * i - 1))
    
    print("\n5. Number Pattern:")
    for i in range(1, 6):
        for j in range(1, i + 1):
            print(j, end=" ")
        print()

# Usage
print_patterns()
```

## ⚠️ Common Mistakes and Tips

### 1. Infinite Loops
```python
# Wrong - infinite loop
# count = 0
# while count < 10:
#     print(count)
#     # Forgot to increment count!

# Correct - always update the loop variable
count = 0
while count < 10:
    print(count)
    count += 1
```

### 2. Off-by-One Errors
```python
# Wrong - prints 0 to 4 instead of 1 to 5
# for i in range(5):
#     print(i + 1)

# Correct - more intuitive
for i in range(1, 6):
    print(i)
```

### 3. Modifying Lists While Iterating
```python
# Wrong - modifying list while iterating
# numbers = [1, 2, 3, 4, 5]
# for num in numbers:
#     if num % 2 == 0:
#         numbers.remove(num)  # Can cause issues

# Correct - iterate over a copy or use list comprehension
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]
print(numbers)  # [1, 3, 5]
```

### 4. Forgetting Indentation
```python
# Wrong - indentation error
# for i in range(5):
# print(i)  # IndentationError

# Correct - proper indentation
for i in range(5):
    print(i)
```

## 📚 Next Steps

Now that you understand loops, you're ready to learn:

1. **Functions**: Creating reusable code blocks
2. **Exception Handling**: Managing errors gracefully
3. **File I/O**: Reading and writing files
4. **Object-Oriented Programming**: Advanced programming concepts

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: When should I use `for` vs `while` loops?
A: Use `for` loops when you know the number of iterations or are iterating over a sequence. Use `while` loops when the number of iterations depends on a condition.

### Q: What's the difference between `break` and `continue`?
A: `break` exits the loop completely, while `continue` skips the rest of the current iteration and moves to the next iteration.

### Q: What is the `else` clause in loops used for?
A: The `else` clause executes only if the loop completes normally (without a `break` statement).

### Q: Are list comprehensions always better than loops?
A: List comprehensions are more concise and often faster, but regular loops are more readable for complex logic.

---

**Practice writing different types of loops with various scenarios to build your skills!** 🐍