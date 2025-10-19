# Loop Examples

# While Loops
print("=== While Loops ===")
count = 0
while count < 5:
    print(f"Count is: {count}")
    count += 1

# While loop with break
print("\nWhile loop with break:")
number = 1
while True:
    if number > 5:
        break
    print(number)
    number += 1

# While loop with continue
print("\nWhile loop with continue:")
count = 0
while count < 10:
    count += 1
    if count % 2 == 0:
        continue  # Skip even numbers
    print(f"Odd number: {count}")

# For Loops
print("\n=== For Loops ===")
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(f"I like {fruit}")

# For loop with string
print("\nFor loop with string:")
for char in "Python":
    print(char)

# Using range() function
print("\nUsing range() function:")
for i in range(5):
    print(i)

print("\nRange with start and stop:")
for i in range(2, 8):
    print(i)

print("\nRange with start, stop, and step:")
for i in range(0, 10, 2):
    print(i)

print("\nReverse counting:")
for i in range(10, 0, -1):
    print(i)

# Enumerate function
print("\n=== Enumerate function ===")
fruits = ["apple", "banana", "orange"]
for index, fruit in enumerate(fruits):
    print(f"Index {index}: {fruit}")

# Loop Control Statements
print("\n=== Loop Control Statements ===")

# Break statement
print("Break statement:")
numbers = [1, 3, 5, 7, 8, 9, 11]
for num in numbers:
    if num % 2 == 0:
        print(f"First even number found: {num}")
        break
else:
    print("No even numbers found")

# Continue statement
print("\nContinue statement:")
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for num in numbers:
    if num % 2 == 0:
        continue  # Skip even numbers
    print(f"Odd number: {num}")

# Pass statement
print("\nPass statement:")
for i in range(5):
    if i == 2:
        pass  # Do nothing
    else:
        print(i)

# For-Else and While-Else
print("\n=== For-Else and While-Else ===")

# For-else
numbers = [1, 3, 5, 7, 9]
for num in numbers:
    if num % 2 == 0:
        print(f"Found even number: {num}")
        break
else:
    print("No even numbers found in the list")

# While-else
count = 1
while count < 5:
    if count == 3:
        print("Found 3!")
        break
    count += 1
else:
    print("Loop completed without finding 3")

# Nested Loops
print("\n=== Nested Loops ===")
for i in range(3):
    for j in range(2):
        print(f"i={i}, j={j}")

# Multiplication table
print("\nMultiplication Table:")
for i in range(1, 6):
    for j in range(1, 6):
        print(f"{i*j:3}", end=" ")
    print()  # New line after each row

# List Comprehensions
print("\n=== List Comprehensions ===")

# Basic list comprehension
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

# List comprehension with condition
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"Even squares: {even_squares}")

# List comprehension with transformation
fruits = ["apple", "banana", "cherry"]
uppercase_fruits = [fruit.upper() for fruit in fruits]
print(f"Uppercase fruits: {uppercase_fruits}")

# Nested list comprehension
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
print(f"Matrix: {matrix}")

# Practical Example: Prime Number Finder
print("\n=== Practical Example: Prime Number Finder ===")

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

# Display primes up to 50
display_primes(50)

# Practical Example: Pattern Printing
print("\n=== Practical Example: Pattern Printing ===")

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
    
    print("\n4. Number Pattern:")
    for i in range(1, 6):
        for j in range(1, i + 1):
            print(j, end=" ")
        print()

# Print patterns
print_patterns()

# Common Mistakes Demonstration
print("\n=== Common Mistakes Demonstration ===")

# Correct way to avoid infinite loops
print("Correct way to avoid infinite loops:")
count = 0
while count < 5:
    print(count)
    count += 1

# Proper indentation
print("\nProper indentation:")
for i in range(3):
    print(f"Outer loop: {i}")
    for j in range(2):
        print(f"  Inner loop: {j}")