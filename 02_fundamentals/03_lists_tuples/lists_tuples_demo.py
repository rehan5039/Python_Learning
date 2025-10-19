# Lists and Tuples Demo

# Creating Lists
print("=== Creating Lists ===")
empty_list = []
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "orange"]
mixed = [1, "hello", 3.14, True]

print(f"Empty list: {empty_list}")
print(f"Numbers: {numbers}")
print(f"Fruits: {fruits}")
print(f"Mixed: {mixed}")

# Creating Tuples
print("\n=== Creating Tuples ===")
empty_tuple = ()
coordinates = (10, 20)
colors = ("red", "green", "blue")
single_element = (42,)  # Note the comma for single element tuple
mixed_tuple = (1, "hello", 3.14, True)

print(f"Empty tuple: {empty_tuple}")
print(f"Coordinates: {coordinates}")
print(f"Colors: {colors}")
print(f"Single element: {single_element}")
print(f"Mixed tuple: {mixed_tuple}")

# Indexing and Slicing
print("\n=== Indexing and Slicing ===")
fruits = ["apple", "banana", "orange", "grape", "kiwi"]
print(f"Fruits list: {fruits}")
print(f"First fruit: {fruits[0]}")
print(f"Last fruit: {fruits[-1]}")
print(f"Second and third: {fruits[1:3]}")
print(f"First three: {fruits[:3]}")
print(f"Last two: {fruits[-2:]}")

# List Methods - Adding Elements
print("\n=== List Methods - Adding Elements ===")
shopping_list = ["milk", "bread"]
print(f"Initial list: {shopping_list}")

shopping_list.append("eggs")
print(f"After append('eggs'): {shopping_list}")

shopping_list.insert(1, "cheese")
print(f"After insert(1, 'cheese'): {shopping_list}")

shopping_list.extend(["butter", "jam"])
print(f"After extend(['butter', 'jam']): {shopping_list}")

# List Methods - Removing Elements
print("\n=== List Methods - Removing Elements ===")
fruits = ["apple", "banana", "orange", "banana", "grape"]
print(f"Initial fruits: {fruits}")

fruits.remove("banana")  # Removes first occurrence
print(f"After remove('banana'): {fruits}")

popped_item = fruits.pop()  # Removes and returns last item
print(f"Popped item: {popped_item}")
print(f"After pop(): {fruits}")

popped_item = fruits.pop(1)  # Removes and returns item at index 1
print(f"Popped item at index 1: {popped_item}")
print(f"After pop(1): {fruits}")

# List Methods - Other Operations
print("\n=== List Methods - Other Operations ===")
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"Original numbers: {numbers}")

print(f"Count of 1s: {numbers.count(1)}")
print(f"Index of 5: {numbers.index(5)}")

numbers.sort()
print(f"After sort(): {numbers}")

numbers.reverse()
print(f"After reverse(): {numbers}")

# List Operations
print("\n=== List Operations ===")
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# Concatenation
combined = list1 + list2
print(f"Concatenation: {list1} + {list2} = {combined}")

# Repetition
repeated = list1 * 3
print(f"Repetition: {list1} * 3 = {repeated}")

# Membership testing
print(f"Is 2 in {list1}? {2 in list1}")
print(f"Is 5 not in {list1}? {5 not in list1}")

# Built-in functions
print(f"Length of {numbers}: {len(numbers)}")
print(f"Max of {numbers}: {max(numbers)}")
print(f"Min of {numbers}: {min(numbers)}")
print(f"Sum of {numbers}: {sum(numbers)}")

# List vs Tuple Comparison
print("\n=== List vs Tuple Comparison ===")
# Lists are mutable
mutable_list = [1, 2, 3]
print(f"Original list: {mutable_list}")
mutable_list[0] = 99
print(f"Modified list: {mutable_list}")

# Tuples are immutable
immutable_tuple = (1, 2, 3)
print(f"Original tuple: {immutable_tuple}")
# immutable_tuple[0] = 99  # This would raise an error

# Tuples can be dictionary keys
locations = {
    (0, 0): "origin",
    (10, 20): "point A",
    (30, 40): "point B"
}
print(f"Locations dictionary: {locations}")

# Iterating Through Collections
print("\n=== Iterating Through Collections ===")
fruits = ["apple", "banana", "orange"]

print("Simple iteration:")
for fruit in fruits:
    print(f"  {fruit}")

print("\nWith enumerate:")
for index, fruit in enumerate(fruits):
    print(f"  {index}: {fruit}")

print("\nWith range and len:")
for i in range(len(fruits)):
    print(f"  {i}: {fruits[i]}")

# List Comprehensions
print("\n=== List Comprehensions ===")
# Basic
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

# With condition
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"Even squares: {even_squares}")

# With transformation
fruits = ["apple", "banana", "cherry"]
uppercase_fruits = [fruit.upper() for fruit in fruits]
print(f"Uppercase fruits: {uppercase_fruits}")

# Practical Example: Grade Statistics
print("\n=== Practical Example: Grade Statistics ===")
def calculate_grades(scores):
    """Calculate statistics for a list of scores"""
    if not scores:
        return None
    
    return {
        "count": len(scores),
        "sum": sum(scores),
        "average": sum(scores) / len(scores),
        "highest": max(scores),
        "lowest": min(scores),
        "sorted": sorted(scores)
    }

student_scores = [85, 92, 78, 96, 88, 91, 87]
stats = calculate_grades(student_scores)
print("Student Scores:", student_scores)
print("Statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key.capitalize()}: {value:.2f}")
    else:
        print(f"  {key.capitalize()}: {value}")

# Common Mistakes Demonstration
print("\n=== Common Mistakes ===")

# Shallow copy issue
print("Shallow copy issue:")
original = [[1, 2], [3, 4]]
copied = original.copy()  # Shallow copy
original[0][0] = 99
print(f"Original: {original}")
print(f"Copied (affected by shallow copy): {copied}")

# Deep copy solution
print("\nDeep copy solution:")
import copy
original = [[1, 2], [3, 4]]
deep_copied = copy.deepcopy(original)
original[0][0] = 100
print(f"Original: {original}")
print(f"Deep copied (unaffected): {deep_copied}")