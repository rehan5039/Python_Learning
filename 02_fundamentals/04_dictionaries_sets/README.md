# 📚 Dictionaries and Sets in Python

Dictionaries and sets are powerful data structures in Python that provide efficient ways to store and manipulate data. This guide will teach you how to use both effectively.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Create and manipulate dictionaries and sets
- Understand the differences between these data structures
- Use dictionary methods and set operations
- Apply these structures to solve real-world problems

## 📚 What are Dictionaries?

Dictionaries are unordered collections of key-value pairs. Each key is unique and maps to a value, making dictionaries ideal for storing related data.

### Creating Dictionaries

```python
# Empty dictionary
empty_dict = {}

# Dictionary with key-value pairs
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "courses": ["Math", "Science", "English"]
}

# Using dict() constructor
person = dict(name="Bob", age=25, city="New York")

# Dictionary with mixed key types (not recommended)
mixed_keys = {
    "name": "Charlie",
    1: "one",
    (2, 3): "tuple key"
}

print(student)
print(person)
```

### Dictionary Characteristics
- **Key-Value Pairs**: Data stored as key-value mappings
- **Unique Keys**: Each key can appear only once
- **Mutable**: Can be modified after creation
- **Unordered**: (In Python < 3.7) Order not guaranteed
- **Dynamic**: Can grow or shrink in size

## 🔑 Dictionary Operations

### Accessing Values
```python
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A"
}

# Access by key
print(student["name"])     # Alice
print(student["age"])      # 20

# Using get() method (safer)
print(student.get("name"))        # Alice
print(student.get("height"))      # None
print(student.get("height", 0))   # 0 (default value)
```

### Adding and Modifying Values
```python
student = {
    "name": "Alice",
    "age": 20
}

# Adding new key-value pairs
student["grade"] = "A"
student["courses"] = ["Math", "Science"]

# Modifying existing values
student["age"] = 21

# Using update() method
student.update({"city": "Boston", "year": 2023})

print(student)
```

### Removing Items
```python
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "courses": ["Math", "Science"]
}

# pop() - removes and returns value
grade = student.pop("grade")
print(f"Removed grade: {grade}")

# popitem() - removes and returns last inserted key-value pair
last_item = student.popitem()
print(f"Removed last item: {last_item}")

# del - removes specific key
del student["age"]

# clear() - removes all items
# student.clear()

print(student)
```

## 🛠 Dictionary Methods

### Common Dictionary Methods
```python
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "courses": ["Math", "Science"]
}

# keys() - returns all keys
print("Keys:", list(student.keys()))

# values() - returns all values
print("Values:", list(student.values()))

# items() - returns key-value pairs as tuples
print("Items:", list(student.items()))

# copy() - creates a shallow copy
student_copy = student.copy()

# setdefault() - gets value or sets default
height = student.setdefault("height", 170)
print(f"Height: {height}")
print(student)
```

### Dictionary Comprehensions
```python
# Basic dictionary comprehension
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# With condition
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)  # {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}

# From two lists
keys = ["name", "age", "city"]
values = ["Bob", 25, "New York"]
person = {k: v for k, v in zip(keys, values)}
print(person)  # {'name': 'Bob', 'age': 25, 'city': 'New York'}
```

## 🔢 What are Sets?

Sets are unordered collections of unique elements. They are useful for membership testing and eliminating duplicate entries.

### Creating Sets
```python
# Empty set
empty_set = set()

# Set with elements
fruits = {"apple", "banana", "orange"}

# From a list (removes duplicates)
numbers = set([1, 2, 2, 3, 3, 4, 5])
print(numbers)  # {1, 2, 3, 4, 5}

# Using set() constructor
vowels = set("aeiou")
print(vowels)  # {'e', 'u', 'o', 'a', 'i'}

print(fruits)
```

### Set Characteristics
- **Unique Elements**: No duplicates allowed
- **Unordered**: No guaranteed order
- **Mutable**: Can be modified after creation (frozenset is immutable)
- **Fast Membership Testing**: O(1) average case
- **Mathematical Operations**: Union, intersection, difference

## 🛠 Set Operations

### Adding and Removing Elements
```python
fruits = {"apple", "banana"}

# add() - adds single element
fruits.add("orange")
print(fruits)  # {'apple', 'banana', 'orange'}

# update() - adds multiple elements
fruits.update(["grape", "kiwi", "mango"])
print(fruits)

# remove() - removes element (raises error if not found)
fruits.remove("banana")
print(fruits)

# discard() - removes element (no error if not found)
fruits.discard("pineapple")  # No error
print(fruits)

# pop() - removes and returns arbitrary element
popped = fruits.pop()
print(f"Popped: {popped}")
print(fruits)
```

### Set Operations
```python
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Union - all elements from both sets
union = set1 | set2  # or set1.union(set2)
print(f"Union: {union}")  # {1, 2, 3, 4, 5, 6, 7, 8}

# Intersection - common elements
intersection = set1 & set2  # or set1.intersection(set2)
print(f"Intersection: {intersection}")  # {4, 5}

# Difference - elements in set1 but not in set2
difference = set1 - set2  # or set1.difference(set2)
print(f"Difference: {difference}")  # {1, 2, 3}

# Symmetric Difference - elements in either set but not both
symmetric_diff = set1 ^ set2  # or set1.symmetric_difference(set2)
print(f"Symmetric Difference: {symmetric_diff}")  # {1, 2, 3, 6, 7, 8}

# Subset and Superset
print(f"set1 subset of union: {set1 <= union}")  # True
print(f"union superset of set2: {union >= set2}")  # True
```

### Set Methods
```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

# copy() - creates a copy
set_copy = set1.copy()

# isdisjoint() - True if no common elements
print(set1.isdisjoint(set2))  # False

# issubset() - True if all elements in set1 are in set2
print({1, 2}.issubset(set1))  # True

# issuperset() - True if set1 contains all elements of set2
print(set1.issuperset({1, 2}))  # True

# clear() - removes all elements
# set1.clear()
```

## 🔄 Iterating Through Collections

### Iterating Through Dictionaries
```python
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "courses": ["Math", "Science"]
}

# Iterate through keys
print("Keys:")
for key in student:
    print(f"  {key}")

# Iterate through values
print("\nValues:")
for value in student.values():
    print(f"  {value}")

# Iterate through key-value pairs
print("\nKey-Value Pairs:")
for key, value in student.items():
    print(f"  {key}: {value}")
```

### Iterating Through Sets
```python
fruits = {"apple", "banana", "orange"}

print("Fruits:")
for fruit in fruits:
    print(f"  {fruit}")
```

## 🧪 Practical Examples

### Example 1: Student Grade Management
```python
class GradeBook:
    def __init__(self):
        self.students = {}
    
    def add_student(self, name, grades=None):
        if grades is None:
            grades = []
        self.students[name] = grades
    
    def add_grade(self, name, grade):
        if name in self.students:
            self.students[name].append(grade)
        else:
            print(f"Student {name} not found")
    
    def get_average(self, name):
        if name in self.students and self.students[name]:
            return sum(self.students[name]) / len(self.students[name])
        return 0
    
    def get_top_students(self):
        averages = {name: self.get_average(name) 
                   for name in self.students 
                   if self.students[name]}
        if not averages:
            return []
        max_avg = max(averages.values())
        return [name for name, avg in averages.items() if avg == max_avg]
    
    def display_grades(self):
        for name, grades in self.students.items():
            avg = self.get_average(name)
            print(f"{name}: {grades} (Average: {avg:.2f})")

# Usage
gradebook = GradeBook()
gradebook.add_student("Alice", [85, 92, 78])
gradebook.add_student("Bob", [90, 88, 95])
gradebook.add_student("Charlie", [76, 82, 79])

gradebook.add_grade("Alice", 95)
gradebook.display_grades()

print(f"\nTop students: {gradebook.get_top_students()}")
```

### Example 2: Word Frequency Counter
```python
def count_words(text):
    """Count frequency of words in text"""
    # Remove punctuation and convert to lowercase
    import string
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Split into words
    words = text.split()
    
    # Count frequencies using dictionary
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    return word_count

def get_most_common_words(word_count, n=5):
    """Get n most common words"""
    # Sort by frequency (descending)
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:n]

# Usage
text = """
Python is a powerful programming language. Python is easy to learn.
Many developers love Python because Python is versatile and readable.
Python can be used for web development, data science, and automation.
"""

word_freq = count_words(text)
print("Word Frequencies:")
for word, count in list(word_freq.items())[:10]:
    print(f"  {word}: {count}")

print("\nMost Common Words:")
common_words = get_most_common_words(word_freq, 5)
for word, count in common_words:
    print(f"  {word}: {count}")
```

### Example 3: Duplicate Finder
```python
def find_duplicates(items):
    """Find duplicate items in a list"""
    seen = set()
    duplicates = set()
    
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    
    return list(duplicates)

def find_common_elements(list1, list2):
    """Find elements common to both lists"""
    return list(set(list1) & set(list2))

# Usage
numbers = [1, 2, 3, 2, 4, 5, 3, 6, 1]
duplicates = find_duplicates(numbers)
print(f"Original list: {numbers}")
print(f"Duplicates: {duplicates}")

list_a = [1, 2, 3, 4, 5]
list_b = [4, 5, 6, 7, 8]
common = find_common_elements(list_a, list_b)
print(f"\nList A: {list_a}")
print(f"List B: {list_b}")
print(f"Common elements: {common}")
```

## ⚠️ Common Mistakes and Tips

### 1. Dictionary Key Errors
```python
# Wrong - KeyError if key doesn't exist
student = {"name": "Alice", "age": 20}
# print(student["grade"])  # KeyError

# Correct - use get() or check key existence
print(student.get("grade", "Not specified"))
if "grade" in student:
    print(student["grade"])
```

### 2. Mutable Default Arguments
```python
# Wrong - don't use mutable default arguments
def add_student_wrong(name, grades=[]):
    grades.append(100)
    return {name: grades}

# Correct - use None as default
def add_student_correct(name, grades=None):
    if grades is None:
        grades = []
    grades.append(100)
    return {name: grades}
```

### 3. Set vs Dictionary Confusion
```python
# Empty set vs empty dictionary
empty_set = set()      # This is a set
empty_dict = {}        # This is a dictionary

# Set with elements
number_set = {1, 2, 3}  # This is a set
number_dict = {1: "one", 2: "two", 3: "three"}  # This is a dictionary
```

### 4. Unhashable Keys
```python
# Wrong - lists can't be dictionary keys (unhashable)
# data = {[1, 2]: "value"}  # TypeError

# Correct - use tuples instead
data = {(1, 2): "value"}  # This works
```

## 📚 Next Steps

Now that you understand dictionaries and sets, you're ready to learn:

1. **Functions**: Creating reusable code blocks
2. **File I/O**: Reading and writing files
3. **Exception Handling**: Managing errors gracefully
4. **Object-Oriented Programming**: Advanced programming concepts

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: When should I use a dictionary vs a list?
A: Use dictionaries when you need to associate keys with values. Use lists when you need ordered collections accessed by index.

### Q: What's the difference between a set and a list?
A: Sets contain unique elements and are unordered, while lists can contain duplicates and maintain order.

### Q: Why are set operations fast?
A: Sets are implemented using hash tables, providing O(1) average case for membership testing.

### Q: Can dictionary keys be any data type?
A: Keys must be hashable (immutable). Strings, numbers, and tuples work; lists and dictionaries don't.

---

**Practice creating and manipulating dictionaries and sets with different examples to build your skills!** 🐍