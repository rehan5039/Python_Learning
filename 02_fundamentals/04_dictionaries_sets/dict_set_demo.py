# Dictionaries and Sets Demo

# Creating Dictionaries
print("=== Creating Dictionaries ===")
empty_dict = {}
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "courses": ["Math", "Science", "English"]
}
person = dict(name="Bob", age=25, city="New York")

print(f"Empty dictionary: {empty_dict}")
print(f"Student dictionary: {student}")
print(f"Person dictionary: {person}")

# Dictionary Operations
print("\n=== Dictionary Operations ===")
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A"
}

# Accessing values
print(f"Name: {student['name']}")
print(f"Age: {student.get('age')}")
print(f"Height: {student.get('height', 'Not specified')}")

# Adding and modifying values
student["courses"] = ["Math", "Science"]
student["age"] = 21
student.update({"city": "Boston", "year": 2023})
print(f"Updated student: {student}")

# Removing items
grade = student.pop("grade")
print(f"Removed grade: {grade}")
print(f"Student after pop: {student}")

# Dictionary Methods
print("\n=== Dictionary Methods ===")
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "courses": ["Math", "Science"]
}

print(f"Keys: {list(student.keys())}")
print(f"Values: {list(student.values())}")
print(f"Items: {list(student.items())}")

# Dictionary Comprehensions
print("\n=== Dictionary Comprehensions ===")
squares = {x: x**2 for x in range(1, 6)}
print(f"Squares: {squares}")

even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(f"Even squares: {even_squares}")

keys = ["name", "age", "city"]
values = ["Bob", 25, "New York"]
person = {k: v for k, v in zip(keys, values)}
print(f"Person from lists: {person}")

# Creating Sets
print("\n=== Creating Sets ===")
empty_set = set()
fruits = {"apple", "banana", "orange"}
numbers = set([1, 2, 2, 3, 3, 4, 5])
vowels = set("aeiou")

print(f"Empty set: {empty_set}")
print(f"Fruits set: {fruits}")
print(f"Numbers set (duplicates removed): {numbers}")
print(f"Vowels set: {vowels}")

# Set Operations
print("\n=== Set Operations ===")
fruits = {"apple", "banana"}
print(f"Initial fruits: {fruits}")

# Adding elements
fruits.add("orange")
print(f"After add('orange'): {fruits}")

fruits.update(["grape", "kiwi"])
print(f"After update(['grape', 'kiwi']): {fruits}")

# Removing elements
fruits.remove("banana")
print(f"After remove('banana'): {fruits}")

fruits.discard("pineapple")  # No error if not found
print(f"After discard('pineapple'): {fruits}")

# Set Operations - Mathematical
print("\n=== Set Operations - Mathematical ===")
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

print(f"Set 1: {set1}")
print(f"Set 2: {set2}")

union = set1 | set2
print(f"Union: {union}")

intersection = set1 & set2
print(f"Intersection: {intersection}")

difference = set1 - set2
print(f"Difference (set1 - set2): {difference}")

symmetric_diff = set1 ^ set2
print(f"Symmetric Difference: {symmetric_diff}")

# Set Methods
print("\n=== Set Methods ===")
set1 = {1, 2, 3}
set2 = {3, 4, 5}

print(f"Set 1: {set1}")
print(f"Set 2: {set2}")
print(f"Is disjoint: {set1.isdisjoint(set2)}")
print(f"Is subset: {{1, 2}}.issubset(set1) = {{1, 2}}.issubset(set1)")
print(f"Is superset: {set1.issuperset({1, 2})}")

# Iterating Through Collections
print("\n=== Iterating Through Collections ===")
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "courses": ["Math", "Science"]
}

print("Dictionary iteration:")
for key, value in student.items():
    print(f"  {key}: {value}")

fruits = {"apple", "banana", "orange"}
print("\nSet iteration:")
for fruit in fruits:
    print(f"  {fruit}")

# Practical Example: Word Frequency Counter
print("\n=== Practical Example: Word Frequency Counter ===")
def count_words(text):
    """Count frequency of words in text"""
    import string
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = text.split()
    
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    return word_count

text = "Python is great. Python is powerful. I love Python programming."
word_freq = count_words(text)
print(f"Text: {text}")
print("Word frequencies:")
for word, count in word_freq.items():
    print(f"  {word}: {count}")

# Practical Example: Duplicate Finder
print("\n=== Practical Example: Duplicate Finder ===")
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

numbers = [1, 2, 3, 2, 4, 5, 3, 6, 1]
duplicates = find_duplicates(numbers)
print(f"Original list: {numbers}")
print(f"Duplicates: {duplicates}")

# Common Mistakes Demonstration
print("\n=== Common Mistakes ===")

# KeyError demonstration
student = {"name": "Alice", "age": 20}
print("Safe key access:")
print(f"Grade: {student.get('grade', 'Not specified')}")

# Set vs Dictionary confusion
print("\nSet vs Dictionary:")
number_set = {1, 2, 3}  # This is a set
number_dict = {1: "one", 2: "two", 3: "three"}  # This is a dictionary
print(f"Set: {number_set}")
print(f"Dictionary: {number_dict}")