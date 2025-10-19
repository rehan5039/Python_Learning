# 📋 Lists and Tuples in Python

Lists and tuples are fundamental data structures in Python used to store collections of items. This guide will teach you how to work with both effectively.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Create and manipulate lists and tuples
- Understand the differences between lists and tuples
- Use indexing and slicing with collections
- Apply built-in methods for lists
- Iterate through collections

## 📝 What are Lists?

Lists are ordered, mutable collections of items. They can contain elements of different data types and can be modified after creation.

### Creating Lists

```python
# Empty list
empty_list = []

# List with elements
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "orange"]
mixed = [1, "hello", 3.14, True]

# Using list() constructor
another_list = list((1, 2, 3, 4))

print(numbers)  # [1, 2, 3, 4, 5]
print(fruits)   # ['apple', 'banana', 'orange']
print(mixed)    # [1, 'hello', 3.14, True]
```

### List Characteristics
- **Ordered**: Elements maintain their position
- **Mutable**: Can be changed after creation
- **Allow duplicates**: Same value can appear multiple times
- **Indexed**: Access elements by position
- **Dynamic**: Can grow or shrink in size

## 📝 What are Tuples?

Tuples are ordered, immutable collections of items. Once created, their elements cannot be changed.

### Creating Tuples

```python
# Empty tuple
empty_tuple = ()

# Tuple with elements
coordinates = (10, 20)
colors = ("red", "green", "blue")
mixed_tuple = (1, "hello", 3.14, True)

# Single element tuple (note the comma)
single_element = (42,)

# Using tuple() constructor
another_tuple = tuple([1, 2, 3, 4])

print(coordinates)   # (10, 20)
print(colors)        # ('red', 'green', 'blue')
print(single_element) # (42,)
```

### Tuple Characteristics
- **Ordered**: Elements maintain their position
- **Immutable**: Cannot be changed after creation
- **Allow duplicates**: Same value can appear multiple times
- **Indexed**: Access elements by position
- **Memory efficient**: Use less memory than lists

## 🔢 Indexing and Slicing

Both lists and tuples support indexing and slicing:

```python
# Indexing
fruits = ["apple", "banana", "orange", "grape"]
print(fruits[0])    # apple
print(fruits[-1])   # grape

# Slicing
print(fruits[1:3])  # ['banana', 'orange']
print(fruits[:2])   # ['apple', 'banana']
print(fruits[2:])   # ['orange', 'grape']

# Tuples work the same way
coordinates = (10, 20, 30, 40)
print(coordinates[1])   # 20
print(coordinates[1:3]) # (20, 30)
```

## 🛠 List Methods

Lists have many built-in methods for manipulation:

### Adding Elements
```python
fruits = ["apple", "banana"]

# append() - adds element to the end
fruits.append("orange")
print(fruits)  # ['apple', 'banana', 'orange']

# insert() - adds element at specific position
fruits.insert(1, "grape")
print(fruits)  # ['apple', 'grape', 'banana', 'orange']

# extend() - adds multiple elements
fruits.extend(["kiwi", "mango"])
print(fruits)  # ['apple', 'grape', 'banana', 'orange', 'kiwi', 'mango']
```

### Removing Elements
```python
fruits = ["apple", "banana", "orange", "grape", "kiwi"]

# remove() - removes first occurrence of element
fruits.remove("banana")
print(fruits)  # ['apple', 'orange', 'grape', 'kiwi']

# pop() - removes and returns element at index (default: last)
last_fruit = fruits.pop()
print(last_fruit)  # kiwi
print(fruits)      # ['apple', 'orange', 'grape']

# pop() with index
second_fruit = fruits.pop(1)
print(second_fruit)  # orange
print(fruits)        # ['apple', 'grape']

# clear() - removes all elements
fruits.clear()
print(fruits)  # []
```

### Other List Methods
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# count() - counts occurrences of element
print(numbers.count(1))  # 2

# index() - finds index of first occurrence
print(numbers.index(5))  # 4

# sort() - sorts the list
numbers.sort()
print(numbers)  # [1, 1, 2, 3, 4, 5, 6, 9]

# reverse() - reverses the list
numbers.reverse()
print(numbers)  # [9, 6, 5, 4, 3, 2, 1, 1]

# copy() - creates a copy of the list
original = [1, 2, 3]
copy_list = original.copy()
print(copy_list)  # [1, 2, 3]
```

## 🔁 List Operations

### Concatenation and Repetition
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# Concatenation
combined = list1 + list2
print(combined)  # [1, 2, 3, 4, 5, 6]

# Repetition
repeated = list1 * 3
print(repeated)  # [1, 2, 3, 1, 2, 3, 1, 2, 3]
```

### Membership Testing
```python
fruits = ["apple", "banana", "orange"]

print("apple" in fruits)    # True
print("grape" in fruits)    # False
print("grape" not in fruits) # True
```

### Length and Other Functions
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

print(len(numbers))  # 8
print(max(numbers))  # 9
print(min(numbers))  # 1
print(sum(numbers))  # 31
```

## 🔄 List vs Tuple Comparison

| Feature | List | Tuple |
|---------|------|-------|
| Mutability | Mutable | Immutable |
| Syntax | Square brackets `[]` | Parentheses `()` |
| Performance | Slower | Faster |
| Memory Usage | More | Less |
| Built-in Methods | Many | Few |
| Use Case | When data changes | When data is constant |

### When to Use Lists vs Tuples

```python
# Use lists when you need to modify data
shopping_list = ["milk", "bread", "eggs"]
shopping_list.append("cheese")  # This works

# Use tuples for constant data
coordinates = (10, 20)  # Coordinates don't change
rgb_color = (255, 128, 0)  # RGB values are constant

# Tuples can be dictionary keys (lists cannot)
locations = {
    (0, 0): "origin",
    (10, 20): "point A",
    (30, 40): "point B"
}
```

## 🔄 Iterating Through Collections

### For Loops
```python
fruits = ["apple", "banana", "orange"]

# Simple iteration
for fruit in fruits:
    print(fruit)

# With index using enumerate()
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# Using range and len()
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")
```

### List Comprehensions
```python
# Basic list comprehension
squares = [x**2 for x in range(1, 6)]
print(squares)  # [1, 4, 9, 16, 25]

# With condition
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(even_squares)  # [4, 16, 36, 64, 100]

# With transformation
fruits = ["apple", "banana", "cherry"]
uppercase_fruits = [fruit.upper() for fruit in fruits]
print(uppercase_fruits)  # ['APPLE', 'BANANA', 'CHERRY']
```

## 🧪 Practical Examples

### Example 1: Shopping Cart
```python
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, item, price):
        self.items.append({"item": item, "price": price})
    
    def remove_item(self, item):
        self.items = [i for i in self.items if i["item"] != item]
    
    def total_cost(self):
        return sum(item["price"] for item in self.items)
    
    def display_cart(self):
        if not self.items:
            print("Cart is empty")
            return
        
        print("Shopping Cart:")
        for item in self.items:
            print(f"  {item['item']}: ${item['price']:.2f}")
        print(f"Total: ${self.total_cost():.2f}")

# Usage
cart = ShoppingCart()
cart.add_item("Apple", 0.99)
cart.add_item("Banana", 0.59)
cart.add_item("Orange", 1.29)
cart.display_cart()
```

### Example 2: Grade Calculator
```python
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

# Usage
student_scores = [85, 92, 78, 96, 88, 91, 87]
stats = calculate_grades(student_scores)
print("Grade Statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key.capitalize()}: {value:.2f}")
    else:
        print(f"  {key.capitalize()}: {value}")
```

### Example 3: Data Processing
```python
def process_temperature_data(temperatures):
    """Process temperature data and find anomalies"""
    if not temperatures:
        return []
    
    average = sum(temperatures) / len(temperatures)
    threshold = average * 0.2  # 20% deviation threshold
    
    anomalies = []
    for i, temp in enumerate(temperatures):
        if abs(temp - average) > threshold:
            anomalies.append({
                "index": i,
                "temperature": temp,
                "deviation": temp - average
            })
    
    return anomalies

# Usage
weekly_temps = [72, 75, 73, 95, 74, 71, 69]
anomalies = process_temperature_data(weekly_temps)
print("Temperature Anomalies:")
for anomaly in anomalies:
    print(f"  Day {anomaly['index'] + 1}: {anomaly['temperature']}°F "
          f"(Deviation: {anomaly['deviation']:+.1f}°F)")
```

## ⚠️ Common Mistakes and Tips

### 1. Mutable Default Arguments
```python
# Wrong - don't use mutable default arguments
def add_item_wrong(item, items=[]):
    items.append(item)
    return items

# Correct - use None as default
def add_item_correct(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### 2. Shallow Copy vs Deep Copy
```python
# Shallow copy issue
original = [[1, 2], [3, 4]]
copied = original.copy()  # Shallow copy
original[0][0] = 99
print(copied[0][0])  # 99 (changed!)

# Deep copy solution
import copy
original = [[1, 2], [3, 4]]
deep_copied = copy.deepcopy(original)
original[0][0] = 99
print(deep_copied[0][0])  # 1 (unchanged)
```

### 3. Modifying List While Iterating
```python
# Wrong - don't modify list while iterating
numbers = [1, 2, 3, 4, 5]
# for num in numbers:
#     if num % 2 == 0:
#         numbers.remove(num)  # This can cause issues

# Correct - iterate over a copy or use list comprehension
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]
print(numbers)  # [1, 3, 5]
```

## 📚 Next Steps

Now that you understand lists and tuples, you're ready to learn:

1. **Dictionaries and Sets**: More complex data structures
2. **Functions**: Creating reusable code blocks
3. **File I/O**: Reading and writing files
4. **List Comprehensions**: Advanced list manipulation techniques

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: When should I use a list vs a tuple?
A: Use lists when you need to modify data, use tuples for constant data.

### Q: Can I change a tuple after creation?
A: No, tuples are immutable. You would need to create a new tuple.

### Q: Why are tuples faster than lists?
A: Tuples are immutable, so Python can optimize their storage and access.

### Q: Can tuples contain mutable objects?
A: Yes, but you can't change which objects the tuple contains, only modify the mutable objects themselves.

---

**Practice creating and manipulating lists and tuples with different examples to build your skills!** 🐍