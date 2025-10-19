# 📋 Lists and Tuples Practice Problems with Solutions

A comprehensive set of list and tuple manipulation problems to reinforce your understanding of Python sequences.

## 📖 How to Use These Problems

1. **Attempt each problem independently** before looking at the solutions
2. **Time yourself** to track your progress
3. **Review solutions** to understand different approaches
4. **Modify problems** to create your own variations
5. **Practice regularly** to build sequence manipulation skills

## 🎯 Problem Set: Lists and Tuples

### Problem 1: Data Processing Pipeline
Create a data processing pipeline that:
- Takes a list of numbers
- Filters out negative numbers
- Squares the remaining numbers
- Sorts them in descending order
- Returns the top N numbers

```python
def process_numbers(numbers, top_n=5):
    """
    Process a list of numbers through a pipeline
    
    Args:
        numbers (list): List of numbers to process
        top_n (int): Number of top results to return
    
    Returns:
        list: Processed numbers
    """
    # Your implementation here
    pass

# Test the function
test_numbers = [1, -2, 3, -4, 5, 6, -7, 8, 9, 10, -11, 12]
result = process_numbers(test_numbers, 5)
print(f"Top 5 processed numbers: {result}")
```

### Problem 2: Matrix Operations
Implement common matrix operations using nested lists:
- Matrix addition
- Matrix multiplication
- Transpose
- Determinant (for 2x2 and 3x3 matrices)
- Find maximum and minimum elements

```python
def matrix_add(matrix1, matrix2):
    """
    Add two matrices
    
    Args:
        matrix1 (list): First matrix
        matrix2 (list): Second matrix
    
    Returns:
        list: Resultant matrix
    """
    # Your implementation here
    pass

def matrix_multiply(matrix1, matrix2):
    """
    Multiply two matrices
    
    Args:
        matrix1 (list): First matrix
        matrix2 (list): Second matrix
    
    Returns:
        list: Resultant matrix
    """
    # Your implementation here
    pass

def matrix_transpose(matrix):
    """
    Transpose a matrix
    
    Args:
        matrix (list): Matrix to transpose
    
    Returns:
        list: Transposed matrix
    """
    # Your implementation here
    pass

# Test the functions
matrix_a = [[1, 2, 3], [4, 5, 6]]
matrix_b = [[7, 8], [9, 10], [11, 12]]
```

### Problem 3: Shopping Cart System
Create a shopping cart system using lists and tuples:
- Add items (name, price, quantity) to cart
- Remove items from cart
- Calculate total cost
- Apply discounts based on total amount
- Generate receipt

```python
class ShoppingCart:
    def __init__(self):
        """Initialize empty shopping cart"""
        # Your implementation here
        pass
    
    def add_item(self, name, price, quantity=1):
        """
        Add item to cart
        
        Args:
            name (str): Item name
            price (float): Item price
            quantity (int): Quantity to add
        """
        # Your implementation here
        pass
    
    def remove_item(self, name):
        """
        Remove item from cart
        
        Args:
            name (str): Item name to remove
        """
        # Your implementation here
        pass
    
    def calculate_total(self):
        """
        Calculate total cost of items in cart
        
        Returns:
            float: Total cost
        """
        # Your implementation here
        pass
    
    def apply_discount(self, discount_percent):
        """
        Apply discount to total
        
        Args:
            discount_percent (float): Discount percentage
        
        Returns:
            float: Discounted total
        """
        # Your implementation here
        pass
    
    def generate_receipt(self):
        """
        Generate formatted receipt
        
        Returns:
            str: Formatted receipt
        """
        # Your implementation here
        pass

# Test the class
cart = ShoppingCart()
cart.add_item("Apple", 0.50, 5)
cart.add_item("Banana", 0.30, 3)
cart.add_item("Orange", 0.75, 2)
print(cart.generate_receipt())
```

### Problem 4: Student Grade Management
Create a system to manage student grades using lists and tuples:
- Store student information (name, grades)
- Calculate average grades
- Find top and bottom performing students
- Sort students by grades
- Generate class statistics

```python
def manage_grades(students_data):
    """
    Manage student grades and generate statistics
    
    Args:
        students_data (list): List of tuples (name, [grades])
    
    Returns:
        dict: Dictionary containing statistics
    """
    # Your implementation here
    pass

def find_top_students(students_data, top_n=3):
    """
    Find top performing students
    
    Args:
        students_data (list): List of tuples (name, [grades])
        top_n (int): Number of top students to return
    
    Returns:
        list: List of top students with averages
    """
    # Your implementation here
    pass

def generate_grade_report(students_data):
    """
    Generate detailed grade report
    
    Args:
        students_data (list): List of tuples (name, [grades])
    
    Returns:
        str: Formatted report
    """
    # Your implementation here
    pass

# Test the functions
students = [
    ("Alice", [85, 92, 78, 96, 88]),
    ("Bob", [76, 84, 82, 79, 85]),
    ("Charlie", [95, 98, 92, 94, 97]),
    ("Diana", [68, 72, 75, 70, 74]),
    ("Eve", [88, 85, 90, 87, 89])
]
```

### Problem 5: Advanced List Operations
Implement advanced list operations:
- Flatten nested lists of any depth
- Rotate list elements by N positions
- Find all unique combinations of elements
- Implement custom sorting algorithms
- Find longest increasing subsequence

```python
def flatten_list(nested_list):
    """
    Flatten nested list of any depth
    
    Args:
        nested_list (list): Nested list to flatten
    
    Returns:
        list: Flattened list
    """
    # Your implementation here
    pass

def rotate_list(lst, positions):
    """
    Rotate list elements by N positions
    
    Args:
        lst (list): List to rotate
        positions (int): Number of positions to rotate
    
    Returns:
        list: Rotated list
    """
    # Your implementation here
    pass

def find_combinations(lst, r):
    """
    Find all unique combinations of r elements
    
    Args:
        lst (list): List of elements
        r (int): Number of elements in each combination
    
    Returns:
        list: List of combinations
    """
    # Your implementation here
    pass

def longest_increasing_subsequence(lst):
    """
    Find longest increasing subsequence
    
    Args:
        lst (list): List of numbers
    
    Returns:
        list: Longest increasing subsequence
    """
    # Your implementation here
    pass

# Test the functions
nested = [1, [2, 3], [4, [5, 6]], 7]
print(f"Flattened: {flatten_list(nested)}")

numbers = [1, 2, 3, 4, 5]
print(f"Rotated right by 2: {rotate_list(numbers, 2)}")

elements = ['a', 'b', 'c', 'd']
print(f"Combinations of 2: {find_combinations(elements, 2)}")

sequence = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"LIS: {longest_increasing_subsequence(sequence)}")
```

### Problem 6: Tuple-Based Data Structures
Create data structures using tuples:
- Point class using tuples (x, y coordinates)
- Color class using tuples (RGB values)
- Date class using tuples (year, month, day)
- Implement operations on these structures

```python
def create_point(x, y):
    """
    Create a point tuple
    
    Args:
        x (float): X coordinate
        y (float): Y coordinate
    
    Returns:
        tuple: Point (x, y)
    """
    # Your implementation here
    pass

def point_distance(point1, point2):
    """
    Calculate distance between two points
    
    Args:
        point1 (tuple): First point (x, y)
        point2 (tuple): Second point (x, y)
    
    Returns:
        float: Distance between points
    """
    # Your implementation here
    pass

def create_color(r, g, b):
    """
    Create a color tuple
    
    Args:
        r (int): Red component (0-255)
        g (int): Green component (0-255)
        b (int): Blue component (0-255)
    
    Returns:
        tuple: Color (r, g, b)
    """
    # Your implementation here
    pass

def mix_colors(color1, color2, ratio=0.5):
    """
    Mix two colors
    
    Args:
        color1 (tuple): First color (r, g, b)
        color2 (tuple): Second color (r, g, b)
        ratio (float): Mixing ratio (0.0 to 1.0)
    
    Returns:
        tuple: Mixed color (r, g, b)
    """
    # Your implementation here
    pass

# Test the functions
p1 = create_point(0, 0)
p2 = create_point(3, 4)
print(f"Distance: {point_distance(p1, p2)}")

red = create_color(255, 0, 0)
blue = create_color(0, 0, 255)
purple = mix_colors(red, blue)
print(f"Purple: {purple}")
```

## ✅ Solutions

### Solution 1: Data Processing Pipeline
```python
def process_numbers(numbers, top_n=5):
    """
    Process a list of numbers through a pipeline
    """
    # Filter out negative numbers
    positive_numbers = [num for num in numbers if num >= 0]
    
    # Square the remaining numbers
    squared_numbers = [num ** 2 for num in positive_numbers]
    
    # Sort in descending order
    sorted_numbers = sorted(squared_numbers, reverse=True)
    
    # Return top N numbers
    return sorted_numbers[:top_n]

# Test
test_numbers = [1, -2, 3, -4, 5, 6, -7, 8, 9, 10, -11, 12]
result = process_numbers(test_numbers, 5)
print(f"Top 5 processed numbers: {result}")
# Output: Top 5 processed numbers: [144, 100, 81, 64, 36]
```

### Solution 2: Matrix Operations
```python
def matrix_add(matrix1, matrix2):
    """
    Add two matrices
    """
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions")
    
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[0])):
            row.append(matrix1[i][j] + matrix2[i][j])
        result.append(row)
    
    return result

def matrix_multiply(matrix1, matrix2):
    """
    Multiply two matrices
    """
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in first matrix must equal number of rows in second matrix")
    
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix2[0])):
            sum_product = 0
            for k in range(len(matrix2)):
                sum_product += matrix1[i][k] * matrix2[k][j]
            row.append(sum_product)
        result.append(row)
    
    return result

def matrix_transpose(matrix):
    """
    Transpose a matrix
    """
    result = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        result.append(row)
    
    return result

# Test
matrix_a = [[1, 2, 3], [4, 5, 6]]
matrix_b = [[7, 8], [9, 10], [11, 12]]

print("Matrix A:", matrix_a)
print("Matrix B:", matrix_b)
print("A + A:", matrix_add(matrix_a, matrix_a))
print("A * B:", matrix_multiply(matrix_a, matrix_b))
print("Transpose of A:", matrix_transpose(matrix_a))
```

### Solution 3: Shopping Cart System
```python
class ShoppingCart:
    def __init__(self):
        """Initialize empty shopping cart"""
        self.items = []
    
    def add_item(self, name, price, quantity=1):
        """
        Add item to cart
        """
        # Check if item already exists
        for item in self.items:
            if item[0] == name:
                # Update quantity
                item_index = self.items.index(item)
                self.items[item_index] = (name, price, item[2] + quantity)
                return
        
        # Add new item
        self.items.append((name, price, quantity))
    
    def remove_item(self, name):
        """
        Remove item from cart
        """
        self.items = [item for item in self.items if item[0] != name]
    
    def calculate_total(self):
        """
        Calculate total cost of items in cart
        """
        return sum(item[1] * item[2] for item in self.items)
    
    def apply_discount(self, discount_percent):
        """
        Apply discount to total
        """
        total = self.calculate_total()
        discount = total * (discount_percent / 100)
        return total - discount
    
    def generate_receipt(self):
        """
        Generate formatted receipt
        """
        if not self.items:
            return "Cart is empty"
        
        receipt = "===== RECEIPT =====\n"
        total = 0
        
        for name, price, quantity in self.items:
            item_total = price * quantity
            total += item_total
            receipt += f"{name} x{quantity}: ${item_total:.2f}\n"
        
        receipt += f"-------------------\n"
        receipt += f"Total: ${total:.2f}\n"
        receipt += "==================="
        
        return receipt

# Test
cart = ShoppingCart()
cart.add_item("Apple", 0.50, 5)
cart.add_item("Banana", 0.30, 3)
cart.add_item("Orange", 0.75, 2)
print(cart.generate_receipt())
```

### Solution 4: Student Grade Management
```python
def manage_grades(students_data):
    """
    Manage student grades and generate statistics
    """
    if not students_data:
        return {}
    
    # Calculate averages
    averages = [(name, sum(grades)/len(grades)) for name, grades in students_data]
    
    # Calculate overall statistics
    all_grades = [grade for _, grades in students_data for grade in grades]
    
    return {
        'student_averages': averages,
        'class_average': sum(all_grades) / len(all_grades),
        'highest_grade': max(all_grades),
        'lowest_grade': min(all_grades),
        'total_students': len(students_data)
    }

def find_top_students(students_data, top_n=3):
    """
    Find top performing students
    """
    averages = [(name, sum(grades)/len(grades)) for name, grades in students_data]
    sorted_averages = sorted(averages, key=lambda x: x[1], reverse=True)
    return sorted_averages[:top_n]

def generate_grade_report(students_data):
    """
    Generate detailed grade report
    """
    stats = manage_grades(students_data)
    top_students = find_top_students(students_data)
    
    report = "===== STUDENT GRADE REPORT =====\n"
    report += f"Total Students: {stats['total_students']}\n"
    report += f"Class Average: {stats['class_average']:.2f}\n"
    report += f"Highest Grade: {stats['highest_grade']}\n"
    report += f"Lowest Grade: {stats['lowest_grade']}\n\n"
    
    report += "Top Performers:\n"
    for i, (name, avg) in enumerate(top_students, 1):
        report += f"{i}. {name}: {avg:.2f}\n"
    
    report += "\nAll Students:\n"
    for name, grades in students_data:
        avg = sum(grades) / len(grades)
        report += f"{name}: {avg:.2f} ({', '.join(map(str, grades))})\n"
    
    report += "==============================="
    return report

# Test
students = [
    ("Alice", [85, 92, 78, 96, 88]),
    ("Bob", [76, 84, 82, 79, 85]),
    ("Charlie", [95, 98, 92, 94, 97]),
    ("Diana", [68, 72, 75, 70, 74]),
    ("Eve", [88, 85, 90, 87, 89])
]

print(generate_grade_report(students))
```

### Solution 5: Advanced List Operations
```python
def flatten_list(nested_list):
    """
    Flatten nested list of any depth
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def rotate_list(lst, positions):
    """
    Rotate list elements by N positions
    """
    if not lst:
        return lst
    
    # Normalize positions
    positions = positions % len(lst)
    
    # Rotate right
    if positions > 0:
        return lst[-positions:] + lst[:-positions]
    # Rotate left
    else:
        return lst[abs(positions):] + lst[:abs(positions)]

def find_combinations(lst, r):
    """
    Find all unique combinations of r elements
    """
    if r == 0:
        return [[]]
    if r == 1:
        return [[item] for item in lst]
    if len(lst) < r:
        return []
    
    # Include first element
    with_first = [[lst[0]] + combo for combo in find_combinations(lst[1:], r-1)]
    # Exclude first element
    without_first = find_combinations(lst[1:], r)
    
    return with_first + without_first

def longest_increasing_subsequence(lst):
    """
    Find longest increasing subsequence
    """
    if not lst:
        return []
    
    n = len(lst)
    # dp[i] stores length of LIS ending at index i
    dp = [1] * n
    # parent[i] stores previous index in LIS ending at index i
    parent = [-1] * n
    
    # Fill dp array
    for i in range(1, n):
        for j in range(i):
            if lst[j] < lst[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # Find index of maximum value in dp
    max_length = max(dp)
    max_index = dp.index(max_length)
    
    # Reconstruct LIS
    lis = []
    current = max_index
    while current != -1:
        lis.append(lst[current])
        current = parent[current]
    
    return lis[::-1]

# Test
nested = [1, [2, 3], [4, [5, 6]], 7]
print(f"Flattened: {flatten_list(nested)}")

numbers = [1, 2, 3, 4, 5]
print(f"Rotated right by 2: {rotate_list(numbers, 2)}")

elements = ['a', 'b', 'c', 'd']
print(f"Combinations of 2: {find_combinations(elements, 2)}")

sequence = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"LIS: {longest_increasing_subsequence(sequence)}")
```

### Solution 6: Tuple-Based Data Structures
```python
def create_point(x, y):
    """
    Create a point tuple
    """
    return (x, y)

def point_distance(point1, point2):
    """
    Calculate distance between two points
    """
    import math
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def create_color(r, g, b):
    """
    Create a color tuple
    """
    # Validate RGB values
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return (r, g, b)

def mix_colors(color1, color2, ratio=0.5):
    """
    Mix two colors
    """
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    # Ensure ratio is between 0 and 1
    ratio = max(0, min(1, ratio))
    
    mixed_r = int(r1 * (1 - ratio) + r2 * ratio)
    mixed_g = int(g1 * (1 - ratio) + g2 * ratio)
    mixed_b = int(b1 * (1 - ratio) + b2 * ratio)
    
    return create_color(mixed_r, mixed_g, mixed_b)

# Test
p1 = create_point(0, 0)
p2 = create_point(3, 4)
print(f"Distance: {point_distance(p1, p2)}")

red = create_color(255, 0, 0)
blue = create_color(0, 0, 255)
purple = mix_colors(red, blue)
print(f"Purple: {purple}")

# Additional operations
def point_add(point1, point2):
    """Add two points"""
    return (point1[0] + point2[0], point1[1] + point2[1])

def scale_color(color, factor):
    """Scale color values"""
    r, g, b = color
    return create_color(int(r * factor), int(g * factor), int(b * factor))

# Test additional operations
p3 = point_add(p1, p2)
print(f"Point addition: {p1} + {p2} = {p3}")

bright_red = scale_color(red, 1.2)
print(f"Bright red: {bright_red}")
```

## 🎯 Advanced Challenge Problems

### Challenge 1: Sparse Matrix Implementation
Implement a sparse matrix class that efficiently stores matrices with many zero elements.

### Challenge 2: Custom Data Structure
Create a custom data structure that combines features of lists and dictionaries.

### Challenge 3: Graph Representation
Implement graph data structures using adjacency lists (lists of tuples).

### Challenge 4: Priority Queue
Create a priority queue implementation using lists with custom sorting.

## 📚 Tips for List and Tuple Manipulation

1. **Use list comprehensions**: More readable and often faster than loops
2. **Leverage built-in functions**: `sum()`, `max()`, `min()`, `sorted()`, etc.
3. **Understand mutability**: Lists are mutable, tuples are immutable
4. **Use slicing effectively**: Powerful way to extract and modify sublists
5. **Consider performance**: Lists for dynamic data, tuples for fixed data
6. **Use unpacking**: `a, b = (1, 2)` for clean assignment
7. **Handle edge cases**: Empty lists, single elements, nested structures

## 🎯 Summary

Lists and tuples are fundamental data structures in Python. These problems cover:
- Basic list operations and methods
- Matrix operations with nested lists
- Practical applications like shopping carts and grade management
- Advanced algorithms like flattening and LIS
- Tuple-based data structures for specific use cases

Practice these problems to become proficient in working with Python sequences!

---

**Keep practicing and exploring the power of lists and tuples in Python!** 🐍