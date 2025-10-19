# 🏗️ Classes and Objects in Python

Object-Oriented Programming (OOP) is a programming paradigm that organizes code around objects and classes. This approach helps create more modular, reusable, and maintainable code. This guide will teach you how to create and use classes and objects effectively in Python.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Define and create classes
- Instantiate objects from classes
- Use attributes and methods
- Understand the concept of `self`
- Work with constructors (`__init__`)
- Apply encapsulation principles

## 📦 What are Classes and Objects?

### Classes
A class is a blueprint or template for creating objects. It defines the properties (attributes) and behaviors (methods) that objects of that class will have.

### Objects
An object is an instance of a class. It's a concrete entity based on the class blueprint, with its own set of attribute values.

### Basic Class Definition

```python
# Basic class definition
class Person:
    pass

# Creating an object (instance) of the class
person1 = Person()
person2 = Person()

print(type(person1))  # <class '__main__.Person'>
print(person1 is person2)  # False (different objects)
```

## 🏗️ Class Attributes and Methods

### Class Attributes
Class attributes are variables that belong to the class itself, shared by all instances.

```python
class Person:
    # Class attribute (shared by all instances)
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age

# Creating instances
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

# Accessing class attribute
print(Person.species)    # Homo sapiens
print(person1.species)   # Homo sapiens
print(person2.species)   # Homo sapiens

# Accessing instance attributes
print(person1.name)      # Alice
print(person2.name)      # Bob
```

### Instance Methods
Instance methods are functions defined inside a class that operate on instances of that class.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    # Instance method
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    # Instance method with parameters
    def have_birthday(self):
        self.age += 1
        return f"Happy birthday! {self.name} is now {self.age} years old."

# Creating an instance
person = Person("Alice", 25)

# Calling instance methods
print(person.introduce())        # Hi, I'm Alice and I'm 25 years old.
print(person.have_birthday())    # Happy birthday! Alice is now 26 years old.
```

## 🔧 The `self` Parameter

The `self` parameter refers to the instance of the class and is used to access attributes and methods within the class.

```python
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, number):
        self.result += number
        return self.result
    
    def subtract(self, number):
        self.result -= number
        return self.result
    
    def get_result(self):
        return self.result

# Creating an instance
calc = Calculator()

# Using methods
print(calc.add(5))        # 5
print(calc.subtract(2))   # 3
print(calc.add(10))       # 13
print(calc.get_result())  # 13

# The self parameter is automatically passed
# calc.add(5) is equivalent to Calculator.add(calc, 5)
```

## 🏗️ Constructors (`__init__`)

The `__init__` method is a special method called a constructor. It's automatically called when creating a new instance.

```python
class BankAccount:
    def __init__(self, account_holder, initial_balance=0):
        self.account_holder = account_holder
        self.balance = initial_balance
        self.transaction_history = []
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            self.transaction_history.append(f"Deposited ${amount}")
            return f"Deposited ${amount}. New balance: ${self.balance}"
        else:
            return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            self.transaction_history.append(f"Withdrew ${amount}")
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        else:
            return "Invalid withdrawal amount or insufficient funds"

# Creating instances with different parameters
account1 = BankAccount("Alice")
account2 = BankAccount("Bob", 1000)

print(account1.deposit(500))   # Deposited $500. New balance: $500
print(account2.withdraw(200))  # Withdrew $200. New balance: $800
```

## 🔒 Encapsulation

Encapsulation is the concept of bundling data and methods that operate on that data within a single unit (class).

### Public, Protected, and Private Attributes

```python
class Student:
    def __init__(self, name, student_id):
        self.name = name                    # Public attribute
        self._student_id = student_id       # Protected attribute (convention)
        self.__grades = []                  # Private attribute (name mangling)
    
    def add_grade(self, grade):
        if 0 <= grade <= 100:
            self.__grades.append(grade)
        else:
            print("Invalid grade. Must be between 0 and 100.")
    
    def get_average(self):
        if not self.__grades:
            return 0
        return sum(self.__grades) / len(self.__grades)
    
    def _protected_method(self):
        return "This is a protected method"
    
    def __private_method(self):
        return "This is a private method"

# Creating an instance
student = Student("Alice", "S12345")

# Accessing public attribute
print(student.name)  # Alice

# Accessing protected attribute (convention - still accessible)
print(student._student_id)  # S12345

# Accessing private attribute (name mangling)
# print(student.__grades)  # AttributeError
print(student._Student__grades)  # [] (access through name mangling)

# Using methods to interact with private data
student.add_grade(85)
student.add_grade(92)
student.add_grade(78)
print(f"Average grade: {student.get_average():.2f}")  # Average grade: 85.00
```

## 🧪 Practical Examples

### Example 1: Library Management System
```python
class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_borrowed = False
        self.borrowed_by = None
    
    def borrow(self, borrower_name):
        if not self.is_borrowed:
            self.is_borrowed = True
            self.borrowed_by = borrower_name
            return f"'{self.title}' borrowed by {borrower_name}"
        else:
            return f"'{self.title}' is already borrowed by {self.borrowed_by}"
    
    def return_book(self):
        if self.is_borrowed:
            borrower = self.borrowed_by
            self.is_borrowed = False
            self.borrowed_by = None
            return f"'{self.title}' returned by {borrower}"
        else:
            return f"'{self.title}' was not borrowed"
    
    def __str__(self):
        status = f"Borrowed by {self.borrowed_by}" if self.is_borrowed else "Available"
        return f"'{self.title}' by {self.author} (ISBN: {self.isbn}) - {status}"

class Library:
    def __init__(self, name):
        self.name = name
        self.books = []
    
    def add_book(self, book):
        self.books.append(book)
        return f"Added '{book.title}' to {self.name}"
    
    def find_book(self, title):
        for book in self.books:
            if book.title.lower() == title.lower():
                return book
        return None
    
    def list_books(self):
        if not self.books:
            return "No books in library"
        
        available = [book for book in self.books if not book.is_borrowed]
        borrowed = [book for book in self.books if book.is_borrowed]
        
        result = f"\n=== {self.name} ===\n"
        result += f"Total books: {len(self.books)}\n"
        result += f"Available: {len(available)}\n"
        result += f"Borrowed: {len(borrowed)}\n\n"
        
        if available:
            result += "Available books:\n"
            for book in available:
                result += f"  {book}\n"
        
        if borrowed:
            result += "\nBorrowed books:\n"
            for book in borrowed:
                result += f"  {book}\n"
        
        return result

# Usage
library = Library("City Library")

# Adding books
book1 = Book("Python Programming", "John Smith", "978-1234567890")
book2 = Book("Data Structures", "Jane Doe", "978-0987654321")
book3 = Book("Machine Learning", "Bob Johnson", "978-1111111111")

library.add_book(book1)
library.add_book(book2)
library.add_book(book3)

print(library.list_books())

# Borrowing books
print(book1.borrow("Alice"))
print(book2.borrow("Bob"))
print(book1.borrow("Charlie"))  # Already borrowed

print(library.list_books())

# Returning books
print(book1.return_book())
print(book1.return_book())  # Already returned

print(library.list_books())
```

### Example 2: Shape Calculator
```python
import math

class Shape:
    def __init__(self, name):
        self.name = name
    
    def area(self):
        raise NotImplementedError("Subclass must implement area method")
    
    def perimeter(self):
        raise NotImplementedError("Subclass must implement perimeter method")
    
    def __str__(self):
        return f"{self.name}: Area = {self.area():.2f}, Perimeter = {self.perimeter():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        return 2 * math.pi * self.radius

class Triangle(Shape):
    def __init__(self, side_a, side_b, side_c):
        super().__init__("Triangle")
        self.side_a = side_a
        self.side_b = side_b
        self.side_c = side_c
    
    def area(self):
        # Using Heron's formula
        s = self.perimeter() / 2
        return math.sqrt(s * (s - self.side_a) * (s - self.side_b) * (s - self.side_c))
    
    def perimeter(self):
        return self.side_a + self.side_b + self.side_c

# Usage
shapes = [
    Rectangle(5, 3),
    Circle(4),
    Triangle(3, 4, 5)
]

print("Shape Calculations:")
for shape in shapes:
    print(shape)

# Total area and perimeter
total_area = sum(shape.area() for shape in shapes)
total_perimeter = sum(shape.perimeter() for shape in shapes)

print(f"\nTotal Area: {total_area:.2f}")
print(f"Total Perimeter: {total_perimeter:.2f}")
```

### Example 3: Student Grade Management
```python
class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.grades = {}
    
    def add_grade(self, subject, grade):
        if subject not in self.grades:
            self.grades[subject] = []
        self.grades[subject].append(grade)
    
    def get_subject_average(self, subject):
        if subject not in self.grades or not self.grades[subject]:
            return 0
        return sum(self.grades[subject]) / len(self.grades[subject])
    
    def get_overall_average(self):
        if not self.grades:
            return 0
        
        total_sum = sum(sum(grades) for grades in self.grades.values())
        total_count = sum(len(grades) for grades in self.grades.values())
        return total_sum / total_count if total_count > 0 else 0
    
    def get_grade_report(self):
        report = f"\n=== Grade Report for {self.name} (ID: {self.student_id}) ===\n"
        
        if not self.grades:
            report += "No grades recorded yet.\n"
            return report
        
        for subject, grades in self.grades.items():
            average = sum(grades) / len(grades)
            report += f"{subject}: {grades} (Average: {average:.2f})\n"
        
        overall_avg = self.get_overall_average()
        report += f"\nOverall Average: {overall_avg:.2f}\n"
        
        # Letter grade
        if overall_avg >= 90:
            letter = "A"
        elif overall_avg >= 80:
            letter = "B"
        elif overall_avg >= 70:
            letter = "C"
        elif overall_avg >= 60:
            letter = "D"
        else:
            letter = "F"
        
        report += f"Letter Grade: {letter}\n"
        return report

class Gradebook:
    def __init__(self):
        self.students = {}
    
    def add_student(self, student):
        self.students[student.student_id] = student
    
    def get_student(self, student_id):
        return self.students.get(student_id)
    
    def class_average(self):
        if not self.students:
            return 0
        
        total_avg = sum(student.get_overall_average() for student in self.students.values())
        return total_avg / len(self.students)
    
    def top_students(self, n=5):
        sorted_students = sorted(
            self.students.values(),
            key=lambda s: s.get_overall_average(),
            reverse=True
        )
        return sorted_students[:n]

# Usage
gradebook = Gradebook()

# Adding students
alice = Student("Alice Johnson", "S001")
bob = Student("Bob Smith", "S002")
charlie = Student("Charlie Brown", "S003")

gradebook.add_student(alice)
gradebook.add_student(bob)
gradebook.add_student(charlie)

# Adding grades
alice.add_grade("Math", 85)
alice.add_grade("Math", 92)
alice.add_grade("Science", 78)
alice.add_grade("Science", 88)
alice.add_grade("English", 95)

bob.add_grade("Math", 75)
bob.add_grade("Math", 80)
bob.add_grade("Science", 82)
bob.add_grade("Science", 79)
bob.add_grade("English", 88)

charlie.add_grade("Math", 95)
charlie.add_grade("Math", 98)
charlie.add_grade("Science", 92)
charlie.add_grade("Science", 94)
charlie.add_grade("English", 97)

# Printing reports
print(alice.get_grade_report())
print(bob.get_grade_report())
print(charlie.get_grade_report())

print(f"\nClass Average: {gradebook.class_average():.2f}")

print("\nTop Students:")
for i, student in enumerate(gradebook.top_students(), 1):
    print(f"{i}. {student.name}: {student.get_overall_average():.2f}")
```

## ⚠️ Common Mistakes and Tips

### 1. Forgetting `self` Parameter

```python
# Wrong - missing self parameter
# class Person:
#     def __init__(name, age):  # Missing self
#         self.name = name
#         self.age = age

# Correct - including self parameter
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

### 2. Confusing Class and Instance Attributes

```python
# Wrong - modifying class attribute affects all instances
# class Counter:
#     count = 0  # Class attribute
#     
#     def increment(self):
#         Counter.count += 1  # Modifying class attribute

# Correct - using instance attribute
class Counter:
    def __init__(self):
        self.count = 0  # Instance attribute
    
    def increment(self):
        self.count += 1

# Testing
c1 = Counter()
c2 = Counter()
c1.increment()
print(c1.count)  # 1
print(c2.count)  # 0 (independent)
```

### 3. Not Calling Parent Constructor

```python
# Wrong - not calling parent __init__
# class Animal:
#     def __init__(self, name):
#         self.name = name
# 
# class Dog(Animal):
#     def __init__(self, name, breed):
#         # Forgot to call parent __init__
#         self.breed = breed

# Correct - calling parent __init__
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Call parent constructor
        self.breed = breed
```

### 4. Mutable Default Arguments

```python
# Wrong - mutable default argument
# class Classroom:
#     def __init__(self, students=[]):  # Dangerous!
#         self.students = students

# Correct - using None as default
class Classroom:
    def __init__(self, students=None):
        if students is None:
            students = []
        self.students = students
```

## 📚 Next Steps

Now that you understand classes and objects, you're ready to learn:

1. **Inheritance**: Creating specialized classes from existing ones
2. **Polymorphism**: Using the same interface for different data types
3. **Advanced OOP Concepts**: Class methods, static methods, properties
4. **Design Patterns**: Common solutions to design problems

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: What is the difference between a class and an object?
A: A class is a blueprint or template, while an object is an instance of a class with actual values.

### Q: Why do we need the `self` parameter?
A: `self` refers to the instance of the class and allows access to its attributes and methods.

### Q: What's the difference between `__init__` and `__new__`?
A: `__new__` creates the object, while `__init__` initializes it. `__init__` is used more commonly.

### Q: When should I use class attributes vs instance attributes?
A: Use class attributes for data shared by all instances, and instance attributes for data unique to each instance.

---

**Practice creating different classes and objects with various scenarios to build your OOP skills!** 🐍