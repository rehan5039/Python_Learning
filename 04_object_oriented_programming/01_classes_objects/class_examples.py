# Classes and Objects Examples

# Basic Class Definition
print("=== Basic Class Definition ===")
class Person:
    pass

# Creating objects (instances) of the class
person1 = Person()
person2 = Person()

print(f"Type of person1: {type(person1)}")
print(f"person1 is person2: {person1 is person2}")

# Class Attributes and Methods
print("\n=== Class Attributes and Methods ===")

class Person:
    # Class attribute (shared by all instances)
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age
    
    # Instance method
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    # Instance method with parameters
    def have_birthday(self):
        self.age += 1
        return f"Happy birthday! {self.name} is now {self.age} years old."

# Creating instances
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

# Accessing class attribute
print(f"Person.species: {Person.species}")
print(f"person1.species: {person1.species}")
print(f"person2.species: {person2.species}")

# Accessing instance attributes
print(f"person1.name: {person1.name}")
print(f"person2.name: {person2.name}")

# Calling instance methods
print(person1.introduce())
print(person2.introduce())
print(person1.have_birthday())

# The `self` Parameter
print("\n=== The `self` Parameter ===")

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
print(f"calc.add(5): {calc.add(5)}")
print(f"calc.subtract(2): {calc.subtract(2)}")
print(f"calc.add(10): {calc.add(10)}")
print(f"calc.get_result(): {calc.get_result()}")

# Constructors (__init__)
print("\n=== Constructors (__init__) ===")

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

print(account1.deposit(500))
print(account2.withdraw(200))
print(account1.withdraw(100))

# Encapsulation
print("\n=== Encapsulation ===")

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
print(f"student.name: {student.name}")

# Accessing protected attribute (convention - still accessible)
print(f"student._student_id: {student._student_id}")

# Accessing private attribute (name mangling)
# print(student.__grades)  # AttributeError
print(f"student._Student__grades: {student._Student__grades}")

# Using methods to interact with private data
student.add_grade(85)
student.add_grade(92)
student.add_grade(78)
print(f"Average grade: {student.get_average():.2f}")

# Practical Example: Library Management System
print("\n=== Practical Example: Library Management System ===")

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

# Practical Example: Shape Calculator
print("\n=== Practical Example: Shape Calculator ===")

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

# Usage
shapes = [
    Rectangle(5, 3),
    Circle(4)
]

print("Shape Calculations:")
for shape in shapes:
    print(shape)

# Total area and perimeter
total_area = sum(shape.area() for shape in shapes)
total_perimeter = sum(shape.perimeter() for shape in shapes)

print(f"\nTotal Area: {total_area:.2f}")
print(f"Total Perimeter: {total_perimeter:.2f}")

# Common Mistakes Demonstration
print("\n=== Common Mistakes Demonstration ===")

# Correct way with self parameter
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."

person = Person("Alice", 25)
print(person.introduce())

# Correct way with instance attributes
class Counter:
    def __init__(self):
        self.count = 0  # Instance attribute
    
    def increment(self):
        self.count += 1

c1 = Counter()
c2 = Counter()
c1.increment()
print(f"c1.count: {c1.count}")
print(f"c2.count: {c2.count}")