# Advanced OOP Examples

# Class Methods and Static Methods
print("=== Class Methods and Static Methods ===")

class Person:
    population = 0
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        Person.population += 1
    
    # Instance method
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    # Class method
    @classmethod
    def get_population(cls):
        return cls.population
    
    # Class method as alternative constructor
    @classmethod
    def from_string(cls, person_str):
        name, age = person_str.split('-')
        return cls(name, int(age))
    
    # Class method to create a baby
    @classmethod
    def new_baby(cls):
        return cls("Newborn", 0)

# Usage
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

print(f"Population: {Person.get_population()}")

# Using alternative constructor
person3 = Person.from_string("Charlie-35")
print(person3.introduce())

# Using class method to create a baby
baby = Person.new_baby()
print(baby.introduce())
print(f"Population: {Person.get_population()}")

# Static Methods
print("\n=== Static Methods ===")

class MathUtils:
    @staticmethod
    def add(x, y):
        return x + y
    
    @staticmethod
    def multiply(x, y):
        return x * y
    
    @staticmethod
    def is_even(number):
        return number % 2 == 0
    
    @staticmethod
    def factorial(n):
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n - 1)

# Usage
print(MathUtils.add(5, 3))
print(MathUtils.multiply(4, 6))
print(MathUtils.is_even(10))
print(MathUtils.factorial(5))

# Properties
print("\n=== Properties ===")

class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    # Getter method
    @property
    def radius(self):
        return self._radius
    
    # Setter method
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    # Deleter method
    @radius.deleter
    def radius(self):
        print("Deleting radius")
        del self._radius
    
    # Property with only getter (read-only)
    @property
    def area(self):
        import math
        return math.pi * self._radius ** 2
    
    @property
    def diameter(self):
        return 2 * self._radius
    
    @diameter.setter
    def diameter(self, value):
        if value < 0:
            raise ValueError("Diameter cannot be negative")
        self._radius = value / 2

# Usage
circle = Circle(5)
print(f"Radius: {circle.radius}")
print(f"Area: {circle.area:.2f}")
print(f"Diameter: {circle.diameter}")

# Using setter
circle.radius = 7
print(f"New radius: {circle.radius}")

# Using diameter setter
circle.diameter = 20
print(f"Radius from diameter: {circle.radius}")

# Property Validation Example
print("\n=== Property Validation Example ===")

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
    
    @property
    def age(self):
        return self._age
    
    @age.setter
    def age(self, value):
        if not isinstance(value, int):
            raise TypeError("Age must be an integer")
        if value < 0 or value > 150:
            raise ValueError("Age must be between 0 and 150")
        self._age = value
    
    @property
    def grade(self):
        return self._grade
    
    @grade.setter
    def grade(self, value):
        valid_grades = ['A', 'B', 'C', 'D', 'F']
        if value not in valid_grades:
            raise ValueError(f"Grade must be one of {valid_grades}")
        self._grade = value
    
    @property
    def is_passing(self):
        return self._grade in ['A', 'B', 'C']

# Usage
student = Student("Alice", 20, "B")
print(f"{student.name} is {'passing' if student.is_passing else 'failing'}")

# Operator Overloading
print("\n=== Operator Overloading ===")

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # Addition operator
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented
    
    # Subtraction operator
    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented
    
    # Multiplication operator (scalar)
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented
    
    # Equality operator
    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        return False
    
    # String representation
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    # Length (magnitude) of vector
    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

# Usage
v1 = Vector(2, 3)
v2 = Vector(1, 4)

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v1 + v2: {v1 + v2}")
print(f"v1 - v2: {v1 - v2}")
print(f"v1 * 3: {v1 * 3}")
print(f"|v1|: {abs(v1):.2f}")
print(f"v1 == v2: {v1 == v2}")

# Special Methods (Dunder Methods)
print("\n=== Special Methods ===")

class CustomList:
    def __init__(self, items=None):
        self.items = items or []
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, value):
        self.items[index] = value
    
    def __delitem__(self, index):
        del self.items[index]
    
    def __iter__(self):
        return iter(self.items)
    
    def __contains__(self, item):
        return item in self.items
    
    def __reversed__(self):
        return CustomList(self.items[::-1])
    
    def append(self, item):
        self.items.append(item)
    
    def __str__(self):
        return f"CustomList({self.items})"

# Usage
clist = CustomList([1, 2, 3, 4, 5])
print(f"Length: {len(clist)}")
print(f"Item at index 2: {clist[2]}")
print(f"Contains 3: {3 in clist}")

clist[2] = 10
print(clist)

clist.append(6)
print(clist)

for item in clist:
    print(item, end=" ")
print()

reversed_clist = reversed(clist)
print(reversed_clist)

# Context Manager Protocol
print("\n=== Context Manager Protocol ===")

class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening file {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.file:
            print(f"Closing file {self.filename}")
            self.file.close()

# Usage
# Create a test file first
with open('test.txt', 'w') as f:
    f.write("Hello, World!\n")
    f.write("This is a test file.\n")

# Read the file using our context manager
with FileManager('test.txt', 'r') as file:
    content = file.read()
    print("File content:")
    print(content)

# Practical Example: Enhanced Bank Account System
print("\n=== Practical Example: Enhanced Bank Account System ===")

from abc import ABC, abstractmethod
from datetime import datetime

class Transaction:
    def __init__(self, transaction_type, amount, balance_after):
        self.transaction_type = transaction_type
        self.amount = amount
        self.balance_after = balance_after
        self.timestamp = datetime.now()
    
    def __str__(self):
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {self.transaction_type}: ${self.amount:.2f} (Balance: ${self.balance_after:.2f})"

class BankAccount(ABC):
    account_counter = 1000
    
    def __init__(self, account_holder, initial_balance=0):
        self.account_holder = account_holder
        self._balance = initial_balance
        self.account_number = BankAccount.account_counter
        BankAccount.account_counter += 1
        self.transactions = []
        self._log_transaction("Account Created", 0)
    
    @property
    def balance(self):
        return self._balance
    
    def _log_transaction(self, transaction_type, amount):
        transaction = Transaction(transaction_type, amount, self._balance)
        self.transactions.append(transaction)
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        self._log_transaction("Deposit", amount)
        return f"Deposited ${amount:.2f}. New balance: ${self._balance:.2f}"
    
    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        self._log_transaction("Withdrawal", amount)
        return f"Withdrew ${amount:.2f}. New balance: ${self._balance:.2f}"
    
    @abstractmethod
    def calculate_interest(self):
        pass
    
    def get_statement(self, limit=5):
        statement = f"\n=== Account Statement for {self.account_holder} ===\n"
        statement += f"Account Number: {self.account_number}\n"
        statement += f"Current Balance: ${self._balance:.2f}\n"
        statement += f"Total Transactions: {len(self.transactions)}\n\n"
        statement += "Recent Transactions:\n"
        
        for transaction in self.transactions[-limit:]:
            statement += f"  {transaction}\n"
        
        return statement
    
    def __str__(self):
        return f"Account {self.account_number}: {self.account_holder} - ${self._balance:.2f}"

class SavingsAccount(BankAccount):
    def __init__(self, account_holder, initial_balance=0, interest_rate=0.02):
        super().__init__(account_holder, initial_balance)
        self.interest_rate = interest_rate
    
    def calculate_interest(self):
        interest = self._balance * self.interest_rate
        self._balance += interest
        self._log_transaction("Interest Added", interest)
        return f"Interest added: ${interest:.2f}. New balance: ${self._balance:.2f}"

# Usage
savings = SavingsAccount("Alice", 1000, 0.03)
print(savings.deposit(200))
print(savings.calculate_interest())
print(savings.get_statement())

# Common Mistakes Demonstration
print("\n=== Common Mistakes Demonstration ===")

# Correct property usage
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

circle = Circle(5)
print(f"Circle radius: {circle.radius}")
circle.radius = 7
print(f"New radius: {circle.radius}")

# Cleanup - remove created files
import os
if os.path.exists('test.txt'):
    os.remove('test.txt')
    print("Removed test.txt")