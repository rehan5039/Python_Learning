# ⚡ Advanced OOP Concepts in Python

Python offers several advanced Object-Oriented Programming features that enhance the flexibility and power of your classes. This guide covers class methods, static methods, properties, operator overloading, and other advanced concepts.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Use class methods and static methods
- Implement properties with getters and setters
- Overload operators for custom classes
- Apply special methods (dunder methods)
- Understand abstract base classes
- Work with class decorators

## 🏷️ Class Methods and Static Methods

### Class Methods

Class methods are methods that are bound to the class rather than the instance. They receive the class as the first argument instead of the instance.

```python
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

print(f"Population: {Person.get_population()}")  # Population: 2

# Using alternative constructor
person3 = Person.from_string("Charlie-35")
print(person3.introduce())  # Hi, I'm Charlie and I'm 35 years old.

# Using class method to create a baby
baby = Person.new_baby()
print(baby.introduce())  # Hi, I'm Newborn and I'm 0 years old.
print(f"Population: {Person.get_population()}")  # Population: 4
```

### Static Methods

Static methods don't receive the class or instance as the first argument. They behave like regular functions but belong to the class namespace.

```python
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

# Usage - can be called on class or instance
print(MathUtils.add(5, 3))        # 8
print(MathUtils.multiply(4, 6))   # 24
print(MathUtils.is_even(10))      # True
print(MathUtils.factorial(5))     # 120

# Can also be called on instances
utils = MathUtils()
print(utils.add(2, 3))            # 5
```

## 🏠 Properties

Properties allow you to control access to instance attributes using getter, setter, and deleter methods.

### Basic Property Implementation

```python
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
print(f"Radius: {circle.radius}")    # 5
print(f"Area: {circle.area:.2f}")    # 78.54
print(f"Diameter: {circle.diameter}") # 10

# Using setter
circle.radius = 7
print(f"New radius: {circle.radius}") # 7

# Using diameter setter
circle.diameter = 20
print(f"Radius from diameter: {circle.radius}") # 10

# Using deleter
# del circle.radius  # Deleting radius
```

### Property Validation Example

```python
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

# Validation in action
try:
    student.age = -5  # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")

try:
    student.grade = "X"  # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")
```

## ➕ Operator Overloading

Operator overloading allows you to define custom behavior for operators like `+`, `-`, `*`, etc.

### Basic Operator Overloading

```python
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
    
    # Representation for debugging
    def __repr__(self):
        return f"Vector(x={self.x}, y={self.y})"
    
    # Length (magnitude) of vector
    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

# Usage
v1 = Vector(2, 3)
v2 = Vector(1, 4)

print(f"v1: {v1}")           # v1: Vector(2, 3)
print(f"v2: {v2}")           # v2: Vector(1, 4)
print(f"v1 + v2: {v1 + v2}") # v1 + v2: Vector(3, 7)
print(f"v1 - v2: {v1 - v2}") # v1 - v2: Vector(1, -1)
print(f"v1 * 3: {v1 * 3}")   # v1 * 3: Vector(6, 9)
print(f"|v1|: {abs(v1):.2f}") # |v1|: 3.61
print(f"v1 == v2: {v1 == v2}") # v1 == v2: False
```

### Advanced Operator Overloading

```python
class Matrix:
    def __init__(self, data):
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty")
        self.data = [row[:] for row in data]  # Deep copy
        self.rows = len(data)
        self.cols = len(data[0])
    
    def __add__(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have the same dimensions")
            
            result = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    row.append(self.data[i][j] + other.data[i][j])
                result.append(row)
            return Matrix(result)
        return NotImplemented
    
    def __mul__(self, other):
        # Scalar multiplication
        if isinstance(other, (int, float)):
            result = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    row.append(self.data[i][j] * other)
                result.append(row)
            return Matrix(result)
        
        # Matrix multiplication
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Number of columns in first matrix must equal number of rows in second matrix")
            
            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    sum_product = 0
                    for k in range(self.cols):
                        sum_product += self.data[i][k] * other.data[k][j]
                    row.append(sum_product)
                result.append(row)
            return Matrix(result)
        
        return NotImplemented
    
    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])
    
    def __repr__(self):
        return f"Matrix({self.data})"

# Usage
m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[5, 6], [7, 8]])

print("Matrix 1:")
print(m1)
print("\nMatrix 2:")
print(m2)

print("\nMatrix 1 + Matrix 2:")
print(m1 + m2)

print("\nMatrix 1 * 2:")
print(m1 * 2)

print("\nMatrix 1 * Matrix 2:")
print(m1 * m2)
```

## 🎭 Special Methods (Dunder Methods)

Python provides many special methods that allow you to customize the behavior of your classes.

### Container-like Behavior

```python
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
    
    def __repr__(self):
        return f"CustomList({self.items!r})"

# Usage
clist = CustomList([1, 2, 3, 4, 5])
print(f"Length: {len(clist)}")      # Length: 5
print(f"Item at index 2: {clist[2]}") # Item at index 2: 3
print(f"Contains 3: {3 in clist}")   # Contains 3: True

clist[2] = 10
print(clist)  # CustomList([1, 2, 10, 4, 5])

clist.append(6)
print(clist)  # CustomList([1, 2, 10, 4, 5, 6])

for item in clist:
    print(item, end=" ")  # 1 2 10 4 5 6
print()

reversed_clist = reversed(clist)
print(reversed_clist)  # CustomList([6, 5, 4, 10, 2, 1])
```

### Context Manager Protocol

```python
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
        
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
        
        # Return False to propagate exceptions, True to suppress them
        return False

# Usage
try:
    with FileManager('test.txt', 'w') as file:
        file.write("Hello, World!\n")
        file.write("This is a test file.\n")
        # raise ValueError("Test exception")  # Uncomment to test exception handling
except ValueError as e:
    print(f"Caught exception: {e}")

# Reading the file
with FileManager('test.txt', 'r') as file:
    content = file.read()
    print("File content:")
    print(content)
```

## 🧪 Practical Examples

### Example 1: Enhanced Bank Account System
```python
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
    
    def __eq__(self, other):
        if isinstance(other, BankAccount):
            return self.account_number == other.account_number
        return False
    
    def __lt__(self, other):
        if isinstance(other, BankAccount):
            return self._balance < other._balance
        return NotImplemented

class SavingsAccount(BankAccount):
    def __init__(self, account_holder, initial_balance=0, interest_rate=0.02):
        super().__init__(account_holder, initial_balance)
        self.interest_rate = interest_rate
    
    def calculate_interest(self):
        interest = self._balance * self.interest_rate
        self._balance += interest
        self._log_transaction("Interest Added", interest)
        return f"Interest added: ${interest:.2f}. New balance: ${self._balance:.2f}"
    
    def withdraw(self, amount):
        if amount > 1000:
            raise ValueError("Withdrawal limit exceeded for savings account")
        return super().withdraw(amount)

class CheckingAccount(BankAccount):
    def __init__(self, account_holder, initial_balance=0, overdraft_limit=100):
        super().__init__(account_holder, initial_balance)
        self.overdraft_limit = overdraft_limit
    
    def calculate_interest(self):
        # Checking accounts typically don't earn interest
        return "No interest for checking accounts"
    
    def withdraw(self, amount):
        if amount > (self._balance + self.overdraft_limit):
            raise ValueError("Withdrawal exceeds overdraft limit")
        return super().withdraw(amount)

# Usage
savings = SavingsAccount("Alice", 1000, 0.03)
checking = CheckingAccount("Bob", 500, 200)

print(savings.deposit(200))
print(checking.withdraw(600))
print(savings.calculate_interest())

print(savings.get_statement())
print(checking.get_statement())

# Comparison operators
print(f"Savings balance < Checking balance: {savings < checking}")
```

### Example 2: Custom Data Structure with Advanced Features
```python
class SmartDict:
    def __init__(self, initial_data=None):
        self._data = initial_data or {}
        self._access_count = {}  # Track access frequency
        self._history = []       # Track changes
    
    def __getitem__(self, key):
        if key in self._data:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._data[key]
        raise KeyError(key)
    
    def __setitem__(self, key, value):
        old_value = self._data.get(key, None)
        self._data[key] = value
        self._history.append(('set', key, old_value, value))
    
    def __delitem__(self, key):
        if key in self._data:
            old_value = self._data[key]
            del self._data[key]
            self._history.append(('del', key, old_value, None))
        else:
            raise KeyError(key)
    
    def __contains__(self, key):
        return key in self._data
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    
    def __str__(self):
        return f"SmartDict({self._data})"
    
    def __repr__(self):
        return f"SmartDict({self._data!r})"
    
    def __eq__(self, other):
        if isinstance(other, SmartDict):
            return self._data == other._data
        return False
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def most_accessed(self, n=3):
        """Return the n most accessed keys"""
        sorted_items = sorted(self._access_count.items(), key=lambda x: x[1], reverse=True)
        return [key for key, count in sorted_items[:n]]
    
    def history(self):
        """Return change history"""
        return self._history.copy()
    
    def undo(self):
        """Undo the last operation"""
        if not self._history:
            return "No operations to undo"
        
        operation, key, old_value, new_value = self._history.pop()
        
        if operation == 'set':
            if old_value is None:
                # This was a new key, remove it
                if key in self._data:
                    del self._data[key]
            else:
                # Restore old value
                self._data[key] = old_value
        elif operation == 'del':
            # Restore deleted key
            self._data[key] = old_value
        
        return f"Undid {operation} operation on key '{key}'"

# Usage
smart_dict = SmartDict({'a': 1, 'b': 2, 'c': 3})

# Access items
print(smart_dict['a'])  # 1
print(smart_dict['b'])  # 2
print(smart_dict['a'])  # 1 (accessed again)

# Set items
smart_dict['d'] = 4
smart_dict['a'] = 10

# Delete items
del smart_dict['b']

print(smart_dict)  # SmartDict({'a': 10, 'c': 3, 'd': 4})

# Check most accessed
print(f"Most accessed keys: {smart_dict.most_accessed()}")

# Check history
print("History:")
for op in smart_dict.history():
    print(f"  {op}")

# Undo last operation
print(smart_dict.undo())
print(smart_dict)  # SmartDict({'a': 10, 'c': 3, 'b': 2})
```

### Example 3: Mathematical Expression Evaluator
```python
class Expression:
    def __init__(self, value):
        self.value = value
    
    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(self.value + other.value)
        elif isinstance(other, (int, float)):
            return Expression(self.value + other)
        return NotImplemented
    
    def __radd__(self, other):
        # Handle cases like 5 + Expression(3)
        if isinstance(other, (int, float)):
            return Expression(other + self.value)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Expression):
            return Expression(self.value - other.value)
        elif isinstance(other, (int, float)):
            return Expression(self.value - other)
        return NotImplemented
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Expression(other - self.value)
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, Expression):
            return Expression(self.value * other.value)
        elif isinstance(other, (int, float)):
            return Expression(self.value * other)
        return NotImplemented
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Expression(other * self.value)
        return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, Expression):
            if other.value == 0:
                raise ZeroDivisionError("Division by zero")
            return Expression(self.value / other.value)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return Expression(self.value / other)
        return NotImplemented
    
    def __pow__(self, other):
        if isinstance(other, Expression):
            return Expression(self.value ** other.value)
        elif isinstance(other, (int, float)):
            return Expression(self.value ** other)
        return NotImplemented
    
    def __neg__(self):
        return Expression(-self.value)
    
    def __abs__(self):
        return Expression(abs(self.value))
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"Expression({self.value})"

# Mathematical functions
def sin(expr):
    import math
    if isinstance(expr, Expression):
        return Expression(math.sin(expr.value))
    return Expression(math.sin(expr))

def cos(expr):
    import math
    if isinstance(expr, Expression):
        return Expression(math.cos(expr.value))
    return Expression(math.cos(expr))

def sqrt(expr):
    import math
    if isinstance(expr, Expression):
        if expr.value < 0:
            raise ValueError("Cannot take square root of negative number")
        return Expression(math.sqrt(expr.value))
    if expr < 0:
        raise ValueError("Cannot take square root of negative number")
    return Expression(math.sqrt(expr))

# Usage
x = Expression(3)
y = Expression(4)

print(f"x = {x}")           # x = 3
print(f"y = {y}")           # y = 4
print(f"x + y = {x + y}")   # x + y = 7
print(f"x - y = {x - y}")   # x - y = -1
print(f"x * y = {x * y}")   # x * y = 12
print(f"x / y = {x / y}")   # x / y = 0.75
print(f"x ** 2 = {x ** 2}") # x ** 2 = 9
print(f"-x = {-x}")         # -x = -3
print(f"abs(-x) = {abs(-x)}") # abs(-x) = 3

# Mixed operations with numbers
print(f"x + 5 = {x + 5}")   # x + 5 = 8
print(f"10 - x = {10 - x}") # 10 - x = 7

# Using mathematical functions
print(f"sin(x) = {sin(x)}")     # sin(x) = 0.1411200080598672
print(f"cos(y) = {cos(y)}")     # cos(y) = -0.6536436208636119
print(f"sqrt(x) = {sqrt(x)}")   # sqrt(x) = 1.7320508075688772
```

## ⚠️ Common Mistakes and Tips

### 1. Forgetting `@property` Decorator

```python
# Wrong - missing @property
# class Circle:
#     def __init__(self, radius):
#         self._radius = radius
#     
#     def radius(self):  # Missing @property
#         return self._radius

# Correct - with @property
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
```

### 2. Incorrect Use of `@classmethod` vs `@staticmethod`

```python
class MyClass:
    class_var = "I'm a class variable"
    
    @classmethod
    def class_method(cls):
        return cls.class_var  # Can access class variables
    
    @staticmethod
    def static_method():
        # Cannot access class variables directly
        # return class_var  # This would cause an error
        return "I don't need class or instance"

# Usage
print(MyClass.class_method())  # I'm a class variable
print(MyClass.static_method()) # I don't need class or instance
```

### 3. Not Handling `NotImplemented` in Operator Overloading

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented  # Important: return NotImplemented, not raise NotImplementedError
    
    def __radd__(self, other):
        # Handle reverse addition (e.g., 5 + Vector(1, 2))
        return self.__add__(other)

# Usage
v1 = Vector(1, 2)
v2 = Vector(3, 4)
result = v1 + v2  # Works
# result = v1 + "string"  # Would return NotImplemented, Python handles it gracefully
```

## 📚 Next Steps

Now that you understand advanced OOP concepts, you're ready to learn:

1. **Design Patterns**: Common solutions to design problems
2. **Metaclasses**: Classes that create classes
3. **Descriptors**: Custom attribute access
4. **Decorators**: Advanced function and class decorators

Continue with the course to explore these concepts in depth!

## 🤔 Frequently Asked Questions

### Q: What's the difference between `@classmethod` and `@staticmethod`?
A: Class methods receive the class as the first argument (`cls`), while static methods don't receive the class or instance.

### Q: When should I use properties instead of regular methods?
A: Use properties when you want to control access to attributes but maintain a simple interface.

### Q: What does `NotImplemented` mean in operator overloading?
A: `NotImplemented` tells Python to try the reverse operation or other alternatives, while `NotImplementedError` raises an exception.

### Q: How do I make my class work with `with` statements?
A: Implement the `__enter__` and `__exit__` methods to make your class a context manager.

---

**Practice implementing advanced OOP concepts with different scenarios to build your expertise!** 🐍