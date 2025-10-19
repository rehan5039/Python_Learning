# Inheritance Examples

# Basic Inheritance
print("=== Basic Inheritance ===")

class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "Some generic animal sound"
    
    def info(self):
        return f"{self.name} is a {self.species}"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")
        self.breed = breed
    
    # Override parent method
    def make_sound(self):
        return "Woof! Woof!"
    
    # Add new method
    def fetch(self):
        return f"{self.name} is fetching the ball!"

# Creating instances
generic_animal = Animal("Generic", "Unknown")
dog = Dog("Buddy", "Golden Retriever")

print(generic_animal.info())
print(generic_animal.make_sound())

print(dog.info())
print(dog.make_sound())
print(dog.fetch())

# The `super()` Function
print("\n=== The `super()` Function ===")

class Vehicle:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
        self.mileage = 0
    
    def info(self):
        return f"{self.year} {self.brand} {self.model}"
    
    def drive(self, miles):
        self.mileage += miles
        return f"Drove {miles} miles. Total mileage: {self.mileage}"

class Car(Vehicle):
    def __init__(self, brand, model, year, doors):
        super().__init__(brand, model, year)
        self.doors = doors
    
    def info(self):
        base_info = super().info()
        return f"{base_info} with {self.doors} doors"

class ElectricCar(Car):
    def __init__(self, brand, model, year, doors, battery_capacity):
        super().__init__(brand, model, year, doors)
        self.battery_capacity = battery_capacity
        self.battery_level = 100
    
    def drive(self, miles):
        if self.battery_level <= 0:
            return "Battery is empty! Please charge."
        
        result = super().drive(miles)
        self.battery_level -= miles * 0.5
        return f"{result} Battery level: {self.battery_level}%"
    
    def charge(self):
        self.battery_level = 100
        self.mileage = 0
        return "Car charged successfully!"

# Usage
car = Car("Toyota", "Camry", 2022, 4)
print(car.info())
print(car.drive(50))

electric_car = ElectricCar("Tesla", "Model 3", 2023, 4, 75)
print(electric_car.info())
print(electric_car.drive(20))
print(electric_car.charge())

# Method Overriding
print("\n=== Method Overriding ===")

class Shape:
    def __init__(self, name):
        self.name = name
    
    def area(self):
        raise NotImplementedError("Subclass must implement area method")
    
    def perimeter(self):
        raise NotImplementedError("Subclass must implement perimeter method")
    
    def describe(self):
        return f"This is a {self.name}"

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
    
    def describe(self):
        base_description = super().describe()
        return f"{base_description} with area {self.area()} and perimeter {self.perimeter()}"

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        import math
        return 2 * math.pi * self.radius
    
    def describe(self):
        base_description = super().describe()
        return f"{base_description} with area {self.area():.2f} and perimeter {self.perimeter():.2f}"

# Usage
shapes = [
    Rectangle(5, 3),
    Circle(4)
]

for shape in shapes:
    print(shape.describe())

# Calling Parent Methods from Overridden Methods
print("\n=== Calling Parent Methods from Overridden Methods ===")

class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    
    def get_info(self):
        return f"Employee: {self.name}, Salary: ${self.salary}"
    
    def calculate_bonus(self):
        return self.salary * 0.05

class Manager(Employee):
    def __init__(self, name, salary, department):
        super().__init__(name, salary)
        self.department = department
    
    def get_info(self):
        base_info = super().get_info()
        return f"{base_info}, Department: {self.department}"
    
    def calculate_bonus(self):
        base_bonus = super().calculate_bonus()
        manager_bonus = self.salary * 0.10
        return base_bonus + manager_bonus

class Developer(Employee):
    def __init__(self, name, salary, programming_languages):
        super().__init__(name, salary)
        self.programming_languages = programming_languages
    
    def get_info(self):
        base_info = super().get_info()
        return f"{base_info}, Languages: {', '.join(self.programming_languages)}"
    
    def calculate_bonus(self):
        base_bonus = super().calculate_bonus()
        skill_bonus = len(self.programming_languages) * 100
        return base_bonus + skill_bonus

# Usage
manager = Manager("Alice", 80000, "Engineering")
developer = Developer("Bob", 70000, ["Python", "JavaScript", "Java"])

print(manager.get_info())
print(f"Manager Bonus: ${manager.calculate_bonus()}")

print(developer.get_info())
print(f"Developer Bonus: ${developer.calculate_bonus()}")

# Multiple Inheritance
print("\n=== Multiple Inheritance ===")

class Flyer:
    def __init__(self):
        self.altitude = 0
    
    def fly(self, height):
        self.altitude = height
        return f"Flying at {height} feet"
    
    def land(self):
        self.altitude = 0
        return "Landed safely"

class Swimmer:
    def __init__(self):
        self.depth = 0
    
    def swim(self, depth):
        self.depth = depth
        return f"Swimming at {depth} feet deep"
    
    def surface(self):
        self.depth = 0
        return "Surfaced"

class Duck(Flyer, Swimmer):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def quack(self):
        return f"{self.name} says: Quack!"

# Usage
duck = Duck("Donald")
print(duck.quack())
print(duck.fly(100))
print(duck.land())
print(duck.swim(10))
print(duck.surface())

# Method Resolution Order (MRO)
print("\n=== Method Resolution Order (MRO) ===")

class A:
    def method(self):
        return "Method from A"

class B(A):
    def method(self):
        return "Method from B"

class C(A):
    def method(self):
        return "Method from C"

class D(B, C):
    pass

# Check MRO
print("MRO for class D:")
for cls in D.__mro__:
    print(f"  {cls}")

# Create instance and call method
d = D()
print(d.method())

# Practical Example: Banking System
print("\n=== Practical Example: Banking System ===")

class Account:
    def __init__(self, account_number, account_holder, balance=0):
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = balance
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return f"Deposited ${amount}. New balance: ${self.balance}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        return "Invalid withdrawal amount or insufficient funds"
    
    def get_balance(self):
        return self.balance
    
    def account_info(self):
        return f"Account {self.account_number}: {self.account_holder} - Balance: ${self.balance}"

class SavingsAccount(Account):
    def __init__(self, account_number, account_holder, balance=0, interest_rate=0.02):
        super().__init__(account_number, account_holder, balance)
        self.interest_rate = interest_rate
    
    def add_interest(self):
        interest = self.balance * self.interest_rate
        self.balance += interest
        return f"Interest added: ${interest:.2f}. New balance: ${self.balance}"
    
    def withdraw(self, amount):
        if amount > 1000:
            return "Withdrawal limit exceeded for savings account"
        return super().withdraw(amount)

# Usage
savings = SavingsAccount("SAV001", "Alice", 1000, 0.03)
print(savings.account_info())
print(savings.deposit(200))
print(savings.add_interest())
print(savings.withdraw(1500))
print(savings.withdraw(500))