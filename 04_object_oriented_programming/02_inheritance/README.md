# 🧬 Inheritance in Python

Inheritance is a fundamental concept in Object-Oriented Programming that allows you to create new classes based on existing classes. The new class inherits attributes and methods from the parent class, promoting code reusability and establishing hierarchical relationships. This guide will teach you how to implement inheritance effectively in Python.

## 🎯 Learning Objectives

By the end of this guide, you will be able to:
- Implement single and multiple inheritance
- Use the `super()` function
- Override parent methods
- Understand method resolution order (MRO)
- Apply inheritance principles in practical scenarios

## 🧬 What is Inheritance?

Inheritance allows a child class (subclass) to inherit attributes and methods from a parent class (superclass). This creates an "is-a" relationship between classes.

### Basic Inheritance

```python
# Parent class (superclass)
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "Some generic animal sound"
    
    def info(self):
        return f"{self.name} is a {self.species}"

# Child class (subclass)
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

print(generic_animal.info())        # Generic is a Unknown
print(generic_animal.make_sound())  # Some generic animal sound

print(dog.info())                   # Buddy is a Dog
print(dog.make_sound())             # Woof! Woof!
print(dog.fetch())                  # Buddy is fetching the ball!
```

## 🔼 The `super()` Function

The `super()` function allows you to call methods from the parent class, ensuring proper initialization and method chaining.

### Using `super()` in Constructors

```python
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
        super().__init__(brand, model, year)  # Call parent constructor
        self.doors = doors
    
    def info(self):
        base_info = super().info()  # Call parent method
        return f"{base_info} with {self.doors} doors"

class ElectricCar(Car):
    def __init__(self, brand, model, year, doors, battery_capacity):
        super().__init__(brand, model, year, doors)
        self.battery_capacity = battery_capacity
        self.battery_level = 100
    
    def drive(self, miles):
        # Custom drive method for electric cars
        if self.battery_level <= 0:
            return "Battery is empty! Please charge."
        
        # Call parent drive method
        result = super().drive(miles)
        self.battery_level -= miles * 0.5  # Simplified battery consumption
        return f"{result} Battery level: {self.battery_level}%"
    
    def charge(self):
        self.battery_level = 100
        self.mileage = 0  # Reset mileage after charging
        return "Car charged successfully!"

# Usage
car = Car("Toyota", "Camry", 2022, 4)
print(car.info())  # 2022 Toyota Camry with 4 doors
print(car.drive(50))  # Drove 50 miles. Total mileage: 50

electric_car = ElectricCar("Tesla", "Model 3", 2023, 4, 75)
print(electric_car.info())  # 2023 Tesla Model 3 with 4 doors
print(electric_car.drive(20))  # Drove 20 miles. Total mileage: 20 Battery level: 90.0%
print(electric_car.charge())  # Car charged successfully!
```

## 🔄 Method Overriding

Method overriding allows a child class to provide a specific implementation of a method already defined in its parent class.

### Basic Method Overriding

```python
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
    
    # Override describe method
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
    
    # Override describe method
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
```

### Calling Parent Methods from Overridden Methods

```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    
    def get_info(self):
        return f"Employee: {self.name}, Salary: ${self.salary}"
    
    def calculate_bonus(self):
        return self.salary * 0.05  # 5% bonus

class Manager(Employee):
    def __init__(self, name, salary, department):
        super().__init__(name, salary)
        self.department = department
    
    # Override get_info to include department
    def get_info(self):
        base_info = super().get_info()  # Call parent method
        return f"{base_info}, Department: {self.department}"
    
    # Override calculate_bonus to give higher bonus
    def calculate_bonus(self):
        base_bonus = super().calculate_bonus()  # Call parent method
        manager_bonus = self.salary * 0.10  # Additional 10% for managers
        return base_bonus + manager_bonus

class Developer(Employee):
    def __init__(self, name, salary, programming_languages):
        super().__init__(name, salary)
        self.programming_languages = programming_languages
    
    # Override get_info to include programming languages
    def get_info(self):
        base_info = super().get_info()
        return f"{base_info}, Languages: {', '.join(self.programming_languages)}"
    
    # Override calculate_bonus based on skills
    def calculate_bonus(self):
        base_bonus = super().calculate_bonus()
        skill_bonus = len(self.programming_languages) * 100  # $100 per language
        return base_bonus + skill_bonus

# Usage
manager = Manager("Alice", 80000, "Engineering")
developer = Developer("Bob", 70000, ["Python", "JavaScript", "Java"])

print(manager.get_info())
print(f"Manager Bonus: ${manager.calculate_bonus()}")

print(developer.get_info())
print(f"Developer Bonus: ${developer.calculate_bonus()}")
```

## 🔀 Multiple Inheritance

Python supports multiple inheritance, where a class can inherit from multiple parent classes.

### Basic Multiple Inheritance

```python
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
        super().__init__()  # This calls the first parent's __init__
        self.name = name
    
    def quack(self):
        return f"{self.name} says: Quack!"

# Usage
duck = Duck("Donald")
print(duck.quack())           # Donald says: Quack!
print(duck.fly(100))          # Flying at 100 feet
print(duck.land())            # Landed safely
print(duck.swim(10))          # Swimming at 10 feet deep
print(duck.surface())         # Surfaced
```

### Method Resolution Order (MRO)

Python uses the C3 linearization algorithm to determine the method resolution order.

```python
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
print(d.method())  # Method from B (first in MRO)

# Using super() with multiple inheritance
class A:
    def __init__(self):
        print("Initializing A")
    
    def method(self):
        return "A"

class B:
    def __init__(self):
        print("Initializing B")
    
    def method(self):
        return "B"

class C(A, B):
    def __init__(self):
        super().__init__()  # Calls A.__init__ based on MRO
        print("Initializing C")
    
    def method(self):
        # This will call B.method() because of MRO
        return super().method() + " then C"

# Usage
c = C()  # Initializing A, then Initializing C
print(c.method())  # B then C
```

## 🧪 Practical Examples

### Example 1: Banking System with Inheritance
```python
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
        # Savings accounts may have withdrawal limits
        if amount > 1000:
            return "Withdrawal limit exceeded for savings account"
        return super().withdraw(amount)

class CheckingAccount(Account):
    def __init__(self, account_number, account_holder, balance=0, overdraft_limit=100):
        super().__init__(account_number, account_holder, balance)
        self.overdraft_limit = overdraft_limit
    
    def withdraw(self, amount):
        if 0 < amount <= (self.balance + self.overdraft_limit):
            self.balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        return "Withdrawal exceeds overdraft limit"

class BusinessAccount(Account):
    def __init__(self, account_number, account_holder, balance=0, transaction_fee=1.0):
        super().__init__(account_number, account_holder, balance)
        self.transaction_fee = transaction_fee
        self.transaction_count = 0
    
    def deposit(self, amount):
        result = super().deposit(amount)
        self.transaction_count += 1
        return result
    
    def withdraw(self, amount):
        result = super().withdraw(amount)
        if "Withdrew" in result:
            self.transaction_count += 1
        return result
    
    def get_monthly_fee(self):
        fee = self.transaction_count * self.transaction_fee
        self.transaction_count = 0  # Reset for next month
        return fee

# Usage
savings = SavingsAccount("SAV001", "Alice", 1000, 0.03)
checking = CheckingAccount("CHK001", "Bob", 500, 200)
business = BusinessAccount("BUS001", "Company Inc.", 10000, 0.5)

print("=== Savings Account ===")
print(savings.account_info())
print(savings.deposit(200))
print(savings.add_interest())
print(savings.withdraw(1500))  # Exceeds limit
print(savings.withdraw(500))   # Within limit

print("\n=== Checking Account ===")
print(checking.account_info())
print(checking.withdraw(600))  # Uses overdraft
print(checking.withdraw(200))  # Exceeds overdraft

print("\n=== Business Account ===")
print(business.account_info())
business.deposit(1000)
business.withdraw(500)
business.deposit(200)
fee = business.get_monthly_fee()
print(f"Monthly fee: ${fee}")
print(business.account_info())
```

### Example 2: Media Library with Inheritance
```python
import datetime

class Media:
    def __init__(self, title, creator, year):
        self.title = title
        self.creator = creator
        self.year = year
        self.is_borrowed = False
        self.borrowed_date = None
        self.due_date = None
    
    def borrow(self, borrower_name, days=14):
        if not self.is_borrowed:
            self.is_borrowed = True
            self.borrowed_date = datetime.datetime.now()
            self.due_date = self.borrowed_date + datetime.timedelta(days=days)
            return f"'{self.title}' borrowed by {borrower_name} until {self.due_date.strftime('%Y-%m-%d')}"
        return f"'{self.title}' is already borrowed"
    
    def return_item(self):
        if self.is_borrowed:
            self.is_borrowed = False
            borrowed_date = self.borrowed_date
            self.borrowed_date = None
            self.due_date = None
            return f"'{self.title}' returned. Borrowed on {borrowed_date.strftime('%Y-%m-%d')}"
        return f"'{self.title}' was not borrowed"
    
    def is_overdue(self):
        if self.is_borrowed and self.due_date:
            return datetime.datetime.now() > self.due_date
        return False
    
    def info(self):
        status = "Available"
        if self.is_borrowed:
            status = f"Borrowed until {self.due_date.strftime('%Y-%m-%d')}"
            if self.is_overdue():
                status += " (OVERDUE)"
        return f"'{self.title}' by {self.creator} ({self.year}) - {status}"

class Book(Media):
    def __init__(self, title, author, year, isbn, pages):
        super().__init__(title, author, year)
        self.isbn = isbn
        self.pages = pages
    
    def info(self):
        base_info = super().info()
        return f"{base_info} - Book ({self.pages} pages, ISBN: {self.isbn})"

class DVD(Media):
    def __init__(self, title, director, year, duration, rating):
        super().__init__(title, director, year)
        self.duration = duration  # in minutes
        self.rating = rating
    
    def info(self):
        base_info = super().info()
        return f"{base_info} - DVD ({self.duration} min, Rating: {self.rating})"

class Magazine(Media):
    def __init__(self, title, publisher, year, issue_number):
        super().__init__(title, publisher, year)
        self.issue_number = issue_number
    
    def borrow(self, borrower_name, days=7):  # Magazines have shorter loan period
        return super().borrow(borrower_name, days)
    
    def info(self):
        base_info = super().info()
        return f"{base_info} - Magazine (Issue #{self.issue_number})"

class DigitalMedia(Media):
    def __init__(self, title, creator, year, file_size):
        super().__init__(title, creator, year)
        self.file_size = file_size  # in MB
    
    def borrow(self, borrower_name, days=365):  # Digital media can be borrowed for a year
        return super().borrow(borrower_name, days)
    
    def info(self):
        base_info = super().info()
        return f"{base_info} - Digital Media ({self.file_size} MB)"

# Usage
book = Book("Python Programming", "John Smith", 2023, "978-1234567890", 450)
dvd = DVD("Inception", "Christopher Nolan", 2010, 148, "PG-13")
magazine = Magazine("Tech Weekly", "Tech Publications", 2023, 42)
digital = DigitalMedia("Learn Python Course", "Online Education", 2023, 1200)

media_items = [book, dvd, magazine, digital]

print("=== Media Library ===")
for item in media_items:
    print(item.info())

# Borrowing items
print("\n=== Borrowing Items ===")
print(book.borrow("Alice", 21))  # Extended loan for book
print(dvd.borrow("Bob"))
print(magazine.borrow("Charlie"))  # Shorter loan period

print("\n=== After Borrowing ===")
for item in media_items:
    print(item.info())
```

### Example 3: Game Characters with Inheritance
```python
import random

class Character:
    def __init__(self, name, health, attack_power):
        self.name = name
        self.health = health
        self.max_health = health
        self.attack_power = attack_power
        self.level = 1
        self.experience = 0
    
    def is_alive(self):
        return self.health > 0
    
    def take_damage(self, damage):
        self.health = max(0, self.health - damage)
        return f"{self.name} takes {damage} damage. Health: {self.health}/{self.max_health}"
    
    def heal(self, amount):
        self.health = min(self.max_health, self.health + amount)
        return f"{self.name} heals {amount} HP. Health: {self.health}/{self.max_health}"
    
    def attack(self, target):
        if not target.is_alive():
            return f"{target.name} is already defeated!"
        
        damage = random.randint(self.attack_power - 2, self.attack_power + 2)
        return f"{self.name} attacks {target.name} for {damage} damage!\n{target.take_damage(damage)}"
    
    def gain_experience(self, exp):
        self.experience += exp
        if self.experience >= self.level * 100:
            self.level_up()
        return f"{self.name} gained {exp} experience points."
    
    def level_up(self):
        self.level += 1
        self.max_health += 10
        self.health = self.max_health
        self.attack_power += 2
        return f"{self.name} leveled up to level {self.level}!"
    
    def info(self):
        return f"{self.name} (Level {self.level}): HP {self.health}/{self.max_health}, ATK {self.attack_power}"

class Warrior(Character):
    def __init__(self, name):
        super().__init__(name, health=120, attack_power=15)
        self.armor = 5
    
    def take_damage(self, damage):
        # Reduce damage by armor
        reduced_damage = max(1, damage - self.armor)
        return super().take_damage(reduced_damage)
    
    def berserk_attack(self, target):
        if not target.is_alive():
            return f"{target.name} is already defeated!"
        
        # Double damage but takes recoil damage
        damage = random.randint(self.attack_power * 2 - 3, self.attack_power * 2 + 3)
        recoil = damage // 4
        result = f"{self.name} uses Berserk Attack on {target.name} for {damage} damage!\n"
        result += target.take_damage(damage) + "\n"
        result += self.take_damage(recoil)
        return result

class Mage(Character):
    def __init__(self, name):
        super().__init__(name, health=80, attack_power=12)
        self.mana = 100
        self.max_mana = 100
    
    def cast_fireball(self, target):
        if not target.is_alive():
            return f"{target.name} is already defeated!"
        
        mana_cost = 20
        if self.mana < mana_cost:
            return f"{self.name} doesn't have enough mana!"
        
        self.mana -= mana_cost
        damage = random.randint(20, 30)
        result = f"{self.name} casts Fireball on {target.name} for {damage} damage!\n"
        result += target.take_damage(damage)
        return result
    
    def heal_spell(self, target):
        mana_cost = 30
        if self.mana < mana_cost:
            return f"{self.name} doesn't have enough mana!"
        
        self.mana -= mana_cost
        heal_amount = random.randint(25, 35)
        return target.heal(heal_amount)
    
    def meditate(self):
        mana_restored = 30
        self.mana = min(self.max_mana, self.mana + mana_restored)
        return f"{self.name} meditates and restores {mana_restored} mana. Mana: {self.mana}/{self.max_mana}"

class Archer(Character):
    def __init__(self, name):
        super().__init__(name, health=90, attack_power=14)
        self.arrows = 20
    
    def shoot_arrow(self, target):
        if not target.is_alive():
            return f"{target.name} is already defeated!"
        
        if self.arrows <= 0:
            return f"{self.name} is out of arrows!"
        
        self.arrows -= 1
        # Archer has a chance for critical hit
        if random.random() < 0.3:  # 30% chance
            damage = random.randint(self.attack_power * 2, self.attack_power * 3)
            return f"{self.name} shoots a critical arrow at {target.name} for {damage} damage!\n{target.take_damage(damage)}"
        else:
            damage = random.randint(self.attack_power - 2, self.attack_power + 2)
            return f"{self.name} shoots an arrow at {target.name} for {damage} damage!\n{target.take_damage(damage)}"
    
    def reload(self):
        arrows_added = 10
        self.arrows += arrows_added
        return f"{self.name} reloads and gains {arrows_added} arrows. Arrows: {self.arrows}"

# Usage
warrior = Warrior("Conan")
mage = Mage("Gandalf")
archer = Archer("Legolas")

characters = [warrior, mage, archer]

print("=== Character Info ===")
for char in characters:
    print(char.info())

print("\n=== Battle Simulation ===")
# Warrior attacks Mage
print(warrior.attack(mage))
print(mage.info())

# Mage casts fireball on Warrior
print(mage.cast_fireball(warrior))
print(warrior.info())

# Archer shoots arrow at Mage
print(archer.shoot_arrow(mage))
print(mage.info())

# Warrior uses berserk attack on Archer
print(warrior.berserk_attack(archer))
print(archer.info())

# Mage heals Warrior
print(mage.heal_spell(warrior))
print(warrior.info())

# Archer reloads
print(archer.reload())