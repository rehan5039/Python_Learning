# Conditional Statements Examples

# Basic if statement
print("=== Basic if statement ===")
age = 18
if age >= 18:
    print("You are eligible to vote!")

# if-else statement
print("\n=== if-else statement ===")
age = 16
if age >= 18:
    print("You are eligible to vote!")
else:
    print("You are not eligible to vote yet.")

# if-elif-else statement
print("\n=== if-elif-else statement ===")
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"
print(f"Your grade is: {grade}")

# Comparison operators
print("\n=== Comparison operators ===")
x, y = 10, 20
print(f"{x} == {y}: {x == y}")
print(f"{x} != {y}: {x != y}")
print(f"{x} < {y}: {x < y}")
print(f"{x} > {y}: {x > y}")
print(f"{x} <= {y}: {x <= y}")
print(f"{x} >= {y}: {x >= y}")

# Logical operators
print("\n=== Logical operators ===")
age = 25
has_license = True
has_car = False

if age >= 18 and has_license:
    print("You can drive!")

if has_license or has_car:
    print("You have transportation options!")

if not has_car:
    print("You don't have a car.")

# Complex logical combinations
if (age >= 18 and has_license) and not has_car:
    print("You can drive but don't have a car.")

# Nested conditions
print("\n=== Nested conditions ===")
username = "admin"
password = "secret123"

if username == "admin":
    if password == "secret123":
        print("Welcome, administrator!")
    else:
        print("Incorrect password!")
else:
    print("Unknown user!")

# Cleaner nested conditions with logical operators
if username == "admin" and password == "secret123":
    print("Welcome, administrator! (Cleaner version)")
elif username == "admin":
    print("Incorrect password! (Cleaner version)")
else:
    print("Unknown user! (Cleaner version)")

# Ternary operator
print("\n=== Ternary operator ===")
age = 20
status = "adult" if age >= 18 else "minor"
print(f"Status: {status}")

temperature = 25
weather = "warm" if temperature > 20 else "cold"
print(f"Weather: {weather}")

# Membership operators
print("\n=== Membership operators ===")
fruits = ["apple", "banana", "orange"]
if "apple" in fruits:
    print("Apple is in the list!")

if "grape" not in fruits:
    print("Grape is not in the list!")

text = "Python Programming"
if "Python" in text:
    print("Found 'Python' in the text!")

# Identity operators
print("\n=== Identity operators ===")
list1 = [1, 2, 3]
list2 = [1, 2, 3]
list3 = list1

print(f"list1 == list2: {list1 == list2}")  # True (same values)
print(f"list1 is list2: {list1 is list2}")  # False (different objects)
print(f"list1 is list3: {list1 is list3}")  # True (same object)

# Practical Example: Grade Calculator
print("\n=== Practical Example: Grade Calculator ===")
def calculate_grade(assignments, midterm, final, participation):
    """Calculate final grade with conditions"""
    
    # Validate inputs
    if not (0 <= assignments <= 100):
        return "Invalid assignment score"
    if not (0 <= midterm <= 100):
        return "Invalid midterm score"
    if not (0 <= final <= 100):
        return "Invalid final score"
    if not (0 <= participation <= 100):
        return "Invalid participation score"
    
    # Calculate weighted average
    final_score = (
        assignments * 0.3 +
        midterm * 0.3 +
        final * 0.3 +
        participation * 0.1
    )
    
    # Apply bonus for perfect participation
    if participation == 100:
        final_score = min(100, final_score + 2)
        print("Bonus points added for perfect participation!")
    
    # Determine letter grade
    if final_score >= 90:
        letter_grade = "A"
        status = "Excellent work!"
    elif final_score >= 80:
        letter_grade = "B"
        status = "Good job!"
    elif final_score >= 70:
        letter_grade = "C"
        status = "Satisfactory"
    elif final_score >= 60:
        letter_grade = "D"
        status = "Needs improvement"
    else:
        letter_grade = "F"
        status = "Failed - see instructor"
    
    return {
        "numeric_score": round(final_score, 2),
        "letter_grade": letter_grade,
        "status": status
    }

# Test the grade calculator
result = calculate_grade(95, 87, 92, 100)
print(f"Final Grade: {result['numeric_score']}% ({result['letter_grade']})")
print(result['status'])

# Practical Example: Weather-Based Activity Recommender
print("\n=== Practical Example: Weather-Based Activity Recommender ===")
def recommend_activity(temperature, is_raining, is_weekend):
    """Recommend activities based on weather and day"""
    
    # Temperature categories
    if temperature > 80:
        temp_category = "hot"
    elif temperature > 60:
        temp_category = "warm"
    elif temperature > 40:
        temp_category = "cool"
    else:
        temp_category = "cold"
    
    # Activity recommendations
    if is_raining:
        if temp_category in ["warm", "hot"] and is_weekend:
            return "Visit a museum or indoor pool"
        elif temp_category == "cold":
            return "Stay indoors and read a book"
        else:
            return "Go to a coffee shop or indoor mall"
    else:  # Not raining
        if temp_category == "hot":
            if is_weekend:
                return "Go to the beach or pool"
            else:
                return "Exercise early morning or evening"
        elif temp_category == "warm":
            if is_weekend:
                return "Go hiking or have a picnic"
            else:
                return "Take a walk during lunch break"
        elif temp_category == "cool":
            if is_weekend:
                return "Go apple picking or visit a park"
            else:
                return "Take a walk or exercise outdoors"
        else:  # cold
            if is_weekend:
                return "Go skiing or ice skating"
            else:
                return "Bundle up and take a short walk"

# Test the activity recommender
activities = [
    (85, False, True),   # Hot, not raining, weekend
    (65, True, False),   # Warm, raining, weekday
    (45, False, True),   # Cool, not raining, weekend
    (30, True, False),   # Cold, raining, weekday
]

for temp, raining, weekend in activities:
    activity = recommend_activity(temp, raining, weekend)
    print(f"Temp: {temp}°F, Raining: {raining}, Weekend: {weekend}")
    print(f"Recommended activity: {activity}\n")

# Common mistakes demonstration
print("\n=== Common mistakes demonstration ===")

# Assignment vs Comparison
age = 18
if age == 18:  # Correct comparison
    print("Age is 18 (comparison)")

# Chaining comparisons (Pythonic way)
if 18 <= age <= 65:
    print("Working age (chained comparison)")

# Comparing with None
value = None
if value is None:
    print("Value is not set (identity check)")

# Empty collections in conditions
my_list = []
my_dict = {}
my_string = ""

if not my_list:
    print("List is empty (Pythonic check)")

if not my_dict:
    print("Dictionary is empty (Pythonic check)")

if not my_string:
    print("String is empty (Pythonic check)")