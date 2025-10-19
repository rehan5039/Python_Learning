# Variables and Data Types Demo

# Numeric Types
integer_example = 42
float_example = 3.14159
complex_example = 2 + 3j

print("=== Numeric Types ===")
print(f"Integer: {integer_example} (type: {type(integer_example)})")
print(f"Float: {float_example} (type: {type(float_example)})")
print(f"Complex: {complex_example} (type: {type(complex_example)})")

# Text Type
string_example = "Hello, Python!"
print("\n=== Text Type ===")
print(f"String: {string_example} (type: {type(string_example)})")

# Boolean Type
boolean_true = True
boolean_false = False
print("\n=== Boolean Type ===")
print(f"True: {boolean_true} (type: {type(boolean_true)})")
print(f"False: {boolean_false} (type: {type(boolean_false)})")

# None Type
none_example = None
print("\n=== None Type ===")
print(f"None: {none_example} (type: {type(none_example)})")

# Type Conversion Examples
print("\n=== Type Conversion ===")
number_string = "123"
converted_number = int(number_string)
print(f"String '123' converted to int: {converted_number} (type: {type(converted_number)})")

float_number = 45.67
converted_int = int(float_number)
print(f"Float 45.67 converted to int: {converted_int} (type: {type(converted_int)})")

# User Input Example
print("\n=== User Input Example ===")
# Uncomment the following lines to test user input
# user_name = input("Enter your name: ")
# user_age = int(input("Enter your age: "))
# print(f"Hello {user_name}, you are {user_age} years old!")

# Demonstrating Variable Reassignment
print("\n=== Variable Reassignment ===")
variable = "I'm a string"
print(f"Variable: {variable} (type: {type(variable)})")

variable = 42
print(f"Same variable reassigned: {variable} (type: {type(variable)})")

variable = True
print(f"Same variable reassigned again: {variable} (type: {type(variable)})")