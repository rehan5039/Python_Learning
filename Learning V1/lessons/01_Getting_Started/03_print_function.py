"""
Lesson 01: Understanding the print() Function
===============================================
Master the print function and its various uses.

This program demonstrates:
- Basic printing
- Printing multiple items
- Using separators
- Changing end character
- Escape sequences
"""

print("=" * 60)
print("Exploring the print() Function")
print("=" * 60)

# 1. Basic printing
print("\n1️⃣ Basic Printing:")
print("Hello, Python!")
print("This is a simple text output")

# 2. Printing multiple items
print("\n2️⃣ Printing Multiple Items:")
print("Python", "is", "awesome!")
print("I", "love", "coding", "in", "Python")

# 3. Using custom separators
print("\n3️⃣ Custom Separators:")
print("Python", "is", "awesome!", sep=" ")      # Default: space
print("Python", "is", "awesome!", sep="-")      # Dash separator
print("Python", "is", "awesome!", sep=" | ")    # Pipe separator
print("2024", "10", "15", sep="/")              # Date format

# 4. Changing the end character
print("\n4️⃣ Custom End Character:")
print("Hello", end=" ")
print("World")  # These appear on the same line

print("Loading", end="...")
print("Done!")

print("A", end=", ")
print("B", end=", ")
print("C")

# 5. Printing different data types
print("\n5️⃣ Printing Different Types:")
print(42)                    # Integer
print(3.14159)              # Float
print(True)                 # Boolean
print([1, 2, 3])           # List
print({"name": "Python"})  # Dictionary

# 6. Printing calculations
print("\n6️⃣ Printing Calculations:")
print(5 + 3)
print(10 * 2)
print("Sum of 15 and 27 is:", 15 + 27)

# 7. Escape sequences
print("\n7️⃣ Escape Sequences:")
print("Line 1\nLine 2\nLine 3")        # \n = new line
print("Tab\tSeparated\tText")          # \t = tab
print("He said, \"Python is great!\"") # \" = quote
print("Path: C:\\Users\\Documents")    # \\ = backslash

# 8. Multi-line strings
print("\n8️⃣ Multi-line Strings:")
print("""
This is a multi-line string.
It can span multiple lines.
Very useful for long text!
""")

# 9. Formatted printing (modern way)
print("\n9️⃣ F-strings (Formatted Strings):")
name = "Python"
version = 3.12
print(f"I am learning {name} version {version}")
print(f"The result of 5 + 3 is {5 + 3}")

# 10. Creative examples
print("\n🎨 Creative ASCII Art:")
print("""
    🐍 PYTHON
   ╔═══════════╗
   ║  WELCOME  ║
   ╚═══════════╝
""")

print("\n" + "=" * 60)
print("✅ print() function exploration complete!")
print("=" * 60)

# Practice challenge
print("\n🎯 Challenge: Can you create your own ASCII art?")
print("Try it in a new file!")
