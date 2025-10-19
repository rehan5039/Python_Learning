"""
Lesson 01: Comments in Python
===============================
Learn how to write effective comments to document your code.

This is a docstring - a special multi-line comment used at the beginning
of modules, classes, and functions to describe what they do.
"""

# ============================================================================
# SECTION 1: Single-Line Comments
# ============================================================================

# This is a single-line comment
# It starts with a hash (#) symbol
# Python ignores everything after # on that line

print("Hello, World!")  # This comment is at the end of a line

# Comments are used to:
# 1. Explain what the code does
# 2. Make notes for yourself or other developers
# 3. Temporarily disable code

# print("This line won't run because it's commented out")

# ============================================================================
# SECTION 2: Multi-Line Comments
# ============================================================================

"""
This is a multi-line comment using triple double-quotes.
It can span multiple lines.
Very useful for detailed explanations.
"""

'''
You can also use triple single-quotes.
Both work the same way.
Choose one style and be consistent!
'''

# ============================================================================
# SECTION 3: Best Practices for Comments
# ============================================================================

# ✅ GOOD COMMENT - Explains WHY, not just WHAT
# Calculate area because we need to determine paint required
area = length * width

# ❌ BAD COMMENT - Just restates the code
# Multiply length by width
area = length * width

# ✅ GOOD COMMENT - Provides context
# Convert temperature from Celsius to Fahrenheit for US users
fahrenheit = celsius * 9/5 + 32

# ❌ BAD COMMENT - Obvious from code
# Convert to Fahrenheit
fahrenheit = celsius * 9/5 + 32

# ============================================================================
# SECTION 4: Comment Types
# ============================================================================

# TODO: Add input validation
# FIXME: This breaks when value is negative
# NOTE: This is a temporary solution
# HACK: Workaround for library bug
# XXX: Danger! This needs review

# ============================================================================
# SECTION 5: Documenting Code Example
# ============================================================================

def calculate_circle_area(radius):
    """
    Calculate the area of a circle.
    
    This function takes a radius and returns the area of a circle
    using the formula: A = π * r²
    
    Args:
        radius (float): The radius of the circle
        
    Returns:
        float: The area of the circle
        
    Example:
        >>> calculate_circle_area(5)
        78.53975
    """
    pi = 3.14159
    area = pi * radius ** 2
    return area

# ============================================================================
# SECTION 6: When to Comment
# ============================================================================

# ✅ DO comment when:
# - Code is complex or non-obvious
# - Explaining business logic
# - Documenting functions and classes
# - Warning about potential issues
# - Explaining workarounds

# ❌ DON'T comment when:
# - Code is self-explanatory
# - Just repeating what code does
# - Leaving old/dead code (delete it instead!)
# - Commenting everything (too much noise)

# ============================================================================
# SECTION 7: Code Organization with Comments
# ============================================================================

# Import required modules
import math
import datetime

# Constants
MAX_USERS = 100
DEFAULT_TIMEOUT = 30

# Main program
def main():
    """Main program entry point."""
    # Initialize variables
    user_count = 0
    
    # Process users
    print("Processing users...")
    
    # Display results
    print(f"Total users: {user_count}")

# ============================================================================
# SECTION 8: Commenting Out Code for Testing
# ============================================================================

# Version 1 - Original code
# print("This is the old version")

# Version 2 - New code
print("This is the new version")

# Multi-line code commenting
"""
old_code = True
if old_code:
    print("Old implementation")
    do_something()
"""

# ============================================================================
# END OF LESSON
# ============================================================================

print("\n" + "=" * 70)
print("✅ Comments lesson complete!")
print("=" * 70)
print("""
Key Takeaways:
1. Use # for single-line comments
2. Use triple quotes for multi-line comments and docstrings
3. Comment WHY, not WHAT
4. Keep comments updated with code
5. Don't over-comment - let code speak for itself when possible
""")
