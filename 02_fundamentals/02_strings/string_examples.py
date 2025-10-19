# String Examples and Demonstrations

# Creating Strings
print("=== Creating Strings ===")
single_quotes = 'Hello, World!'
double_quotes = "Python Programming"
triple_quotes = """This is a
multi-line
string"""

print(f"Single quotes: {single_quotes}")
print(f"Double quotes: {double_quotes}")
print(f"Triple quotes:\n{triple_quotes}")

# String Indexing
print("\n=== String Indexing ===")
text = "Python"
print(f"Text: {text}")
print(f"First character (index 0): {text[0]}")
print(f"Last character (index -1): {text[-1]}")
print(f"Second character: {text[1]}")
print(f"Second to last character: {text[-2]}")

# String Slicing
print("\n=== String Slicing ===")
sample = "Python Programming"
print(f"Sample text: {sample}")
print(f"First 6 characters: {sample[0:6]}")
print(f"From index 7 to end: {sample[7:]}")
print(f"From beginning to index 6: {sample[:6]}")
print(f"Last 11 characters: {sample[-11:]}")
print(f"Every second character: {sample[::2]}")
print(f"Reversed string: {sample[::-1]}")

# String Methods
print("\n=== String Methods ===")
message = "  Hello, Python World!  "
print(f"Original: '{message}'")
print(f"Uppercase: '{message.upper()}'")
print(f"Lowercase: '{message.lower()}'")
print(f"Title Case: '{message.title()}'")
print(f"Strip whitespace: '{message.strip()}'")
print(f"Replace 'Python' with 'Java': '{message.replace('Python', 'Java')}'")

# String Methods - Search and Count
text = "Python is great, Python is powerful, Python is fun"
print(f"\nText: {text}")
print(f"Find 'Python': {text.find('Python')}")
print(f"Count 'Python': {text.count('Python')}")
print(f"Find 'Java': {text.find('Java')}")

# Splitting and Joining
print("\n=== Splitting and Joining ===")
sentence = "Python is awesome"
words = sentence.split()
print(f"Sentence: {sentence}")
print(f"Split into words: {words}")

joined = " ".join(words)
print(f"Joined with spaces: {joined}")

data = "apple,banana,orange,grape"
fruits = data.split(",")
print(f"Fruits data: {data}")
print(f"Split by comma: {fruits}")

# String Formatting
print("\n=== String Formatting ===")
name = "Alice"
age = 30
city = "New York"

# f-strings (Python 3.6+)
f_string = f"Hello, {name}! You are {age} years old and live in {city}."
print(f"f-string: {f_string}")

# format() method
format_method = "Hello, {}! You are {} years old and live in {}.".format(name, age, city)
print(f"format() method: {format_method}")

# % formatting (old style)
percent_format = "Hello, %s! You are %d years old and live in %s." % (name, age, city)
print(f"% formatting: {percent_format}")

# Expression in f-strings
print(f"\nExpression in f-string: {10 * 5 = }")
print(f"Next year {name} will be {age + 1} years old.")

# Escape Sequences
print("\n=== Escape Sequences ===")
print("New line: Hello\nWorld")
print("Tab: Name:\tAlice")
print("Backslash: Path: C:\\Users\\Alice")
print("Single quote: It\'s a beautiful day!")
print("Double quote: She said, \"Hello!\"")

# Raw strings
print("\n=== Raw Strings ===")
normal_string = "C:\\Users\\Alice\\Documents"
raw_string = r"C:\Users\Alice\Documents"
print(f"Normal string: {normal_string}")
print(f"Raw string: {raw_string}")

# String Operations
print("\n=== String Operations ===")
first = "Hello"
second = "World"

# Concatenation
concatenated = first + " " + second
print(f"Concatenation: {concatenated}")

# Repetition
repeated = first * 3
print(f"Repetition: {repeated}")

# Membership testing
text = "Python Programming"
print(f"'Python' in '{text}': {'Python' in text}")
print(f"'Java' in '{text}': {'Java' in text}")

# String length
print(f"Length of '{text}': {len(text)}")

# Practical Example: Email Extractor
print("\n=== Practical Example: Email Extractor ===")
def extract_emails(text):
    """Extract email addresses from text"""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails

sample_text = "Contact us at support@example.com or sales@company.org for more information."
found_emails = extract_emails(sample_text)
print(f"Text: {sample_text}")
print(f"Found emails: {found_emails}")

# Practical Example: Text Statistics
print("\n=== Practical Example: Text Statistics ===")
def analyze_text(text):
    """Analyze text properties"""
    words = text.split()
    char_count = len(text)
    word_count = len(words)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    return {
        'characters': char_count,
        'words': word_count,
        'sentences': sentence_count,
        'avg_word_length': sum(len(word) for word in words) / word_count if word_count > 0 else 0
    }

sample_text = "Python is a powerful programming language. It is easy to learn and very versatile!"
stats = analyze_text(sample_text)
print(f"Text: {sample_text}")
print(f"Statistics: {stats}")