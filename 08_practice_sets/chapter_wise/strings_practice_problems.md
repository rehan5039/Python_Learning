# 🔤 Strings Practice Problems with Solutions

A comprehensive set of string manipulation problems to reinforce your understanding of Python strings.

## 📖 How to Use These Problems

1. **Attempt each problem independently** before looking at the solutions
2. **Time yourself** to track your progress
3. **Review solutions** to understand different approaches
4. **Modify problems** to create your own variations
5. **Practice regularly** to build string manipulation skills

## 🎯 Problem Set: Strings

### Problem 1: Text Analysis Tool
Create a program that analyzes a given text and provides statistics:
- Total characters (with and without spaces)
- Total words
- Total lines
- Character frequency analysis
- Most common words

```python
def analyze_text(text):
    """
    Analyze text and return statistics
    
    Args:
        text (str): Text to analyze
    
    Returns:
        dict: Dictionary containing text statistics
    """
    # Your implementation here
    pass

# Test the function
sample_text = """Python is a powerful programming language.
It is widely used for web development, data analysis, 
and artificial intelligence applications."""
```

### Problem 2: String Encryption/Decryption
Implement a simple Caesar cipher for string encryption and decryption:
- Shift each letter by a given number of positions
- Handle both uppercase and lowercase letters
- Preserve non-alphabetic characters
- Implement both encryption and decryption functions

```python
def caesar_encrypt(text, shift):
    """
    Encrypt text using Caesar cipher
    
    Args:
        text (str): Text to encrypt
        shift (int): Number of positions to shift
    
    Returns:
        str: Encrypted text
    """
    # Your implementation here
    pass

def caesar_decrypt(text, shift):
    """
    Decrypt text using Caesar cipher
    
    Args:
        text (str): Text to decrypt
        shift (int): Number of positions to shift back
    
    Returns:
        str: Decrypted text
    """
    # Your implementation here
    pass

# Test the functions
original = "Hello, World!"
encrypted = caesar_encrypt(original, 3)
decrypted = caesar_decrypt(encrypted, 3)
print(f"Original: {original}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")
```

### Problem 3: URL Parser
Create a function that parses a URL and extracts its components:
- Protocol (http, https, ftp, etc.)
- Domain
- Port (if specified)
- Path
- Query parameters
- Fragment (if any)

```python
def parse_url(url):
    """
    Parse URL and extract components
    
    Args:
        url (str): URL to parse
    
    Returns:
        dict: Dictionary containing URL components
    """
    # Your implementation here
    pass

# Test the function
test_urls = [
    "https://www.example.com:8080/path/to/page?param1=value1&param2=value2#section1",
    "http://localhost:3000/api/users",
    "ftp://files.example.com/downloads/"
]
```

### Problem 4: Text Formatter
Create a text formatter that can:
- Justify text to left, right, or center
- Wrap text to a specified width
- Add prefixes or suffixes to each line
- Convert case (title, sentence, uppercase, lowercase)
- Handle indentation

```python
def format_text(text, width=80, alignment='left', indent=0):
    """
    Format text with specified options
    
    Args:
        text (str): Text to format
        width (int): Maximum line width
        alignment (str): 'left', 'right', or 'center'
        indent (int): Number of spaces to indent
    
    Returns:
        str: Formatted text
    """
    # Your implementation here
    pass

# Test the function
sample_text = "Python is an interpreted, high-level, general-purpose programming language."
formatted = format_text(sample_text, width=40, alignment='center', indent=4)
print(formatted)
```

### Problem 5: Palindrome Generator
Create functions to:
- Check if a string is a palindrome (ignoring spaces, punctuation, and case)
- Find all palindromic substrings in a string
- Generate the longest palindromic substring
- Create a palindrome from a given string

```python
def is_palindrome(text):
    """
    Check if text is a palindrome (ignoring spaces, punctuation, and case)
    
    Args:
        text (str): Text to check
    
    Returns:
        bool: True if palindrome, False otherwise
    """
    # Your implementation here
    pass

def find_palindromes(text):
    """
    Find all palindromic substrings in text
    
    Args:
        text (str): Text to search
    
    Returns:
        list: List of palindromic substrings
    """
    # Your implementation here
    pass

def longest_palindrome(text):
    """
    Find the longest palindromic substring
    
    Args:
        text (str): Text to search
    
    Returns:
        str: Longest palindromic substring
    """
    # Your implementation here
    pass

# Test the functions
test_text = "A man, a plan, a canal: Panama"
print(f"Is palindrome: {is_palindrome(test_text)}")
print(f"Palindromes: {find_palindromes('abccba')}")
print(f"Longest: {longest_palindrome('babad')}")
```

### Problem 6: Regex Pattern Matcher
Create a function that uses regular expressions to:
- Find email addresses in text
- Find phone numbers in various formats
- Find URLs in text
- Validate credit card numbers
- Extract hashtags and mentions from social media text

```python
import re

def find_emails(text):
    """
    Find all email addresses in text
    
    Args:
        text (str): Text to search
    
    Returns:
        list: List of email addresses found
    """
    # Your implementation here
    pass

def find_phone_numbers(text):
    """
    Find phone numbers in various formats
    
    Args:
        text (str): Text to search
    
    Returns:
        list: List of phone numbers found
    """
    # Your implementation here
    pass

# Test the functions
sample_text = """
Contact us at support@example.com or call (555) 123-4567.
Visit our website at https://www.example.com.
Follow us @example or use #python hashtag.
Credit card: 1234-5678-9012-3456
"""

print(f"Emails: {find_emails(sample_text)}")
print(f"Phone numbers: {find_phone_numbers(sample_text)}")
```

## ✅ Solutions

### Solution 1: Text Analysis Tool
```python
def analyze_text(text):
    """
    Analyze text and return statistics
    """
    # Basic statistics
    total_chars = len(text)
    total_chars_no_spaces = len(text.replace(' ', ''))
    words = text.split()
    total_words = len(words)
    lines = text.split('\n')
    total_lines = len(lines)
    
    # Character frequency
    char_freq = {}
    for char in text.lower():
        if char.isalpha():
            char_freq[char] = char_freq.get(char, 0) + 1
    
    # Word frequency
    word_freq = {}
    for word in words:
        clean_word = word.lower().strip('.,!?;:"')
        word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
    
    # Most common words (top 5)
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    most_common = sorted_words[:5]
    
    return {
        'total_chars': total_chars,
        'total_chars_no_spaces': total_chars_no_spaces,
        'total_words': total_words,
        'total_lines': total_lines,
        'char_frequency': char_freq,
        'most_common_words': most_common
    }

# Test
sample_text = """Python is a powerful programming language.
It is widely used for web development, data analysis, 
and artificial intelligence applications."""

result = analyze_text(sample_text)
for key, value in result.items():
    print(f"{key}: {value}")
```

### Solution 2: String Encryption/Decryption
```python
def caesar_encrypt(text, shift):
    """
    Encrypt text using Caesar cipher
    """
    result = ""
    for char in text:
        if char.isalpha():
            # Determine if uppercase or lowercase
            ascii_offset = ord('A') if char.isupper() else ord('a')
            # Shift character
            shifted = (ord(char) - ascii_offset + shift) % 26
            result += chr(shifted + ascii_offset)
        else:
            # Non-alphabetic characters remain unchanged
            result += char
    return result

def caesar_decrypt(text, shift):
    """
    Decrypt text using Caesar cipher
    """
    # Decryption is just encryption with negative shift
    return caesar_encrypt(text, -shift)

# Test
original = "Hello, World!"
encrypted = caesar_encrypt(original, 3)
decrypted = caesar_decrypt(encrypted, 3)
print(f"Original: {original}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")
```

### Solution 3: URL Parser
```python
def parse_url(url):
    """
    Parse URL and extract components
    """
    import re
    
    # Regex pattern for URL parsing
    pattern = r'^(?P<protocol>https?|ftp)://(?P<domain>[^:/\s]+)(:(?P<port>\d+))?(?P<path>/[^?#]*)?(\?(?P<query>[^#]*))?(#(?P<fragment>.*))?'
    
    match = re.match(pattern, url)
    if not match:
        return None
    
    components = match.groupdict()
    
    # Parse query parameters
    if components['query']:
        query_params = {}
        for param in components['query'].split('&'):
            if '=' in param:
                key, value = param.split('=', 1)
                query_params[key] = value
            else:
                query_params[param] = ''
        components['query_params'] = query_params
    else:
        components['query_params'] = {}
    
    return components

# Test
test_urls = [
    "https://www.example.com:8080/path/to/page?param1=value1&param2=value2#section1",
    "http://localhost:3000/api/users",
    "ftp://files.example.com/downloads/"
]

for url in test_urls:
    parsed = parse_url(url)
    print(f"URL: {url}")
    if parsed:
        for key, value in parsed.items():
            print(f"  {key}: {value}")
    print()
```

### Solution 4: Text Formatter
```python
def format_text(text, width=80, alignment='left', indent=0):
    """
    Format text with specified options
    """
    import textwrap
    
    # Add indentation
    if indent > 0:
        text = textwrap.fill(text, width=width-indent)
        lines = text.split('\n')
        indented_lines = [' ' * indent + line for line in lines]
        text = '\n'.join(indented_lines)
    
    # Wrap text
    wrapped = textwrap.fill(text, width=width)
    
    # Apply alignment
    lines = wrapped.split('\n')
    formatted_lines = []
    
    for line in lines:
        if alignment == 'left':
            formatted_lines.append(line.ljust(width))
        elif alignment == 'right':
            formatted_lines.append(line.rjust(width))
        elif alignment == 'center':
            formatted_lines.append(line.center(width))
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

# Test
sample_text = "Python is an interpreted, high-level, general-purpose programming language."
formatted = format_text(sample_text, width=40, alignment='center', indent=4)
print(formatted)
```

### Solution 5: Palindrome Generator
```python
def is_palindrome(text):
    """
    Check if text is a palindrome (ignoring spaces, punctuation, and case)
    """
    # Clean the text
    cleaned = ''.join(char.lower() for char in text if char.isalnum())
    return cleaned == cleaned[::-1]

def find_palindromes(text):
    """
    Find all palindromic substrings in text
    """
    palindromes = set()
    n = len(text)
    
    # Check all substrings
    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = text[i:j]
            # Only consider substrings of length 2 or more
            if len(substring) >= 2 and is_palindrome(substring):
                palindromes.add(substring)
    
    return list(palindromes)

def longest_palindrome(text):
    """
    Find the longest palindromic substring
    """
    if not text:
        return ""
    
    longest = ""
    n = len(text)
    
    # Check all substrings
    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = text[i:j]
            if is_palindrome(substring) and len(substring) > len(longest):
                longest = substring
    
    return longest

# Test
test_text = "A man, a plan, a canal: Panama"
print(f"Is palindrome: {is_palindrome(test_text)}")
print(f"Palindromes: {find_palindromes('abccba')}")
print(f"Longest: {longest_palindrome('babad')}")
```

### Solution 6: Regex Pattern Matcher
```python
import re

def find_emails(text):
    """
    Find all email addresses in text
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

def find_phone_numbers(text):
    """
    Find phone numbers in various formats
    """
    phone_patterns = [
        r'\(\d{3}\)\s*\d{3}-\d{4}',  # (555) 123-4567
        r'\d{3}-\d{3}-\d{4}',        # 555-123-4567
        r'\d{3}\.\d{3}\.\d{4}',      # 555.123.4567
        r'\d{10}'                    # 5551234567
    ]
    
    phone_numbers = []
    for pattern in phone_patterns:
        phone_numbers.extend(re.findall(pattern, text))
    
    return phone_numbers

# Test
sample_text = """
Contact us at support@example.com or call (555) 123-4567.
Visit our website at https://www.example.com.
Follow us @example or use #python hashtag.
Credit card: 1234-5678-9012-3456
"""

print(f"Emails: {find_emails(sample_text)}")
print(f"Phone numbers: {find_phone_numbers(sample_text)}")
```

## 🎯 Advanced Challenge Problems

### Challenge 1: Text Compression
Implement a simple text compression algorithm and its decompression counterpart.

### Challenge 2: Fuzzy String Matching
Create a function that finds similar strings using techniques like Levenshtein distance.

### Challenge 3: Text Summarization
Implement a basic text summarization tool that extracts key sentences.

### Challenge 4: Language Detection
Create a function that detects the language of a given text.

## 📚 Tips for String Manipulation

1. **Use built-in methods**: Python's string methods are optimized and readable
2. **Leverage f-strings**: Modern string formatting is more readable
3. **Consider regular expressions**: For complex pattern matching
4. **Use string modules**: `string`, `textwrap`, `re` modules provide powerful tools
5. **Handle encoding**: Be aware of Unicode and encoding issues
6. **Performance matters**: For large texts, consider generators and efficient algorithms

## 🎯 Summary

String manipulation is a fundamental skill in Python programming. These problems cover:
- Basic string operations
- Text analysis and processing
- Pattern matching with regex
- Text formatting and presentation
- Algorithmic string problems

Practice these problems to become proficient in handling text data effectively!

---

**Keep practicing and exploring the power of string manipulation in Python!** 🐍