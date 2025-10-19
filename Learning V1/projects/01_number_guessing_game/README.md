# Project 01: Number Guessing Game 🎯

## 📋 Project Overview

Create a fun number guessing game where the computer generates a random number and the player tries to guess it!

**Difficulty**: 🌱 Beginner  
**Estimated Time**: 30-45 minutes  
**Skills Practiced**: 
- Random number generation
- Loops (while)
- Conditional statements (if/elif/else)
- User input
- Type conversion

---

## 🎯 Project Requirements

### Basic Version (Minimum Requirements)

Your program should:

1. **Generate a random number** between 1 and 100
2. **Ask the user** to guess the number
3. **Provide feedback**:
   - "Too high!" if guess is greater than the number
   - "Too low!" if guess is less than the number
   - "Correct!" if guess is right
4. **Keep track** of the number of attempts
5. **End the game** when the user guesses correctly

### Example Output:
```
=================================
  WELCOME TO NUMBER GUESSING GAME
=================================

I'm thinking of a number between 1 and 100.
Can you guess it?

Enter your guess: 50
Too high! Try again.

Enter your guess: 25
Too low! Try again.

Enter your guess: 37
Too high! Try again.

Enter your guess: 31
Correct! You guessed it in 4 attempts!
```

---

## 🚀 Advanced Features (Optional Challenges)

Once you complete the basic version, try adding:

### Challenge 1: Difficulty Levels
- **Easy**: 1-50, unlimited attempts
- **Medium**: 1-100, 10 attempts
- **Hard**: 1-200, 7 attempts

### Challenge 2: Multiple Rounds
- Ask if player wants to play again
- Keep track of total games played
- Display win/loss statistics

### Challenge 3: Hints
- Give a hint after 3 wrong guesses
- Hints could be: even/odd, divisible by X, etc.

### Challenge 4: High Score
- Track the best score (fewest attempts)
- Display high score at the end
- Save high score to a file (advanced)

### Challenge 5: Input Validation
- Handle non-numeric inputs gracefully
- Prevent guesses outside the valid range
- Display error messages for invalid inputs

---

## 📚 Concepts You'll Use

### 1. Random Module
```python
import random

# Generate random number
number = random.randint(1, 100)
```

### 2. While Loops
```python
while condition:
    # Code here
```

### 3. User Input
```python
guess = int(input("Enter your guess: "))
```

### 4. Conditionals
```python
if guess > number:
    print("Too high!")
elif guess < number:
    print("Too low!")
else:
    print("Correct!")
```

---

## 🛠️ Step-by-Step Guide

### Step 1: Import Required Modules
```python
import random
```

### Step 2: Generate Random Number
```python
secret_number = random.randint(1, 100)
```

### Step 3: Initialize Variables
```python
attempts = 0
guessed = False
```

### Step 4: Create the Game Loop
```python
while not guessed:
    # Your game logic here
```

### Step 5: Get User Input and Validate
```python
guess = int(input("Enter your guess: "))
attempts += 1
```

### Step 6: Compare and Provide Feedback
```python
if guess > secret_number:
    print("Too high!")
# ... etc
```

### Step 7: Display Results
```python
print(f"You won in {attempts} attempts!")
```

---

## 📝 Starter Code Template

Use this template to get started:

```python
"""
Number Guessing Game
====================
A fun game where you guess a random number!
"""

import random

# Constants
MIN_NUMBER = 1
MAX_NUMBER = 100

def main():
    """Main game function"""
    # TODO: Display welcome message
    
    # TODO: Generate random number
    
    # TODO: Initialize game variables
    
    # TODO: Create game loop
    
    # TODO: Display final results
    
if __name__ == "__main__":
    main()
```

---

## ✅ Testing Checklist

Test your program with:

- [ ] Valid guesses
- [ ] Edge cases (1 and 100)
- [ ] Invalid inputs (letters, negative numbers)
- [ ] Winning on first try
- [ ] Multiple rounds (if implemented)
- [ ] Exceeding attempt limit (if implemented)

---

## 🎓 Learning Objectives

After completing this project, you should be able to:

✅ Use the `random` module to generate random numbers  
✅ Implement a game loop with `while`  
✅ Handle user input and validation  
✅ Use conditional statements effectively  
✅ Track and update game state  
✅ Create an engaging user experience  

---

## 💡 Hints

<details>
<summary>Click to see hint 1</summary>

Use a boolean variable to track if the number has been guessed:
```python
guessed = False
while not guessed:
    # game logic
```
</details>

<details>
<summary>Click to see hint 2</summary>

Remember to increment the attempts counter:
```python
attempts = 0
# Inside the loop:
attempts += 1
```
</details>

<details>
<summary>Click to see hint 3</summary>

For input validation, use try-except:
```python
try:
    guess = int(input("Enter guess: "))
except ValueError:
    print("Please enter a valid number!")
```
</details>

---

## 🏆 Bonus Ideas

- Add colors to output (using colorama library)
- Create a graphical version (using tkinter)
- Add sound effects (using pygame)
- Make a two-player version
- Create a "guess the word" variant

---

## 📂 Files in This Project

- `README.md` - This file
- `starter.py` - Template to get you started
- `solution_basic.py` - Basic version solution
- `solution_advanced.py` - Advanced version with all features

---

## 🚀 Next Steps

1. Read through this README completely
2. Try to implement the basic version yourself
3. Test your code thoroughly
4. Add advanced features
5. Compare with the solution
6. Customize and make it your own!

---

**Good luck and have fun coding! 🎮🐍**
