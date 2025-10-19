# 🔢 Guess The Number Game

A fun number guessing game where you try to guess a randomly generated number within a certain range. This project demonstrates random number generation, user input handling, and game loop implementation.

## 🎯 Game Rules

1. The computer generates a random number within a specified range
2. You have a limited number of attempts to guess the number
3. After each guess, you'll receive feedback if your guess was too high or too low
4. Win by guessing the correct number before running out of attempts

## 🚀 How to Play

1. Run the game: `python main.py`
2. Select a difficulty level:
   - **Easy**: Number between 1-50, 10 attempts
   - **Medium**: Number between 1-100, 10 attempts
   - **Hard**: Number between 1-200, 8 attempts
3. Enter your guess when prompted
4. Receive feedback on your guess
5. Continue until you guess correctly or run out of attempts
6. Choose to play again or exit

## 🧠 Learning Concepts

This project demonstrates:
- **Random number generation** using the `random` module
- **User input validation** and error handling
- **Game loop implementation** with while loops
- **Difficulty levels** with different parameters
- **Attempt tracking** and game state management
- **Conditional logic** for comparison and feedback

## 📁 Project Structure

```
guess_the_number/
├── main.py          # Main game logic
└── README.md        # This file
```

## 🎮 Sample Gameplay

```
Welcome to the Number Guessing Game!

Select difficulty level:
1. Easy (1-50, 10 attempts)
2. Medium (1-100, 10 attempts)
3. Hard (1-200, 8 attempts)
Enter your choice (1-3): 2

I'm thinking of a number between 1 and 100.
You have 10 attempts to guess the number.

Attempt 1: Enter your guess: 50
Too high! Try a lower number.
You have 9 attempts left.

Attempt 2: Enter your guess: 25
Too low! Try a higher number.
You have 8 attempts left.

Attempt 3: Enter your guess: 37
🎉 Congratulations! You guessed the number in 3 attempts!

Do you want to play again? (y/n): n
Thanks for playing!
```

## 🛠 Requirements

- Python 3.x

## 🏃‍♂️ Running the Game

```bash
python main.py
```

## 🎯 Educational Value

This project is perfect for beginners to practice:
1. **Random number generation** with `random.randint()`
2. **Exception handling** with try-except blocks
3. **User input validation** and sanitization
4. **Game state management** with variables
5. **Loop control** with while loops and break statements
6. **Function organization** for modular code
7. **Difficulty implementation** with parameterized functions

## 🤔 How It Works

1. **Difficulty Selection**: User chooses game parameters
2. **Number Generation**: Computer generates random number
3. **Game Loop**: Player guesses with feedback
4. **Validation**: Input is checked for validity
5. **Win/Loss Conditions**: Game ends when correct or out of attempts
6. **Replay Option**: User can play multiple rounds

## 📚 Concepts Covered

- **Random Module**: `random.randint()` for number generation
- **Exception Handling**: `try-except` for input validation
- **Functions**: Modular design with multiple functions
- **Loops**: `while` loop for game continuation
- **Conditionals**: Complex if-elif-else for game logic
- **Input/Output**: User interaction and feedback
- **Variables**: State tracking for attempts and numbers

## 🔧 Features

- **Multiple Difficulty Levels**: Easy, Medium, and Hard modes
- **Input Validation**: Handles non-numeric input gracefully
- **Attempt Tracking**: Shows remaining attempts
- **Immediate Feedback**: Tells if guess is too high or too low
- **Replay Functionality**: Play multiple rounds
- **Clear Instructions**: User-friendly prompts and messages

---

**Enjoy the game and happy coding!** 🔢