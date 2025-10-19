"""
Number Guessing Game - Starter Template
========================================

Complete the TODOs to create your number guessing game!

Requirements:
1. Generate a random number between 1 and 100
2. Ask user to guess the number
3. Provide feedback (too high/too low/correct)
4. Track number of attempts
5. End game when guessed correctly

Good luck! 🎯
"""

import random

# Game Constants
MIN_NUMBER = 1
MAX_NUMBER = 100


def display_welcome():
    """Display welcome message"""
    # TODO: Print a nice welcome message with game rules
    pass


def get_user_guess():
    """Get and validate user's guess"""
    # TODO: Get input from user
    # TODO: Convert to integer
    # TODO: Handle invalid input (Optional challenge)
    pass


def check_guess(guess, secret_number):
    """
    Check if guess is correct, too high, or too low
    
    Args:
        guess: The user's guess
        secret_number: The number to guess
        
    Returns:
        str: "high", "low", or "correct"
    """
    # TODO: Compare guess with secret_number
    # TODO: Return appropriate string
    pass


def play_game():
    """Main game logic"""
    # TODO: Display welcome
    
    # TODO: Generate random number
    
    # TODO: Initialize attempts counter
    
    # TODO: Create game loop (while not guessed)
    
    # TODO: Get user guess
    
    # TODO: Increment attempts
    
    # TODO: Check guess and provide feedback
    
    # TODO: Display final message with attempts count
    
    pass


def main():
    """Main program entry point"""
    # TODO: Call play_game()
    
    # TODO: (Optional) Ask if player wants to play again
    
    pass


# Run the program
if __name__ == "__main__":
    main()


# ============================================
# HINTS (Delete these when you're done!)
# ============================================

# Hint 1: Generating random number
# secret = random.randint(MIN_NUMBER, MAX_NUMBER)

# Hint 2: While loop structure
# guessed = False
# while not guessed:
#     # your code here

# Hint 3: Getting input
# guess = int(input("Enter your guess: "))

# Hint 4: Comparing numbers
# if guess > secret_number:
#     print("Too high!")
# elif guess < secret_number:
#     print("Too low!")
# else:
#     print("Correct!")
#     guessed = True
