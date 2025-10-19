"""
Number Guessing Game - Complete Solution (Basic Version)
=========================================================

This is a complete, working solution for the number guessing game.
Try to build it yourself first before looking at this solution!

Features:
- Random number generation
- User input and validation
- Feedback on guesses
- Attempt tracking
- Clean, readable code with comments
"""

import random

# Game Constants
MIN_NUMBER = 1
MAX_NUMBER = 100


def display_welcome():
    """Display welcome message and game rules"""
    print("=" * 50)
    print(" " * 10 + "NUMBER GUESSING GAME" + " " * 10)
    print("=" * 50)
    print(f"\nI'm thinking of a number between {MIN_NUMBER} and {MAX_NUMBER}.")
    print("Can you guess it? Let's find out!\n")


def get_user_guess():
    """
    Get and validate user's guess
    
    Returns:
        int: The user's guess
    """
    while True:
        try:
            guess = int(input(f"Enter your guess ({MIN_NUMBER}-{MAX_NUMBER}): "))
            
            # Validate range
            if guess < MIN_NUMBER or guess > MAX_NUMBER:
                print(f"⚠️  Please enter a number between {MIN_NUMBER} and {MAX_NUMBER}!")
                continue
                
            return guess
            
        except ValueError:
            print("⚠️  Invalid input! Please enter a number.")


def check_guess(guess, secret_number, attempts):
    """
    Check if guess is correct and provide feedback
    
    Args:
        guess: The user's guess
        secret_number: The secret number to guess
        attempts: Current number of attempts
        
    Returns:
        bool: True if guess is correct, False otherwise
    """
    if guess > secret_number:
        difference = guess - secret_number
        print(f"📉 Too high! (Off by {difference})")
        return False
        
    elif guess < secret_number:
        difference = secret_number - guess
        print(f"📈 Too low! (Off by {difference})")
        return False
        
    else:
        # Correct guess!
        print(f"\n🎉 Congratulations! You guessed it!")
        print(f"✅ The number was {secret_number}")
        print(f"🎯 You won in {attempts} attempts!")
        
        # Give performance feedback
        if attempts == 1:
            print("🏆 AMAZING! First try!")
        elif attempts <= 5:
            print("⭐ Excellent! Great guessing!")
        elif attempts <= 10:
            print("👍 Good job!")
        else:
            print("✨ You got there in the end!")
            
        return True


def play_game():
    """Main game logic"""
    # Display welcome message
    display_welcome()
    
    # Generate secret number
    secret_number = random.randint(MIN_NUMBER, MAX_NUMBER)
    
    # Initialize game variables
    attempts = 0
    guessed = False
    
    # Main game loop
    while not guessed:
        # Get user's guess
        guess = get_user_guess()
        
        # Increment attempts
        attempts += 1
        
        # Check if guess is correct
        guessed = check_guess(guess, secret_number, attempts)
        
        # Add spacing for readability
        if not guessed:
            print()


def main():
    """Main program entry point"""
    # Play the game
    play_game()
    
    # Thank you message
    print("\n" + "=" * 50)
    print("Thanks for playing! 🎮")
    print("=" * 50)


# Run the program
if __name__ == "__main__":
    main()
