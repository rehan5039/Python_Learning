import random

def play_game():
    """Main game function"""
    print("Welcome to the Number Guessing Game!")
    print("I'm thinking of a number between 1 and 100.")
    
    # Generate random number
    secret_number = random.randint(1, 100)
    attempts = 0
    max_attempts = 10
    
    print(f"You have {max_attempts} attempts to guess the number.")
    
    while attempts < max_attempts:
        try:
            # Get user input
            guess = int(input(f"\nAttempt {attempts + 1}: Enter your guess: "))
            attempts += 1
            
            # Check the guess
            if guess == secret_number:
                print(f"🎉 Congratulations! You guessed the number in {attempts} attempts!")
                return
            elif guess < secret_number:
                print("Too low! Try a higher number.")
            else:
                print("Too high! Try a lower number.")
            
            # Show remaining attempts
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"You have {remaining} attempts left.")
            
        except ValueError:
            print("Please enter a valid number!")
            attempts -= 1  # Don't count invalid input as an attempt
    
    # Game over
    print(f"\n😢 Game Over! You've used all {max_attempts} attempts.")
    print(f"The number was {secret_number}.")

def set_difficulty():
    """Set game difficulty level"""
    print("\nSelect difficulty level:")
    print("1. Easy (1-50, 10 attempts)")
    print("2. Medium (1-100, 10 attempts)")
    print("3. Hard (1-200, 8 attempts)")
    
    while True:
        try:
            choice = int(input("Enter your choice (1-3): "))
            if choice == 1:
                return 1, 50, 10
            elif choice == 2:
                return 1, 100, 10
            elif choice == 3:
                return 1, 200, 8
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number!")

def play_game_with_difficulty():
    """Play game with difficulty selection"""
    print("Welcome to the Number Guessing Game!")
    
    # Set difficulty
    min_num, max_num, max_attempts = set_difficulty()
    
    print(f"\nI'm thinking of a number between {min_num} and {max_num}.")
    print(f"You have {max_attempts} attempts to guess the number.")
    
    # Generate random number
    secret_number = random.randint(min_num, max_num)
    attempts = 0
    
    while attempts < max_attempts:
        try:
            # Get user input
            guess = int(input(f"\nAttempt {attempts + 1}: Enter your guess: "))
            attempts += 1
            
            # Check the guess
            if guess == secret_number:
                print(f"🎉 Congratulations! You guessed the number in {attempts} attempts!")
                return
            elif guess < secret_number:
                print("Too low! Try a higher number.")
            else:
                print("Too high! Try a lower number.")
            
            # Show remaining attempts
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"You have {remaining} attempts left.")
            
        except ValueError:
            print("Please enter a valid number!")
            attempts -= 1  # Don't count invalid input as an attempt
    
    # Game over
    print(f"\n😢 Game Over! You've used all {max_attempts} attempts.")
    print(f"The number was {secret_number}.")

def main():
    """Main function"""
    while True:
        play_game_with_difficulty()
        
        # Ask to play again
        play_again = input("\nDo you want to play again? (y/n): ").lower()
        if play_again != 'y':
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    main()