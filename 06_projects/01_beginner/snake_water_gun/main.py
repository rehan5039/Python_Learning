import random

def game_win(comp, you):
    # If two values are equal, declare a tie!
    if comp == you:
        return None

    # Check for all possibilities when computer chose s
    elif comp == 's':
        if you == 'w':
            return False
        elif you == 'g':
            return True

    # Check for all possibilities when computer chose w
    elif comp == 'w':
        if you == 'g':
            return False
        elif you == 's':
            return True

    # Check for all possibilities when computer chose g
    elif comp == 'g':
        if you == 's':
            return False
        elif you == 'w':
            return True

def play_game():
    print("Welcome to Snake, Water, Gun Game!")
    print("Computer's turn: Snake(s), Water(w), or Gun(g)?")
    
    # Computer's turn
    rand_no = random.randint(1, 3)
    if rand_no == 1:
        comp = 's'
    elif rand_no == 2:
        comp = 'w'
    else:
        comp = 'g'

    # Player's turn
    you = input("Your turn: Snake(s), Water(w), or Gun(g)? ").lower()
    
    # Validate input
    if you not in ['s', 'w', 'g']:
        print("Invalid input! Please enter 's', 'w', or 'g'.")
        return

    print(f"Computer chose: {comp}")
    print(f"You chose: {you}")

    # Determine the winner
    result = game_win(comp, you)

    if result is None:
        print("It's a tie!")
    elif result:
        print("You win!")
    else:
        print("You lose!")

def main():
    while True:
        play_game()
        play_again = input("Do you want to play again? (y/n): ").lower()
        if play_again != 'y':
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    main()