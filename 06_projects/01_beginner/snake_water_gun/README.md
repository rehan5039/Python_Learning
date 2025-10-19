# 🐍 Snake Water Gun Game

A classic hand game implementation in Python where you play against the computer. This project demonstrates basic programming concepts including conditionals, loops, and random number generation.

## 🎯 Game Rules

- **Snake (s) vs Water (w)**: Snake drinks water → Snake wins
- **Water (w) vs Gun (g)**: Gun sinks in water → Water wins
- **Gun (g) vs Snake (s)**: Gun shoots snake → Gun wins
- If both players choose the same option, it's a tie

## 🚀 How to Play

1. Run the game: `python main.py`
2. The computer will automatically choose Snake, Water, or Gun
3. You will be prompted to enter your choice:
   - `s` for Snake
   - `w` for Water
   - `g` for Gun
4. The game will display both choices and declare the winner
5. You can choose to play again or exit

## 🧠 Learning Concepts

This project demonstrates:
- **Random number generation** using the `random` module
- **Conditional statements** for game logic
- **Input validation** to handle user input
- **Functions** to organize code
- **Loops** for replay functionality
- **String comparison** for determining winners

## 📁 Project Structure

```
snake_water_gun/
├── main.py          # Main game logic
└── README.md        # This file
```

## 🎮 Sample Gameplay

```
Welcome to Snake, Water, Gun Game!
Computer's turn: Snake(s), Water(w), or Gun(g)?
Your turn: Snake(s), Water(w), or Gun(g)? s
Computer chose: w
You chose: s
You win!
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
1. **Basic syntax** and structure
2. **Control flow** with if-elif-else statements
3. **User interaction** with input/output
4. **Randomization** for computer moves
5. **Game logic** implementation
6. **Code organization** with functions

## 🤔 How It Works

1. **Computer Choice**: Randomly selects between snake, water, or gun
2. **Player Choice**: Takes input from the user
3. **Validation**: Ensures user input is valid
4. **Comparison**: Uses conditional logic to determine the winner
5. **Result**: Displays the outcome and asks to play again

## 📚 Concepts Covered

- **Random Module**: `random.randint()` for computer moves
- **Functions**: `game_win()` to determine winner, `play_game()` for game flow
- **Loops**: `while` loop for replay functionality
- **Conditionals**: Complex if-elif-else structure for game rules
- **Input/Output**: User interaction and feedback

---

**Enjoy the game and happy coding!** 🐍