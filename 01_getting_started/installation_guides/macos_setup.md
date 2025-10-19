# 🍎 Python Installation Guide for macOS

This guide will walk you through installing Python on macOS and setting up your development environment.

## 📋 Prerequisites

- macOS 10.9 or later
- Administrator privileges
- Internet connection

## 🚀 Step-by-Step Installation

### Step 1: Check if Python is Already Installed

1. Open Terminal (Applications > Utilities > Terminal)
2. Type the following command and press Enter:
   ```bash
   python3 --version
   ```
3. If Python is installed, you'll see output like:
   ```
   Python 3.x.x
   ```
4. If not installed, proceed to the next steps

### Step 2: Download Python

1. Open your web browser and go to [python.org/downloads](https://www.python.org/downloads/)
2. Click on the yellow "Download Python" button (this will download the latest version for macOS)
3. The download will start automatically

### Step 3: Install Python

1. Locate the downloaded `.pkg` file (usually in your Downloads folder)
2. Double-click the file to start the installation
3. Follow the installation wizard:
   - Click "Continue"
   - Review the license agreement and click "Continue" then "Agree"
   - Click "Install"
   - Enter your password when prompted
   - Wait for the installation to complete

### Step 4: Verify Installation

1. Open Terminal
2. Type the following command and press Enter:
   ```bash
   python3 --version
   ```
3. You should see output similar to:
   ```
   Python 3.x.x
   ```

## ⚙️ Setting Up Your Development Environment

### Installing Visual Studio Code

1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download the macOS version
3. Open the downloaded `.zip` file
4. Drag Visual Studio Code to your Applications folder
5. Open Visual Studio Code from your Applications folder

### Installing Python Extension for VS Code

1. Open Visual Studio Code
2. Click on the Extensions icon in the sidebar (or press `Cmd+Shift+X`)
3. Search for "Python" (by Microsoft)
4. Click "Install"

## 🧪 Testing Your Setup

### Create Your First Python Program

1. Open Visual Studio Code
2. Create a new file (`Cmd+N`)
3. Save it as `hello.py` (`Cmd+S`)
4. Type the following code:
   ```python
   print("Hello, World!")
   print("Welcome to Python Programming!")
   ```
5. Save the file (`Cmd+S`)
6. Run the program in one of these ways:
   - Right-click in the editor and select "Run Python File in Terminal"
   - Press `Ctrl+F5`
   - Open terminal (`Ctrl+`` `) and type: `python3 hello.py`

You should see the output in the terminal:
```
Hello, World!
Welcome to Python Programming!
```

## 🛠 Common Issues and Solutions

### Issue 1: Command 'python3' Not Found

**Solution**:
1. Try using `python` instead of `python3`
2. If that doesn't work, add Python to your PATH:
   - Open Terminal
   - Type `nano ~/.bash_profile` (or `nano ~/.zshrc` if using zsh)
   - Add this line: `export PATH="/usr/local/bin/python3:$PATH"`
   - Save and exit (`Ctrl+X`, then `Y`, then `Enter`)
   - Restart Terminal

### Issue 2: Permission Error During Installation

**Solution**:
1. Make sure you're using an account with administrator privileges
2. If prompted, enter your password correctly

## 📚 Next Steps

Now that Python is installed, you're ready to start learning! Here's what to do next:

1. **Explore Python REPL**:
   - Open Terminal
   - Type `python3` and press Enter
   - Try simple commands like `print("Hello")` or `2 + 2`
   - Type `exit()` to quit

2. **Learn Basic Commands**:
   ```bash
   # Check Python version
   python3 --version
   
   # Run a Python script
   python3 script_name.py
   
   # Install packages
   pip3 install package_name
   ```

3. **Start with the Basics**:
   - Variables and data types
   - Basic operations
   - Input/output functions

## 🆘 Getting Help

If you encounter any issues:
1. Check the [Python Documentation](https://docs.python.org/3/)
2. Visit the [Python Discord](https://pythondiscord.com/)
3. Search for solutions on [Stack Overflow](https://stackoverflow.com/questions/tagged/python)

---

**Happy Coding!** 🐍