# 🐍 Complete Python Installation Guide

This comprehensive guide covers Python installation for all major operating systems: Windows, macOS, and Linux. Follow these step-by-step instructions to set up Python and your development environment.

## 🖥️ Windows Installation

### Prerequisites
- Windows 7 or later (64-bit recommended)
- Administrator privileges
- Internet connection

### Step 1: Download Python
1. Open your web browser and go to [python.org](https://www.python.org/downloads/)
2. Click on the yellow "Download Python" button (this will download the latest version)
3. Alternatively, click "Downloads" in the navigation menu and select a specific version

### Step 2: Run the Installer
1. Locate the downloaded `.exe` file (usually in your Downloads folder)
2. Double-click the file to start the installation
3. **Important**: Check both of these boxes:
   - ✅ "Add Python to PATH" (This is crucial for running Python from the command line)
   - ✅ "Install launcher for all users (recommended)"

### Step 3: Choose Installation Type
You have two options:
- **Install Now**: Uses default settings (recommended for beginners)
- **Customize Installation**: Allows you to select specific features

For beginners, select "Install Now".

### Step 4: Complete Installation
1. The installer will automatically download and install Python
2. Wait for the installation to complete
3. You should see a "Setup was successful" message
4. Click "Close" to finish

### Step 5: Verify Installation
1. Open Command Prompt (Press `Win + R`, type `cmd`, and press Enter)
2. Type the following command and press Enter:
   ```bash
   python --version
   ```
3. You should see output similar to:
   ```
   Python 3.x.x
   ```

## 🍏 macOS Installation

### Prerequisites
- macOS 10.9 or later
- Internet connection

### Option 1: Using Python.org (Recommended)
1. Open your web browser and go to [python.org/downloads](https://www.python.org/downloads/)
2. Click on the yellow "Download Python" button
3. Once downloaded, open the `.pkg` file
4. Follow the installation wizard:
   - Click "Continue"
   - Click "Continue" again
   - Click "Continue" to read the license
   - Click "Agree"
   - Click "Install" (you may need to enter your password)
   - Wait for installation to complete
   - Click "Close"

### Option 2: Using Homebrew (For Advanced Users)
1. Install Homebrew if you haven't already:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python:
   ```bash
   brew install python
   ```

### Verify Installation
1. Open Terminal (Applications > Utilities > Terminal)
2. Type the following command and press Enter:
   ```bash
   python3 --version
   ```
3. You should see output similar to:
   ```
   Python 3.x.x
   ```

## 🐧 Linux Installation

### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python 3
sudo apt install python3

# Install pip (Python package manager)
sudo apt install python3-pip

# Verify installation
python3 --version
```

### Fedora
```bash
# Install Python 3
sudo dnf install python3

# Install pip
sudo dnf install python3-pip

# Verify installation
python3 --version
```

### CentOS/RHEL
```bash
# Install Python 3
sudo yum install python3

# Install pip
sudo yum install python3-pip

# Verify installation
python3 --version
```

### Arch Linux
```bash
# Install Python 3
sudo pacman -S python

# Install pip
sudo pacman -S python-pip

# Verify installation
python --version
```

## ⚙️ Setting Up Your Development Environment

### Installing Visual Studio Code (Recommended IDE)

#### Windows
1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download the Windows version
3. Run the installer and follow the setup wizard
4. During installation, make sure to check:
   - ✅ "Add to PATH" (required for command line integration)
   - ✅ "Register as default editor for supported file types"
   - ✅ "Add to explorer context menu"

#### macOS
1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download the macOS version
3. Open the downloaded `.zip` file
4. Drag Visual Studio Code to your Applications folder

#### Linux
1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download the appropriate version for your distribution (.deb for Ubuntu/Debian, .rpm for Fedora/CentOS)
3. Install using your package manager or by double-clicking the downloaded file

### Installing Python Extension for VS Code
1. Open Visual Studio Code
2. Click on the Extensions icon in the sidebar (or press `Ctrl+Shift+X` / `Cmd+Shift+X`)
3. Search for "Python" (by Microsoft)
4. Click "Install"

## 🧪 Testing Your Setup

### Create Your First Python Program

1. Open Visual Studio Code
2. Create a new file (`Ctrl+N` / `Cmd+N`)
3. Save it as `hello.py` (`Ctrl+S` / `Cmd+S`)
4. Type the following code:
   ```python
   print("Hello, World!")
   print("Welcome to Python Programming!")
   ```
5. Save the file (`Ctrl+S` / `Cmd+S`)
6. Run the program in one of these ways:
   - Right-click in the editor and select "Run Python File in Terminal"
   - Press `Ctrl+F5` / `Cmd+F5`
   - Open terminal (`Ctrl+`` ` / `Ctrl+Shift+`` `) and type: `python hello.py` (Windows) or `python3 hello.py` (macOS/Linux)

You should see the output in the terminal:
```
Hello, World!
Welcome to Python Programming!
```

## 🛠 Common Issues and Solutions

### Issue 1: "'python' is not recognized as an internal or external command" (Windows)
**Solution**:
1. Make sure you checked "Add Python to PATH" during installation
2. Restart your command prompt
3. If still not working, add Python to PATH manually:
   - Search for "Environment Variables" in Windows search
   - Click "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "System Variables", find and select "Path", then click "Edit"
   - Click "New" and add these paths:
     - `C:\Users\[YourUsername]\AppData\Local\Programs\Python\Python3x\`
     - `C:\Users\[YourUsername]\AppData\Local\Programs\Python\Python3x\Scripts\`
   - Replace `[YourUsername]` with your actual username and `3x` with your Python version

### Issue 2: "command not found: python3" (macOS/Linux)
**Solution**:
1. Try using `python` instead of `python3`
2. If that doesn't work, you may need to install Python or add it to your PATH

### Issue 3: Permission Error During Installation
**Solution**:
1. Right-click the Python installer (Windows)
2. Select "Run as administrator"
3. Proceed with installation

## 📚 Next Steps

Now that Python is installed, you're ready to start learning! Here's what to do next:

1. **Explore Python REPL**:
   - Open Command Prompt/Terminal
   - Type `python` (Windows) or `python3` (macOS/Linux) and press Enter
   - Try simple commands like `print("Hello")` or `2 + 2`
   - Type `exit()` to quit

2. **Learn Basic Commands**:
   ```bash
   # Check Python version
   python --version   # Windows
   python3 --version  # macOS/Linux
   
   # Run a Python script
   python script_name.py    # Windows
   python3 script_name.py   # macOS/Linux
   
   # Install packages
   pip install package_name    # Windows
   pip3 install package_name   # macOS/Linux
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