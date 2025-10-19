# 🐧 Python Installation Guide for Linux

This guide will walk you through installing Python on Linux and setting up your development environment.

## 📋 Prerequisites

- Linux distribution (Ubuntu, Fedora, CentOS, etc.)
- Administrator privileges (sudo access)
- Internet connection

## 🚀 Installation Methods

### Method 1: Using Package Manager (Recommended)

#### For Ubuntu/Debian-based distributions:

1. Update package list:
   ```bash
   sudo apt update
   ```

2. Install Python 3:
   ```bash
   sudo apt install python3
   ```

3. Install pip (Python package manager):
   ```bash
   sudo apt install python3-pip
   ```

4. Verify installation:
   ```bash
   python3 --version
   pip3 --version
   ```

#### For Fedora:

1. Install Python 3:
   ```bash
   sudo dnf install python3
   ```

2. Install pip:
   ```bash
   sudo dnf install python3-pip
   ```

3. Verify installation:
   ```bash
   python3 --version
   pip3 --version
   ```

#### For CentOS/RHEL:

1. Install Python 3:
   ```bash
   sudo yum install python3
   ```

2. Install pip:
   ```bash
   sudo yum install python3-pip
   ```

3. Verify installation:
   ```bash
   python3 --version
   pip3 --version
   ```

### Method 2: Building from Source (Advanced)

1. Download Python source code from [python.org](https://www.python.org/downloads/source/)
2. Extract the archive:
   ```bash
   tar -xf Python-3.x.x.tgz
   ```
3. Navigate to the directory:
   ```bash
   cd Python-3.x.x
   ```
4. Configure the build:
   ```bash
   ./configure --enable-optimizations
   ```
5. Compile and install:
   ```bash
   make -j 8
   sudo make altinstall
   ```

## ⚙️ Setting Up Your Development Environment

### Installing Visual Studio Code

#### Using Snap (Ubuntu/Debian):
```bash
sudo snap install code --classic
```

#### Using Package Manager:

**Ubuntu/Debian:**
```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /usr/share/keyrings/
sudo sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code
```

**Fedora:**
```bash
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
sudo dnf config-manager --add-repo https://packages.microsoft.com/yumrepos/vscode
sudo dnf install code
```

### Installing Python Extension for VS Code

1. Open Visual Studio Code
2. Click on the Extensions icon in the sidebar (or press `Ctrl+Shift+X`)
3. Search for "Python" (by Microsoft)
4. Click "Install"

## 🧪 Testing Your Setup

### Create Your First Python Program

1. Open Visual Studio Code
2. Create a new file (`Ctrl+N`)
3. Save it as `hello.py` (`Ctrl+S`)
4. Type the following code:
   ```python
   print("Hello, World!")
   print("Welcome to Python Programming!")
   ```
5. Save the file (`Ctrl+S`)
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

### Issue 1: Command 'python' Not Found

**Solution**:
Create a symlink:
```bash
sudo ln -s /usr/bin/python3 /usr/bin/python
```

### Issue 2: Permission Denied When Installing Packages

**Solution**:
Use the `--user` flag:
```bash
pip3 install --user package_name
```

### Issue 3: Python Version Conflicts

**Solution**:
Use `python3` and `pip3` explicitly instead of `python` and `pip`:
```bash
python3 script.py
pip3 install package_name
```

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